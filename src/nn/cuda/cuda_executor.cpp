/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_executor.h"

#include "cuda_buffers.h"
#include "cuda_execution_schedule.h"
#include "cuda_execution_tape.h"
#include "cuda_output_mapping.h"
#include "cuda_stage_executor.h"
#include "cuda_weight_buffers.h"
#include "cuda_workspace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

thread_local bool g_cuda_profile_suppressed = false;

std::string CudaErrorMessage(const char *op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

bool EnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] != '\0' && std::string(value) != "0";
}

bool EnvAnyFlagEnabled(std::initializer_list<const char *> names) {
  for (const char *name : names) {
    if (EnvFlagEnabled(name))
      return true;
  }
  return false;
}

int EnvIntOrDefault(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

int ReserveCudaProfileReportIndex() {
  if (g_cuda_profile_suppressed)
    return -1;
  if (!EnvFlagEnabled("METALFISH_CUDA_PROFILE"))
    return -1;
  const int limit = EnvIntOrDefault("METALFISH_CUDA_PROFILE_LIMIT", 1);
  if (limit <= 0)
    return -1;
  static std::atomic<int> reports{0};
  const int index = reports.fetch_add(1, std::memory_order_relaxed);
  return index < limit ? index : -1;
}

double MillisSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

void ReserveExecutionWorkspace(const CudaExecutionTape &tape,
                               CudaExecutionWorkspace &workspace) {
  for (const auto &binding : tape.Bindings()) {
    if (!tape.Reserve(workspace, binding)) {
      throw std::runtime_error("CUDA execution tape reserve returned null: " +
                               binding.name);
    }
  }
}

void PrepareExecutionWorkspace(const CudaExecutionTape &tape,
                               CudaExecutionWorkspace &workspace) {
  ReserveExecutionWorkspace(tape, workspace);
  workspace.Clear(workspace.Stream());
}

void CheckCudaSuccess(const char *op, cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(op, status));
}

bool CudaGraphExecutionRequested() {
  if (const char *value = std::getenv("METALFISH_CUDA_GRAPH");
      value && std::string(value) == "0") {
    return false;
  }
  if (const char *value = std::getenv("METALFISH_CUDA_GRAPH_EXECUTION");
      value && std::string(value) == "0") {
    return false;
  }
  return true;
}

bool CudaGraphExecutionCompatible() {
  return !EnvAnyFlagEnabled({
      "METALFISH_CUDA_PROFILE",
      "METALFISH_CUDA_TRACE_STAGE_OUTPUTS",
      "METALFISH_CUDA_TRACE_ATTENTION_INTERNALS",
      "METALFISH_CUDA_TRACE_DYNAMIC_PE_INTERNALS",
      "METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN",
      "METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN",
  });
}

void EndFailedCudaGraphCapture(cudaStream_t stream) {
  cudaGraph_t abandoned = nullptr;
  const cudaError_t status = cudaStreamEndCapture(stream, &abandoned);
  if (abandoned)
    cudaGraphDestroy(abandoned);
  if (status == cudaSuccess || status == cudaErrorStreamCaptureInvalidated)
    return;
  cudaGetLastError();
}

cudaError_t InstantiateCudaGraph(cudaGraphExec_t *graph_exec,
                                 cudaGraph_t graph) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
  return cudaGraphInstantiate(graph_exec, graph, 0);
#else
  return cudaGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
#endif
}

struct CudaGraphExecutionKey {
  int batch_size = 0;
  std::uint64_t workspace_generation = 0;
  std::uint64_t buffer_generation = 0;
  cudaStream_t stream = nullptr;
  std::uintptr_t input_masks = 0;
  std::uintptr_t input_values = 0;
  std::uintptr_t policy = 0;
  std::uintptr_t value = 0;
  std::uintptr_t moves_left = 0;
  std::uintptr_t raw_policy = 0;

  bool IsValid() const { return batch_size > 0 && stream != nullptr; }

  bool operator==(const CudaGraphExecutionKey &other) const {
    return batch_size == other.batch_size &&
           workspace_generation == other.workspace_generation &&
           buffer_generation == other.buffer_generation &&
           stream == other.stream && input_masks == other.input_masks &&
           input_values == other.input_values && policy == other.policy &&
           value == other.value && moves_left == other.moves_left &&
           raw_policy == other.raw_policy;
  }
};

std::uintptr_t DevicePtrKey(const void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr);
}

CudaGraphExecutionKey
MakeCudaGraphExecutionKey(int batch_size, std::uint64_t workspace_generation,
                          std::uint64_t buffer_generation, cudaStream_t stream,
                          const CudaInferenceBuffers &buffers) {
  return CudaGraphExecutionKey{
      batch_size,
      workspace_generation,
      buffer_generation,
      stream,
      DevicePtrKey(buffers.input_masks),
      DevicePtrKey(buffers.input_values),
      DevicePtrKey(buffers.policy),
      DevicePtrKey(buffers.value),
      DevicePtrKey(buffers.moves_left),
      DevicePtrKey(buffers.raw_policy),
  };
}

struct CudaGraphExecutionCache {
  CudaGraphExecutionKey key;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  CudaGraphExecutionCache() = default;
  CudaGraphExecutionCache(const CudaGraphExecutionCache &) = delete;
  CudaGraphExecutionCache &operator=(const CudaGraphExecutionCache &) = delete;

  ~CudaGraphExecutionCache() { Reset(); }

  bool Matches(const CudaGraphExecutionKey &candidate) const {
    return graph_exec && key == candidate;
  }

  void Reset() {
    if (graph_exec) {
      cudaGraphExecDestroy(graph_exec);
      graph_exec = nullptr;
    }
    if (graph) {
      cudaGraphDestroy(graph);
      graph = nullptr;
    }
    key = {};
  }
};

struct CudaProfileBucket {
  CudaExecutionScheduleKind kind = CudaExecutionScheduleKind::Unsupported;
  int count = 0;
  double millis = 0.0;
};

void AddCudaProfileBucket(std::vector<CudaProfileBucket> &buckets,
                          const CudaStageTimingRecord &record) {
  for (auto &bucket : buckets) {
    if (bucket.kind == record.kind) {
      ++bucket.count;
      bucket.millis += record.millis;
      return;
    }
  }
  buckets.push_back(CudaProfileBucket{record.kind, 1, record.millis});
}

void PrintCudaProfileReport(int report_index, int batch_size,
                            const CudaStageTimingCollector &timings,
                            double sequence_ms, double output_ms,
                            std::size_t workspace_bytes) {
  std::vector<CudaProfileBucket> buckets;
  for (const auto &record : timings.Records())
    AddCudaProfileBucket(buckets, record);
  std::sort(
      buckets.begin(), buckets.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.millis > rhs.millis; });

  std::vector<CudaStageTimingRecord> slowest = timings.Records();
  std::sort(
      slowest.begin(), slowest.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.millis > rhs.millis; });

  std::cerr << std::fixed << std::setprecision(3)
            << "CUDA profile report=" << report_index << " batch=" << batch_size
            << " sequence_ms=" << sequence_ms << " output_sync_ms=" << output_ms
            << " workspace_mb="
            << (static_cast<double>(workspace_bytes) / (1024.0 * 1024.0))
            << " stages=" << timings.Records().size() << '\n';

  std::cerr << "CUDA profile buckets:";
  for (const auto &bucket : buckets) {
    std::cerr << ' ' << CudaExecutionScheduleKindName(bucket.kind) << '='
              << bucket.millis << "ms/" << bucket.count;
  }
  std::cerr << '\n';

  std::cerr << "CUDA profile slowest:";
  const std::size_t limit = std::min<std::size_t>(slowest.size(), 8);
  for (std::size_t i = 0; i < limit; ++i) {
    std::cerr << ' ' << slowest[i].name << '=' << slowest[i].millis << "ms";
  }
  std::cerr << std::endl;
}

void UploadDeviceFloats(float *ptr, const std::vector<float> &host,
                        const char *name) {
  if (host.empty())
    return;
  if (!ptr) {
    throw std::runtime_error(std::string("CUDA output buffer is missing: ") +
                             name);
  }
  const cudaError_t status = cudaMemcpy(
      ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

class MissingCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &, const NetworkResolvedExecutionPlan &,
               const CudaWeightBuffers &, CudaInferenceBuffers &,
               CudaExecutionWorkspace &, int) override {
    throw std::runtime_error(
        "CUDA transformer executor is not implemented yet");
  }

  std::string Name() const override { return "missing"; }
};

class NullCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &plan,
               const NetworkResolvedExecutionPlan &, const CudaWeightBuffers &,
               CudaInferenceBuffers &buffers, CudaExecutionWorkspace &,
               int batch_size) override {
    std::vector<float> policy(plan.PolicyEntries(batch_size), 0.0f);
    std::vector<float> value(plan.ValueEntries(batch_size), 0.0f);
    std::vector<float> moves_left(plan.MovesLeftEntries(batch_size), 0.0f);
    std::vector<float> raw_policy(plan.RawPolicyEntries(batch_size), 0.0f);

    for (int b = 0; b < batch_size; ++b) {
      const size_t policy_offset = static_cast<size_t>(b) * plan.policy_outputs;
      policy[policy_offset] = 0.25f + static_cast<float>(b);
      policy[policy_offset + plan.policy_outputs - 1] =
          -0.75f - static_cast<float>(b);

      const size_t value_offset = static_cast<size_t>(b) * 3;
      value[value_offset + 0] = 0.70f;
      value[value_offset + 1] = 0.20f;
      value[value_offset + 2] = 0.10f + 0.05f * static_cast<float>(b);

      moves_left[static_cast<size_t>(b)] = 12.0f + static_cast<float>(b);
      if (!raw_policy.empty()) {
        const size_t raw_offset =
            static_cast<size_t>(b) * plan.raw_policy_outputs;
        raw_policy[raw_offset] = 3.0f + static_cast<float>(b);
      }
    }

    UploadDeviceFloats(buffers.policy, policy, "cudaMemcpy(policy)");
    UploadDeviceFloats(buffers.value, value, "cudaMemcpy(value)");
    UploadDeviceFloats(buffers.moves_left, moves_left,
                       "cudaMemcpy(moves_left)");
    UploadDeviceFloats(buffers.raw_policy, raw_policy,
                       "cudaMemcpy(raw_policy)");
  }

  std::string Name() const override { return "null-smoke"; }
};

class PlanSmokeCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &plan,
               const NetworkResolvedExecutionPlan &execution_plan,
               const CudaWeightBuffers &weights, CudaInferenceBuffers &buffers,
               CudaExecutionWorkspace &workspace, int batch_size) override {
    if (batch_size <= 0)
      throw std::runtime_error("CUDA plan smoke executor received empty batch");
    if (!buffers.input_values || !buffers.policy)
      throw std::runtime_error("CUDA plan smoke executor received no buffers");

    const auto tape =
        CreatePlanSmokeExecutionTape(plan, execution_plan, batch_size);
    PrepareExecutionWorkspace(tape, workspace);
    const auto schedule = CreateCudaExecutionSchedule(execution_plan);
    const auto stage_inputs =
        CreateCudaStageInputBindings(execution_plan, schedule);
    const auto sequence = ExecuteDenseActivationLayerNormSequence(
        execution_plan, weights, buffers.input_values, nullptr, nullptr, tape,
        workspace, batch_size, stage_inputs, schedule);
    CudaOutputMappingOptions options;
    options.allow_partial_policy_rows = true;
    options.allow_partial_raw_policy_rows = true;
    const auto mapping =
        CreateCudaOutputMapping(plan, execution_plan, schedule, options);
    CopyMappedOutputs(mapping, sequence, buffers, workspace, batch_size);
    workspace.Synchronize();
  }

  std::string Name() const override { return "plan-smoke"; }
};

class ResolvedCudaExecutor final : public CudaExecutor {
public:
  ResolvedCudaExecutor(CudaExecutionSchedule schedule,
                       CudaOutputMapping output_mapping)
      : schedule_(std::move(schedule)),
        output_mapping_(std::move(output_mapping)),
        graph_requested_(CudaGraphExecutionRequested()) {}

  void Execute(const NetworkTensorPlan &,
               const NetworkResolvedExecutionPlan &execution_plan,
               const CudaWeightBuffers &weights, CudaInferenceBuffers &buffers,
               CudaExecutionWorkspace &workspace, int batch_size) override {
    Validate(execution_plan, buffers, batch_size);

    if (GraphExecutionEnabled()) {
      ExecuteGraph(execution_plan, weights, buffers, workspace, batch_size);
      return;
    }

    ExecuteUncaptured(execution_plan, weights, buffers, workspace, batch_size,
                      true);
  }

  std::string Name() const override {
    if (!graph_requested_)
      return "resolved";
    if (!CudaGraphExecutionCompatible())
      return "resolved+graph-incompatible";
    if (graph_disabled_) {
      if (EnvFlagEnabled("METALFISH_CUDA_GRAPH_STATUS_DETAIL") &&
          !graph_disabled_reason_.empty()) {
        return "resolved+graph-fallback(" + graph_disabled_reason_ + ")";
      }
      return "resolved+graph-fallback";
    }
    if (graph_replay_count_ > 0)
      return "resolved+graph-replay";
    if (graph_capture_count_ > 0)
      return "resolved+graph-captured";
    if (graph_primed_key_.IsValid())
      return "resolved+graph-primed";
    return "resolved+graph";
  }

private:
  void Validate(const NetworkResolvedExecutionPlan &,
                const CudaInferenceBuffers &buffers, int batch_size) const {
    if (batch_size <= 0)
      throw std::runtime_error("CUDA resolved executor received empty batch");
    if (!buffers.input_masks || !buffers.input_values || !buffers.policy)
      throw std::runtime_error("CUDA resolved executor received no buffers");
    if (!schedule_.FullySupported()) {
      throw std::runtime_error(
          "CUDA resolved executor schedule is unsupported: " +
          schedule_.Summary());
    }
    if (!output_mapping_.ok()) {
      throw std::runtime_error(
          "CUDA resolved executor output mapping failed: " +
          output_mapping_.Summary());
    }
  }

  CudaDenseStageSequenceOutput RunSequence(
      const NetworkResolvedExecutionPlan &execution_plan,
      const CudaWeightBuffers &weights, const CudaInferenceBuffers &buffers,
      const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
      int batch_size, CudaStageTimingCollector *timing_collector) const {
    const auto stage_inputs =
        CreateCudaStageInputBindings(execution_plan, schedule_);
    return ExecuteDenseActivationLayerNormSequence(
        execution_plan, weights, buffers.input_values, buffers.input_masks,
        buffers.input_values, tape, workspace, batch_size, stage_inputs,
        schedule_, timing_collector);
  }

  void ExecuteUncaptured(const NetworkResolvedExecutionPlan &execution_plan,
                         const CudaWeightBuffers &weights,
                         CudaInferenceBuffers &buffers,
                         CudaExecutionWorkspace &workspace, int batch_size,
                         bool allow_profile) const {
    const auto tape = CreateResolvedExecutionTape(execution_plan, batch_size);
    PrepareExecutionWorkspace(tape, workspace);
    const int profile_index =
        allow_profile ? ReserveCudaProfileReportIndex() : -1;
    CudaStageTimingCollector timings;
    CudaStageTimingCollector *timing_collector =
        profile_index >= 0 ? &timings : nullptr;
    const auto sequence_start = std::chrono::steady_clock::now();
    const auto sequence = RunSequence(execution_plan, weights, buffers, tape,
                                      workspace, batch_size, timing_collector);
    const double sequence_ms =
        profile_index >= 0 ? MillisSince(sequence_start) : 0.0;
    const auto output_start = std::chrono::steady_clock::now();
    CopyMappedOutputs(output_mapping_, sequence, buffers, workspace,
                      batch_size);
    workspace.Synchronize();
    const double output_ms =
        profile_index >= 0 ? MillisSince(output_start) : 0.0;
    if (profile_index >= 0) {
      PrintCudaProfileReport(profile_index, batch_size, timings, sequence_ms,
                             output_ms, workspace.TotalBytes());
    }
  }

  bool GraphExecutionEnabled() const {
    return graph_requested_ && !graph_disabled_ &&
           CudaGraphExecutionCompatible();
  }

  void DisableGraphExecution(const std::string &reason) {
    graph_cache_.Reset();
    graph_primed_key_ = {};
    graph_disabled_ = true;
    graph_disabled_reason_ = reason;
    cudaGetLastError();
  }

  void ExecuteGraphFallback(const NetworkResolvedExecutionPlan &execution_plan,
                            const CudaWeightBuffers &weights,
                            CudaInferenceBuffers &buffers,
                            CudaExecutionWorkspace &workspace, int batch_size,
                            const std::string &reason) {
    DisableGraphExecution(reason);
    ExecuteUncaptured(execution_plan, weights, buffers, workspace, batch_size,
                      false);
  }

  void ExecuteCapturedWork(const NetworkResolvedExecutionPlan &execution_plan,
                           const CudaWeightBuffers &weights,
                           CudaInferenceBuffers &buffers,
                           CudaExecutionWorkspace &workspace, int batch_size,
                           const CudaExecutionTape &tape) const {
    PrepareExecutionWorkspace(tape, workspace);
    const auto sequence = RunSequence(execution_plan, weights, buffers, tape,
                                      workspace, batch_size, nullptr);
    CopyMappedOutputs(output_mapping_, sequence, buffers, workspace,
                      batch_size);
  }

  void ExecuteGraph(const NetworkResolvedExecutionPlan &execution_plan,
                    const CudaWeightBuffers &weights,
                    CudaInferenceBuffers &buffers,
                    CudaExecutionWorkspace &workspace, int batch_size) {
    const auto tape = CreateResolvedExecutionTape(execution_plan, batch_size);
    ReserveExecutionWorkspace(tape, workspace);
    cudaStream_t stream = workspace.Stream();
    const auto key =
        MakeCudaGraphExecutionKey(batch_size, workspace.Generation(),
                                  buffers.Generation(), stream, buffers);

    if (graph_cache_.Matches(key)) {
      const cudaError_t launch_status =
          cudaGraphLaunch(graph_cache_.graph_exec, stream);
      if (launch_status != cudaSuccess) {
        ExecuteGraphFallback(
            execution_plan, weights, buffers, workspace, batch_size,
            CudaErrorMessage("cudaGraphLaunch", launch_status));
        return;
      }
      try {
        workspace.Synchronize();
        ++graph_replay_count_;
      } catch (const std::exception &e) {
        ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                             batch_size, e.what());
      } catch (...) {
        ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                             batch_size, "cudaGraphLaunch failed");
      }
      return;
    }

    if (graph_cache_.graph_exec && !graph_cache_.Matches(key)) {
      graph_cache_.Reset();
      graph_primed_key_ = {};
    }

    if (!(graph_primed_key_ == key)) {
      ExecuteUncaptured(execution_plan, weights, buffers, workspace, batch_size,
                        false);
      graph_primed_key_ = key;
      return;
    }

    const cudaError_t capture_status =
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (capture_status != cudaSuccess) {
      ExecuteGraphFallback(
          execution_plan, weights, buffers, workspace, batch_size,
          CudaErrorMessage("cudaStreamBeginCapture", capture_status));
      return;
    }
    try {
      ExecuteCapturedWork(execution_plan, weights, buffers, workspace,
                          batch_size, tape);
    } catch (const std::exception &e) {
      EndFailedCudaGraphCapture(stream);
      ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                           batch_size, e.what());
      return;
    } catch (...) {
      EndFailedCudaGraphCapture(stream);
      ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                           batch_size, "CUDA graph capture work failed");
      return;
    }

    cudaGraph_t graph = nullptr;
    cudaError_t graph_status = cudaStreamEndCapture(stream, &graph);
    if (graph_status != cudaSuccess) {
      if (graph)
        cudaGraphDestroy(graph);
      ExecuteGraphFallback(
          execution_plan, weights, buffers, workspace, batch_size,
          CudaErrorMessage("cudaStreamEndCapture", graph_status));
      return;
    }

    cudaGraphExec_t graph_exec = nullptr;
    graph_status = InstantiateCudaGraph(&graph_exec, graph);
    if (graph_status != cudaSuccess) {
      if (graph)
        cudaGraphDestroy(graph);
      ExecuteGraphFallback(
          execution_plan, weights, buffers, workspace, batch_size,
          CudaErrorMessage("cudaGraphInstantiate", graph_status));
      return;
    }

    graph_cache_.Reset();
    graph_cache_.graph = graph;
    graph_cache_.graph_exec = graph_exec;
    graph_cache_.key = key;

    const cudaError_t launch_status =
        cudaGraphLaunch(graph_cache_.graph_exec, stream);
    if (launch_status != cudaSuccess) {
      ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                           batch_size,
                           CudaErrorMessage("cudaGraphLaunch", launch_status));
      return;
    }
    try {
      workspace.Synchronize();
      ++graph_capture_count_;
    } catch (const std::exception &e) {
      ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                           batch_size, e.what());
    } catch (...) {
      ExecuteGraphFallback(execution_plan, weights, buffers, workspace,
                           batch_size, "cudaGraphLaunch failed");
    }
  }

  CudaExecutionSchedule schedule_;
  CudaOutputMapping output_mapping_;
  bool graph_requested_ = false;
  bool graph_disabled_ = false;
  std::uint64_t graph_capture_count_ = 0;
  std::uint64_t graph_replay_count_ = 0;
  std::string graph_disabled_reason_;
  CudaGraphExecutionKey graph_primed_key_;
  CudaGraphExecutionCache graph_cache_;
};

} // namespace

CudaProfileSuppressionScope::CudaProfileSuppressionScope()
    : previous_(g_cuda_profile_suppressed) {
  g_cuda_profile_suppressed = true;
}

CudaProfileSuppressionScope::~CudaProfileSuppressionScope() {
  g_cuda_profile_suppressed = previous_;
}

std::unique_ptr<CudaExecutor> CreateMissingCudaExecutor() {
  return std::make_unique<MissingCudaExecutor>();
}

std::unique_ptr<CudaExecutor> CreateNullCudaExecutorForSmoke() {
  return std::make_unique<NullCudaExecutor>();
}

std::unique_ptr<CudaExecutor> CreatePlanSmokeCudaExecutor() {
  return std::make_unique<PlanSmokeCudaExecutor>();
}

std::unique_ptr<CudaExecutor>
CreateResolvedCudaExecutor(CudaExecutionSchedule schedule,
                           CudaOutputMapping output_mapping) {
  return std::make_unique<ResolvedCudaExecutor>(std::move(schedule),
                                                std::move(output_mapping));
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
