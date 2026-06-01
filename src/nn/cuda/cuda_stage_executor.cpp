/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_stage_executor.h"

#include "cuda_attention_plan.h"
#include "cuda_device_copy.h"
#include "cuda_execution_schedule.h"
#include "cuda_execution_tape.h"
#include "cuda_kernels.h"
#include "cuda_plan_analysis.h"
#include "cuda_stage_bindings.h"
#include "cuda_weight_buffers.h"
#include "cuda_workspace.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

std::string CudaErrorMessage(std::string_view op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

CudaActivationKind ActivationFromString(const std::string &activation) {
  if (activation == "relu_2")
    return CudaActivationKind::Relu2;
  if (activation == "tanh")
    return CudaActivationKind::Tanh;
  if (activation == "sigmoid")
    return CudaActivationKind::Sigmoid;
  if (activation == "swish")
    return CudaActivationKind::Swish;
  if (activation == "mish")
    return CudaActivationKind::Mish;
  if (activation == "selu")
    return CudaActivationKind::Selu;
  return CudaActivationKind::Relu;
}

struct DenseStageActivation {
  enum class Mode {
    Linear,
    Elementwise,
    Softmax,
  };

  Mode mode = Mode::Linear;
  CudaActivationKind kind = CudaActivationKind::Relu;
};

void RequireDenseTensors(const NetworkResolvedExecutionStep &dense) {
  if (dense.kind != NetworkExecutionOpKind::Dense || dense.tensors.size() < 2) {
    throw std::runtime_error("CUDA dense stage has missing tensors");
  }
}

void RequireConvolutionTensors(
    const NetworkResolvedExecutionStep &convolution) {
  if (convolution.kind != NetworkExecutionOpKind::Convolution ||
      convolution.tensors.size() < 2) {
    throw std::runtime_error("CUDA convolution stage has missing tensors");
  }
}

void RequireLayerNormTensors(const NetworkResolvedExecutionStep &norm) {
  if (norm.kind != NetworkExecutionOpKind::LayerNorm ||
      norm.tensors.size() < 2) {
    throw std::runtime_error("CUDA layernorm stage has missing tensors");
  }
}

void RequireGateTensors(const NetworkResolvedExecutionStep &gate) {
  if (gate.kind != NetworkExecutionOpKind::Gate || gate.tensors.empty()) {
    throw std::runtime_error("CUDA gate stage has missing tensors");
  }
}

void RequireFeedForwardTensors(const NetworkResolvedExecutionStep &ffn) {
  if (ffn.kind != NetworkExecutionOpKind::FeedForward ||
      ffn.tensors.size() < 4) {
    throw std::runtime_error("CUDA feed-forward stage has missing tensors");
  }
}

void RequirePolicyMapTensors(const NetworkResolvedExecutionStep &policy_map) {
  if (policy_map.kind != NetworkExecutionOpKind::PolicyMap ||
      policy_map.tensors.empty()) {
    throw std::runtime_error("CUDA policy-map stage has missing tensors");
  }
}

bool IsMultiplyGate(std::string_view name) {
  return name.find("mult_gate") != std::string_view::npos;
}

bool IsAddGate(std::string_view name) {
  return name.find("add_gate") != std::string_view::npos;
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.ends_with(suffix);
}

bool EnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] != '\0' && std::string(value) != "0";
}

bool EnvFlagOrDefault(const char *name, bool fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  return std::string(value) != "0";
}

int EnvIntOrDefault(const char *name, int fallback, int min_value,
                    int max_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  try {
    return std::clamp(std::stoi(value), min_value, max_value);
  } catch (...) {
    return fallback;
  }
}

float EnvFloatOrDefault(const char *name, float fallback, float min_value,
                        float max_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  char *end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (!end || *end != '\0' || !std::isfinite(parsed))
    return fallback;
  return std::clamp(parsed, min_value, max_value);
}

bool EnvSubstringMatches(const char *name, std::string_view value) {
  const char *filter = std::getenv(name);
  if (!filter || filter[0] == '\0')
    return true;
  return value.find(filter) != std::string_view::npos;
}

int ReserveCudaStageTraceReport(int batch_size) {
  if (!EnvFlagEnabled("METALFISH_CUDA_TRACE_STAGE_OUTPUTS"))
    return -1;
  const int target_batch =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_STAGE_BATCH", -1, -1, 4096);
  if (target_batch >= 0 && target_batch != batch_size)
    return -1;

  static std::atomic<int> run_counter{0};
  const int run = run_counter.fetch_add(1, std::memory_order_relaxed);
  const int skip =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_STAGE_SKIP", 0, 0, 1000000);
  const int limit =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_STAGE_LIMIT", 1, 1, 1000000);
  if (run < skip || run >= skip + limit)
    return -1;
  return run;
}

struct CudaTraceBaseline {
  std::vector<float> values;
  int rows = 0;
  int width = 0;
  int run = -1;
};

void CompareCudaTraceBuffer(int run, int stage_index, std::string_view name,
                            const std::vector<float> &host, int rows,
                            int width) {
  const int base_run =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_COMPARE_BASE_RUN", -1, -1, 1000000);
  if (base_run < 0 || run < base_run)
    return;

  const float min_delta = EnvFloatOrDefault(
      "METALFISH_CUDA_TRACE_COMPARE_MIN_DELTA", 1.0e-7f, 0.0f, 1.0e6f);
  std::string key;
  key.reserve(name.size() + 16);
  key.append(std::to_string(stage_index));
  key.push_back('|');
  key.append(name);

  static std::mutex mutex;
  static std::unordered_map<std::string, CudaTraceBaseline> baselines;
  std::lock_guard<std::mutex> lock(mutex);
  auto baseline_it = baselines.find(key);
  if (run == base_run || (base_run == 0 && baseline_it == baselines.end())) {
    baselines[key] = {host, rows, width, run};
    return;
  }

  baseline_it = baselines.find(key);
  if (baseline_it == baselines.end())
    return;
  const CudaTraceBaseline &baseline = baseline_it->second;
  const std::size_t count = std::min(host.size(), baseline.values.size());
  if (count == 0)
    return;

  double sum_abs_delta = 0.0;
  double sum_square_delta = 0.0;
  float max_abs_delta = 0.0f;
  std::size_t max_index = 0;
  std::size_t changed = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const float delta = std::fabs(host[i] - baseline.values[i]);
    sum_abs_delta += delta;
    sum_square_delta += static_cast<double>(delta) * delta;
    if (delta > min_delta)
      ++changed;
    if (delta > max_abs_delta) {
      max_abs_delta = delta;
      max_index = i;
    }
  }
  if (max_abs_delta < min_delta)
    return;

  const int row = width > 0 ? static_cast<int>(max_index / width) : 0;
  const int col = width > 0 ? static_cast<int>(max_index % width)
                            : static_cast<int>(max_index);
  std::ostringstream out;
  out << std::fixed << std::setprecision(9)
      << "CUDA_STAGE_TRACE_COMPARE run=" << run << " base_run=" << base_run
      << " stage=" << stage_index << " name=" << name << " rows=" << rows
      << " width=" << width << " sampled=" << count
      << " max_abs_delta=" << max_abs_delta
      << " mean_abs_delta=" << (sum_abs_delta / static_cast<double>(count))
      << " rms_delta="
      << std::sqrt(sum_square_delta / static_cast<double>(count))
      << " changed=" << changed << " max_index=" << max_index << " row=" << row
      << " col=" << col << " baseline=" << baseline.values[max_index]
      << " actual=" << host[max_index]
      << " baseline_actual_run=" << baseline.run
      << " baseline_rows=" << baseline.rows
      << " baseline_width=" << baseline.width;
  std::cerr << out.str() << std::endl;
}

void TraceCudaBufferOutput(int run, int stage_index, std::string_view name,
                           CudaExecutionScheduleKind kind, const float *buffer,
                           int rows, int width, int batch_size,
                           cudaStream_t stream) {
  if (run < 0 || !buffer || rows <= 0 || width <= 0)
    return;
  if (!EnvSubstringMatches("METALFISH_CUDA_TRACE_STAGE_FILTER", name))
    return;

  int traced_rows = rows;
  int traced_entry = -1;
  std::size_t entry_offset = 0;
  const int requested_entry =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_STAGE_ENTRY", -1, -1, 1000000);
  if (requested_entry >= 0 && batch_size > 0 && rows % batch_size == 0) {
    const int rows_per_entry = rows / batch_size;
    if (rows_per_entry > 0) {
      traced_entry = std::min(requested_entry, batch_size - 1);
      traced_rows = rows_per_entry;
      entry_offset = static_cast<std::size_t>(traced_entry) *
                     static_cast<std::size_t>(rows_per_entry) *
                     static_cast<std::size_t>(width);
    }
  }

  const std::size_t entries =
      static_cast<std::size_t>(traced_rows) * static_cast<std::size_t>(width);
  const int max_entries =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_STAGE_MAX_FLOATS", 0, 0, 100000000);
  const std::size_t sampled_entries =
      max_entries > 0 ? std::min<std::size_t>(entries, max_entries) : entries;
  std::vector<float> host(sampled_entries);
  const cudaError_t copy_status = cudaMemcpyAsync(
      host.data(), buffer + entry_offset, sampled_entries * sizeof(float),
      cudaMemcpyDeviceToHost, stream);
  if (copy_status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("cudaMemcpyAsync(stage_trace)", copy_status));
  }
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("cudaStreamSynchronize(stage_trace)", sync_status));
  }

  double sum = 0.0;
  double abs_sum = 0.0;
  double weighted_sum = 0.0;
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  std::size_t max_index = 0;
  int nonfinite = 0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    const float value = host[i];
    if (!std::isfinite(value)) {
      ++nonfinite;
      continue;
    }
    sum += value;
    abs_sum += std::fabs(static_cast<double>(value));
    weighted_sum += value * static_cast<double>((i % 997) + 1);
    if (value < min_value)
      min_value = value;
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }
  if (host.empty()) {
    min_value = 0.0f;
    max_value = 0.0f;
  }
  CompareCudaTraceBuffer(run, stage_index, name, host, traced_rows, width);

  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "CUDA_STAGE_TRACE run=" << run
      << " stage=" << stage_index
      << " kind=" << CudaExecutionScheduleKindName(kind) << " name=" << name
      << " rows=" << traced_rows << " width=" << width
      << " source_rows=" << rows << " batch=" << batch_size;
  if (traced_entry >= 0)
    out << " entry=" << traced_entry;
  out << " entries=" << entries << " sampled=" << sampled_entries
      << " sum=" << sum << " abs_sum=" << abs_sum
      << " weighted_sum=" << weighted_sum << " min=" << min_value
      << " max=" << max_value << " max_index=" << max_index
      << " nonfinite=" << nonfinite;
  const int sample_count = std::min<int>(3, static_cast<int>(host.size()));
  if (sample_count > 0) {
    out << " first=[";
    for (int i = 0; i < sample_count; ++i) {
      if (i != 0)
        out << ',';
      out << host[static_cast<std::size_t>(i)];
    }
    out << ']';
  }
  std::cerr << out.str() << std::endl;
}

void TraceCudaStageOutput(int run, int stage_index, std::string_view name,
                          CudaExecutionScheduleKind kind,
                          const CudaDenseStageOutput &stage, int batch_size,
                          cudaStream_t stream) {
  TraceCudaBufferOutput(run, stage_index, name, kind, stage.output, stage.rows,
                        stage.output_width, batch_size, stream);
}

void TraceCudaAttentionBuffer(int run, int stage_index, int sub_index,
                              std::string_view name,
                              CudaExecutionScheduleKind kind,
                              const float *buffer, int rows, int width,
                              int batch_size, cudaStream_t stream) {
  if (!EnvFlagEnabled("METALFISH_CUDA_TRACE_ATTENTION_INTERNALS"))
    return;
  TraceCudaBufferOutput(run, stage_index * 100 + sub_index, name, kind, buffer,
                        rows, width, batch_size, stream);
}

void TraceCudaDynamicPositionBuffer(int run, int stage_index, int sub_index,
                                    std::string_view name,
                                    CudaExecutionScheduleKind kind,
                                    const float *buffer, int rows, int width,
                                    int batch_size, cudaStream_t stream) {
  if (!EnvFlagEnabled("METALFISH_CUDA_TRACE_DYNAMIC_PE_INTERNALS"))
    return;
  TraceCudaBufferOutput(run, stage_index * 100 + sub_index, name, kind, buffer,
                        rows, width, batch_size, stream);
}

DenseStageActivation ActivationFromName(const std::string &activation) {
  if (activation.empty())
    return {};
  if (activation == "softmax")
    return {DenseStageActivation::Mode::Softmax, CudaActivationKind::Relu};
  return {DenseStageActivation::Mode::Elementwise,
          ActivationFromString(activation)};
}

DenseStageActivation
DenseStageActivationForName(const NetworkResolvedExecutionPlan &execution_plan,
                            std::string_view name) {
  return ActivationFromName(
      NetworkDenseStageActivationName(execution_plan, name));
}

void CreateTimingEvent(cudaEvent_t *event) {
  const cudaError_t status = cudaEventCreateWithFlags(event, cudaEventDefault);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaEventCreate", status));
}

class CudaStageTimer {
public:
  CudaStageTimer(CudaStageTimingCollector *collector, std::string name,
                 CudaExecutionScheduleKind kind, cudaStream_t stream)
      : collector_(collector), name_(std::move(name)), kind_(kind),
        stream_(stream) {
    if (!collector_)
      return;
    CreateTimingEvent(&start_);
    CreateTimingEvent(&stop_);
    const cudaError_t status = cudaEventRecord(start_, stream_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventRecord(start)", status));
  }

  CudaStageTimer(const CudaStageTimer &) = delete;
  CudaStageTimer &operator=(const CudaStageTimer &) = delete;

  ~CudaStageTimer() {
    if (start_)
      cudaEventDestroy(start_);
    if (stop_)
      cudaEventDestroy(stop_);
  }

  void Stop() {
    if (!collector_ || stopped_)
      return;
    cudaError_t status = cudaEventRecord(stop_, stream_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventRecord(stop)", status));
    status = cudaEventSynchronize(stop_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventSynchronize", status));
    float millis = 0.0f;
    status = cudaEventElapsedTime(&millis, start_, stop_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventElapsedTime", status));
    collector_->Add(name_, kind_, millis);
    stopped_ = true;
  }

private:
  CudaStageTimingCollector *collector_ = nullptr;
  std::string name_;
  CudaExecutionScheduleKind kind_ = CudaExecutionScheduleKind::Unsupported;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool stopped_ = false;
};

bool IsDynamicPositionPreprocessName(std::string_view name) {
  return name == "body.input_embedding_preprocess";
}

bool IsStaticPositionEmbeddingStage(const NetworkResolvedExecutionPlan &plan,
                                    const NetworkResolvedExecutionStep &step) {
  return plan.format.input_embedding == INPUT_EMBEDDING_PE_MAP &&
         step.kind == NetworkExecutionOpKind::Dense &&
         step.name == "body.input_embedding";
}

bool ConvolutionUsesPackedInput(std::string_view name) {
  return name == "body.input";
}

std::string ResidualBlockName(std::string_view name) {
  if (EndsWith(name, ".conv1"))
    return std::string(name.substr(0, name.size() - 6));
  if (EndsWith(name, ".conv2"))
    return std::string(name.substr(0, name.size() - 6));
  return {};
}

bool IsResidualSqueezeExciteFor(const NetworkResolvedExecutionStep &step,
                                std::string_view block_name) {
  return step.kind == NetworkExecutionOpKind::Dense &&
         step.name == std::string(block_name) + ".se";
}

bool ConvolutionAppliesActivation(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &step) {
  const std::string conv_policy_output =
      "policy." + execution_plan.policy_head + ".policy";
  return !(execution_plan.format.conv_policy &&
           step.name == conv_policy_output);
}

const NetworkResolvedTensorRef &
RequireTensorSuffix(const NetworkResolvedExecutionStep &step,
                    std::string_view suffix) {
  if (const auto *tensor = FindNetworkTensorSuffix(step, suffix))
    return *tensor;
  throw std::runtime_error("CUDA stage is missing tensor " +
                           std::string(suffix) + " in " + step.name);
}

CudaDeviceTensorView TensorBySuffix(const NetworkResolvedExecutionStep &step,
                                    const CudaWeightBuffers &weights,
                                    std::string_view suffix) {
  const auto &ref = RequireTensorSuffix(step, suffix);
  return weights.TensorAt(ref.inventory_index);
}

void RequireTapeShape(const CudaExecutionBufferBinding &binding, int rows,
                      int width, std::string_view role) {
  if (binding.rows != rows || binding.width != width ||
      binding.entries != static_cast<std::size_t>(rows) * width) {
    throw std::runtime_error("CUDA " + std::string(role) +
                             " tape size mismatch");
  }
}

std::string ShapeString(int rows, int width) {
  return std::to_string(rows) + "x" + std::to_string(width);
}

std::string BindingShapeString(const CudaExecutionBufferBinding &binding) {
  return ShapeString(binding.rows, binding.width);
}

int DenseStageInputWidth(const NetworkResolvedExecutionStep &dense) {
  if (dense.kind != NetworkExecutionOpKind::Dense || dense.tensors.empty() ||
      dense.tensors[0].dims.size() != 2) {
    return 0;
  }
  return static_cast<int>(dense.tensors[0].dims[1]);
}

void MaybeFlattenSquareRowsForDenseStage(
    const NetworkResolvedExecutionStep &step, CudaExecutionScheduleKind kind,
    int batch_size, int &stage_input_rows, int &stage_input_width) {
  if (!IsCudaDenseScheduleEntry(kind) ||
      IsDynamicPositionPreprocessName(step.name) || batch_size <= 0 ||
      stage_input_rows <= 0 || stage_input_width <= 0 ||
      stage_input_rows % batch_size != 0) {
    return;
  }

  const int expected_width = DenseStageInputWidth(step);
  if (expected_width <= 0 || expected_width == stage_input_width)
    return;

  const int rows_per_batch = stage_input_rows / batch_size;
  if (expected_width == rows_per_batch * stage_input_width) {
    stage_input_rows = batch_size;
    stage_input_width = expected_width;
  }
}

struct CudaFeedForwardTensors {
  CudaDeviceTensorView dense1_weight;
  CudaDeviceTensorView dense1_bias;
  CudaDeviceTensorView dense2_weight;
  CudaDeviceTensorView dense2_bias;
  int input_width = 0;
  int hidden_width = 0;
  int output_width = 0;
};

struct CudaSmolgenStageOutput {
  float *compressed = nullptr;
  float *dense1 = nullptr;
  float *activation1 = nullptr;
  float *norm1 = nullptr;
  float *dense2 = nullptr;
  float *activation2 = nullptr;
  float *norm2 = nullptr;
  float *global_bias = nullptr;
};

CudaFeedForwardTensors
ResolveFeedForwardTensors(const NetworkResolvedExecutionStep &ffn,
                          const CudaWeightBuffers &weights) {
  RequireFeedForwardTensors(ffn);
  CudaFeedForwardTensors resolved;
  resolved.dense1_weight = weights.TensorAt(ffn.tensors[0].inventory_index);
  resolved.dense1_bias = weights.TensorAt(ffn.tensors[1].inventory_index);
  resolved.dense2_weight = weights.TensorAt(ffn.tensors[2].inventory_index);
  resolved.dense2_bias = weights.TensorAt(ffn.tensors[3].inventory_index);
  if (resolved.dense1_weight.dims.size() != 2 ||
      resolved.dense1_bias.dims.size() != 1 ||
      resolved.dense2_weight.dims.size() != 2 ||
      resolved.dense2_bias.dims.size() != 1) {
    throw std::runtime_error("CUDA feed-forward tensor shape is invalid");
  }

  resolved.hidden_width = static_cast<int>(resolved.dense1_weight.dims[0]);
  resolved.input_width = static_cast<int>(resolved.dense1_weight.dims[1]);
  resolved.output_width = static_cast<int>(resolved.dense2_weight.dims[0]);
  const int dense2_input_width =
      static_cast<int>(resolved.dense2_weight.dims[1]);
  if (resolved.input_width <= 0 || resolved.hidden_width <= 0 ||
      resolved.output_width <= 0 ||
      dense2_input_width != resolved.hidden_width ||
      resolved.dense1_bias.elements !=
          static_cast<std::size_t>(resolved.hidden_width) ||
      resolved.dense2_bias.elements !=
          static_cast<std::size_t>(resolved.output_width)) {
    throw std::runtime_error("CUDA feed-forward tensor dimensions mismatch");
  }
  return resolved;
}

float FeedForwardResidualScale(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  return NetworkFeedForwardResidualScale(execution_plan, stage_name);
}

float FeedForwardLayerNormEpsilon(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  return NetworkFeedForwardLayerNormEpsilon(execution_plan, stage_name);
}

float DenseLayerNormEpsilon(const NetworkResolvedExecutionPlan &execution_plan,
                            std::string_view stage_name) {
  return NetworkDenseLayerNormEpsilon(execution_plan, stage_name);
}

float AttentionLayerNormEpsilon(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  return NetworkAttentionLayerNormEpsilon(execution_plan, stage_name);
}

const NetworkResolvedExecutionStep *
FindStep(const NetworkResolvedExecutionPlan &execution_plan,
         std::string_view name) {
  for (const auto &step : execution_plan.steps) {
    if (step.name == name)
      return &step;
  }
  return nullptr;
}

} // namespace

void CopyDeviceFloatRows(float *dst, int dst_stride, const float *src,
                         int src_stride, int rows, int width,
                         std::string_view name, cudaStream_t stream) {
  if (rows <= 0 || width <= 0)
    return;
  if (!dst || !src)
    throw std::runtime_error("CUDA row copy missing buffer: " +
                             std::string(name));
  if (dst_stride < width || src_stride < width)
    throw std::runtime_error("CUDA row copy stride too small: " +
                             std::string(name));

  cudaError_t status = cudaSuccess;
  if (stream) {
    status = cudaMemcpy2DAsync(
        dst, static_cast<std::size_t>(dst_stride) * sizeof(float), src,
        static_cast<std::size_t>(src_stride) * sizeof(float),
        static_cast<std::size_t>(width) * sizeof(float), rows,
        cudaMemcpyDeviceToDevice, stream);
  } else {
    status =
        cudaMemcpy2D(dst, static_cast<std::size_t>(dst_stride) * sizeof(float),
                     src, static_cast<std::size_t>(src_stride) * sizeof(float),
                     static_cast<std::size_t>(width) * sizeof(float), rows,
                     cudaMemcpyDeviceToDevice);
  }
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

const CudaDenseStageOutput *
CudaDenseStageSequenceOutput::FindStage(std::string_view name) const {
  for (const auto &stage : stages) {
    if (stage.first == name)
      return &stage.second;
  }
  return nullptr;
}

void CudaStageTimingCollector::Add(std::string name,
                                   CudaExecutionScheduleKind kind,
                                   float millis) {
  records_.push_back(CudaStageTimingRecord{std::move(name), kind, millis});
}

CudaDenseStageOutput ExecuteConvolutionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &convolution,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, int input_rows, int input_width) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA convolution stage received empty batch");
  RequireConvolutionTensors(convolution);

  const auto conv_weight = TensorBySuffix(convolution, weights, ".weights");
  const auto conv_bias = TensorBySuffix(convolution, weights, ".biases");
  if (conv_weight.dims.size() != 4 || conv_bias.dims.size() != 1) {
    throw std::runtime_error("CUDA convolution tensor shape is invalid");
  }

  const int output_channels = static_cast<int>(conv_weight.dims[0]);
  const int input_channels = static_cast<int>(conv_weight.dims[1]);
  const int kernel_size = static_cast<int>(conv_weight.dims[2]);
  const int kernel_width = static_cast<int>(conv_weight.dims[3]);
  const int squares = kCudaAttentionSquares;
  if (output_channels <= 0 || input_channels <= 0 || kernel_size <= 0 ||
      kernel_size != kernel_width ||
      conv_bias.elements != static_cast<std::size_t>(output_channels)) {
    throw std::runtime_error("CUDA convolution tensor dimensions mismatch");
  }

  const bool uses_packed_input = ConvolutionUsesPackedInput(convolution.name);
  float *stage_input = const_cast<float *>(input);
  if (uses_packed_input) {
    if (!input_masks || !input_values) {
      throw std::runtime_error(
          "CUDA convolution input stage requires packed input buffers");
    }
    if (execution_plan.tensors.input_planes != input_channels ||
        execution_plan.tensors.input_squares != squares) {
      throw std::runtime_error("CUDA convolution input tensor plan mismatch");
    }
    const auto &expanded_binding =
        tape.RequireName(convolution.name + ".expanded");
    RequireTapeShape(expanded_binding, batch_size * input_channels, squares,
                     "convolution input expansion");
    stage_input = tape.Reserve(workspace, expanded_binding);
    LaunchExpandPackedInputPlanesNchwKernel(
        input_masks, input_values, stage_input, batch_size, input_channels,
        squares, workspace.Stream());
  } else {
    if (!stage_input)
      throw std::runtime_error("CUDA convolution stage input is missing");
    if (input_rows != batch_size * input_channels || input_width != squares) {
      throw std::runtime_error("CUDA convolution stage input shape mismatch");
    }
  }

  const auto &conv_binding =
      tape.RequireName(convolution.name + ".convolution");
  RequireTapeShape(conv_binding, batch_size * output_channels, squares,
                   "convolution output");

  CudaDenseStageOutput output;
  output.expanded_input = uses_packed_input ? stage_input : nullptr;
  output.convolution = tape.Reserve(workspace, conv_binding);
  output.output = output.convolution;
  output.input_width = squares;
  output.output_width = squares;
  output.rows = batch_size * output_channels;

  LaunchConvolution2DKernel(
      stage_input, conv_weight.data, conv_bias.data, output.convolution,
      batch_size, squares, input_channels, output_channels, kernel_size,
      ActivationFromString(
          execution_plan.format.activations.default_activation),
      ConvolutionAppliesActivation(execution_plan, convolution),
      workspace.Stream());
  return output;
}

CudaDenseStageOutput ExecuteResidualConvolutionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &conv1,
    const NetworkResolvedExecutionStep &conv2,
    const NetworkResolvedExecutionStep *squeeze_excite,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, int input_rows, int input_width) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA residual convolution received empty batch");
  if (!input)
    throw std::runtime_error("CUDA residual convolution input is missing");
  RequireConvolutionTensors(conv1);
  RequireConvolutionTensors(conv2);

  const std::string block_name = ResidualBlockName(conv1.name);
  if (block_name.empty() || conv2.name != block_name + ".conv2") {
    throw std::runtime_error("CUDA residual convolution block names mismatch");
  }
  if (input_rows <= 0 || input_width != kCudaAttentionSquares ||
      input_rows % batch_size != 0) {
    throw std::runtime_error("CUDA residual convolution input shape mismatch");
  }

  const int skip_channels = input_rows / batch_size;
  const auto conv1_weight = TensorBySuffix(conv1, weights, ".weights");
  const auto conv1_bias = TensorBySuffix(conv1, weights, ".biases");
  const auto conv2_weight = TensorBySuffix(conv2, weights, ".weights");
  const auto conv2_bias = TensorBySuffix(conv2, weights, ".biases");
  CudaDeviceTensorView se_w1;
  CudaDeviceTensorView se_b1;
  CudaDeviceTensorView se_w2;
  CudaDeviceTensorView se_b2;
  const bool has_se = squeeze_excite != nullptr;
  if (has_se) {
    if (!IsResidualSqueezeExciteFor(*squeeze_excite, block_name))
      throw std::runtime_error(
          "CUDA residual convolution squeeze-excite name mismatch");
    se_w1 = TensorBySuffix(*squeeze_excite, weights, ".w1");
    se_b1 = TensorBySuffix(*squeeze_excite, weights, ".b1");
    se_w2 = TensorBySuffix(*squeeze_excite, weights, ".w2");
    se_b2 = TensorBySuffix(*squeeze_excite, weights, ".b2");
  }
  if (conv1_weight.dims.size() != 4 || conv1_bias.dims.size() != 1 ||
      conv2_weight.dims.size() != 4 || conv2_bias.dims.size() != 1) {
    throw std::runtime_error(
        "CUDA residual convolution tensor shape is invalid");
  }

  const int conv1_channels = static_cast<int>(conv1_weight.dims[0]);
  const int conv1_input_channels = static_cast<int>(conv1_weight.dims[1]);
  const int conv1_kernel = static_cast<int>(conv1_weight.dims[2]);
  const int conv1_kernel_width = static_cast<int>(conv1_weight.dims[3]);
  const int conv2_channels = static_cast<int>(conv2_weight.dims[0]);
  const int conv2_input_channels = static_cast<int>(conv2_weight.dims[1]);
  const int conv2_kernel = static_cast<int>(conv2_weight.dims[2]);
  const int conv2_kernel_width = static_cast<int>(conv2_weight.dims[3]);
  if (conv1_channels <= 0 || conv1_input_channels != skip_channels ||
      conv1_kernel <= 0 || conv1_kernel != conv1_kernel_width ||
      conv1_bias.elements != static_cast<std::size_t>(conv1_channels) ||
      conv2_channels != skip_channels ||
      conv2_input_channels != conv1_channels || conv2_kernel <= 0 ||
      conv2_kernel != conv2_kernel_width ||
      conv2_bias.elements != static_cast<std::size_t>(conv2_channels)) {
    throw std::runtime_error(
        "CUDA residual convolution tensor dimensions mismatch");
  }
  int se_hidden = 0;
  int se_output_channels = 0;
  if (has_se) {
    if (se_w1.dims.size() != 2 || se_b1.dims.size() != 1 ||
        se_w2.dims.size() != 2 || se_b2.dims.size() != 1) {
      throw std::runtime_error(
          "CUDA residual squeeze-excite tensor shape is invalid");
    }
    se_hidden = static_cast<int>(se_w1.dims[0]);
    const int se_input_channels = static_cast<int>(se_w1.dims[1]);
    se_output_channels = static_cast<int>(se_w2.dims[0]);
    const int se_fc2_input = static_cast<int>(se_w2.dims[1]);
    if (se_hidden <= 0 || se_input_channels != skip_channels ||
        se_b1.elements != static_cast<std::size_t>(se_hidden) ||
        se_output_channels != skip_channels * 2 || se_fc2_input != se_hidden ||
        se_b2.elements != static_cast<std::size_t>(se_output_channels)) {
      throw std::runtime_error(
          "CUDA residual squeeze-excite tensor dimensions mismatch");
    }
  }

  const auto &conv1_binding = tape.RequireName(conv1.name + ".convolution");
  const auto &conv2_binding = tape.RequireName(conv2.name + ".convolution");
  const auto &residual_binding = tape.RequireName(block_name + ".residual");
  const auto &activation_binding = tape.RequireName(block_name + ".activation");
  const CudaExecutionBufferBinding *se_pool_binding = nullptr;
  const CudaExecutionBufferBinding *se_fc1_binding = nullptr;
  const CudaExecutionBufferBinding *se_activation_binding = nullptr;
  const CudaExecutionBufferBinding *se_fc2_binding = nullptr;
  if (has_se) {
    se_pool_binding = &tape.RequireName(squeeze_excite->name + ".pool");
    se_fc1_binding = &tape.RequireName(squeeze_excite->name + ".fc1.dense");
    se_activation_binding =
        &tape.RequireName(squeeze_excite->name + ".fc1.activation");
    se_fc2_binding = &tape.RequireName(squeeze_excite->name + ".fc2.dense");
  }
  RequireTapeShape(conv1_binding, batch_size * conv1_channels,
                   kCudaAttentionSquares, "residual conv1 output");
  RequireTapeShape(conv2_binding, batch_size * conv2_channels,
                   kCudaAttentionSquares, "residual conv2 output");
  RequireTapeShape(residual_binding, input_rows, input_width,
                   "residual convolution add");
  RequireTapeShape(activation_binding, input_rows, input_width,
                   "residual convolution activation");
  if (has_se) {
    RequireTapeShape(*se_pool_binding, batch_size, skip_channels,
                     "residual squeeze-excite pool");
    RequireTapeShape(*se_fc1_binding, batch_size, se_hidden,
                     "residual squeeze-excite fc1");
    RequireTapeShape(*se_activation_binding, batch_size, se_hidden,
                     "residual squeeze-excite activation");
    RequireTapeShape(*se_fc2_binding, batch_size, se_output_channels,
                     "residual squeeze-excite fc2");
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, conv1_binding);
  output.convolution = tape.Reserve(workspace, conv2_binding);
  output.residual = tape.Reserve(workspace, residual_binding);
  if (has_se) {
    output.squeeze_excite_pool = tape.Reserve(workspace, *se_pool_binding);
    output.squeeze_excite_hidden = tape.Reserve(workspace, *se_fc1_binding);
    output.squeeze_excite_activation =
        tape.Reserve(workspace, *se_activation_binding);
    output.squeeze_excite_output = tape.Reserve(workspace, *se_fc2_binding);
  }
  output.activation = tape.Reserve(workspace, activation_binding);
  output.output = output.activation;
  output.input_width = input_width;
  output.output_width = input_width;
  output.rows = input_rows;

  const CudaActivationKind activation = ActivationFromString(
      execution_plan.format.activations.default_activation);
  cudaStream_t stream = workspace.Stream();
  LaunchConvolution2DKernel(input, conv1_weight.data, conv1_bias.data,
                            output.dense, batch_size, kCudaAttentionSquares,
                            conv1_input_channels, conv1_channels, conv1_kernel,
                            activation, true, stream);
  LaunchConvolution2DKernel(
      output.dense, conv2_weight.data, conv2_bias.data, output.convolution,
      batch_size, kCudaAttentionSquares, conv2_input_channels, conv2_channels,
      conv2_kernel, activation, false, stream);
  if (has_se) {
    LaunchGlobalAveragePoolNchwKernel(output.convolution,
                                      output.squeeze_excite_pool, batch_size,
                                      skip_channels, input_width, stream);
    LaunchDenseAffineKernel(output.squeeze_excite_pool, se_w1.data, se_b1.data,
                            output.squeeze_excite_hidden, batch_size,
                            skip_channels, se_hidden, stream);
    LaunchActivationKernel(output.squeeze_excite_hidden,
                           output.squeeze_excite_activation,
                           batch_size * se_hidden, activation, stream);
    LaunchDenseAffineKernel(output.squeeze_excite_activation, se_w2.data,
                            se_b2.data, output.squeeze_excite_output,
                            batch_size, se_hidden, se_output_channels, stream);
    LaunchSqueezeExciteResidualKernel(
        input, output.convolution, output.squeeze_excite_output,
        output.residual, output.activation, batch_size, skip_channels,
        input_width, activation, stream);
  } else {
    LaunchResidualAddKernel(input, output.convolution, output.residual,
                            input_rows, input_width, 1.0f, stream);
    LaunchActivationKernel(output.residual, output.activation,
                           input_rows * input_width, activation, stream);
  }
  return output;
}

CudaDenseStageOutput
ExecuteDenseActivationStage(const NetworkResolvedExecutionPlan &execution_plan,
                            const NetworkResolvedExecutionStep &dense,
                            const CudaWeightBuffers &weights,
                            const float *input, const CudaExecutionTape &tape,
                            CudaExecutionWorkspace &workspace, int rows) {
  if (rows <= 0)
    throw std::runtime_error("CUDA dense stage received empty batch");
  if (!input)
    throw std::runtime_error("CUDA dense stage input is missing");
  RequireDenseTensors(dense);

  const auto dense_weight = weights.TensorAt(dense.tensors[0].inventory_index);
  const auto dense_bias = weights.TensorAt(dense.tensors[1].inventory_index);
  if (dense_weight.dims.size() != 2 || dense_bias.dims.size() != 1) {
    throw std::runtime_error("CUDA dense stage tensor shape is invalid");
  }

  const int output_width = static_cast<int>(dense_weight.dims[0]);
  const int input_width = static_cast<int>(dense_weight.dims[1]);
  if (input_width <= 0 || output_width <= 0 ||
      dense_bias.elements != static_cast<std::size_t>(output_width)) {
    throw std::runtime_error("CUDA dense stage tensor dimensions mismatch");
  }

  const auto &dense_binding = tape.RequireName(dense.name + ".dense");
  const auto &activation_binding = tape.RequireName(dense.name + ".activation");
  const std::size_t entries =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(output_width);
  if (dense_binding.rows != rows || dense_binding.width != output_width ||
      dense_binding.entries != entries || activation_binding.rows != rows ||
      activation_binding.width != output_width ||
      activation_binding.entries != entries) {
    throw std::runtime_error(
        "CUDA dense stage tape size mismatch: " + dense.name +
        " binding=" + BindingShapeString(dense_binding) +
        " activation=" + BindingShapeString(activation_binding) +
        " actual=" + ShapeString(rows, output_width));
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense_binding);
  output.input_width = input_width;
  output.output_width = output_width;
  output.rows = rows;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, dense_weight.data, dense_bias.data,
                          output.dense, rows, input_width, output_width,
                          stream);
  const DenseStageActivation activation =
      DenseStageActivationForName(execution_plan, dense.name);
  if (activation.mode == DenseStageActivation::Mode::Linear) {
    output.activation = output.dense;
    output.output = output.dense;
  } else {
    output.activation = tape.Reserve(workspace, activation_binding);
    output.output = output.activation;
    if (activation.mode == DenseStageActivation::Mode::Softmax) {
      LaunchAttentionSoftmaxKernel(output.dense, output.activation, rows,
                                   output_width, stream);
    } else {
      LaunchActivationKernel(output.dense, output.activation,
                             static_cast<int>(entries), activation.kind,
                             stream);
    }
  }
  return output;
}

CudaDenseStageOutput ExecuteDenseActivationLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int rows) {
  RequireLayerNormTensors(norm);

  CudaDenseStageOutput output = ExecuteDenseActivationStage(
      execution_plan, dense, weights, input, tape, workspace, rows);
  const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
  const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
  if (gamma.dims.size() != 1 || beta.dims.size() != 1) {
    throw std::runtime_error("CUDA layernorm stage tensor shape is invalid");
  }

  if (gamma.elements != static_cast<std::size_t>(output.output_width) ||
      beta.elements != static_cast<std::size_t>(output.output_width)) {
    throw std::runtime_error("CUDA layernorm stage tensor dimensions mismatch");
  }

  const auto &norm_binding = tape.RequireName(norm.name + ".normalized");
  const std::size_t entries = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(output.output_width);
  if (norm_binding.entries != entries) {
    throw std::runtime_error("CUDA layernorm stage tape size mismatch");
  }

  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchLayerNormKernel(output.activation, gamma.data, beta.data,
                        output.normalized, rows, output.output_width,
                        DenseLayerNormEpsilon(execution_plan, norm.name),
                        stream);
  return output;
}

CudaDenseStageOutput ExecuteDynamicPositionEncodingStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const CudaWeightBuffers &weights,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA dynamic PE stage received empty batch");
  if (!input_masks || !input_values)
    throw std::runtime_error("CUDA dynamic PE stage input is missing");
  if (!IsDynamicPositionPreprocessName(dense.name))
    throw std::runtime_error("CUDA dynamic PE stage name is invalid");
  RequireDenseTensors(dense);

  const auto dense_weight = weights.TensorAt(dense.tensors[0].inventory_index);
  const auto dense_bias = weights.TensorAt(dense.tensors[1].inventory_index);
  if (dense_weight.dims.size() != 2 || dense_bias.dims.size() != 1)
    throw std::runtime_error("CUDA dynamic PE dense tensor shape is invalid");

  const auto geometry =
      ResolveDynamicPositionEncodingGeometry(execution_plan, dense);
  if (dense_weight.dims[0] !=
          static_cast<std::uint32_t>(geometry.dense_output_width) ||
      dense_weight.dims[1] !=
          static_cast<std::uint32_t>(geometry.dense_input_width) ||
      dense_bias.elements !=
          static_cast<std::size_t>(geometry.dense_output_width)) {
    throw std::runtime_error("CUDA dynamic PE resolved shape mismatch");
  }
  const int square_rows = batch_size * geometry.input_squares;

  const auto &expanded_binding = tape.RequireName(dense.name + ".expanded");
  const auto &position_input_binding =
      tape.RequireName(dense.name + ".position_input");
  const auto &dense_binding = tape.RequireName(dense.name + ".dense");
  const auto &concat_binding = tape.RequireName(dense.name + ".concat");
  RequireTapeShape(expanded_binding, square_rows, geometry.input_planes,
                   "dynamic PE expanded input");
  RequireTapeShape(position_input_binding, batch_size,
                   geometry.dense_input_width, "dynamic PE dense input");
  RequireTapeShape(dense_binding, batch_size, geometry.dense_output_width,
                   "dynamic PE dense output");
  RequireTapeShape(concat_binding, square_rows, geometry.concat_width,
                   "dynamic PE concat output");

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense_binding);
  output.expanded_input = tape.Reserve(workspace, expanded_binding);
  output.position_input = tape.Reserve(workspace, position_input_binding);
  output.normalized = tape.Reserve(workspace, concat_binding);
  output.output = output.normalized;
  output.input_width = geometry.dense_input_width;
  output.output_width = geometry.concat_width;
  output.rows = square_rows;

  cudaStream_t stream = workspace.Stream();
  LaunchExpandPackedInputPlanesWithPositionInputKernel(
      input_masks, input_values, output.expanded_input, output.position_input,
      batch_size, geometry.input_planes, geometry.position_planes,
      geometry.input_squares, stream);
  LaunchDenseAffineKernel(output.position_input, dense_weight.data,
                          dense_bias.data, output.dense, batch_size,
                          geometry.dense_input_width,
                          geometry.dense_output_width, stream);
  LaunchDynamicPositionEncodingConcatKernel(
      output.expanded_input, output.dense, output.output, batch_size,
      geometry.input_planes, geometry.position_width, geometry.input_squares,
      stream);
  return output;
}

float *PrepareStaticPositionEncodingInput(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const std::uint64_t *input_masks,
    const float *input_values, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA static PE stage received empty batch");
  if (!input_masks || !input_values)
    throw std::runtime_error("CUDA static PE stage input is missing");
  if (!IsStaticPositionEmbeddingStage(execution_plan, dense))
    throw std::runtime_error("CUDA static PE stage name is invalid");
  RequireDenseTensors(dense);

  const auto &dense_ref = dense.tensors[0];
  if (dense_ref.dims.size() != 2)
    throw std::runtime_error("CUDA static PE dense tensor shape is invalid");
  const auto geometry =
      ResolveStaticPositionEncodingGeometry(execution_plan, dense);

  const int square_rows = batch_size * geometry.input_squares;
  const auto &concat_binding =
      tape.RequireName(dense.name + ".static_pe_concat");
  RequireTapeShape(concat_binding, square_rows, geometry.concat_width,
                   "static PE concat output");

  float *output = tape.Reserve(workspace, concat_binding);
  LaunchStaticPositionEncodingConcatKernel(
      input_masks, input_values, output, batch_size, geometry.input_planes,
      geometry.position_width, geometry.input_squares, workspace.Stream());
  return output;
}

CudaDenseStageOutput ExecuteGateStage(const NetworkResolvedExecutionStep &gate,
                                      const CudaWeightBuffers &weights,
                                      const float *input, int input_width,
                                      const CudaExecutionTape &tape,
                                      CudaExecutionWorkspace &workspace,
                                      int rows) {
  if (rows <= 0)
    throw std::runtime_error("CUDA gate stage received empty batch");
  if (!input || input_width <= 0)
    throw std::runtime_error("CUDA gate stage input is missing");
  RequireGateTensors(gate);

  const auto &gate_binding = tape.RequireName(gate.name + ".gated");
  const std::size_t entries =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(input_width);
  if (gate_binding.rows != rows || gate_binding.width != input_width ||
      gate_binding.entries != entries) {
    std::ostringstream out;
    out << "CUDA gate stage tape size mismatch: binding=" << gate_binding.rows
        << "x" << gate_binding.width << ", actual=" << rows << "x"
        << input_width;
    throw std::runtime_error(out.str());
  }

  CudaDenseStageOutput output;
  output.gated = tape.Reserve(workspace, gate_binding);
  output.output = output.gated;
  output.input_width = input_width;
  output.output_width = input_width;
  output.rows = rows;

  const float *source = input;
  cudaStream_t stream = workspace.Stream();
  bool launched = false;
  for (const auto &tensor : gate.tensors) {
    const auto gate_weights = weights.TensorAt(tensor.inventory_index);
    int gate_rows = 0;
    if (gate_weights.elements == static_cast<std::size_t>(input_width)) {
      gate_rows = 1;
    } else if (gate_weights.elements % static_cast<std::size_t>(input_width) ==
               0) {
      gate_rows = static_cast<int>(gate_weights.elements /
                                   static_cast<std::size_t>(input_width));
    }
    if (gate_rows <= 0 || rows % gate_rows != 0) {
      throw std::runtime_error("CUDA gate stage tensor dimensions mismatch");
    }
    if (IsMultiplyGate(tensor.name)) {
      LaunchGateKernel(source, gate_weights.data, output.gated, rows,
                       input_width, gate_rows, CudaGateKind::Multiply, stream);
    } else if (IsAddGate(tensor.name)) {
      LaunchGateKernel(source, gate_weights.data, output.gated, rows,
                       input_width, gate_rows, CudaGateKind::Add, stream);
    } else {
      throw std::runtime_error("CUDA gate stage tensor is unknown: " +
                               tensor.name);
    }
    source = output.gated;
    launched = true;
  }
  if (!launched)
    throw std::runtime_error("CUDA gate stage launched no operations");
  return output;
}

static CudaDenseStageOutput
ExecuteFeedForwardStageImpl(const NetworkResolvedExecutionPlan &execution_plan,
                            const NetworkResolvedExecutionStep &ffn,
                            const CudaFeedForwardTensors &tensors,
                            const float *input, const CudaExecutionTape &tape,
                            CudaExecutionWorkspace &workspace, int rows,
                            bool defer_dense2_bias) {
  if (rows <= 0)
    throw std::runtime_error("CUDA feed-forward stage received empty batch");
  if (!input)
    throw std::runtime_error("CUDA feed-forward stage input is missing");

  const auto &dense1_binding = tape.RequireName(ffn.name + ".dense1");
  const auto &activation_binding = tape.RequireName(ffn.name + ".activation");
  const auto &dense2_binding = tape.RequireName(ffn.name + ".dense2");
  const std::size_t hidden_entries =
      static_cast<std::size_t>(rows) *
      static_cast<std::size_t>(tensors.hidden_width);
  const std::size_t output_entries =
      static_cast<std::size_t>(rows) *
      static_cast<std::size_t>(tensors.output_width);
  if (dense1_binding.entries != hidden_entries ||
      activation_binding.entries != hidden_entries ||
      dense2_binding.entries != output_entries) {
    throw std::runtime_error("CUDA feed-forward stage tape size mismatch");
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense1_binding);
  output.activation = tape.Reserve(workspace, activation_binding);
  output.feed_forward = tape.Reserve(workspace, dense2_binding);
  output.output = output.feed_forward;
  output.input_width = tensors.input_width;
  output.output_width = tensors.output_width;
  output.rows = rows;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, tensors.dense1_weight.data, nullptr,
                          output.dense, rows, tensors.input_width,
                          tensors.hidden_width, stream);
  LaunchBiasActivationKernel(
      output.dense, tensors.dense1_bias.data, output.activation, rows,
      tensors.hidden_width,
      ActivationFromString(execution_plan.format.activations.ffn_activation),
      stream);
  const float *dense2_bias =
      defer_dense2_bias ? nullptr : tensors.dense2_bias.data;
  LaunchDenseAffineKernel(output.activation, tensors.dense2_weight.data,
                          dense2_bias, output.feed_forward, rows,
                          tensors.hidden_width, tensors.output_width, stream);
  return output;
}

CudaDenseStageOutput
ExecuteFeedForwardStage(const NetworkResolvedExecutionPlan &execution_plan,
                        const NetworkResolvedExecutionStep &ffn,
                        const CudaWeightBuffers &weights, const float *input,
                        const CudaExecutionTape &tape,
                        CudaExecutionWorkspace &workspace, int rows) {
  const auto tensors = ResolveFeedForwardTensors(ffn, weights);
  return ExecuteFeedForwardStageImpl(execution_plan, ffn, tensors, input, tape,
                                     workspace, rows, false);
}

CudaDenseStageOutput ExecuteFeedForwardLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &ffn,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int rows) {
  RequireLayerNormTensors(norm);

  const auto tensors = ResolveFeedForwardTensors(ffn, weights);
  CudaDenseStageOutput output = ExecuteFeedForwardStageImpl(
      execution_plan, ffn, tensors, input, tape, workspace, rows, true);
  if (output.input_width != output.output_width) {
    throw std::runtime_error("CUDA feed-forward residual width mismatch");
  }

  const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
  const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
  if (gamma.dims.size() != 1 || beta.dims.size() != 1 ||
      gamma.elements != static_cast<std::size_t>(output.output_width) ||
      beta.elements != static_cast<std::size_t>(output.output_width)) {
    throw std::runtime_error(
        "CUDA feed-forward layernorm tensor dimensions mismatch");
  }

  const auto &residual_binding = tape.RequireName(norm.name + ".residual");
  const auto &norm_binding = tape.RequireName(norm.name + ".normalized");
  const std::size_t entries = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(output.output_width);
  if (residual_binding.entries != entries || norm_binding.entries != entries) {
    throw std::runtime_error("CUDA feed-forward layernorm tape size mismatch");
  }

  output.residual = tape.Reserve(workspace, residual_binding);
  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchResidualBiasLayerNormKernel(
      input, output.feed_forward, tensors.dense2_bias.data, gamma.data,
      beta.data, output.feed_forward, output.residual, output.normalized, rows,
      output.output_width, FeedForwardResidualScale(execution_plan, ffn.name),
      FeedForwardLayerNormEpsilon(execution_plan, ffn.name), stream);
  return output;
}

CudaDenseStageOutput
ExecutePolicyMapStage(const NetworkResolvedExecutionPlan &execution_plan,
                      const NetworkResolvedExecutionStep &policy_map,
                      const CudaWeightBuffers &weights,
                      const CudaDenseStageSequenceOutput &sequence,
                      const CudaExecutionTape &tape,
                      CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA policy-map stage received empty batch");
  if (!execution_plan.format.attention_policy &&
      !execution_plan.format.conv_policy) {
    throw std::runtime_error("CUDA policy-map stage requires attention or "
                             "convolution policy format");
  }

  const std::string suffix = ".policy_map";
  if (!EndsWith(policy_map.name, suffix)) {
    throw std::runtime_error("CUDA policy-map stage name is invalid");
  }
  const std::string policy_prefix =
      policy_map.name.substr(0, policy_map.name.size() - suffix.size());

  if (execution_plan.format.conv_policy) {
    const CudaDenseStageOutput *raw_stage =
        sequence.FindStage(policy_prefix + ".policy");
    if (!raw_stage || !raw_stage->output) {
      throw std::runtime_error("CUDA convolution policy-map source is missing");
    }
    constexpr int kConvPolicyChannels =
        kNetworkConvPolicyScratch / kPackedInputSquareCount;
    if (raw_stage->rows != batch_size * kConvPolicyChannels ||
        raw_stage->output_width != kPackedInputSquareCount) {
      throw std::runtime_error(
          "CUDA convolution policy-map source dimensions mismatch");
    }

    const auto &mapped_binding = tape.RequireName(policy_map.name + ".mapped");
    RequireTapeShape(mapped_binding, batch_size, kNetworkPolicyOutputs,
                     "convolution policy mapped logits");

    CudaDenseStageOutput output;
    output.activation = tape.Reserve(workspace, mapped_binding);
    output.output = output.activation;
    output.input_width = kNetworkConvPolicyScratch;
    output.output_width = kNetworkPolicyOutputs;
    output.rows = batch_size;

    LaunchConvolutionPolicyMapKernel(raw_stage->output, output.output,
                                     batch_size, workspace.Stream());
    return output;
  }

  RequirePolicyMapTensors(policy_map);
  const CudaDenseStageOutput *query_stage =
      sequence.FindStage(policy_prefix + ".dense2");
  const CudaDenseStageOutput *key_stage =
      sequence.FindStage(policy_prefix + ".dense3");
  if (!query_stage || !query_stage->output || !key_stage ||
      !key_stage->output) {
    throw std::runtime_error("CUDA policy-map Q/K stages are missing");
  }
  if (query_stage->output_width <= 0 ||
      query_stage->output_width != key_stage->output_width) {
    throw std::runtime_error("CUDA policy-map Q/K dimensions mismatch");
  }

  const auto promotion_weights =
      weights.TensorAt(policy_map.tensors[0].inventory_index);
  const int channels = query_stage->output_width;
  if (promotion_weights.elements != static_cast<std::size_t>(4 * channels)) {
    throw std::runtime_error(
        "CUDA policy-map promotion tensor dimensions mismatch");
  }

  const auto &raw_binding = tape.RequireName(policy_map.name + ".raw");
  const auto &mapped_binding = tape.RequireName(policy_map.name + ".mapped");
  RequireTapeShape(raw_binding, batch_size, kNetworkAttentionPolicyScratch,
                   "attention policy raw map");
  RequireTapeShape(mapped_binding, batch_size, kNetworkPolicyOutputs,
                   "attention policy mapped logits");

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, raw_binding);
  output.activation = tape.Reserve(workspace, mapped_binding);
  output.output = output.activation;
  output.input_width = channels;
  output.output_width = kNetworkPolicyOutputs;
  output.rows = batch_size;

  LaunchAttentionPolicyMapKernel(
      query_stage->output, key_stage->output, promotion_weights.data,
      output.dense, output.output, batch_size, channels, workspace.Stream());
  return output;
}

CudaAttentionProjectionOutput ExecuteAttentionInputProjectionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention projection received empty batch");
  if (!input)
    throw std::runtime_error("CUDA attention projection input is missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention projection index is out of range");

  const auto &step = execution_plan.steps[attention_step_index];
  const auto attention = ResolveCudaAttentionStagePlan(
      execution_plan, attention_step_index,
      NetworkAttentionHeadCount(execution_plan, step.name));
  const int rows = batch_size * attention.squares;

  const auto q_weight = TensorBySuffix(step, weights, ".q_w");
  const auto q_bias = TensorBySuffix(step, weights, ".q_b");
  const auto k_weight = TensorBySuffix(step, weights, ".k_w");
  const auto k_bias = TensorBySuffix(step, weights, ".k_b");
  const auto v_weight = TensorBySuffix(step, weights, ".v_w");
  const auto v_bias = TensorBySuffix(step, weights, ".v_b");
  const auto &q_binding = tape.RequireName(step.name + ".q");
  const auto &k_binding = tape.RequireName(step.name + ".k");
  const auto &v_binding = tape.RequireName(step.name + ".v");
  RequireTapeShape(q_binding, rows, attention.qkv_width, "query");
  RequireTapeShape(k_binding, rows, attention.qkv_width, "key");
  RequireTapeShape(v_binding, rows, attention.qkv_width, "value");

  CudaAttentionProjectionOutput output;
  output.query = tape.Reserve(workspace, q_binding);
  output.key = tape.Reserve(workspace, k_binding);
  output.value = tape.Reserve(workspace, v_binding);
  output.rows = rows;
  output.input_width = attention.input_width;
  output.qkv_width = attention.qkv_width;
  output.output_width = attention.output_width;
  output.heads = attention.heads;
  output.head_depth = attention.head_depth;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, q_weight.data, q_bias.data, output.query, rows,
                          attention.input_width, attention.qkv_width, stream);
  LaunchDenseAffineKernel(input, k_weight.data, k_bias.data, output.key, rows,
                          attention.input_width, attention.qkv_width, stream);
  LaunchDenseAffineKernel(input, v_weight.data, v_bias.data, output.value, rows,
                          attention.input_width, attention.qkv_width, stream);
  return output;
}

CudaAttentionProjectionOutput ExecuteAttentionOutputProjectionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaWeightBuffers &weights,
    const float *context, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size,
    bool defer_projection_bias) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention output received empty batch");
  if (!context)
    throw std::runtime_error("CUDA attention output context is missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention output index is out of range");

  const auto &step = execution_plan.steps[attention_step_index];
  const auto attention = ResolveCudaAttentionStagePlan(
      execution_plan, attention_step_index,
      NetworkAttentionHeadCount(execution_plan, step.name));
  const int rows = batch_size * attention.squares;
  const auto dense_weight = TensorBySuffix(step, weights, ".dense_w");
  const auto dense_bias = TensorBySuffix(step, weights, ".dense_b");
  const auto &projection_binding = tape.RequireName(step.name + ".projection");
  RequireTapeShape(projection_binding, rows, attention.output_width,
                   "attention output projection");

  CudaAttentionProjectionOutput output;
  output.projection = tape.Reserve(workspace, projection_binding);
  output.rows = rows;
  output.input_width = attention.input_width;
  output.qkv_width = attention.qkv_width;
  output.output_width = attention.output_width;
  output.heads = attention.heads;
  output.head_depth = attention.head_depth;
  output.projection_bias = defer_projection_bias ? dense_bias.data : nullptr;

  LaunchDenseAffineKernel(context, dense_weight.data,
                          defer_projection_bias ? nullptr : dense_bias.data,
                          output.projection, rows, attention.qkv_width,
                          attention.output_width, workspace.Stream());
  return output;
}

CudaSmolgenStageOutput ExecuteAttentionSmolgenStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaAttentionStagePlan &attention,
    const CudaWeightBuffers &weights, const float *parent,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size) {
  if (!attention.smolgen.present)
    return {};
  if (!attention.smolgen.has_global_positional_weights) {
    throw std::runtime_error(
        "CUDA attention smolgen requires global positional weights");
  }
  if (!parent)
    throw std::runtime_error("CUDA attention smolgen parent is missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention smolgen index is out of range");

  const auto &attention_step = execution_plan.steps[attention_step_index];
  const auto *dense =
      FindStep(execution_plan, attention_step.name + ".smolgen.dense");
  const auto *norm =
      FindStep(execution_plan, attention_step.name + ".smolgen.norm");
  const auto *global_step =
      FindCudaGlobalPositionalEncodingStep(execution_plan);
  if (!dense || !norm || !global_step) {
    throw std::runtime_error("CUDA attention smolgen steps are incomplete");
  }

  const auto compress = TensorBySuffix(*dense, weights, ".compress");
  const auto dense1_weight = TensorBySuffix(*dense, weights, ".dense1_w");
  const auto dense1_bias = TensorBySuffix(*dense, weights, ".dense1_b");
  const auto dense2_weight = TensorBySuffix(*dense, weights, ".dense2_w");
  const auto dense2_bias = TensorBySuffix(*dense, weights, ".dense2_b");
  const auto ln1_gamma = TensorBySuffix(*norm, weights, ".ln1_gammas");
  const auto ln1_beta = TensorBySuffix(*norm, weights, ".ln1_betas");
  const auto ln2_gamma = TensorBySuffix(*norm, weights, ".ln2_gammas");
  const auto ln2_beta = TensorBySuffix(*norm, weights, ".ln2_betas");
  const auto global = TensorBySuffix(*global_step, weights, "body.smolgen_w");

  const int square_rows = batch_size * attention.squares;
  const int flattened_width =
      attention.squares * attention.smolgen.compressed_channels;
  const int global_rows = batch_size * attention.heads;
  const int global_width = attention.squares * attention.squares;

  const auto &compress_binding =
      tape.RequireName(attention.name + ".smolgen.compress");
  const auto &dense1_binding =
      tape.RequireName(attention.name + ".smolgen.dense1");
  const auto &activation1_binding =
      tape.RequireName(attention.name + ".smolgen.activation1");
  const auto &norm1_binding =
      tape.RequireName(attention.name + ".smolgen.norm1");
  const auto &dense2_binding =
      tape.RequireName(attention.name + ".smolgen.dense2");
  const auto &activation2_binding =
      tape.RequireName(attention.name + ".smolgen.activation2");
  const auto &norm2_binding =
      tape.RequireName(attention.name + ".smolgen.norm2");
  const auto &global_binding =
      tape.RequireName(attention.name + ".smolgen.global");
  RequireTapeShape(compress_binding, square_rows,
                   attention.smolgen.compressed_channels, "smolgen compress");
  RequireTapeShape(dense1_binding, batch_size, attention.smolgen.dense1_width,
                   "smolgen dense1");
  RequireTapeShape(activation1_binding, batch_size,
                   attention.smolgen.dense1_width, "smolgen activation1");
  RequireTapeShape(norm1_binding, batch_size, attention.smolgen.dense1_width,
                   "smolgen norm1");
  RequireTapeShape(dense2_binding, batch_size, attention.smolgen.dense2_width,
                   "smolgen dense2");
  RequireTapeShape(activation2_binding, batch_size,
                   attention.smolgen.dense2_width, "smolgen activation2");
  RequireTapeShape(norm2_binding, batch_size, attention.smolgen.dense2_width,
                   "smolgen norm2");
  RequireTapeShape(global_binding, global_rows, global_width, "smolgen global");

  CudaSmolgenStageOutput output;
  output.compressed = tape.Reserve(workspace, compress_binding);
  output.dense1 = tape.Reserve(workspace, dense1_binding);
  output.activation1 = tape.Reserve(workspace, activation1_binding);
  output.norm1 = tape.Reserve(workspace, norm1_binding);
  output.dense2 = tape.Reserve(workspace, dense2_binding);
  output.activation2 = tape.Reserve(workspace, activation2_binding);
  output.norm2 = tape.Reserve(workspace, norm2_binding);
  output.global_bias = tape.Reserve(workspace, global_binding);

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(parent, compress.data, nullptr, output.compressed,
                          square_rows, attention.input_width,
                          attention.smolgen.compressed_channels, stream);
  LaunchDenseAffineKernel(output.compressed, dense1_weight.data, nullptr,
                          output.dense1, batch_size, flattened_width,
                          attention.smolgen.dense1_width, stream);
  LaunchBiasActivationKernel(
      output.dense1, dense1_bias.data, output.activation1, batch_size,
      attention.smolgen.dense1_width,
      ActivationFromString(
          execution_plan.format.activations.smolgen_activation),
      stream);
  LaunchLayerNormKernel(output.activation1, ln1_gamma.data, ln1_beta.data,
                        output.norm1, batch_size,
                        attention.smolgen.dense1_width, 1e-3f, stream);
  LaunchDenseAffineKernel(
      output.norm1, dense2_weight.data, nullptr, output.dense2, batch_size,
      attention.smolgen.dense1_width, attention.smolgen.dense2_width, stream);
  LaunchBiasActivationKernel(
      output.dense2, dense2_bias.data, output.activation2, batch_size,
      attention.smolgen.dense2_width,
      ActivationFromString(
          execution_plan.format.activations.smolgen_activation),
      stream);
  LaunchLayerNormKernel(output.activation2, ln2_gamma.data, ln2_beta.data,
                        output.norm2, batch_size,
                        attention.smolgen.dense2_width, 1e-3f, stream);
  LaunchDenseAffineKernel(
      output.norm2, global.data, nullptr, output.global_bias, global_rows,
      attention.smolgen.dense2_width_per_head, global_width, stream);
  return output;
}

CudaDenseStageOutput ExecuteAttentionResidualLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &norm, const float *parent,
    const CudaAttentionProjectionOutput &attention_output,
    const CudaWeightBuffers &weights, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention layernorm received empty batch");
  if (!parent)
    throw std::runtime_error("CUDA attention layernorm parent is missing");
  if (!attention_output.projection)
    throw std::runtime_error("CUDA attention layernorm projection is missing");
  RequireLayerNormTensors(norm);

  const int rows = batch_size * kCudaAttentionSquares;
  const int width = attention_output.output_width;
  if (width <= 0 || attention_output.rows != rows ||
      attention_output.input_width != width) {
    throw std::runtime_error(
        "CUDA attention layernorm projection shape mismatch");
  }

  const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
  const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
  if (gamma.dims.size() != 1 || beta.dims.size() != 1 ||
      gamma.elements != static_cast<std::size_t>(width) ||
      beta.elements != static_cast<std::size_t>(width)) {
    throw std::runtime_error(
        "CUDA attention layernorm tensor dimensions mismatch");
  }

  const auto &residual_binding =
      tape.RequireName(norm.name + ".attention_residual");
  const auto &norm_binding = tape.RequireName(norm.name + ".normalized");
  RequireTapeShape(residual_binding, rows, width, "attention residual");
  RequireTapeShape(norm_binding, rows, width, "attention layernorm");

  CudaDenseStageOutput output;
  output.residual = tape.Reserve(workspace, residual_binding);
  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;
  output.input_width = width;
  output.output_width = width;
  output.rows = rows;

  cudaStream_t stream = workspace.Stream();
  if (attention_output.projection_bias) {
    LaunchResidualBiasLayerNormKernel(
        parent, attention_output.projection, attention_output.projection_bias,
        gamma.data, beta.data, attention_output.projection, output.residual,
        output.normalized, rows, width,
        FeedForwardResidualScale(execution_plan, norm.name),
        AttentionLayerNormEpsilon(execution_plan, norm.name), stream);
  } else {
    LaunchResidualLayerNormKernel(
        parent, attention_output.projection, gamma.data, beta.data,
        output.residual, output.normalized, rows, width,
        FeedForwardResidualScale(execution_plan, norm.name),
        AttentionLayerNormEpsilon(execution_plan, norm.name), stream);
  }
  return output;
}

CudaAttentionCoreOutput ExecuteAttentionCoreStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index,
    const CudaAttentionProjectionOutput &projections,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaWeightBuffers *weights, const float *parent,
    int trace_run, int trace_stage_index, CudaExecutionScheduleKind trace_kind,
    bool deterministic_attention_softmax) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention core received empty batch");
  if (!projections.query || !projections.key || !projections.value)
    throw std::runtime_error("CUDA attention core projections are missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention core index is out of range");

  const auto &step = execution_plan.steps[attention_step_index];
  const auto attention = ResolveCudaAttentionStagePlan(
      execution_plan, attention_step_index,
      NetworkAttentionHeadCount(execution_plan, step.name));
  const int square_rows = batch_size * attention.squares;
  const int score_rows = batch_size * attention.heads * attention.squares;
  if (projections.rows != square_rows ||
      projections.qkv_width != attention.qkv_width ||
      projections.heads != attention.heads ||
      projections.head_depth != attention.head_depth) {
    throw std::runtime_error("CUDA attention core projection shape mismatch");
  }

  const auto &scores_binding = tape.RequireName(step.name + ".scores");
  const auto &probabilities_binding =
      tape.RequireName(step.name + ".probabilities");
  const auto &context_binding = tape.RequireName(step.name + ".context");
  RequireTapeShape(scores_binding, score_rows, attention.squares,
                   "attention scores");
  RequireTapeShape(probabilities_binding, score_rows, attention.squares,
                   "attention probabilities");
  RequireTapeShape(context_binding, square_rows, attention.qkv_width,
                   "attention context");

  CudaAttentionCoreOutput output;
  output.scores = tape.Reserve(workspace, scores_binding);
  output.probabilities = tape.Reserve(workspace, probabilities_binding);
  output.context = tape.Reserve(workspace, context_binding);
  output.score_rows = score_rows;
  output.score_width = attention.squares;
  output.smolgen_dense1_width = attention.smolgen.dense1_width;
  output.smolgen_dense2_width = attention.smolgen.dense2_width;
  output.rows = square_rows;
  output.qkv_width = attention.qkv_width;
  output.heads = attention.heads;
  output.head_depth = attention.head_depth;

  cudaStream_t stream = workspace.Stream();
  LaunchAttentionScoreKernel(
      projections.query, projections.key, output.scores, batch_size,
      attention.heads, attention.squares, attention.head_depth,
      attention.qkv_width,
      1.0f / std::sqrt(static_cast<float>(attention.head_depth)), stream);
  const bool deterministic_softmax =
      deterministic_attention_softmax &&
      EnvFlagOrDefault("METALFISH_CUDA_DETERMINISTIC_ATTENTION_SOFTMAX", true);
  bool applied_bias_with_softmax = false;
  if (attention.smolgen.present) {
    if (!weights || !parent) {
      throw std::runtime_error(
          "CUDA attention smolgen requires weights and parent input");
    }
    const auto smolgen = ExecuteAttentionSmolgenStage(
        execution_plan, attention_step_index, attention, *weights, parent, tape,
        workspace, batch_size);
    output.attention_bias = smolgen.global_bias;
    output.smolgen_dense1 = smolgen.dense1;
    output.smolgen_norm1 = smolgen.norm1;
    output.smolgen_dense2 = smolgen.dense2;
    output.smolgen_activation2 = smolgen.activation2;
    output.smolgen_norm2 = smolgen.norm2;
    LaunchAttentionBiasSoftmaxKernel(
        output.scores, smolgen.global_bias, output.probabilities, score_rows,
        attention.squares, stream, deterministic_softmax);
    applied_bias_with_softmax = true;
  }
  if (!applied_bias_with_softmax) {
    LaunchAttentionSoftmaxKernel(output.scores, output.probabilities,
                                 score_rows, attention.squares, stream,
                                 deterministic_softmax);
  }
  TraceCudaAttentionBuffer(trace_run, trace_stage_index, 14,
                           step.name + ".probabilities.post_softmax",
                           trace_kind, output.probabilities, score_rows,
                           attention.squares, batch_size, stream);
  LaunchAttentionContextKernel(output.probabilities, projections.value,
                               output.context, batch_size, attention.heads,
                               attention.squares, attention.head_depth,
                               attention.qkv_width, stream);
  TraceCudaAttentionBuffer(trace_run, trace_stage_index, 15,
                           step.name + ".probabilities.post_context",
                           trace_kind, output.probabilities, score_rows,
                           attention.squares, batch_size, stream);
  return output;
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, CudaStageTimingCollector *timings) {
  const CudaStageInputBindings input_bindings;
  return ExecuteDenseActivationLayerNormSequence(execution_plan, weights, input,
                                                 tape, workspace, batch_size,
                                                 input_bindings, timings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    CudaStageTimingCollector *timings) {
  return ExecuteDenseActivationLayerNormSequence(
      execution_plan, weights, input, nullptr, nullptr, tape, workspace,
      batch_size, input_bindings, timings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    CudaStageTimingCollector *timings) {
  const auto schedule = CreateCudaExecutionSchedule(execution_plan);
  return ExecuteDenseActivationLayerNormSequence(
      execution_plan, weights, input, input_masks, input_values, tape,
      workspace, batch_size, input_bindings, schedule, timings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    const CudaExecutionSchedule &schedule, CudaStageTimingCollector *timings,
    CudaStageExecutionOptions options) {
  if (!input)
    throw std::runtime_error("CUDA dense stage sequence input is missing");

  if (!schedule.FullySupported()) {
    throw std::runtime_error("CUDA dense stage sequence is unsupported: " +
                             schedule.Summary());
  }

  CudaDenseStageSequenceOutput sequence;
  const float *current_input = input;
  int current_width = 0;
  int current_rows = batch_size;
  const int stage_trace_run = ReserveCudaStageTraceReport(batch_size);
  int traced_stage_index = 0;
  for (const auto &entry : schedule.entries) {
    if (entry.kind != CudaExecutionScheduleKind::ConvolutionStage &&
        entry.kind != CudaExecutionScheduleKind::ResidualConvolutionStage &&
        entry.kind != CudaExecutionScheduleKind::DenseLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::DenseActivationStage &&
        entry.kind != CudaExecutionScheduleKind::GateStage &&
        entry.kind != CudaExecutionScheduleKind::AttentionLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::FeedForwardStage &&
        entry.kind != CudaExecutionScheduleKind::FeedForwardLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::PolicyMapStage) {
      continue;
    }

    const auto &step = execution_plan.steps[entry.first_step];
    const float *stage_input = current_input;
    int stage_input_width = current_width;
    int stage_input_rows = current_rows;
    if (const std::string *source = input_bindings.FindSource(step.name)) {
      if (source->empty()) {
        stage_input = input;
        stage_input_width = 0;
        stage_input_rows = batch_size;
      } else {
        const CudaDenseStageOutput *source_stage = sequence.FindStage(*source);
        if (!source_stage || !source_stage->output) {
          throw std::runtime_error("CUDA stage input source is missing for " +
                                   step.name + ": " + *source);
        }
        stage_input = source_stage->output;
        stage_input_width = source_stage->output_width;
        stage_input_rows = source_stage->rows;
      }
    }
    MaybeFlattenSquareRowsForDenseStage(step, entry.kind, batch_size,
                                        stage_input_rows, stage_input_width);
    float *static_position_input = nullptr;
    if (IsStaticPositionEmbeddingStage(execution_plan, step)) {
      static_position_input = PrepareStaticPositionEncodingInput(
          execution_plan, step, input_masks, input_values, tape, workspace,
          batch_size);
      const auto geometry =
          ResolveStaticPositionEncodingGeometry(execution_plan, step);
      stage_input = static_position_input;
      stage_input_width = geometry.concat_width;
      stage_input_rows = batch_size * geometry.input_squares;
    }

    CudaDenseStageOutput stage;
    CudaStageTimer timer(timings, step.name, entry.kind, workspace.Stream());
    if (entry.kind == CudaExecutionScheduleKind::ConvolutionStage) {
      stage = ExecuteConvolutionStage(
          execution_plan, step, weights, stage_input, input_masks, input_values,
          tape, workspace, batch_size, stage_input_rows, stage_input_width);
    } else if (entry.kind ==
               CudaExecutionScheduleKind::ResidualConvolutionStage) {
      if (entry.second_step >= execution_plan.steps.size()) {
        throw std::runtime_error(
            "CUDA residual convolution schedule index is invalid");
      }
      const NetworkResolvedExecutionStep *squeeze_excite = nullptr;
      const std::size_t se_index = entry.second_step + 1;
      if (se_index < execution_plan.steps.size() &&
          IsResidualSqueezeExciteFor(execution_plan.steps[se_index],
                                     ResidualBlockName(step.name))) {
        squeeze_excite = &execution_plan.steps[se_index];
      }
      stage = ExecuteResidualConvolutionStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          squeeze_excite, weights, stage_input, tape, workspace, batch_size,
          stage_input_rows, stage_input_width);
    } else if (entry.kind == CudaExecutionScheduleKind::DenseActivationStage &&
               IsDynamicPositionPreprocessName(step.name)) {
      stage = ExecuteDynamicPositionEncodingStage(execution_plan, step, weights,
                                                  input_masks, input_values,
                                                  tape, workspace, batch_size);
      const int pe_width =
          stage.output_width - execution_plan.tensors.input_planes;
      const int dense_width =
          pe_width > 0 ? pe_width * execution_plan.tensors.input_squares : 0;
      TraceCudaDynamicPositionBuffer(
          stage_trace_run, traced_stage_index, 1, step.name + ".expanded",
          entry.kind, stage.expanded_input, stage.rows,
          execution_plan.tensors.input_planes, batch_size, workspace.Stream());
      TraceCudaDynamicPositionBuffer(
          stage_trace_run, traced_stage_index, 2, step.name + ".position_input",
          entry.kind, stage.position_input, batch_size, stage.input_width,
          batch_size, workspace.Stream());
      TraceCudaDynamicPositionBuffer(stage_trace_run, traced_stage_index, 3,
                                     step.name + ".dense", entry.kind,
                                     stage.dense, batch_size, dense_width,
                                     batch_size, workspace.Stream());
    } else if (entry.kind == CudaExecutionScheduleKind::GateStage) {
      stage = ExecuteGateStage(step, weights, stage_input, stage_input_width,
                               tape, workspace, stage_input_rows);
    } else if (entry.kind ==
               CudaExecutionScheduleKind::AttentionLayerNormStage) {
      TraceCudaAttentionBuffer(stage_trace_run, traced_stage_index, 0,
                               step.name + ".input", entry.kind, stage_input,
                               stage_input_rows, stage_input_width, batch_size,
                               workspace.Stream());
      const auto input_projection = ExecuteAttentionInputProjectionStage(
          execution_plan, entry.first_step, weights, stage_input, tape,
          workspace, batch_size);
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 1, step.name + ".query",
          entry.kind, input_projection.query, input_projection.rows,
          input_projection.qkv_width, batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 2, step.name + ".key",
          entry.kind, input_projection.key, input_projection.rows,
          input_projection.qkv_width, batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 3, step.name + ".value",
          entry.kind, input_projection.value, input_projection.rows,
          input_projection.qkv_width, batch_size, workspace.Stream());
      const auto core = ExecuteAttentionCoreStage(
          execution_plan, entry.first_step, input_projection, tape, workspace,
          batch_size, &weights, stage_input, stage_trace_run,
          traced_stage_index, entry.kind,
          options.deterministic_attention_softmax);
      TraceCudaAttentionBuffer(stage_trace_run, traced_stage_index, 4,
                               step.name + ".scores", entry.kind, core.scores,
                               core.score_rows, core.score_width, batch_size,
                               workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 5, step.name + ".smolgen.bias",
          entry.kind, core.attention_bias, core.score_rows, core.score_width,
          batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 6, step.name + ".smolgen.dense1",
          entry.kind, core.smolgen_dense1, batch_size,
          core.smolgen_dense1_width, batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 7, step.name + ".smolgen.norm1",
          entry.kind, core.smolgen_norm1, batch_size, core.smolgen_dense1_width,
          batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 8, step.name + ".smolgen.dense2",
          entry.kind, core.smolgen_dense2, batch_size,
          core.smolgen_dense2_width, batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(stage_trace_run, traced_stage_index, 9,
                               step.name + ".smolgen.activation2", entry.kind,
                               core.smolgen_activation2, batch_size,
                               core.smolgen_dense2_width, batch_size,
                               workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 10, step.name + ".smolgen.norm2",
          entry.kind, core.smolgen_norm2, batch_size, core.smolgen_dense2_width,
          batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 11, step.name + ".probabilities",
          entry.kind, core.probabilities, core.score_rows, core.score_width,
          batch_size, workspace.Stream());
      TraceCudaAttentionBuffer(stage_trace_run, traced_stage_index, 12,
                               step.name + ".context", entry.kind, core.context,
                               core.rows, core.qkv_width, batch_size,
                               workspace.Stream());
      const auto output_projection = ExecuteAttentionOutputProjectionStage(
          execution_plan, entry.first_step, weights, core.context, tape,
          workspace, batch_size, true);
      stage = ExecuteAttentionResidualLayerNormStage(
          execution_plan, execution_plan.steps[entry.second_step], stage_input,
          output_projection, weights, tape, workspace, batch_size);
      TraceCudaAttentionBuffer(
          stage_trace_run, traced_stage_index, 13, step.name + ".projection",
          entry.kind, output_projection.projection, output_projection.rows,
          output_projection.output_width, batch_size, workspace.Stream());
    } else if (entry.kind ==
               CudaExecutionScheduleKind::FeedForwardLayerNormStage) {
      stage = ExecuteFeedForwardLayerNormStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          weights, stage_input, tape, workspace, stage_input_rows);
    } else if (entry.kind == CudaExecutionScheduleKind::FeedForwardStage) {
      stage =
          ExecuteFeedForwardStage(execution_plan, step, weights, stage_input,
                                  tape, workspace, stage_input_rows);
    } else if (entry.kind == CudaExecutionScheduleKind::PolicyMapStage) {
      stage = ExecutePolicyMapStage(execution_plan, step, weights, sequence,
                                    tape, workspace, batch_size);
    } else if (entry.kind == CudaExecutionScheduleKind::DenseLayerNormStage) {
      stage = ExecuteDenseActivationLayerNormStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          weights, stage_input, tape, workspace, stage_input_rows);
    } else {
      stage = ExecuteDenseActivationStage(execution_plan, step, weights,
                                          stage_input, tape, workspace,
                                          stage_input_rows);
    }
    if (static_position_input)
      stage.expanded_input = static_position_input;
    timer.Stop();
    TraceCudaStageOutput(stage_trace_run, traced_stage_index, step.name,
                         entry.kind, stage, batch_size, workspace.Stream());
    ++traced_stage_index;
    if (stage_input_width != 0 && stage.input_width != stage_input_width) {
      throw std::runtime_error(
          "CUDA dense stage sequence input width mismatch");
    }

    sequence.last = stage;
    sequence.stages.push_back({step.name, stage});
    ++sequence.stage_count;
    current_input = stage.output;
    current_width = stage.output_width;
    current_rows = stage.rows;
  }

  if (sequence.stage_count == 0)
    throw std::runtime_error("CUDA dense stage sequence found no stages");
  return sequence;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
