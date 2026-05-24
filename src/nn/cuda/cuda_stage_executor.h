/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_execution_schedule.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace MetalFish {
namespace NN {
struct NetworkResolvedExecutionPlan;
struct NetworkResolvedExecutionStep;

namespace Cuda {
class CudaExecutionTape;
class CudaExecutionWorkspace;
class CudaStageInputBindings;
class CudaWeightBuffers;

struct CudaDenseStageOutput {
  float *dense = nullptr;
  float *convolution = nullptr;
  float *activation = nullptr;
  float *normalized = nullptr;
  float *gated = nullptr;
  float *feed_forward = nullptr;
  float *residual = nullptr;
  float *expanded_input = nullptr;
  float *position_input = nullptr;
  float *output = nullptr;
  int input_width = 0;
  int output_width = 0;
  int rows = 0;
};

struct CudaDenseStageSequenceOutput {
  CudaDenseStageOutput last;
  std::vector<std::pair<std::string, CudaDenseStageOutput>> stages;
  int stage_count = 0;

  const CudaDenseStageOutput *FindStage(std::string_view name) const;
};

struct CudaAttentionProjectionOutput {
  float *query = nullptr;
  float *key = nullptr;
  float *value = nullptr;
  float *projection = nullptr;
  const float *projection_bias = nullptr;
  int rows = 0;
  int input_width = 0;
  int qkv_width = 0;
  int output_width = 0;
  int heads = 0;
  int head_depth = 0;
};

struct CudaAttentionCoreOutput {
  float *scores = nullptr;
  float *attention_bias = nullptr;
  float *smolgen_dense1 = nullptr;
  float *smolgen_norm1 = nullptr;
  float *smolgen_dense2 = nullptr;
  float *smolgen_activation2 = nullptr;
  float *smolgen_norm2 = nullptr;
  float *probabilities = nullptr;
  float *context = nullptr;
  int score_rows = 0;
  int score_width = 0;
  int smolgen_dense1_width = 0;
  int smolgen_dense2_width = 0;
  int rows = 0;
  int qkv_width = 0;
  int heads = 0;
  int head_depth = 0;
};

struct CudaStageTimingRecord {
  std::string name;
  CudaExecutionScheduleKind kind = CudaExecutionScheduleKind::Unsupported;
  float millis = 0.0f;
};

class CudaStageTimingCollector {
public:
  void Add(std::string name, CudaExecutionScheduleKind kind, float millis);
  const std::vector<CudaStageTimingRecord> &Records() const { return records_; }
  void Clear() { records_.clear(); }

private:
  std::vector<CudaStageTimingRecord> records_;
};

CudaDenseStageOutput
ExecuteConvolutionStage(const NetworkResolvedExecutionPlan &execution_plan,
                        const NetworkResolvedExecutionStep &convolution,
                        const CudaWeightBuffers &weights, const float *input,
                        const std::uint64_t *input_masks,
                        const float *input_values,
                        const CudaExecutionTape &tape,
                        CudaExecutionWorkspace &workspace, int batch_size,
                        int input_rows, int input_width);

CudaDenseStageOutput
ExecuteDenseActivationStage(const NetworkResolvedExecutionPlan &execution_plan,
                            const NetworkResolvedExecutionStep &dense,
                            const CudaWeightBuffers &weights,
                            const float *input, const CudaExecutionTape &tape,
                            CudaExecutionWorkspace &workspace, int rows);

CudaDenseStageOutput ExecuteDenseActivationLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int rows);

CudaDenseStageOutput ExecuteDynamicPositionEncodingStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const CudaWeightBuffers &weights,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size);

CudaDenseStageOutput ExecuteGateStage(const NetworkResolvedExecutionStep &gate,
                                      const CudaWeightBuffers &weights,
                                      const float *input, int input_width,
                                      const CudaExecutionTape &tape,
                                      CudaExecutionWorkspace &workspace,
                                      int rows);

CudaDenseStageOutput
ExecuteFeedForwardStage(const NetworkResolvedExecutionPlan &execution_plan,
                        const NetworkResolvedExecutionStep &ffn,
                        const CudaWeightBuffers &weights, const float *input,
                        const CudaExecutionTape &tape,
                        CudaExecutionWorkspace &workspace, int rows);

CudaDenseStageOutput ExecuteFeedForwardLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &ffn,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int rows);

CudaDenseStageOutput ExecuteAttentionPolicyMapStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &policy_map,
    const CudaWeightBuffers &weights,
    const CudaDenseStageSequenceOutput &sequence, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size);

CudaAttentionProjectionOutput ExecuteAttentionInputProjectionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size);

CudaAttentionProjectionOutput ExecuteAttentionOutputProjectionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaWeightBuffers &weights,
    const float *context, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size,
    bool defer_projection_bias = false);

CudaDenseStageOutput ExecuteAttentionResidualLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &norm, const float *parent,
    const CudaAttentionProjectionOutput &attention_output,
    const CudaWeightBuffers &weights, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size);

CudaAttentionCoreOutput ExecuteAttentionCoreStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index,
    const CudaAttentionProjectionOutput &projections,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaWeightBuffers *weights = nullptr,
    const float *parent = nullptr, int trace_run = -1,
    int trace_stage_index = -1,
    CudaExecutionScheduleKind trace_kind =
        CudaExecutionScheduleKind::AttentionLayerNormStage);

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, CudaStageTimingCollector *timings = nullptr);

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    CudaStageTimingCollector *timings = nullptr);

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    CudaStageTimingCollector *timings = nullptr);

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    const CudaExecutionSchedule &schedule,
    CudaStageTimingCollector *timings = nullptr);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
