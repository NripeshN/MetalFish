/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network_execution_plan.h"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {

enum class CudaExecutionScheduleKind {
  Boundary,
  ConvolutionStage,
  ResidualConvolutionStage,
  DenseActivationStage,
  DenseLayerNormStage,
  GateStage,
  AttentionLayerNormStage,
  FeedForwardStage,
  FeedForwardLayerNormStage,
  PolicyMapStage,
  PositionalEncodingStage,
  Unsupported,
};

std::string CudaExecutionScheduleKindName(CudaExecutionScheduleKind kind);

struct CudaExecutionScheduleEntry {
  CudaExecutionScheduleKind kind = CudaExecutionScheduleKind::Unsupported;
  std::size_t first_step = std::numeric_limits<std::size_t>::max();
  std::size_t second_step = std::numeric_limits<std::size_t>::max();
  NetworkExecutionOpKind op_kind = NetworkExecutionOpKind::Dense;
  std::string name;
  std::string reason;
};

struct CudaExecutionSchedule {
  std::vector<CudaExecutionScheduleEntry> entries;
  int boundary_count = 0;
  int convolution_stage_count = 0;
  int residual_convolution_stage_count = 0;
  int dense_activation_stage_count = 0;
  int dense_layernorm_stage_count = 0;
  int gate_stage_count = 0;
  int attention_layernorm_stage_count = 0;
  int feed_forward_stage_count = 0;
  int feed_forward_layernorm_stage_count = 0;
  int policy_map_stage_count = 0;
  int positional_encoding_stage_count = 0;
  int unsupported_count = 0;

  bool FullySupported() const { return unsupported_count == 0; }
  const CudaExecutionScheduleEntry *FirstUnsupported() const;
  std::string Summary() const;
};

CudaExecutionSchedule
CreateCudaExecutionSchedule(const NetworkResolvedExecutionPlan &plan);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
