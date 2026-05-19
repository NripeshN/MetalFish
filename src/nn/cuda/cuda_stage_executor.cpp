/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_stage_executor.h"

#include "cuda_execution_schedule.h"
#include "cuda_kernels.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

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

void RequireDenseTensors(const NetworkResolvedExecutionStep &dense) {
  if (dense.kind != NetworkExecutionOpKind::Dense ||
      dense.tensors.size() < 2) {
    throw std::runtime_error("CUDA dense stage has missing tensors");
  }
}

void RequireLayerNormTensors(const NetworkResolvedExecutionStep &norm) {
  if (norm.kind != NetworkExecutionOpKind::LayerNorm ||
      norm.tensors.size() < 2) {
    throw std::runtime_error("CUDA layernorm stage has missing tensors");
  }
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
    status = cudaMemcpy2D(
        dst, static_cast<std::size_t>(dst_stride) * sizeof(float), src,
        static_cast<std::size_t>(src_stride) * sizeof(float),
        static_cast<std::size_t>(width) * sizeof(float), rows,
        cudaMemcpyDeviceToDevice);
  }
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

CudaDenseStageOutput ExecuteDenseActivationStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
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
  const auto &activation_binding =
      tape.RequireName(dense.name + ".activation");
  const std::size_t entries = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(output_width);
  if (dense_binding.entries != entries ||
      activation_binding.entries != entries) {
    throw std::runtime_error("CUDA dense stage tape size mismatch");
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense_binding);
  output.activation = tape.Reserve(workspace, activation_binding);
  output.output = output.activation;
  output.input_width = input_width;
  output.output_width = output_width;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, dense_weight.data, dense_bias.data,
                          output.dense, batch_size, input_width, output_width,
                          stream);
  LaunchActivationKernel(
      output.dense, output.activation, static_cast<int>(entries),
      ActivationFromString(execution_plan.format.activations.ffn_activation),
      stream);
  return output;
}

CudaDenseStageOutput ExecuteDenseActivationLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  RequireLayerNormTensors(norm);

  CudaDenseStageOutput output = ExecuteDenseActivationStage(
      execution_plan, dense, weights, input, tape, workspace, batch_size);
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
  const std::size_t entries = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(output.output_width);
  if (norm_binding.entries != entries) {
    throw std::runtime_error("CUDA layernorm stage tape size mismatch");
  }

  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchLayerNormKernel(output.activation, gamma.data, beta.data,
                        output.normalized, batch_size, output.output_width,
                        1e-5f, stream);
  return output;
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size) {
  if (!input)
    throw std::runtime_error("CUDA dense stage sequence input is missing");

  const auto schedule = CreateCudaExecutionSchedule(execution_plan);
  if (!schedule.FullySupported()) {
    throw std::runtime_error("CUDA dense stage sequence is unsupported: " +
                             schedule.Summary());
  }

  CudaDenseStageSequenceOutput sequence;
  const float *current_input = input;
  int current_width = 0;
  for (const auto &entry : schedule.entries) {
    if (entry.kind != CudaExecutionScheduleKind::DenseLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::DenseActivationStage) {
      continue;
    }

    const auto &step = execution_plan.steps[entry.first_step];
    const CudaDenseStageOutput stage =
        entry.kind == CudaExecutionScheduleKind::DenseLayerNormStage
            ? ExecuteDenseActivationLayerNormStage(
                  execution_plan, step, execution_plan.steps[entry.second_step],
                  weights, current_input, tape, workspace, batch_size)
            : ExecuteDenseActivationStage(execution_plan, step, weights,
                                          current_input, tape, workspace,
                                          batch_size);
    if (current_width != 0 && stage.input_width != current_width) {
      throw std::runtime_error(
          "CUDA dense stage sequence input width mismatch");
    }

    sequence.last = stage;
    ++sequence.stage_count;
    current_input = stage.output;
    current_width = stage.output_width;
  }

  if (sequence.stage_count == 0)
    throw std::runtime_error("CUDA dense stage sequence found no stages");
  return sequence;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
