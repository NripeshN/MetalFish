/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_buffers.h"

#include "../network_output_decoder.h"
#include "cuda_executor.h"
#include "cuda_runtime_probe.h"

#include <cmath>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
CudaBufferSmokeResult RunNullExecutorPipelineSmokeRaw(const float *inputs,
                                                      int batch_size) {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }
  if (inputs == nullptr || batch_size <= 0) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = "CUDA null executor smoke received invalid input";
    return result;
  }

  NetworkFormatDescriptor format;
  format.wdl = true;
  format.moves_left = true;
  format.attention_policy = true;
  const auto plan = CreateNetworkTensorPlan(format);
  const auto layout = LayoutFromTensorPlan(plan, batch_size);

  std::vector<std::uint64_t> masks;
  std::vector<float> values;
  PackInputPlanesHostRaw(inputs, batch_size, masks, values);

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(layout);
    buffers.UploadPackedInputs(masks, values, batch_size);
    buffers.ClearOutputs(batch_size);
    auto executor = CreateNullCudaExecutorForSmoke();
    NetworkResolvedExecutionPlan resolved_plan;
    CudaWeightBuffers weights;
    executor->Execute(plan, resolved_plan, weights, buffers, batch_size);

    result.allocation_bytes = buffers.AllocationBytes();
    const auto downloaded = buffers.DownloadOutputs(batch_size);
    const float *decoded_moves_left =
        downloaded.moves_left.empty() ? nullptr : downloaded.moves_left.data();
    const auto decoded = DecodeNetworkOutputBatch(
        plan, downloaded.policy.data(), downloaded.policy.size(),
        downloaded.value.data(), downloaded.value.size(), decoded_moves_left,
        downloaded.moves_left.size(), batch_size);

    if (decoded.size() != static_cast<size_t>(batch_size)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA null executor decoded batch size mismatch";
      return result;
    }
    for (int b = 0; b < batch_size; ++b) {
      const auto &out = decoded[static_cast<size_t>(b)];
      const float expected_value = 0.60f - 0.05f * static_cast<float>(b);
      if (!out.has_wdl || !out.has_moves_left ||
          std::fabs(out.value - expected_value) > 1e-6f ||
          out.moves_left != 12.0f + static_cast<float>(b) ||
          out.policy[0] != 0.25f + static_cast<float>(b) ||
          out.policy[plan.policy_outputs - 1] !=
              -0.75f - static_cast<float>(b)) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA null executor decoded output mismatch";
        return result;
      }
      if (!downloaded.raw_policy.empty()) {
        const size_t raw_offset =
            static_cast<size_t>(b) * plan.raw_policy_outputs;
        if (downloaded.raw_policy[raw_offset] != 3.0f + static_cast<float>(b)) {
          result.status = CudaSmokeStatus::Mismatch;
          result.message = "CUDA null executor raw policy download mismatch";
          return result;
        }
      }
    }
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
