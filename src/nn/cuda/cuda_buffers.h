/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "../network_format_types.h"
#include "../network_tensor_plan.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

constexpr int kCudaPolicyOutputs = kNetworkPolicyOutputs;
constexpr int kCudaAttentionPolicyScratch = kNetworkAttentionPolicyScratch;
constexpr int kCudaConvPolicyScratch = kNetworkConvPolicyScratch;

struct CudaBufferLayout {
  int max_batch_size = 0;
  bool wdl = false;
  bool moves_left = false;
  bool conv_policy = false;
  bool attention_policy = false;
  NetworkTensorPlan tensor_plan;

  size_t InputPlaneEntries() const;
  size_t PolicyEntries() const;
  size_t ValueEntries() const;
  size_t MovesLeftEntries() const;
  size_t RawPolicyEntries() const;
  size_t TotalBytes() const;
};

CudaBufferLayout LayoutFromTensorPlan(const NetworkTensorPlan &plan,
                                      int max_batch_size);
CudaBufferLayout LayoutFromNetworkFormat(const NetworkFormatDescriptor &format,
                                         int max_batch_size);

struct CudaOutputDownload {
  std::vector<float> policy;
  std::vector<float> value;
  std::vector<float> moves_left;
  std::vector<float> raw_policy;
};

class CudaInferenceBuffers {
public:
  CudaInferenceBuffers() = default;
  CudaInferenceBuffers(const CudaInferenceBuffers &) = delete;
  CudaInferenceBuffers &operator=(const CudaInferenceBuffers &) = delete;
  CudaInferenceBuffers(CudaInferenceBuffers &&other) noexcept;
  CudaInferenceBuffers &operator=(CudaInferenceBuffers &&other) noexcept;
  ~CudaInferenceBuffers();

  void Allocate(const CudaBufferLayout &layout);
  void UploadPackedInputs(const std::vector<std::uint64_t> &masks,
                          const std::vector<float> &values, int batch_size,
                          cudaStream_t stream = nullptr);
  void ClearAll(cudaStream_t stream = nullptr);
  void ClearOutputs(int batch_size, cudaStream_t stream = nullptr);
  CudaOutputDownload DownloadOutputs(int batch_size,
                                     cudaStream_t stream = nullptr,
                                     bool include_raw_policy = true) const;
  void Release();

  const CudaBufferLayout &Layout() const { return layout_; }
  size_t AllocationBytes() const { return allocation_bytes_; }
  std::uint64_t Generation() const { return generation_; }

  std::uint64_t *input_masks = nullptr;
  float *input_values = nullptr;
  float *policy = nullptr;
  float *value = nullptr;
  float *moves_left = nullptr;
  float *raw_policy = nullptr;

private:
  CudaBufferLayout layout_;
  size_t allocation_bytes_ = 0;
  std::uint64_t generation_ = 1;
};

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
