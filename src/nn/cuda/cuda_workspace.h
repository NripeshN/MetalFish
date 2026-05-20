/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "cuda_input_packing.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

enum class CudaWorkspaceSlot {
  Dense,
  Activation,
  Norm,
  Count,
};

struct CudaNamedWorkspaceBuffer {
  std::string name;
  float *buffer = nullptr;
  std::size_t capacity = 0;
};

struct CudaNamedByteWorkspaceBuffer {
  std::string name;
  void *buffer = nullptr;
  std::size_t capacity_bytes = 0;
};

class CudaExecutionWorkspace {
public:
  CudaExecutionWorkspace() = default;
  CudaExecutionWorkspace(const CudaExecutionWorkspace &) = delete;
  CudaExecutionWorkspace &operator=(const CudaExecutionWorkspace &) = delete;
  ~CudaExecutionWorkspace();

  float *ReserveFloats(CudaWorkspaceSlot slot, std::size_t entries);
  float *ReserveNamedFloats(std::string_view name, std::size_t entries);
  void *ReserveNamedBytes(std::string_view name, std::size_t bytes);
  cudaStream_t Stream();
  void Synchronize();
  std::size_t CapacityFloats(CudaWorkspaceSlot slot) const;
  std::size_t NamedCapacityFloats(std::string_view name) const;
  std::size_t NamedCapacityBytes(std::string_view name) const;
  std::size_t NamedBufferCount() const;
  std::size_t TotalCapacityFloats() const;
  std::size_t TotalBytes() const;
  void Clear(cudaStream_t stream = nullptr);
  void Release();

private:
  static constexpr std::size_t kSlotCount =
      static_cast<std::size_t>(CudaWorkspaceSlot::Count);

  std::array<float *, kSlotCount> buffers_{};
  std::array<std::size_t, kSlotCount> capacities_{};
  std::vector<CudaNamedWorkspaceBuffer> named_buffers_;
  std::vector<CudaNamedByteWorkspaceBuffer> named_byte_buffers_;
  cudaStream_t stream_ = nullptr;
};

struct CudaWorkspaceSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
  std::size_t allocation_bytes = 0;
};

CudaWorkspaceSmokeResult RunExecutionWorkspaceSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
