/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <array>
#include <cstddef>
#include <string>

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

class CudaExecutionWorkspace {
public:
  CudaExecutionWorkspace() = default;
  CudaExecutionWorkspace(const CudaExecutionWorkspace &) = delete;
  CudaExecutionWorkspace &operator=(const CudaExecutionWorkspace &) = delete;
  ~CudaExecutionWorkspace();

  float *ReserveFloats(CudaWorkspaceSlot slot, std::size_t entries);
  std::size_t CapacityFloats(CudaWorkspaceSlot slot) const;
  std::size_t TotalCapacityFloats() const;
  std::size_t TotalBytes() const;
  void Release();

private:
  static constexpr std::size_t kSlotCount =
      static_cast<std::size_t>(CudaWorkspaceSlot::Count);

  std::array<float *, kSlotCount> buffers_{};
  std::array<std::size_t, kSlotCount> capacities_{};
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
