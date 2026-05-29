/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_smoke_status.h"

#include <cstddef>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaWorkspaceSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
  std::size_t allocation_bytes = 0;
};

CudaWorkspaceSmokeResult RunExecutionWorkspaceSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
