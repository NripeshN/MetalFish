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
struct NetworkWeightInventory;

namespace Cuda {

struct CudaWeightBufferSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
  std::size_t allocation_bytes = 0;
  std::size_t tensor_count = 0;
};

CudaWeightBufferSmokeResult
RunWeightUploadSmoke(const NetworkWeightInventory &inventory);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
