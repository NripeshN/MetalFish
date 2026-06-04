/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

namespace MetalFish {
namespace NN {
namespace Cuda {

enum class CudaSmokeStatus {
  Success,
  NoDevice,
  RuntimeError,
  Mismatch,
};

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
