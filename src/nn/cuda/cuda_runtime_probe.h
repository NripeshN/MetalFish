/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

int CompiledCudaRuntimeVersion();
int RuntimeCudaDeviceCount();
std::string RuntimeCudaDeviceSummary();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
