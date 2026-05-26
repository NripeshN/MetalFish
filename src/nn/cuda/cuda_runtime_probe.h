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

struct CudaDeviceSelection {
  bool ok = false;
  int device = -1;
  std::string message;
};

int CompiledCudaRuntimeVersion();
int RuntimeCudaDeviceCount();
CudaDeviceSelection SelectCudaDevice();
CudaDeviceSelection SelectCudaDevice(int requested_device);
std::string RuntimeCudaDeviceSummary();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
