/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

int CompiledCudaRuntimeVersion() { return CUDART_VERSION; }

int RuntimeCudaDeviceCount() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return -static_cast<int>(status);
  }
  return count;
}

std::string RuntimeCudaDeviceSummary() {
  std::ostringstream out;
  out << "CUDA runtime " << CompiledCudaRuntimeVersion();

  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    out << ", device query failed: " << cudaGetErrorString(status);
    cudaGetLastError();
    return out.str();
  }

  out << ", devices=" << count;
  if (count <= 0)
    return out.str();

  int current = 0;
  if (cudaGetDevice(&current) != cudaSuccess) {
    cudaGetLastError();
    current = 0;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, current) == cudaSuccess) {
    out << ", active=" << current << " " << prop.name << " sm_" << prop.major
        << prop.minor
        << ", memory=" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << "MiB";
  } else {
    out << ", active=" << current << " properties unavailable";
    cudaGetLastError();
  }
  return out.str();
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
