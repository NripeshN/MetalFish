/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

std::string CudaErrorText(cudaError_t status) {
  return cudaGetErrorString(status);
}

bool ParseDeviceIndex(const char *raw, int &device) {
  if (!raw || !*raw)
    return false;

  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value < 0 ||
      value > INT_MAX) {
    return false;
  }

  device = static_cast<int>(value);
  return true;
}

long long DeviceScore(const cudaDeviceProp &prop) {
  return static_cast<long long>(prop.major) * 1'000'000'000'000LL +
         static_cast<long long>(prop.minor) * 10'000'000'000LL +
         static_cast<long long>(prop.multiProcessorCount) * 1'000'000LL +
         static_cast<long long>(prop.clockRate) * 100LL +
         static_cast<long long>(prop.totalGlobalMem / (1024ULL * 1024ULL));
}

std::string DeviceDescription(int device, const cudaDeviceProp &prop) {
  std::ostringstream out;
  out << device << " " << prop.name << " sm_" << prop.major << prop.minor
      << ", sms=" << prop.multiProcessorCount
      << ", memory=" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << "MiB";
  return out.str();
}

} // namespace

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

CudaDeviceSelection SelectCudaDevice(int requested_device) {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    const std::string error = CudaErrorText(status);
    cudaGetLastError();
    return {false, -1, "device query failed: " + error};
  }
  if (count <= 0)
    return {false, -1, "no CUDA device is available"};

  bool requested_from_env = false;
  if (requested_device < 0) {
    const char *requested_raw = std::getenv("METALFISH_CUDA_DEVICE");
    if (requested_raw && *requested_raw) {
      requested_from_env = true;
      if (!ParseDeviceIndex(requested_raw, requested_device)) {
        return {false, -1,
                "METALFISH_CUDA_DEVICE must be a non-negative integer"};
      }
    }
  }

  if (requested_device >= 0) {
    const int requested = requested_device;
    if (requested >= count) {
      std::ostringstream out;
      out << (requested_from_env ? "METALFISH_CUDA_DEVICE="
                                 : "configured CUDA device ")
          << requested
          << " is outside the visible CUDA device range 0.." << (count - 1);
      return {false, -1, out.str()};
    }
    status = cudaSetDevice(requested);
    if (status != cudaSuccess) {
      const std::string error = CudaErrorText(status);
      cudaGetLastError();
      return {false, requested, "cudaSetDevice failed: " + error};
    }

    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, requested);
    if (status == cudaSuccess) {
      std::string message =
          "selected CUDA device " + DeviceDescription(requested, prop);
      message += requested_from_env ? " from METALFISH_CUDA_DEVICE"
                                    : " from backend config";
      return {true, requested, message};
    }
    const std::string error = CudaErrorText(status);
    cudaGetLastError();
    std::string message = "selected CUDA device " + std::to_string(requested);
    message += requested_from_env ? " from METALFISH_CUDA_DEVICE"
                                  : " from backend config";
    message += "; properties unavailable: " + error;
    return {true, requested, message};
  }

  int best_device = 0;
  long long best_score = std::numeric_limits<long long>::min();
  cudaDeviceProp best_prop{};
  bool have_properties = false;
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    const long long score = DeviceScore(prop);
    if (!have_properties || score > best_score) {
      best_device = device;
      best_score = score;
      best_prop = prop;
      have_properties = true;
    }
  }

  status = cudaSetDevice(best_device);
  if (status != cudaSuccess) {
    const std::string error = CudaErrorText(status);
    cudaGetLastError();
    return {false, best_device, "cudaSetDevice failed: " + error};
  }

  if (have_properties) {
    return {true, best_device,
            "selected best visible CUDA device " +
                DeviceDescription(best_device, best_prop)};
  }
  return {true, best_device,
          "selected CUDA device 0; device properties unavailable"};
}

CudaDeviceSelection SelectCudaDevice() { return SelectCudaDevice(-1); }

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
