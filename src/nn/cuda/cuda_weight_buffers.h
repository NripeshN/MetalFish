/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../network_weight_inventory.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaDeviceTensorView {
  const float *data = nullptr;
  std::size_t elements = 0;
  std::vector<std::uint32_t> dims;
  NetworkWeightTensorKind kind = NetworkWeightTensorKind::Generic;
};

class CudaWeightBuffers {
public:
  CudaWeightBuffers() = default;
  CudaWeightBuffers(const CudaWeightBuffers &) = delete;
  CudaWeightBuffers &operator=(const CudaWeightBuffers &) = delete;
  CudaWeightBuffers(CudaWeightBuffers &&other) noexcept;
  CudaWeightBuffers &operator=(CudaWeightBuffers &&other) noexcept;
  ~CudaWeightBuffers();

  void Upload(const NetworkWeightInventory &inventory);
  bool DownloadMatches(const NetworkWeightInventory &inventory,
                       std::string *error) const;
  void Release();

  std::size_t AllocationBytes() const { return allocation_bytes_; }
  std::size_t TensorCount() const { return device_tensors_.size(); }
  CudaDeviceTensorView TensorAt(std::size_t index) const;

private:
  struct DeviceTensor {
    float *data = nullptr;
    std::size_t elements = 0;
    std::vector<std::uint32_t> dims;
    NetworkWeightTensorKind kind = NetworkWeightTensorKind::Generic;
  };

  std::vector<DeviceTensor> device_tensors_;
  std::size_t allocation_bytes_ = 0;
};

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
