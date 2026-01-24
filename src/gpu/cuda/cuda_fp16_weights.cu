/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  FP16 Weight Storage Implementation
*/

#ifdef USE_CUDA

#include "cuda_fp16_weights.h"
#include <iostream>
#include <unordered_map>
#include <string>

namespace MetalFish {
namespace GPU {
namespace CUDA {

FP16WeightManager::~FP16WeightManager() {
  clear_all();
}

half* FP16WeightManager::convert_and_store_weights(
    const int16_t* int16_weights, size_t size, float scale) {
  
  // Allocate host memory for FP16 conversion
  std::vector<half> fp16_host(size);
  
  // Convert INT16 to FP16
  for (size_t i = 0; i < size; i++) {
    float val = static_cast<float>(int16_weights[i]) / scale;
    fp16_host[i] = __float2half(val);
  }
  
  // Allocate device memory
  half* device_ptr = nullptr;
  cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(half));
  if (err != cudaSuccess) {
    std::cerr << "[FP16 Weights] Failed to allocate device memory: "
              << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }
  
  // Copy to device
  err = cudaMemcpy(device_ptr, fp16_host.data(), size * sizeof(half),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[FP16 Weights] Failed to copy to device: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(device_ptr);
    return nullptr;
  }
  
  total_memory_ += size * sizeof(half);
  return device_ptr;
}

half* FP16WeightManager::convert_and_store_biases(
    const int32_t* int32_biases, size_t size, float scale) {
  
  // Allocate host memory for FP16 conversion
  std::vector<half> fp16_host(size);
  
  // Convert INT32 to FP16
  for (size_t i = 0; i < size; i++) {
    float val = static_cast<float>(int32_biases[i]) / scale;
    fp16_host[i] = __float2half(val);
  }
  
  // Allocate device memory
  half* device_ptr = nullptr;
  cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(half));
  if (err != cudaSuccess) {
    std::cerr << "[FP16 Biases] Failed to allocate device memory: "
              << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }
  
  // Copy to device
  err = cudaMemcpy(device_ptr, fp16_host.data(), size * sizeof(half),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[FP16 Biases] Failed to copy to device: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(device_ptr);
    return nullptr;
  }
  
  total_memory_ += size * sizeof(half);
  return device_ptr;
}

half* FP16WeightManager::get_fp16_weights(const std::string& layer_name) {
  auto it = weights_.find(layer_name);
  return (it != weights_.end()) ? it->second.device_ptr : nullptr;
}

half* FP16WeightManager::get_fp16_biases(const std::string& layer_name) {
  auto it = biases_.find(layer_name);
  return (it != biases_.end()) ? it->second.device_ptr : nullptr;
}

void FP16WeightManager::clear_all() {
  for (auto& [name, data] : weights_) {
    if (data.device_ptr) {
      cudaFree(data.device_ptr);
    }
  }
  
  for (auto& [name, data] : biases_) {
    if (data.device_ptr) {
      cudaFree(data.device_ptr);
    }
  }
  
  weights_.clear();
  biases_.clear();
  total_memory_ = 0;
}

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
