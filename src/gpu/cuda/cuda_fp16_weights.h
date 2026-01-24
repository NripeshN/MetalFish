/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  FP16 Weight Storage

  Provides FP16 weight storage and conversion for tensor core compatibility.
*/

#ifndef CUDA_FP16_WEIGHTS_H
#define CUDA_FP16_WEIGHTS_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>

namespace MetalFish {
namespace GPU {
namespace CUDA {

/**
 * FP16 Weight Manager
 * 
 * Manages conversion and storage of network weights in FP16 format
 * for tensor core acceleration.
 */
class FP16WeightManager {
public:
  FP16WeightManager() = default;
  ~FP16WeightManager();

  /**
   * Convert and store weights in FP16 format
   * @param int16_weights Original INT16 weights
   * @param size Number of weight elements
   * @param scale Scale factor for conversion
   * @return Device pointer to FP16 weights
   */
  half* convert_and_store_weights(const int16_t* int16_weights, 
                                   size_t size, float scale = 64.0f);

  /**
   * Convert and store biases in FP16 format
   * @param int32_biases Original INT32 biases
   * @param size Number of bias elements
   * @param scale Scale factor for conversion
   * @return Device pointer to FP16 biases
   */
  half* convert_and_store_biases(const int32_t* int32_biases,
                                  size_t size, float scale = 64.0f);

  /**
   * Get FP16 weights for a layer
   */
  half* get_fp16_weights(const std::string& layer_name);

  /**
   * Get FP16 biases for a layer
   */
  half* get_fp16_biases(const std::string& layer_name);

  /**
   * Free all FP16 weights
   */
  void clear_all();

  /**
   * Get total memory used by FP16 weights
   */
  size_t get_memory_usage() const { return total_memory_; }

private:
  struct WeightData {
    half* device_ptr = nullptr;
    size_t size = 0;
  };

  std::unordered_map<std::string, WeightData> weights_;
  std::unordered_map<std::string, WeightData> biases_;
  size_t total_memory_ = 0;
};

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
#endif // CUDA_FP16_WEIGHTS_H
