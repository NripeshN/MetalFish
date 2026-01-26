/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Multi-GPU Support

  Enables batch distribution across multiple NVIDIA GPUs.
*/

#ifndef CUDA_MULTI_GPU_H
#define CUDA_MULTI_GPU_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace MetalFish {
namespace GPU {
namespace CUDA {

/**
 * GPU Device Information
 */
struct GPUInfo {
  int device_id;
  std::string name;
  int compute_major;
  int compute_minor;
  size_t total_memory;
  int multiprocessor_count;
  bool has_tensor_cores;
  bool has_peer_access;
};

/**
 * Multi-GPU Manager
 * 
 * Manages multiple GPUs for parallel batch processing.
 */
class MultiGPUManager {
public:
  MultiGPUManager();
  ~MultiGPUManager();

  /**
   * Initialize multi-GPU support
   * @param use_all If true, use all available GPUs. Otherwise, use best GPU only.
   * @return true if at least one GPU is available
   */
  bool initialize(bool use_all = false);

  /**
   * Get number of active GPUs
   */
  int get_num_gpus() const { return static_cast<int>(gpu_info_.size()); }

  /**
   * Get GPU information
   */
  const GPUInfo& get_gpu_info(int gpu_index) const;

  /**
   * Get best GPU (highest compute capability)
   */
  int get_best_gpu() const;

  /**
   * Enable peer-to-peer access between GPUs
   */
  bool enable_peer_access();

  /**
   * Distribute batch across GPUs
   * Returns the batch size for each GPU
   */
  std::vector<int> distribute_batch(int total_batch_size) const;

  /**
   * Set current device
   */
  bool set_device(int gpu_index);

  /**
   * Get current device
   */
  int get_current_device() const;

  /**
   * Synchronize all GPUs
   */
  void synchronize_all();

  /**
   * Check if multi-GPU is enabled
   */
  bool is_multi_gpu_enabled() const { return gpu_info_.size() > 1; }

private:
  std::vector<GPUInfo> gpu_info_;
  bool initialized_;
  int original_device_;
};

/**
 * RAII helper to switch GPU device temporarily
 */
class ScopedDevice {
public:
  ScopedDevice(int device_id);
  ~ScopedDevice();

private:
  int saved_device_;
};

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
#endif // CUDA_MULTI_GPU_H
