/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Multi-GPU Implementation
*/

#ifdef USE_CUDA

#include "cuda_multi_gpu.h"
#include <iostream>
#include <algorithm>

namespace MetalFish {
namespace GPU {
namespace CUDA {

MultiGPUManager::MultiGPUManager() : initialized_(false), original_device_(0) {
  cudaGetDevice(&original_device_);
}

MultiGPUManager::~MultiGPUManager() {
  if (initialized_) {
    cudaSetDevice(original_device_);
  }
}

bool MultiGPUManager::initialize(bool use_all) {
  if (initialized_) {
    return true;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "[Multi-GPU] No CUDA devices found" << std::endl;
    return false;
  }

  std::cout << "[Multi-GPU] Found " << device_count << " CUDA device(s)" << std::endl;

  // Collect GPU information
  std::vector<GPUInfo> all_gpus;
  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    GPUInfo info;
    info.device_id = i;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.has_tensor_cores = (prop.major >= 7);
    info.has_peer_access = false;

    all_gpus.push_back(info);
    
    std::cout << "[Multi-GPU] GPU " << i << ": " << info.name 
              << " (SM " << info.compute_major << "." << info.compute_minor << ")" << std::endl;
  }

  if (use_all) {
    // Use all GPUs
    gpu_info_ = all_gpus;
  } else {
    // Use only the best GPU
    auto best_gpu = std::max_element(all_gpus.begin(), all_gpus.end(),
      [](const GPUInfo& a, const GPUInfo& b) {
        int score_a = a.compute_major * 100 + a.compute_minor;
        int score_b = b.compute_major * 100 + b.compute_minor;
        return score_a < score_b;
      });
    gpu_info_.push_back(*best_gpu);
  }

  initialized_ = true;
  std::cout << "[Multi-GPU] Using " << gpu_info_.size() << " GPU(s)" << std::endl;
  
  return true;
}

const GPUInfo& MultiGPUManager::get_gpu_info(int gpu_index) const {
  return gpu_info_[gpu_index];
}

int MultiGPUManager::get_best_gpu() const {
  if (gpu_info_.empty()) {
    return 0;
  }

  int best_idx = 0;
  int best_score = gpu_info_[0].compute_major * 100 + gpu_info_[0].compute_minor;
  
  for (size_t i = 1; i < gpu_info_.size(); i++) {
    int score = gpu_info_[i].compute_major * 100 + gpu_info_[i].compute_minor;
    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }
  
  return best_idx;
}

bool MultiGPUManager::enable_peer_access() {
  if (gpu_info_.size() < 2) {
    return true;  // Nothing to do with single GPU
  }

  std::cout << "[Multi-GPU] Enabling peer-to-peer access..." << std::endl;
  
  for (size_t i = 0; i < gpu_info_.size(); i++) {
    cudaSetDevice(gpu_info_[i].device_id);
    
    for (size_t j = 0; j < gpu_info_.size(); j++) {
      if (i == j) continue;
      
      int can_access = 0;
      cudaDeviceCanAccessPeer(&can_access, gpu_info_[i].device_id, 
                              gpu_info_[j].device_id);
      
      if (can_access) {
        cudaError_t err = cudaDeviceEnablePeerAccess(gpu_info_[j].device_id, 0);
        if (err == cudaSuccess) {
          gpu_info_[i].has_peer_access = true;
          std::cout << "[Multi-GPU] Enabled P2P: GPU " << i << " -> GPU " << j << std::endl;
        } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
          std::cerr << "[Multi-GPU] Failed to enable P2P: " 
                    << cudaGetErrorString(err) << std::endl;
        } else {
          // Already enabled, clear the error
          cudaGetLastError();
        }
      }
    }
  }
  
  cudaSetDevice(original_device_);
  return true;
}

std::vector<int> MultiGPUManager::distribute_batch(int total_batch_size) const {
  std::vector<int> batch_sizes(gpu_info_.size());
  
  if (gpu_info_.size() == 1) {
    batch_sizes[0] = total_batch_size;
    return batch_sizes;
  }

  // Distribute based on relative compute capability
  std::vector<int> scores;
  int total_score = 0;
  
  for (const auto& info : gpu_info_) {
    int score = info.multiprocessor_count * (info.compute_major * 10 + info.compute_minor);
    scores.push_back(score);
    total_score += score;
  }

  // Distribute proportionally
  int remaining = total_batch_size;
  for (size_t i = 0; i < gpu_info_.size(); i++) {
    if (i == gpu_info_.size() - 1) {
      // Last GPU gets all remaining
      batch_sizes[i] = remaining;
    } else {
      int size = (total_batch_size * scores[i]) / total_score;
      batch_sizes[i] = size;
      remaining -= size;
    }
  }

  return batch_sizes;
}

bool MultiGPUManager::set_device(int gpu_index) {
  if (gpu_index < 0 || gpu_index >= static_cast<int>(gpu_info_.size())) {
    return false;
  }

  cudaError_t err = cudaSetDevice(gpu_info_[gpu_index].device_id);
  return err == cudaSuccess;
}

int MultiGPUManager::get_current_device() const {
  int device;
  cudaGetDevice(&device);
  
  // Find index in our list
  for (size_t i = 0; i < gpu_info_.size(); i++) {
    if (gpu_info_[i].device_id == device) {
      return static_cast<int>(i);
    }
  }
  
  return 0;
}

void MultiGPUManager::synchronize_all() {
  int current_device;
  cudaGetDevice(&current_device);
  
  for (const auto& info : gpu_info_) {
    cudaSetDevice(info.device_id);
    cudaDeviceSynchronize();
  }
  
  cudaSetDevice(current_device);
}

ScopedDevice::ScopedDevice(int device_id) : saved_device_(0) {
  cudaGetDevice(&saved_device_);
  cudaSetDevice(device_id);
}

ScopedDevice::~ScopedDevice() {
  cudaSetDevice(saved_device_);
}

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
