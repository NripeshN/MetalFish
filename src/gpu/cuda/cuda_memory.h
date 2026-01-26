/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Advanced Memory Management Header

  Interface for optimized memory management utilities.
*/

#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include <mutex>
#include <vector>

namespace MetalFish {
namespace GPU {
namespace CUDA {

/**
 * Unified Memory Manager
 * 
 * Provides optimized unified memory allocation with hints
 */
class UnifiedMemoryManager {
public:
  static void *allocate_unified(size_t size, int device_id);
  static void *allocate_unified_readonly(size_t size, int device_id);
  static void prefetch_to_device(void *ptr, size_t size, int device_id, 
                                 cudaStream_t stream = 0);
  static void prefetch_to_host(void *ptr, size_t size, 
                               cudaStream_t stream = 0);
  static void free_unified(void *ptr);
};

/**
 * Pinned Memory Manager
 * 
 * Manages pinned (page-locked) host memory for faster transfers
 */
class PinnedMemoryManager {
public:
  static void *allocate_pinned(size_t size);
  static void free_pinned(void *ptr);
};

/**
 * Double Buffer
 * 
 * Implements double buffering for overlapping transfers and computation
 */
template <typename T>
class DoubleBuffer {
public:
  DoubleBuffer(size_t size, int device_id);
  ~DoubleBuffer();
  
  bool is_valid() const { return valid_; }
  T *get_host_buffer(int index) const;
  T *get_device_buffer(int index) const;
  void swap_buffers();
  void transfer_to_device(int index, cudaStream_t stream);
  void transfer_from_device(int index, cudaStream_t stream);
  
private:
  T *host_buffers_[2];
  T *device_buffers_[2];
  cudaStream_t streams_[2];
  size_t size_;
  int current_index_;
  bool valid_;
};

/**
 * Memory Pool
 * 
 * Simple memory pool allocator for reducing allocation overhead
 */
class MemoryPool {
public:
  MemoryPool(size_t pool_size, int device_id);
  ~MemoryPool();
  
  void *allocate(size_t size);
  void reset();
  size_t get_allocated() const { return allocated_; }
  
private:
  void *pool_base_;
  size_t pool_size_;
  size_t allocated_;
  int device_id_;
};

/**
 * Cache-Aligned Allocator
 * 
 * Allocates memory with specified alignment for optimal cache performance
 */
class CacheAlignedAllocator {
public:
  static void *allocate_aligned(size_t size, size_t alignment);
  static void free_aligned(void *ptr);
};

// ============================================================================
// Template Implementation for DoubleBuffer
// ============================================================================

template <typename T>
DoubleBuffer<T>::DoubleBuffer(size_t size, int device_id) 
    : size_(size), current_index_(0),
      host_buffers_{nullptr, nullptr}, device_buffers_{nullptr, nullptr},
      streams_{nullptr, nullptr}, valid_(false) {
  
  // Allocate two pinned host buffers
  host_buffers_[0] = static_cast<T*>(PinnedMemoryManager::allocate_pinned(size * sizeof(T)));
  if (!host_buffers_[0]) return;
  
  host_buffers_[1] = static_cast<T*>(PinnedMemoryManager::allocate_pinned(size * sizeof(T)));
  if (!host_buffers_[1]) return;
  
  // Allocate device buffers
  if (cudaMalloc(&device_buffers_[0], size * sizeof(T)) != cudaSuccess) return;
  if (cudaMalloc(&device_buffers_[1], size * sizeof(T)) != cudaSuccess) return;
  
  // Create streams for concurrent operations
  if (cudaStreamCreate(&streams_[0]) != cudaSuccess) return;
  if (cudaStreamCreate(&streams_[1]) != cudaSuccess) return;
  
  valid_ = true;
}

template <typename T>
DoubleBuffer<T>::~DoubleBuffer() {
  // Free host buffers (check for nullptr in case construction failed partway)
  if (host_buffers_[0]) PinnedMemoryManager::free_pinned(host_buffers_[0]);
  if (host_buffers_[1]) PinnedMemoryManager::free_pinned(host_buffers_[1]);
  
  // Free device buffers
  if (device_buffers_[0]) cudaFree(device_buffers_[0]);
  if (device_buffers_[1]) cudaFree(device_buffers_[1]);
  
  // Destroy streams
  if (streams_[0]) cudaStreamDestroy(streams_[0]);
  if (streams_[1]) cudaStreamDestroy(streams_[1]);
}

template <typename T>
T *DoubleBuffer<T>::get_host_buffer(int index) const {
  return host_buffers_[index];
}

template <typename T>
T *DoubleBuffer<T>::get_device_buffer(int index) const {
  return device_buffers_[index];
}

template <typename T>
void DoubleBuffer<T>::swap_buffers() {
  current_index_ = 1 - current_index_;
}

template <typename T>
void DoubleBuffer<T>::transfer_to_device(int index, cudaStream_t stream) {
  cudaMemcpyAsync(device_buffers_[index], host_buffers_[index],
                  size_ * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void DoubleBuffer<T>::transfer_from_device(int index, cudaStream_t stream) {
  cudaMemcpyAsync(host_buffers_[index], device_buffers_[index],
                  size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // CUDA_MEMORY_H
