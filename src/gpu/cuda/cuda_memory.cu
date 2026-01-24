/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Advanced Memory Management

  Optimized memory management including:
  - Unified memory with hints and prefetching
  - Pinned memory for faster transfers
  - Double buffering for async operations
  - Memory pool management
*/

#ifndef CUDA_MEMORY_CU
#define CUDA_MEMORY_CU

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace MetalFish {
namespace GPU {
namespace CUDA {

// ============================================================================
// Unified Memory Manager
// ============================================================================

class UnifiedMemoryManager {
public:
  /**
   * Allocate unified memory with optimal hints
   */
  static void *allocate_unified(size_t size, int device_id) {
    void *ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    
    if (err != cudaSuccess) {
      std::cerr << "[CUDA Memory] Failed to allocate unified memory: "
                << cudaGetErrorString(err) << std::endl;
      return nullptr;
    }
    
    // Set memory access hints for better performance
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device_id);
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    
    return ptr;
  }
  
  /**
   * Allocate unified memory with read-mostly hint
   * Useful for weight buffers that are rarely modified
   */
  static void *allocate_unified_readonly(size_t size, int device_id) {
    void *ptr = allocate_unified(size, device_id);
    
    if (ptr) {
      // Mark as read-mostly for better caching
      cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device_id);
    }
    
    return ptr;
  }
  
  /**
   * Prefetch data to device asynchronously
   */
  static void prefetch_to_device(void *ptr, size_t size, int device_id, 
                                 cudaStream_t stream = 0) {
    cudaMemPrefetchAsync(ptr, size, device_id, stream);
  }
  
  /**
   * Prefetch data to CPU asynchronously
   */
  static void prefetch_to_host(void *ptr, size_t size, 
                               cudaStream_t stream = 0) {
    cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream);
  }
  
  /**
   * Free unified memory
   */
  static void free_unified(void *ptr) {
    if (ptr) {
      cudaFree(ptr);
    }
  }
};

// ============================================================================
// Pinned Memory Manager
// ============================================================================

class PinnedMemoryManager {
public:
  /**
   * Allocate pinned (page-locked) host memory
   * Provides faster CPU-GPU transfers
   */
  static void *allocate_pinned(size_t size) {
    void *ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    
    if (err != cudaSuccess) {
      std::cerr << "[CUDA Memory] Failed to allocate pinned memory: "
                << cudaGetErrorString(err) << std::endl;
      return nullptr;
    }
    
    return ptr;
  }
  
  /**
   * Register existing host memory as pinned
   * Useful for making existing allocations DMA-capable
   */
  static bool register_pinned(void *ptr, size_t size) {
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    
    if (err != cudaSuccess) {
      std::cerr << "[CUDA Memory] Failed to register pinned memory: "
                << cudaGetErrorString(err) << std::endl;
      return false;
    }
    
    return true;
  }
  
  /**
   * Unregister pinned memory
   */
  static void unregister_pinned(void *ptr) {
    if (ptr) {
      cudaHostUnregister(ptr);
    }
  }
  
  /**
   * Free pinned memory
   */
  static void free_pinned(void *ptr) {
    if (ptr) {
      cudaFreeHost(ptr);
    }
  }
};

// ============================================================================
// Double Buffer for Async Operations
// ============================================================================

template <typename T>
class DoubleBuffer {
public:
  DoubleBuffer(size_t size, int device_id) 
      : size_(size), device_id_(device_id), current_buffer_(0) {
    
    // Allocate two pinned host buffers
    host_buffers_[0] = static_cast<T*>(PinnedMemoryManager::allocate_pinned(size * sizeof(T)));
    host_buffers_[1] = static_cast<T*>(PinnedMemoryManager::allocate_pinned(size * sizeof(T)));
    
    // Allocate device buffers
    cudaMalloc(&device_buffers_[0], size * sizeof(T));
    cudaMalloc(&device_buffers_[1], size * sizeof(T));
    
    // Create streams for concurrent operations
    cudaStreamCreate(&compute_stream_);
    cudaStreamCreate(&copy_stream_);
  }
  
  ~DoubleBuffer() {
    // Free host buffers
    PinnedMemoryManager::free_pinned(host_buffers_[0]);
    PinnedMemoryManager::free_pinned(host_buffers_[1]);
    
    // Free device buffers
    cudaFree(device_buffers_[0]);
    cudaFree(device_buffers_[1]);
    
    // Destroy streams
    cudaStreamDestroy(compute_stream_);
    cudaStreamDestroy(copy_stream_);
  }
  
  /**
   * Get current host buffer for writing
   */
  T *get_host_buffer() {
    return host_buffers_[current_buffer_];
  }
  
  /**
   * Get current device buffer for compute
   */
  T *get_device_buffer() {
    return device_buffers_[current_buffer_];
  }
  
  /**
   * Swap buffers and initiate async transfer
   * While computing on buffer N, prefetch buffer N+1
   */
  void swap_and_transfer() {
    int next_buffer = 1 - current_buffer_;
    
    // Copy next buffer to device asynchronously
    cudaMemcpyAsync(device_buffers_[next_buffer], 
                   host_buffers_[next_buffer],
                   size_ * sizeof(T),
                   cudaMemcpyHostToDevice,
                   copy_stream_);
    
    // Swap for next iteration
    current_buffer_ = next_buffer;
  }
  
  /**
   * Wait for all operations to complete
   */
  void synchronize() {
    cudaStreamSynchronize(compute_stream_);
    cudaStreamSynchronize(copy_stream_);
  }
  
  cudaStream_t get_compute_stream() { return compute_stream_; }
  cudaStream_t get_copy_stream() { return copy_stream_; }
  
private:
  size_t size_;
  int device_id_;
  int current_buffer_;
  
  T *host_buffers_[2];
  T *device_buffers_[2];
  
  cudaStream_t compute_stream_;
  cudaStream_t copy_stream_;
};

// ============================================================================
// Memory Pool for Efficient Allocation
// ============================================================================

class MemoryPool {
public:
  MemoryPool(size_t pool_size, int device_id) 
      : pool_size_(pool_size), device_id_(device_id), allocated_(0) {
    
    // Allocate large contiguous block
    cudaError_t err = cudaMalloc(&pool_base_, pool_size);
    if (err != cudaSuccess) {
      std::cerr << "[CUDA Memory Pool] Failed to allocate pool: "
                << cudaGetErrorString(err) << std::endl;
      pool_base_ = nullptr;
    }
  }
  
  ~MemoryPool() {
    if (pool_base_) {
      cudaFree(pool_base_);
    }
  }
  
  /**
   * Allocate from pool (simple bump allocator)
   */
  void *allocate(size_t size, size_t alignment = 256) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!pool_base_) return nullptr;
    
    // Align allocation
    size_t aligned_offset = (allocated_ + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + size > pool_size_) {
      std::cerr << "[CUDA Memory Pool] Out of pool memory" << std::endl;
      return nullptr;
    }
    
    void *ptr = static_cast<char*>(pool_base_) + aligned_offset;
    allocated_ = aligned_offset + size;
    
    return ptr;
  }
  
  /**
   * Reset pool (invalidates all previous allocations)
   */
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_ = 0;
  }
  
  size_t get_allocated() const { return allocated_; }
  size_t get_available() const { return pool_size_ - allocated_; }
  
private:
  void *pool_base_;
  size_t pool_size_;
  size_t allocated_;
  int device_id_;
  std::mutex mutex_;
};

// ============================================================================
// Cache-Aligned Allocator
// ============================================================================

/**
 * Allocate memory with specific cache line alignment
 * Important for avoiding false sharing and optimizing cache usage
 * Note: alignment must be a power of 2
 */
class CacheAlignedAllocator {
public:
  /**
   * Allocate device memory aligned to cache line (128 bytes default)
   * @param size Size to allocate in bytes
   * @param alignment Alignment in bytes (must be power of 2, default 128)
   * @return Aligned device pointer or nullptr on failure
   */
  static void *allocate_aligned(size_t size, size_t alignment = 128) {
    // Validate alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      std::cerr << "[CUDA Memory] Alignment must be a power of 2" << std::endl;
      return nullptr;
    }
    
    // CUDA allocations are already 256-byte aligned, but we can ensure it
    void *ptr = nullptr;
    
    // Calculate aligned size (alignment must be power of 2)
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    
    cudaError_t err = cudaMalloc(&ptr, aligned_size);
    if (err != cudaSuccess) {
      std::cerr << "[CUDA Memory] Failed to allocate aligned memory: "
                << cudaGetErrorString(err) << std::endl;
      return nullptr;
    }
    
    return ptr;
  }
  
  static void free_aligned(void *ptr) {
    if (ptr) {
      cudaFree(ptr);
    }
  }
};

// ============================================================================
// Async Memory Operations Helper
// ============================================================================

class AsyncMemoryOps {
public:
  /**
   * Async memcpy with event synchronization
   */
  static void copy_async_with_event(void *dst, const void *src, size_t size,
                                    cudaMemcpyKind kind, cudaStream_t stream,
                                    cudaEvent_t *completion_event = nullptr) {
    cudaMemcpyAsync(dst, src, size, kind, stream);
    
    if (completion_event) {
      cudaEventRecord(*completion_event, stream);
    }
  }
  
  /**
   * Async memset
   */
  static void memset_async(void *ptr, int value, size_t size, 
                          cudaStream_t stream) {
    cudaMemsetAsync(ptr, value, size, stream);
  }
  
  /**
   * 2D memcpy for efficient matrix transfers
   */
  static void copy_2d_async(void *dst, size_t dpitch,
                           const void *src, size_t spitch,
                           size_t width, size_t height,
                           cudaMemcpyKind kind, cudaStream_t stream) {
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
  }
};

// ============================================================================
// Memory Statistics
// ============================================================================

class MemoryStats {
public:
  static void print_memory_info(int device_id) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "[CUDA Memory Stats] Device " << device_id << std::endl;
    std::cout << "  Total:     " << (total_mem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Used:      " << (used_mem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Free:      " << (free_mem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Utilization: " << (100.0 * used_mem / total_mem) << "%" << std::endl;
  }
  
  static size_t get_free_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
  }
  
  static size_t get_total_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
  }
};

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // CUDA_MEMORY_CU
