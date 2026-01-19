/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Utilities Header

  Common utilities and helpers for CUDA operations.
*/

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <iostream>

namespace MetalFish {
namespace GPU {
namespace CUDA {

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_SAFE_CALL(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << " at "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
    }                                                                          \
  } while (0)

#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaError_t err = cudaDeviceSynchronize();                                 \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA Sync Error] " << cudaGetErrorString(err) << " at "   \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
    }                                                                          \
  } while (0)

// ============================================================================
// Device Query Utilities
// ============================================================================

inline int get_device_count() {
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}

inline bool has_cuda_device() {
  return get_device_count() > 0;
}

inline int get_best_device() {
  int device_count = get_device_count();
  if (device_count == 0) return -1;

  int best_device = 0;
  int best_sm = 0;

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    int sm = prop.major * 100 + prop.minor;
    if (sm > best_sm) {
      best_sm = sm;
      best_device = i;
    }
  }

  return best_device;
}

// ============================================================================
// Memory Utilities
// ============================================================================

template <typename T>
T* cuda_malloc(size_t count) {
  T* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

template <typename T>
T* cuda_malloc_managed(size_t count) {
  T* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, count * sizeof(T));
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

template <typename T>
void cuda_free(T* ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

template <typename T>
void cuda_memcpy_to_device(T* dst, const T* src, size_t count) {
  cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void cuda_memcpy_to_host(T* dst, const T* src, size_t count) {
  cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
void cuda_memcpy_async_to_device(T* dst, const T* src, size_t count,
                                 cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void cuda_memcpy_async_to_host(T* dst, const T* src, size_t count,
                               cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

// ============================================================================
// Kernel Launch Utilities
// ============================================================================

inline dim3 calculate_grid_1d(size_t total_threads, size_t block_size = 256) {
  return dim3((total_threads + block_size - 1) / block_size);
}

inline dim3 calculate_grid_2d(size_t width, size_t height,
                              dim3 block_size = dim3(16, 16)) {
  return dim3((width + block_size.x - 1) / block_size.x,
              (height + block_size.y - 1) / block_size.y);
}

// ============================================================================
// RAII Wrappers
// ============================================================================

class CUDAStream {
public:
  CUDAStream() { cudaStreamCreate(&stream_); }
  ~CUDAStream() { cudaStreamDestroy(stream_); }

  cudaStream_t get() const { return stream_; }
  operator cudaStream_t() const { return stream_; }

  void synchronize() { cudaStreamSynchronize(stream_); }

private:
  cudaStream_t stream_;
};

class CUDAEvent {
public:
  CUDAEvent() { cudaEventCreate(&event_); }
  ~CUDAEvent() { cudaEventDestroy(event_); }

  cudaEvent_t get() const { return event_; }
  operator cudaEvent_t() const { return event_; }

  void record(cudaStream_t stream = 0) { cudaEventRecord(event_, stream); }
  void synchronize() { cudaEventSynchronize(event_); }

  float elapsed_ms(const CUDAEvent& start) const {
    float ms = 0;
    cudaEventElapsedTime(&ms, start.event_, event_);
    return ms;
  }

private:
  cudaEvent_t event_;
};

template <typename T>
class CUDADeviceBuffer {
public:
  CUDADeviceBuffer() : ptr_(nullptr), size_(0) {}
  
  explicit CUDADeviceBuffer(size_t count) : ptr_(nullptr), size_(count) {
    if (count > 0) {
      cudaMalloc(&ptr_, count * sizeof(T));
    }
  }

  ~CUDADeviceBuffer() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  // Move semantics
  CUDADeviceBuffer(CUDADeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  CUDADeviceBuffer& operator=(CUDADeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // No copy
  CUDADeviceBuffer(const CUDADeviceBuffer&) = delete;
  CUDADeviceBuffer& operator=(const CUDADeviceBuffer&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return size_; }
  bool valid() const { return ptr_ != nullptr; }

  void copy_from_host(const T* src, size_t count) {
    cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copy_to_host(T* dst, size_t count) const {
    cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
  }

private:
  T* ptr_;
  size_t size_;
};

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
