/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Backend Header

  Provides CUDA implementation of the GPU backend interface.
  Supports NVIDIA GPUs for accelerated NNUE evaluation.
*/

#pragma once

#ifdef USE_CUDA

#include "../backend.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace MetalFish {
namespace GPU {

// Forward declarations
class CUDABuffer;
class CUDAKernel;
class CUDACommandEncoder;

/**
 * CUDA Buffer Implementation
 *
 * Manages GPU memory with optional unified memory support
 * for newer NVIDIA GPUs with managed memory.
 */
class CUDABuffer : public Buffer {
public:
  CUDABuffer(void *device_ptr, void *host_ptr, size_t size, bool unified);
  ~CUDABuffer() override;

  void *data() override;
  const void *data() const override;
  size_t size() const override { return size_; }
  bool valid() const override { return device_ptr_ != nullptr; }

  // CUDA-specific accessors
  void *device_data() { return device_ptr_; }
  const void *device_data() const { return device_ptr_; }

  // Synchronize host and device memory (for non-unified memory)
  void sync_to_device();
  void sync_to_host();

private:
  void *device_ptr_;
  void *host_ptr_;
  size_t size_;
  bool unified_;
};

/**
 * CUDA Compute Kernel
 *
 * Represents a CUDA kernel function loaded from a module.
 */
class CUDAKernel : public ComputeKernel {
public:
  CUDAKernel(const std::string &name, void *function);
  ~CUDAKernel() override;

  const std::string &name() const override { return name_; }
  bool valid() const override { return function_ != nullptr; }
  size_t max_threads_per_threadgroup() const override;

  void *cuda_function() const { return function_; }

private:
  std::string name_;
  void *function_;
  int max_threads_per_block_;
};

/**
 * CUDA Command Encoder
 *
 * Records and executes CUDA kernel launches.
 */
class CUDACommandEncoder : public CommandEncoder {
public:
  CUDACommandEncoder(cudaStream_t stream);
  ~CUDACommandEncoder() override;

  void set_kernel(ComputeKernel *kernel) override;
  void set_buffer(Buffer *buffer, int index, size_t offset = 0) override;
  void set_bytes(const void *data, size_t size, int index) override;
  void dispatch_threads(size_t width, size_t height = 1,
                        size_t depth = 1) override;
  void dispatch_threadgroups(size_t groups_x, size_t groups_y, size_t groups_z,
                             size_t threads_x, size_t threads_y,
                             size_t threads_z) override;
  void barrier() override;

  cudaStream_t stream() const { return stream_; }

private:
  cudaStream_t stream_;
  CUDAKernel *current_kernel_;
  std::vector<void *> buffer_args_;
  std::vector<std::vector<uint8_t>> const_data_;
  bool owns_stream_;
};

/**
 * CUDA Backend Implementation
 *
 * Singleton backend for NVIDIA GPU operations.
 */
class CUDABackend : public Backend {
public:
  static CUDABackend &instance();
  static bool is_available();

  BackendType type() const override { return BackendType::CUDA; }

  std::string device_name() const override;
  bool has_unified_memory() const override;
  size_t max_buffer_size() const override;
  size_t max_threadgroup_memory() const override;

  // Hardware capabilities
  size_t recommended_working_set_size() const override;
  size_t total_system_memory() const override;
  int gpu_core_count() const override;
  int max_threads_per_simd_group() const override;
  int recommended_batch_size() const override;

  std::unique_ptr<Buffer>
  create_buffer(size_t size, MemoryMode mode = MemoryMode::Shared,
                BufferUsage usage = BufferUsage::Default) override;
  std::unique_ptr<Buffer>
  create_buffer(const void *data, size_t size,
                MemoryMode mode = MemoryMode::Shared) override;

  std::unique_ptr<ComputeKernel>
  create_kernel(const std::string &name,
                const std::string &library = "") override;

  bool compile_library(const std::string &name,
                       const std::string &source) override;
  bool load_library(const std::string &name, const std::string &path) override;

  std::unique_ptr<CommandEncoder> create_encoder() override;
  std::unique_ptr<CommandEncoder> create_parallel_encoder() override;
  size_t num_parallel_queues() const override;

  void submit_and_wait(CommandEncoder *encoder) override;
  void submit(CommandEncoder *encoder) override;
  void submit_async(CommandEncoder *encoder,
                    std::function<void()> completion_handler) override;
  void synchronize() override;

  size_t allocated_memory() const override { return allocated_memory_; }
  size_t peak_memory() const override { return peak_memory_; }
  void reset_peak_memory() override { peak_memory_ = allocated_memory_; }

  // CUDA-specific methods
  int device_id() const { return device_id_; }
  int compute_capability_major() const { return compute_capability_major_; }
  int compute_capability_minor() const { return compute_capability_minor_; }
  size_t total_memory() const { return total_memory_; }
  int multiprocessor_count() const { return multiprocessor_count_; }
  bool has_tensor_cores() const { return tensor_cores_available_; }
  bool has_int8_tensor_cores() const { return int8_tensor_cores_available_; }
  bool has_warp_shuffle() const { return compute_capability_major_ >= 3; }
  bool has_cooperative_groups() const { return compute_capability_major_ >= 6; }

private:
  CUDABackend();
  ~CUDABackend();

  bool initialize();
  void cleanup();
  void detect_architecture_features();

  int device_id_;
  std::string device_name_;
  int compute_capability_major_;
  int compute_capability_minor_;
  size_t total_memory_;
  int multiprocessor_count_;
  bool unified_memory_supported_;
  bool tensor_cores_available_;
  bool int8_tensor_cores_available_;

  cudaStream_t default_stream_;
  std::vector<cudaStream_t> parallel_streams_;
  size_t stream_index_;

  std::unordered_map<std::string, void *> modules_;
  std::unordered_map<std::string, void *> kernels_;

  size_t allocated_memory_;
  size_t peak_memory_;
  bool initialized_;
};

} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
