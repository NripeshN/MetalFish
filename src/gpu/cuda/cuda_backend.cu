/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Backend Implementation

  Implements the GPU backend interface for NVIDIA CUDA.
  Provides similar functionality to Metal backend for NVIDIA GPUs.
*/

#ifdef USE_CUDA

#include "../backend.h"
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace MetalFish {
namespace GPU {

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(err) << std::endl;                       \
    }                                                                          \
  } while (0)

// ============================================================================
// CUDA Buffer Implementation
// ============================================================================

class CUDABuffer : public Buffer {
public:
  CUDABuffer(void *device_ptr, void *host_ptr, size_t size, MemoryMode mode)
      : device_ptr_(device_ptr), host_ptr_(host_ptr), size_(size),
        mode_(mode) {}

  ~CUDABuffer() override {
    if (mode_ == MemoryMode::Shared) {
      // Unified memory - single free
      if (device_ptr_) {
        CUDA_CHECK(cudaFree(device_ptr_));
      }
    } else {
      // Separate device and host memory
      if (device_ptr_) {
        CUDA_CHECK(cudaFree(device_ptr_));
      }
      if (host_ptr_) {
        delete[] static_cast<uint8_t *>(host_ptr_);
      }
    }
  }

  void *data() override {
    // For unified memory, device_ptr is accessible from CPU
    // For private memory, return host staging buffer
    return (mode_ == MemoryMode::Shared) ? device_ptr_ : host_ptr_;
  }

  const void *data() const override {
    return (mode_ == MemoryMode::Shared) ? device_ptr_ : host_ptr_;
  }

  size_t size() const override { return size_; }

  bool valid() const override { return device_ptr_ != nullptr; }

  void *device_ptr() const { return device_ptr_; }

  // Synchronize host to device (for non-unified memory)
  void sync_to_device() {
    if (mode_ != MemoryMode::Shared && host_ptr_ && device_ptr_) {
      CUDA_CHECK(
          cudaMemcpy(device_ptr_, host_ptr_, size_, cudaMemcpyHostToDevice));
    }
  }

  // Synchronize device to host (for non-unified memory)
  void sync_to_host() {
    if (mode_ != MemoryMode::Shared && host_ptr_ && device_ptr_) {
      CUDA_CHECK(
          cudaMemcpy(host_ptr_, device_ptr_, size_, cudaMemcpyDeviceToHost));
    }
  }

private:
  void *device_ptr_;
  void *host_ptr_;
  size_t size_;
  MemoryMode mode_;
};

// ============================================================================
// CUDA Compute Kernel Implementation
// ============================================================================

class CUDAKernel : public ComputeKernel {
public:
  CUDAKernel(const std::string &name, CUfunction function,
             int max_threads_per_block)
      : name_(name), function_(function),
        max_threads_per_block_(max_threads_per_block) {}

  ~CUDAKernel() override {
    // CUDA function is owned by module, don't free here
  }

  const std::string &name() const override { return name_; }

  bool valid() const override { return function_ != nullptr; }

  size_t max_threads_per_threadgroup() const override {
    return max_threads_per_block_;
  }

  CUfunction cu_function() const { return function_; }

private:
  std::string name_;
  CUfunction function_;
  int max_threads_per_block_;
};

// ============================================================================
// CUDA Command Encoder Implementation
// ============================================================================

class CUDACommandEncoder : public CommandEncoder {
public:
  CUDACommandEncoder(CUstream stream) : stream_(stream), current_kernel_(nullptr) {}

  ~CUDACommandEncoder() override {
    // Stream is owned by backend, don't destroy here
  }

  void set_kernel(ComputeKernel *kernel) override {
    current_kernel_ = static_cast<CUDAKernel *>(kernel);
  }

  void set_buffer(Buffer *buffer, int index, size_t offset = 0) override {
    auto *cuda_buffer = static_cast<CUDABuffer *>(buffer);
    if (cuda_buffer) {
      void *ptr = static_cast<uint8_t *>(cuda_buffer->device_ptr()) + offset;
      kernel_args_[index] = ptr;
    }
  }

  void set_bytes(const void *data, size_t size, int index) override {
    // Store inline data in a persistent buffer
    inline_data_storage_.emplace_back(size);
    std::memcpy(inline_data_storage_.back().data(), data, size);
    kernel_args_[index] = inline_data_storage_.back().data();
  }

  void dispatch_threads(size_t width, size_t height = 1,
                        size_t depth = 1) override {
    if (!current_kernel_)
      return;

    // Calculate grid and block dimensions
    // Use a reasonable block size (256 threads)
    const int block_size = 256;
    size_t total_threads = width * height * depth;
    size_t num_blocks = (total_threads + block_size - 1) / block_size;

    dim3 grid(num_blocks);
    dim3 block(block_size);

    // Prepare kernel arguments
    std::vector<void *> args;
    for (auto &kv : kernel_args_) {
      args.push_back(&kv.second);
    }

    // Launch kernel
    CUresult result = cuLaunchKernel(
        current_kernel_->cu_function(), grid.x, grid.y, grid.z, block.x,
        block.y, block.z, 0, stream_, args.data(), nullptr);

    if (result != CUDA_SUCCESS) {
      const char *error_str;
      cuGetErrorString(result, &error_str);
      std::cerr << "CUDA kernel launch failed: " << error_str << std::endl;
    }
  }

  void dispatch_threadgroups(size_t groups_x, size_t groups_y, size_t groups_z,
                             size_t threads_x, size_t threads_y,
                             size_t threads_z) override {
    if (!current_kernel_)
      return;

    dim3 grid(groups_x, groups_y, groups_z);
    dim3 block(threads_x, threads_y, threads_z);

    // Prepare kernel arguments
    std::vector<void *> args;
    for (auto &kv : kernel_args_) {
      args.push_back(&kv.second);
    }

    // Launch kernel
    CUresult result = cuLaunchKernel(
        current_kernel_->cu_function(), grid.x, grid.y, grid.z, block.x,
        block.y, block.z, 0, stream_, args.data(), nullptr);

    if (result != CUDA_SUCCESS) {
      const char *error_str;
      cuGetErrorString(result, &error_str);
      std::cerr << "CUDA kernel launch failed: " << error_str << std::endl;
    }
  }

  void barrier() override {
    // Insert a stream synchronization
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  CUstream cu_stream() const { return stream_; }

private:
  CUstream stream_;
  CUDAKernel *current_kernel_;
  std::unordered_map<int, void *> kernel_args_;
  std::vector<std::vector<uint8_t>> inline_data_storage_;
};

// ============================================================================
// CUDA Backend Implementation
// ============================================================================

class CUDABackend : public Backend {
public:
  static CUDABackend &instance() {
    static CUDABackend instance;
    return instance;
  }

  BackendType type() const override { return BackendType::CUDA; }

  std::string device_name() const override {
    if (!initialized_)
      return "CUDA (not initialized)";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    return std::string(prop.name);
  }

  bool has_unified_memory() const override {
    if (!initialized_)
      return false;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    return prop.unifiedAddressing != 0;
  }

  size_t max_buffer_size() const override {
    if (!initialized_)
      return 0;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    return prop.totalGlobalMem;
  }

  size_t max_threadgroup_memory() const override {
    if (!initialized_)
      return 0;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    return prop.sharedMemPerBlock;
  }

  std::unique_ptr<Buffer> create_buffer(size_t size, MemoryMode mode,
                                        BufferUsage usage) override {
    if (!initialized_ || size == 0)
      return nullptr;

    void *device_ptr = nullptr;
    void *host_ptr = nullptr;

    if (mode == MemoryMode::Shared && has_unified_memory()) {
      // Use CUDA managed memory (unified memory)
      CUDA_CHECK(cudaMallocManaged(&device_ptr, size));
      host_ptr = device_ptr; // Same pointer for unified memory
    } else {
      // Allocate device memory
      CUDA_CHECK(cudaMalloc(&device_ptr, size));

      // Allocate host staging buffer for non-shared modes
      if (mode != MemoryMode::Private) {
        host_ptr = new uint8_t[size];
      }
    }

    if (device_ptr) {
      allocated_memory_ += size;
      peak_memory_ = std::max(peak_memory_.load(), allocated_memory_.load());
      return std::make_unique<CUDABuffer>(device_ptr, host_ptr, size, mode);
    }

    return nullptr;
  }

  std::unique_ptr<Buffer> create_buffer(const void *data, size_t size,
                                        MemoryMode mode) override {
    if (!initialized_ || !data || size == 0)
      return nullptr;

    auto buffer = create_buffer(size, mode, BufferUsage::Default);
    if (buffer) {
      // Copy initial data
      std::memcpy(buffer->data(), data, size);

      // For non-unified memory, sync to device
      auto *cuda_buffer = static_cast<CUDABuffer *>(buffer.get());
      cuda_buffer->sync_to_device();
    }

    return buffer;
  }

  std::unique_ptr<ComputeKernel>
  create_kernel(const std::string &name,
                const std::string &library_name) override {
    if (!initialized_)
      return nullptr;

    // Look up module
    auto it = modules_.find(library_name.empty() ? "default" : library_name);
    if (it == modules_.end()) {
      std::cerr << "CUDA module not found: "
                << (library_name.empty() ? "default" : library_name)
                << std::endl;
      return nullptr;
    }

    // Get function from module
    CUfunction function;
    CUresult result = cuModuleGetFunction(&function, it->second, name.c_str());
    if (result != CUDA_SUCCESS) {
      const char *error_str;
      cuGetErrorString(result, &error_str);
      std::cerr << "Failed to get CUDA function '" << name
                << "': " << error_str << std::endl;
      return nullptr;
    }

    // Get kernel attributes
    int max_threads;
    cuFuncGetAttribute(&max_threads,
                       CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);

    return std::make_unique<CUDAKernel>(name, function, max_threads);
  }

  bool compile_library(const std::string &name,
                       const std::string &source) override {
    if (!initialized_)
      return false;

    // Create NVRTC program
    nvrtcProgram prog;
    nvrtcResult result =
        nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, nullptr, nullptr);

    if (result != NVRTC_SUCCESS) {
      std::cerr << "Failed to create NVRTC program: "
                << nvrtcGetErrorString(result) << std::endl;
      return false;
    }

    // Compile options
    std::vector<const char *> opts = {
        "--gpu-architecture=compute_50", // Minimum compute capability
        "--std=c++14"};

    result = nvrtcCompileProgram(prog, opts.size(), opts.data());

    if (result != NVRTC_SUCCESS) {
      // Get compilation log
      size_t log_size;
      nvrtcGetProgramLogSize(prog, &log_size);
      std::vector<char> log(log_size);
      nvrtcGetProgramLog(prog, log.data());
      std::cerr << "CUDA compilation failed:\n" << log.data() << std::endl;
      nvrtcDestroyProgram(&prog);
      return false;
    }

    // Get PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());

    // Load module from PTX
    CUmodule module;
    CUresult cu_result = cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr);

    if (cu_result != CUDA_SUCCESS) {
      const char *error_str;
      cuGetErrorString(cu_result, &error_str);
      std::cerr << "Failed to load CUDA module: " << error_str << std::endl;
      nvrtcDestroyProgram(&prog);
      return false;
    }

    // Store module
    modules_[name] = module;

    nvrtcDestroyProgram(&prog);
    return true;
  }

  bool load_library(const std::string &name,
                    const std::string &path) override {
    if (!initialized_)
      return false;

    // Load module from file (PTX or cubin)
    CUmodule module;
    CUresult result = cuModuleLoad(&module, path.c_str());

    if (result != CUDA_SUCCESS) {
      const char *error_str;
      cuGetErrorString(result, &error_str);
      std::cerr << "Failed to load CUDA module from file: " << error_str
                << std::endl;
      return false;
    }

    modules_[name] = module;
    return true;
  }

  std::unique_ptr<CommandEncoder> create_encoder() override {
    if (!initialized_)
      return nullptr;

    return std::make_unique<CUDACommandEncoder>(stream_);
  }

  void submit_and_wait(CommandEncoder *encoder) override {
    if (!initialized_ || !encoder)
      return;

    auto *cuda_encoder = static_cast<CUDACommandEncoder *>(encoder);
    CUDA_CHECK(cudaStreamSynchronize(cuda_encoder->cu_stream()));
  }

  void submit(CommandEncoder *encoder) override {
    // Commands are already submitted in dispatch calls
    // No additional work needed
  }

  void synchronize() override {
    if (!initialized_)
      return;

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  size_t allocated_memory() const override { return allocated_memory_; }

  size_t peak_memory() const override { return peak_memory_; }

  void reset_peak_memory() override { peak_memory_ = allocated_memory_; }

private:
  CUDABackend() : initialized_(false), device_id_(0), allocated_memory_(0), peak_memory_(0) {
    // Initialize CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
      std::cerr << "[GPU Backend] No CUDA devices found" << std::endl;
      return;
    }

    // Initialize CUDA driver API
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
      std::cerr << "[GPU Backend] Failed to initialize CUDA driver API"
                << std::endl;
      return;
    }

    // Use device 0 by default
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Create default stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    initialized_ = true;

    std::cout << "[GPU Backend] CUDA initialized successfully" << std::endl;
    std::cout << "[GPU Backend] Device: " << device_name() << std::endl;
    std::cout << "[GPU Backend] Unified Memory: "
              << (has_unified_memory() ? "Yes" : "No") << std::endl;
  }

  ~CUDABackend() {
    if (initialized_) {
      // Destroy stream
      CUDA_CHECK(cudaStreamDestroy(stream_));

      // Unload modules
      for (auto &kv : modules_) {
        cuModuleUnload(kv.second);
      }

      // Reset device
      CUDA_CHECK(cudaDeviceReset());
    }
  }

  bool initialized_;
  int device_id_;
  CUstream stream_;
  std::unordered_map<std::string, CUmodule> modules_;
  std::atomic<size_t> allocated_memory_;
  std::atomic<size_t> peak_memory_;
};

// ============================================================================
// Backend static methods
// ============================================================================

Backend &Backend::get() { return CUDABackend::instance(); }

bool Backend::available() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

// ============================================================================
// ScopedTimer implementation
// ============================================================================

struct ScopedTimer::Impl {
  std::string name;
  std::chrono::high_resolution_clock::time_point start;
  std::function<void(double)> callback;
};

ScopedTimer::ScopedTimer(const std::string &name,
                         std::function<void(double)> callback)
    : impl_(std::make_unique<Impl>()) {
  impl_->name = name;
  impl_->start = std::chrono::high_resolution_clock::now();
  impl_->callback = callback;
}

ScopedTimer::~ScopedTimer() {
  double ms = elapsed_ms();
  if (impl_->callback) {
    impl_->callback(ms);
  }
}

double ScopedTimer::elapsed_ms() const {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(now - impl_->start);
  return duration.count() / 1000.0;
}

} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
