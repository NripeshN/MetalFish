/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  ROCm/HIP Backend Implementation

  Implements the GPU backend interface for AMD ROCm/HIP.
  Supports discrete and APU configurations.
*/

#ifdef USE_ROCM

#include "../backend.h"
#include <atomic>
#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace MetalFish {
namespace GPU {

// Helper macro for HIP error checking
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << hipGetErrorString(err) << std::endl;                        \
    }                                                                          \
  } while (0)

// ============================================================================
// ROCm Buffer Implementation
// ============================================================================

class ROCmBuffer : public Buffer {
public:
  ROCmBuffer(void *ptr, size_t size, bool is_device_memory)
      : ptr_(ptr), size_(size), is_device_memory_(is_device_memory),
        owns_memory_(true) {}

  ~ROCmBuffer() override {
    if (owns_memory_ && ptr_) {
      if (is_device_memory_) {
        HIP_CHECK(hipFree(ptr_));
      } else {
        HIP_CHECK(hipHostFree(ptr_));
      }
    }
  }

  void *data() override { return ptr_; }

  const void *data() const override { return ptr_; }

  size_t size() const override { return size_; }

  bool valid() const override { return ptr_ != nullptr; }

private:
  void *ptr_;
  size_t size_;
  bool is_device_memory_;
  bool owns_memory_;
};

// ============================================================================
// ROCm Compute Kernel Implementation
// ============================================================================

class ROCmKernel : public ComputeKernel {
public:
  ROCmKernel(const std::string &name, hipFunction_t function)
      : name_(name), function_(function) {}

  ~ROCmKernel() override {
    // HIP functions are managed by the module, not individually
  }

  const std::string &name() const override { return name_; }

  bool valid() const override { return function_ != nullptr; }

  size_t max_threads_per_threadgroup() const override {
    // Query device properties for max threads per block
    int max_threads;
    HIP_CHECK(hipFuncGetAttribute(&max_threads,
                                  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  function_));
    return max_threads;
  }

  hipFunction_t hip_function() const { return function_; }

private:
  std::string name_;
  hipFunction_t function_;
};

// ============================================================================
// ROCm Command Encoder Implementation
// ============================================================================

class ROCmCommandEncoder : public CommandEncoder {
public:
  ROCmCommandEncoder(hipStream_t stream)
      : stream_(stream), current_kernel_(nullptr) {}

  ~ROCmCommandEncoder() override {
    // Stream cleanup handled by backend
  }

  void set_kernel(ComputeKernel *kernel) override {
    current_kernel_ = static_cast<ROCmKernel *>(kernel);
    kernel_params_.clear();
  }

  void set_buffer(Buffer *buffer, int index, size_t offset = 0) override {
    auto *rocm_buffer = static_cast<ROCmBuffer *>(buffer);
    if (rocm_buffer) {
      void *ptr = static_cast<char *>(rocm_buffer->data()) + offset;
      while (kernel_params_.size() <= static_cast<size_t>(index)) {
        kernel_params_.push_back(nullptr);
      }
      kernel_params_[index] = ptr;
    }
  }

  void set_bytes(const void *data, size_t size, int index) override {
    // For small constant data, we need to allocate device memory
    // This is a simplified implementation
    void *d_ptr;
    HIP_CHECK(hipMalloc(&d_ptr, size));
    HIP_CHECK(hipMemcpyAsync(d_ptr, data, size, hipMemcpyHostToDevice, stream_));
    
    while (kernel_params_.size() <= static_cast<size_t>(index)) {
      kernel_params_.push_back(nullptr);
    }
    kernel_params_[index] = d_ptr;
    temp_allocations_.push_back(d_ptr);
  }

  void dispatch_threads(size_t width, size_t height, size_t depth) override {
    if (!current_kernel_)
      return;

    // Calculate grid and block dimensions
    size_t max_threads = current_kernel_->max_threads_per_threadgroup();
    size_t block_x = std::min(width, max_threads);
    size_t block_y = std::min(height, max_threads / block_x);
    size_t block_z = std::min(depth, max_threads / (block_x * block_y));

    dim3 block_dim(block_x, block_y, block_z);
    dim3 grid_dim((width + block_x - 1) / block_x,
                  (height + block_y - 1) / block_y,
                  (depth + block_z - 1) / block_z);

    HIP_CHECK(hipModuleLaunchKernel(
        current_kernel_->hip_function(), grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z, 0, stream_,
        kernel_params_.data(), nullptr));
  }

  void dispatch_threadgroups(size_t groups_x, size_t groups_y, size_t groups_z,
                             size_t threads_x, size_t threads_y,
                             size_t threads_z) override {
    if (!current_kernel_)
      return;

    dim3 grid_dim(groups_x, groups_y, groups_z);
    dim3 block_dim(threads_x, threads_y, threads_z);

    HIP_CHECK(hipModuleLaunchKernel(
        current_kernel_->hip_function(), grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z, 0, stream_,
        kernel_params_.data(), nullptr));
  }

  void barrier() override {
    // Insert a stream synchronization point
    HIP_CHECK(hipStreamSynchronize(stream_));
  }

  hipStream_t get_stream() const { return stream_; }

  void cleanup_temp_allocations() {
    for (void *ptr : temp_allocations_) {
      HIP_CHECK(hipFree(ptr));
    }
    temp_allocations_.clear();
  }

private:
  hipStream_t stream_;
  ROCmKernel *current_kernel_;
  std::vector<void *> kernel_params_;
  std::vector<void *> temp_allocations_;
};

// ============================================================================
// ROCm Backend Implementation
// ============================================================================

class ROCmBackend : public Backend {
public:
  static ROCmBackend &instance() {
    static ROCmBackend instance;
    return instance;
  }

  BackendType type() const override { return BackendType::ROCm; }

  std::string device_name() const override {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_));
    return std::string(props.name);
  }

  bool has_unified_memory() const override {
    // Check if device supports unified memory (APUs typically do)
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_));
    return props.integrated != 0;
  }

  size_t max_buffer_size() const override {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_));
    return props.totalGlobalMem;
  }

  size_t max_threadgroup_memory() const override {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_));
    return props.sharedMemPerBlock;
  }

  std::unique_ptr<Buffer> create_buffer(size_t size, MemoryMode mode,
                                        BufferUsage) override {
    void *ptr = nullptr;
    bool is_device = false;

    switch (mode) {
    case MemoryMode::Shared:
      // Use host-accessible memory
      HIP_CHECK(hipHostMalloc(&ptr, size, hipHostMallocMapped));
      break;
    case MemoryMode::Private:
      // Device-only memory
      HIP_CHECK(hipMalloc(&ptr, size));
      is_device = true;
      break;
    case MemoryMode::Managed:
      // Managed/unified memory
      HIP_CHECK(hipMallocManaged(&ptr, size));
      is_device = true;
      break;
    }

    if (ptr) {
      allocated_ += size;
      peak_memory_ = std::max(peak_memory_, allocated_);
      return std::make_unique<ROCmBuffer>(ptr, size, is_device);
    }

    return nullptr;
  }

  std::unique_ptr<Buffer> create_buffer(const void *data, size_t size,
                                        MemoryMode mode) override {
    auto buffer = create_buffer(size, mode);
    if (buffer && data) {
      HIP_CHECK(
          hipMemcpy(buffer->data(), data, size, hipMemcpyHostToDevice));
    }
    return buffer;
  }

  std::unique_ptr<ComputeKernel> create_kernel(const std::string &name,
                                               const std::string &library) override {
    std::lock_guard<std::mutex> lock(modules_mutex_);

    // Find the module
    auto mod_it = modules_.find(library.empty() ? "default" : library);
    if (mod_it == modules_.end()) {
      std::cerr << "ROCm module not found: " << library << std::endl;
      return nullptr;
    }

    // Get the kernel function
    hipFunction_t function;
    hipError_t err =
        hipModuleGetFunction(&function, mod_it->second, name.c_str());
    if (err != hipSuccess) {
      std::cerr << "Failed to get kernel function '" << name
                << "': " << hipGetErrorString(err) << std::endl;
      return nullptr;
    }

    return std::make_unique<ROCmKernel>(name, function);
  }

  bool compile_library(const std::string &name,
                       const std::string &source) override {
    // ROCm uses offline compilation via hipcc/rocm-clang
    // Runtime compilation would require hiprtc library
    // For now, this is a placeholder
    std::cerr << "ROCm runtime compilation not yet implemented" << std::endl;
    return false;
  }

  bool load_library(const std::string &name, const std::string &path) override {
    std::lock_guard<std::mutex> lock(modules_mutex_);

    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, path.c_str());
    if (err != hipSuccess) {
      std::cerr << "Failed to load ROCm module from " << path << ": "
                << hipGetErrorString(err) << std::endl;
      return false;
    }

    modules_[name] = module;
    return true;
  }

  std::unique_ptr<CommandEncoder> create_encoder() override {
    return std::make_unique<ROCmCommandEncoder>(stream_);
  }

  void submit_and_wait(CommandEncoder *encoder) override {
    auto *rocm_encoder = static_cast<ROCmCommandEncoder *>(encoder);
    HIP_CHECK(hipStreamSynchronize(rocm_encoder->get_stream()));
    rocm_encoder->cleanup_temp_allocations();
  }

  void submit(CommandEncoder *encoder) override {
    // Asynchronous submission - no wait
    (void)encoder;
  }

  void synchronize() override { HIP_CHECK(hipStreamSynchronize(stream_)); }

  size_t allocated_memory() const override { return allocated_; }

  size_t peak_memory() const override { return peak_memory_; }

  void reset_peak_memory() override { peak_memory_ = allocated_; }

private:
  ROCmBackend() : device_(0), allocated_(0), peak_memory_(0) {
    // Initialize HIP
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
      std::cerr << "No ROCm devices found" << std::endl;
      throw std::runtime_error("No ROCm devices available");
    }

    // Use the first device
    HIP_CHECK(hipSetDevice(device_));

    // Create a stream for async operations
    HIP_CHECK(hipStreamCreate(&stream_));

    std::cout << "ROCm backend initialized with device: " << device_name()
              << std::endl;
  }

  ~ROCmBackend() {
    // Cleanup modules
    for (auto &[name, module] : modules_) {
      hipModuleUnload(module);
    }

    // Destroy stream
    if (stream_) {
      hipStreamDestroy(stream_);
    }
  }

  int device_;
  hipStream_t stream_;
  std::unordered_map<std::string, hipModule_t> modules_;
  std::mutex modules_mutex_;
  std::atomic<size_t> allocated_;
  std::atomic<size_t> peak_memory_;
};

// ============================================================================
// Backend Factory
// ============================================================================

Backend &Backend::get() {
#ifdef USE_ROCM
  return ROCmBackend::instance();
#else
  throw std::runtime_error("ROCm backend not compiled");
#endif
}

bool Backend::available() {
#ifdef USE_ROCM
  int device_count;
  hipError_t err = hipGetDeviceCount(&device_count);
  return err == hipSuccess && device_count > 0;
#else
  return false;
#endif
}

// ============================================================================
// Scoped Timer Implementation
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
  impl_->callback = callback;
  impl_->start = std::chrono::high_resolution_clock::now();
}

ScopedTimer::~ScopedTimer() {
  auto elapsed = elapsed_ms();
  if (impl_->callback) {
    impl_->callback(elapsed);
  }
}

double ScopedTimer::elapsed_ms() const {
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - impl_->start).count();
}

} // namespace GPU
} // namespace MetalFish

#endif // USE_ROCM
