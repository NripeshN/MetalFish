/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Backend Implementation

  Implements the GPU backend interface for NVIDIA CUDA.
  Optimized for modern NVIDIA GPUs with tensor cores when available.

  Note: This implementation uses only the CUDA Runtime API to avoid
  dependency on libcuda.so (driver library) which requires an actual GPU.
  Runtime kernel compilation (NVRTC) is optional and guarded.
*/

#ifdef USE_CUDA

#include "cuda_backend.h"
#include <atomic>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>

// Only include driver API and NVRTC if not building without them
#ifndef NO_CUDA_DRIVER_API
#include <cuda.h>
#endif

#ifndef NO_NVRTC
#include <nvrtc.h>
#endif

namespace MetalFish {
namespace GPU {

// ============================================================================
// CUDA Error Checking Utilities
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << " at "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_VOID(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << " at "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return;                                                                  \
    }                                                                          \
  } while (0)

#ifndef NO_NVRTC
#define NVRTC_CHECK(call)                                                      \
  do {                                                                         \
    nvrtcResult result = call;                                                 \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "[NVRTC Error] " << nvrtcGetErrorString(result) << " at "   \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return false;                                                            \
    }                                                                          \
  } while (0)
#endif

// ============================================================================
// CUDABuffer Implementation
// ============================================================================

CUDABuffer::CUDABuffer(void *device_ptr, void *host_ptr, size_t size,
                       bool unified)
    : device_ptr_(device_ptr), host_ptr_(host_ptr), size_(size),
      unified_(unified) {}

CUDABuffer::~CUDABuffer() {
  if (device_ptr_) {
    if (unified_) {
      cudaFree(device_ptr_);
    } else {
      cudaFree(device_ptr_);
      if (host_ptr_) {
        cudaFreeHost(host_ptr_);
      }
    }
  }
}

void *CUDABuffer::data() {
  if (unified_) {
    return device_ptr_;
  }
  return host_ptr_;
}

const void *CUDABuffer::data() const {
  if (unified_) {
    return device_ptr_;
  }
  return host_ptr_;
}

void CUDABuffer::sync_to_device() {
  if (!unified_ && host_ptr_ && device_ptr_) {
    cudaMemcpy(device_ptr_, host_ptr_, size_, cudaMemcpyHostToDevice);
  }
}

void CUDABuffer::sync_to_host() {
  if (!unified_ && host_ptr_ && device_ptr_) {
    cudaMemcpy(host_ptr_, device_ptr_, size_, cudaMemcpyDeviceToHost);
  }
}

// ============================================================================
// CUDAKernel Implementation
// ============================================================================

CUDAKernel::CUDAKernel(const std::string &name, void *function)
    : name_(name), function_(function), max_threads_per_block_(1024) {
  if (function_) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, function_) == cudaSuccess) {
      max_threads_per_block_ = attr.maxThreadsPerBlock;
    }
  }
}

CUDAKernel::~CUDAKernel() {
  // Kernels are managed by the module, don't free here
}

size_t CUDAKernel::max_threads_per_threadgroup() const {
  return max_threads_per_block_;
}

// ============================================================================
// CUDACommandEncoder Implementation
// ============================================================================

CUDACommandEncoder::CUDACommandEncoder(cudaStream_t stream)
    : stream_(stream), current_kernel_(nullptr), owns_stream_(false) {
  if (stream_ == nullptr) {
    cudaStreamCreate(&stream_);
    owns_stream_ = true;
  }
  buffer_args_.resize(16, nullptr);
  const_data_.resize(16);
}

CUDACommandEncoder::~CUDACommandEncoder() {
  if (owns_stream_ && stream_) {
    cudaStreamDestroy(stream_);
  }
}

void CUDACommandEncoder::set_kernel(ComputeKernel *kernel) {
  current_kernel_ = static_cast<CUDAKernel *>(kernel);
}

void CUDACommandEncoder::set_buffer(Buffer *buffer, int index, size_t offset) {
  if (index < 0 || index >= static_cast<int>(buffer_args_.size())) {
    return;
  }
  auto *cuda_buffer = static_cast<CUDABuffer *>(buffer);
  if (cuda_buffer) {
    buffer_args_[index] =
        static_cast<char *>(cuda_buffer->device_data()) + offset;
  }
}

void CUDACommandEncoder::set_bytes(const void *data, size_t size, int index) {
  if (index < 0 || index >= static_cast<int>(const_data_.size())) {
    return;
  }
  const_data_[index].resize(size);
  std::memcpy(const_data_[index].data(), data, size);
  buffer_args_[index] = const_data_[index].data();
}

void CUDACommandEncoder::dispatch_threads(size_t width, size_t height,
                                          size_t depth) {
  if (!current_kernel_ || !current_kernel_->valid()) {
    return;
  }

  // Calculate optimal block dimensions
  int max_threads = current_kernel_->max_threads_per_threadgroup();
  dim3 block_dim;
  dim3 grid_dim;

  if (depth > 1) {
    // 3D dispatch
    block_dim = dim3(8, 8, 8);
    grid_dim = dim3((width + block_dim.x - 1) / block_dim.x,
                    (height + block_dim.y - 1) / block_dim.y,
                    (depth + block_dim.z - 1) / block_dim.z);
  } else if (height > 1) {
    // 2D dispatch
    block_dim = dim3(16, 16, 1);
    grid_dim = dim3((width + block_dim.x - 1) / block_dim.x,
                    (height + block_dim.y - 1) / block_dim.y, 1);
  } else {
    // 1D dispatch
    block_dim = dim3(std::min(static_cast<size_t>(max_threads), width), 1, 1);
    grid_dim = dim3((width + block_dim.x - 1) / block_dim.x, 1, 1);
  }

  // Prepare kernel arguments
  std::vector<void *> args;
  for (size_t i = 0; i < buffer_args_.size(); ++i) {
    if (buffer_args_[i]) {
      args.push_back(&buffer_args_[i]);
    }
  }

  // Launch kernel
  cudaLaunchKernel(current_kernel_->cuda_function(), grid_dim, block_dim,
                   args.data(), 0, stream_);
}

void CUDACommandEncoder::dispatch_threadgroups(size_t groups_x, size_t groups_y,
                                               size_t groups_z,
                                               size_t threads_x,
                                               size_t threads_y,
                                               size_t threads_z) {
  if (!current_kernel_ || !current_kernel_->valid()) {
    return;
  }

  dim3 grid_dim(groups_x, groups_y, groups_z);
  dim3 block_dim(threads_x, threads_y, threads_z);

  std::vector<void *> args;
  for (size_t i = 0; i < buffer_args_.size(); ++i) {
    if (buffer_args_[i]) {
      args.push_back(&buffer_args_[i]);
    }
  }

  cudaLaunchKernel(current_kernel_->cuda_function(), grid_dim, block_dim,
                   args.data(), 0, stream_);
}

void CUDACommandEncoder::barrier() { cudaStreamSynchronize(stream_); }

// ============================================================================
// CUDABackend Implementation
// ============================================================================

CUDABackend::CUDABackend()
    : device_id_(-1), compute_capability_major_(0),
      compute_capability_minor_(0), total_memory_(0), multiprocessor_count_(0),
      unified_memory_supported_(false), default_stream_(nullptr),
      stream_index_(0), allocated_memory_(0), peak_memory_(0),
      initialized_(false) {}

CUDABackend::~CUDABackend() { cleanup(); }

CUDABackend &CUDABackend::instance() {
  static CUDABackend instance;
  if (!instance.initialized_) {
    instance.initialize();
  }
  return instance;
}

bool CUDABackend::is_available() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

bool CUDABackend::initialize() {
  if (initialized_) {
    return true;
  }

  // Get device count
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "[CUDA Backend] No CUDA devices found" << std::endl;
    return false;
  }

  // Select best device (highest compute capability)
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

  device_id_ = best_device;
  cudaSetDevice(device_id_);

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id_);

  device_name_ = prop.name;
  compute_capability_major_ = prop.major;
  compute_capability_minor_ = prop.minor;
  total_memory_ = prop.totalGlobalMem;
  multiprocessor_count_ = prop.multiProcessorCount;
  unified_memory_supported_ = prop.managedMemory != 0;

  // Create default stream
  cudaStreamCreate(&default_stream_);

  // Create parallel streams
  const int num_streams = 4;
  parallel_streams_.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamCreate(&parallel_streams_[i]);
  }

  initialized_ = true;

  std::cout << "[CUDA Backend] Initialized: " << device_name_ << std::endl;
  std::cout << "[CUDA Backend] Compute Capability: "
            << compute_capability_major_ << "." << compute_capability_minor_
            << std::endl;
  std::cout << "[CUDA Backend] Total Memory: " << total_memory_ / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "[CUDA Backend] Multiprocessors: " << multiprocessor_count_
            << std::endl;
  std::cout << "[CUDA Backend] Unified Memory: "
            << (unified_memory_supported_ ? "Yes" : "No") << std::endl;

  return true;
}

void CUDABackend::cleanup() {
  if (!initialized_) {
    return;
  }

  // Destroy streams
  if (default_stream_) {
    cudaStreamDestroy(default_stream_);
    default_stream_ = nullptr;
  }
  for (auto &stream : parallel_streams_) {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  parallel_streams_.clear();

  // Unload modules (only if driver API is available)
#ifndef NO_CUDA_DRIVER_API
  for (auto &[name, module] : modules_) {
    if (module) {
      cuModuleUnload(static_cast<CUmodule>(module));
    }
  }
#endif
  modules_.clear();
  kernels_.clear();

  initialized_ = false;
}

std::string CUDABackend::device_name() const { return device_name_; }

bool CUDABackend::has_unified_memory() const {
  return unified_memory_supported_;
}

size_t CUDABackend::max_buffer_size() const { return total_memory_; }

size_t CUDABackend::max_threadgroup_memory() const {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id_);
  return prop.sharedMemPerBlock;
}

std::unique_ptr<Buffer> CUDABackend::create_buffer(size_t size, MemoryMode mode,
                                                   BufferUsage usage) {
  if (!initialized_ || size == 0) {
    return nullptr;
  }

  void *device_ptr = nullptr;
  void *host_ptr = nullptr;
  bool unified = false;

  if (mode == MemoryMode::Shared && unified_memory_supported_) {
    // Use unified memory
    cudaError_t err = cudaMallocManaged(&device_ptr, size);
    if (err != cudaSuccess) {
      return nullptr;
    }
    unified = true;
  } else {
    // Allocate device and host memory separately
    cudaError_t err = cudaMalloc(&device_ptr, size);
    if (err != cudaSuccess) {
      return nullptr;
    }

    if (mode != MemoryMode::Private) {
      err = cudaMallocHost(&host_ptr, size);
      if (err != cudaSuccess) {
        cudaFree(device_ptr);
        return nullptr;
      }
    }
  }

  allocated_memory_ += size;
  peak_memory_ = std::max(peak_memory_, allocated_memory_);

  return std::make_unique<CUDABuffer>(device_ptr, host_ptr, size, unified);
}

std::unique_ptr<Buffer>
CUDABackend::create_buffer(const void *data, size_t size, MemoryMode mode) {
  auto buffer = create_buffer(size, mode);
  if (buffer && data) {
    auto *cuda_buffer = static_cast<CUDABuffer *>(buffer.get());
    if (cuda_buffer->data()) {
      std::memcpy(cuda_buffer->data(), data, size);
      cuda_buffer->sync_to_device();
    }
  }
  return buffer;
}

std::unique_ptr<ComputeKernel>
CUDABackend::create_kernel(const std::string &name,
                           const std::string &library_name) {
  if (!initialized_) {
    return nullptr;
  }

  // Look up kernel in cache
  std::string key = library_name.empty() ? name : library_name + "::" + name;
  auto it = kernels_.find(key);
  if (it != kernels_.end()) {
    return std::make_unique<CUDAKernel>(name, it->second);
  }

#ifndef NO_CUDA_DRIVER_API
  // Try to get from module (requires driver API)
  auto mod_it = modules_.find(library_name);
  if (mod_it != modules_.end()) {
    CUfunction func;
    CUresult result = cuModuleGetFunction(
        &func, static_cast<CUmodule>(mod_it->second), name.c_str());
    if (result == CUDA_SUCCESS) {
      kernels_[key] = func;
      return std::make_unique<CUDAKernel>(name, func);
    }
  }
#endif

  // Kernel not found - this is expected when using pre-compiled kernels
  // The actual kernel dispatch happens through the cuda_* host functions
  return nullptr;
}

bool CUDABackend::compile_library(const std::string &name,
                                  const std::string &source) {
#if defined(NO_NVRTC) || defined(NO_CUDA_DRIVER_API)
  // Runtime compilation not available without NVRTC and driver API
  std::cerr << "[CUDA] Runtime compilation not available (NO_NVRTC or "
               "NO_CUDA_DRIVER_API defined)"
            << std::endl;
  return false;
#else
  if (!initialized_) {
    return false;
  }

  // Create NVRTC program
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0,
                                 nullptr, nullptr));

  // Set compilation options
  std::string arch_opt = "--gpu-architecture=compute_" +
                         std::to_string(compute_capability_major_) +
                         std::to_string(compute_capability_minor_);
  const char *opts[] = {arch_opt.c_str(), "--std=c++17", "-default-device"};

  // Compile
  nvrtcResult compile_result = nvrtcCompileProgram(prog, 3, opts);

  // Get log
  size_t log_size;
  nvrtcGetProgramLogSize(prog, &log_size);
  if (log_size > 1) {
    std::vector<char> log(log_size);
    nvrtcGetProgramLog(prog, log.data());
    if (compile_result != NVRTC_SUCCESS) {
      std::cerr << "[CUDA] Compilation log for " << name << ":\n"
                << log.data() << std::endl;
    }
  }

  if (compile_result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    return false;
  }

  // Get PTX
  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
  std::vector<char> ptx(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

  nvrtcDestroyProgram(&prog);

  // Load module
  CUmodule module;
  CUresult result =
      cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr);
  if (result != CUDA_SUCCESS) {
    std::cerr << "[CUDA] Failed to load module: " << name << std::endl;
    return false;
  }

  // Store module
  if (modules_[name]) {
    cuModuleUnload(static_cast<CUmodule>(modules_[name]));
  }
  modules_[name] = module;

  return true;
#endif
}

bool CUDABackend::load_library(const std::string &name,
                               const std::string &path) {
#ifdef NO_CUDA_DRIVER_API
  // Library loading not available without driver API
  std::cerr
      << "[CUDA] Library loading not available (NO_CUDA_DRIVER_API defined)"
      << std::endl;
  return false;
#else
  if (!initialized_) {
    return false;
  }

  CUmodule module;
  CUresult result = cuModuleLoad(&module, path.c_str());
  if (result != CUDA_SUCCESS) {
    std::cerr << "[CUDA] Failed to load library: " << path << std::endl;
    return false;
  }

  if (modules_[name]) {
    cuModuleUnload(static_cast<CUmodule>(modules_[name]));
  }
  modules_[name] = module;

  return true;
#endif
}

std::unique_ptr<CommandEncoder> CUDABackend::create_encoder() {
  if (!initialized_) {
    return nullptr;
  }
  return std::make_unique<CUDACommandEncoder>(default_stream_);
}

std::unique_ptr<CommandEncoder> CUDABackend::create_parallel_encoder() {
  if (!initialized_ || parallel_streams_.empty()) {
    return create_encoder();
  }
  size_t idx = stream_index_++ % parallel_streams_.size();
  return std::make_unique<CUDACommandEncoder>(parallel_streams_[idx]);
}

size_t CUDABackend::num_parallel_queues() const {
  return parallel_streams_.size();
}

void CUDABackend::submit_and_wait(CommandEncoder *encoder) {
  auto *cuda_encoder = static_cast<CUDACommandEncoder *>(encoder);
  if (cuda_encoder) {
    cudaStreamSynchronize(cuda_encoder->stream());
  }
}

void CUDABackend::submit(CommandEncoder *encoder) {
  // Commands are already submitted when dispatch is called
  // This is a no-op for CUDA
}

void CUDABackend::submit_async(CommandEncoder *encoder,
                               std::function<void()> completion_handler) {
  auto *cuda_encoder = static_cast<CUDACommandEncoder *>(encoder);
  if (cuda_encoder && completion_handler) {
    cudaStreamAddCallback(
        cuda_encoder->stream(),
        [](cudaStream_t stream, cudaError_t status, void *userData) {
          auto *handler = static_cast<std::function<void()> *>(userData);
          (*handler)();
          delete handler;
        },
        new std::function<void()>(completion_handler), 0);
  }
}

void CUDABackend::synchronize() { cudaDeviceSynchronize(); }

// ============================================================================
// Backend Interface Implementation (when CUDA is the active backend)
// ============================================================================

#ifndef USE_METAL
// Only implement these if Metal is not available

Backend &Backend::get() { return CUDABackend::instance(); }

bool Backend::available() { return CUDABackend::is_available(); }

// ScopedTimer implementation
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
  return std::chrono::duration<double, std::milli>(now - impl_->start).count();
}

#endif // !USE_METAL

} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
