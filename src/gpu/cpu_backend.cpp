/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CPU Fallback Backend Implementation

  This provides stub implementations when no GPU backend is available.
  All GPU operations gracefully fall back to CPU.
*/

#ifndef USE_METAL

#include "backend.h"
#include "gpu_nnue_integration.h"
#include <chrono>
#include <cstring>
#include <iostream>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace MetalFish {
namespace GPU {

// Null buffer implementation
class NullBuffer : public Buffer {
public:
  NullBuffer(size_t size) : size_(size) {
    if (size > 0) {
      data_ = std::make_unique<uint8_t[]>(size);
    }
  }

  void *data() override { return data_.get(); }
  const void *data() const override { return data_.get(); }
  size_t size() const override { return size_; }
  bool valid() const override { return data_ != nullptr; }

private:
  std::unique_ptr<uint8_t[]> data_;
  size_t size_;
};

// Null kernel implementation
class NullKernel : public ComputeKernel {
public:
  NullKernel(const std::string &name) : name_(name) {}

  const std::string &name() const override { return name_; }
  bool valid() const override { return false; }
  size_t max_threads_per_threadgroup() const override { return 0; }

private:
  std::string name_;
};

// Null encoder implementation
class NullEncoder : public CommandEncoder {
public:
  void set_kernel(ComputeKernel *) override {}
  void set_buffer(Buffer *, int, size_t) override {}
  void set_bytes(const void *, size_t, int) override {}
  void dispatch_threads(size_t, size_t, size_t) override {}
  void dispatch_threadgroups(size_t, size_t, size_t, size_t, size_t,
                             size_t) override {}
  void barrier() override {}
};

// CPU fallback backend
class CPUBackend : public Backend {
public:
  static CPUBackend &instance() {
    static CPUBackend instance;
    return instance;
  }

  BackendType type() const override { return BackendType::None; }

  std::string device_name() const override { return "CPU (no GPU available)"; }
  bool has_unified_memory() const override { return true; }
  size_t max_buffer_size() const override { return SIZE_MAX; }
  size_t max_threadgroup_memory() const override { return 0; }

  size_t recommended_working_set_size() const override {
    // For CPU, return a reasonable portion of system memory
    return total_system_memory() / 4;
  }

  size_t total_system_memory() const override {
#ifdef __APPLE__
    return [[NSProcessInfo processInfo] physicalMemory];
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
#else
    return 8ULL * 1024 * 1024 * 1024; // Default 8GB
#endif
  }

  int gpu_core_count() const override {
    // No GPU, return 0
    return 0;
  }

  int max_threads_per_simd_group() const override {
    // For CPU, return typical SIMD width (AVX-512 = 16 floats, AVX2 = 8)
#ifdef __AVX512F__
    return 16;
#elif defined(__AVX2__) || defined(__AVX__)
    return 8;
#else
    return 4; // SSE
#endif
  }

  int recommended_batch_size() const override {
    // For CPU fallback, use smaller batches
    return 1;
  }

  std::unique_ptr<Buffer> create_buffer(size_t size, MemoryMode,
                                        BufferUsage) override {
    allocated_ += size;
    return std::make_unique<NullBuffer>(size);
  }

  std::unique_ptr<Buffer> create_buffer(const void *data, size_t size,
                                        MemoryMode) override {
    auto buffer = std::make_unique<NullBuffer>(size);
    if (data && buffer->data()) {
      std::memcpy(buffer->data(), data, size);
    }
    allocated_ += size;
    return buffer;
  }

  std::unique_ptr<ComputeKernel> create_kernel(const std::string &name,
                                               const std::string &) override {
    return std::make_unique<NullKernel>(name);
  }

  bool compile_library(const std::string &, const std::string &) override {
    return false;
  }

  bool load_library(const std::string &, const std::string &) override {
    return false;
  }

  std::unique_ptr<CommandEncoder> create_encoder() override {
    return std::make_unique<NullEncoder>();
  }

  void submit_and_wait(CommandEncoder *) override {}
  void submit(CommandEncoder *) override {}
  void submit_async(CommandEncoder *,
                    std::function<void()> completion_handler) override {
    // CPU fallback: just call the completion handler immediately
    if (completion_handler) {
      completion_handler();
    }
  }
  void synchronize() override {}

  size_t allocated_memory() const override { return allocated_; }
  size_t peak_memory() const override { return allocated_; }
  void reset_peak_memory() override {}

private:
  CPUBackend() {
    std::cout << "[GPU Backend] No GPU available, using CPU fallback"
              << std::endl;
  }

  size_t allocated_ = 0;
};

// Backend static methods
Backend &Backend::get() { return CPUBackend::instance(); }

bool Backend::available() { return false; }

// Shutdown functions - no-op for CPU backend since there's nothing to clean up
void shutdown_gpu_backend() {
  // No GPU resources to release in CPU fallback mode
}

bool gpu_backend_shutdown() {
  // CPU backend is always "shutdown" safe since there's no actual GPU
  return false;
}

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

} // namespace GPU
} // namespace MetalFish

// ============================================================================
// GPUTuningParams Implementation (CPU fallback)
// ============================================================================

namespace MetalFish::GPU {

EvalStrategy GPUTuningParams::select_strategy(int batch_size) const {
  // CPU fallback always uses CPU
  (void)batch_size;
  return EvalStrategy::CPU_FALLBACK;
}

} // namespace MetalFish::GPU

#endif // !USE_METAL
