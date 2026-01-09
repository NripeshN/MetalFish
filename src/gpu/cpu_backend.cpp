/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CPU Fallback Backend Implementation
  
  This provides stub implementations when no GPU backend is available.
  All GPU operations gracefully fall back to CPU.
*/

#ifndef USE_METAL

#include "backend.h"
#include <chrono>
#include <iostream>

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

#endif // !USE_METAL
