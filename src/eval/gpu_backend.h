/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace MetalFish {
namespace GPU {

// Cross-platform helper to compute next power of 2
namespace detail {
inline int next_power_of_2(int v) {
  if (v <= 0)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}
} // namespace detail

// Forward declarations
class Buffer;
class CommandEncoder;
class ComputeKernel;

enum class BackendType {
  None,  // CPU fallback
  Metal, // Apple Metal
  CUDA   // NVIDIA CUDA (future)
};

// Buffer usage hints for optimal memory allocation
enum class BufferUsage {
  Default,
  Transient,
  Persistent,
  Streaming
};

// Memory access mode (relevant for unified memory systems)
enum class MemoryMode {
  Shared,  // CPU and GPU can access (unified memory)
  Private, // GPU only (fastest for GPU-only data)
  Managed  // System manages CPU/GPU synchronization
};

class Buffer {
public:
  virtual ~Buffer() = default;

  virtual void *data() = 0;
  virtual const void *data() const = 0;
  virtual size_t size() const = 0;
  virtual bool valid() const = 0;

  template <typename T> T *as() { return static_cast<T *>(data()); }

  template <typename T> const T *as() const {
    return static_cast<const T *>(data());
  }

  template <typename T> size_t count() const { return size() / sizeof(T); }
};

class ComputeKernel {
public:
  virtual ~ComputeKernel() = default;

  virtual const std::string &name() const = 0;
  virtual bool valid() const = 0;
  virtual size_t max_threads_per_threadgroup() const = 0;
};

class CommandEncoder {
public:
  virtual ~CommandEncoder() = default;

  virtual void set_kernel(ComputeKernel *kernel) = 0;
  virtual void set_buffer(Buffer *buffer, int index, size_t offset = 0) = 0;
  virtual void set_bytes(const void *data, size_t size, int index) = 0;

  template <typename T> void set_value(const T &value, int index) {
    set_bytes(&value, sizeof(T), index);
  }

  virtual void dispatch_threads(size_t width, size_t height = 1,
                                size_t depth = 1) = 0;
  virtual void dispatch_threadgroups(size_t groups_x, size_t groups_y,
                                     size_t groups_z, size_t threads_x,
                                     size_t threads_y, size_t threads_z) = 0;

  virtual void barrier() = 0;
};

class Backend {
public:
  virtual ~Backend() = default;

  static Backend &get();
  static bool available();

  virtual BackendType type() const = 0;

  virtual std::string device_name() const = 0;
  virtual bool has_unified_memory() const = 0;
  virtual size_t max_buffer_size() const = 0;
  virtual size_t max_threadgroup_memory() const = 0;

  // Hardware capabilities (for dynamic tuning)
  virtual size_t recommended_working_set_size() const = 0;
  virtual size_t total_system_memory() const = 0;
  virtual int gpu_core_count() const = 0;
  virtual int max_threads_per_simd_group() const = 0;

  virtual int recommended_batch_size() const {
    int cores = gpu_core_count();
    if (cores <= 0)
      return 128;
    // ~16 positions per GPU core is a good heuristic
    int batch = detail::next_power_of_2(cores * 16);
    return std::max(32, std::min(512, batch));
  }

  virtual std::unique_ptr<Buffer>
  create_buffer(size_t size, MemoryMode mode = MemoryMode::Shared,
                BufferUsage usage = BufferUsage::Default) = 0;

  virtual std::unique_ptr<Buffer>
  create_buffer(const void *data, size_t size,
                MemoryMode mode = MemoryMode::Shared) = 0;

  template <typename T>
  std::unique_ptr<Buffer> create_buffer(const std::vector<T> &data,
                                        MemoryMode mode = MemoryMode::Shared) {
    return create_buffer(data.data(), data.size() * sizeof(T), mode);
  }

  virtual std::unique_ptr<ComputeKernel>
  create_kernel(const std::string &name, const std::string &library = "") = 0;

  virtual bool compile_library(const std::string &name,
                               const std::string &source) = 0;

  virtual bool load_library(const std::string &name,
                            const std::string &path) = 0;

  virtual std::unique_ptr<CommandEncoder> create_encoder() = 0;

  // Create encoder on a parallel queue for async work (multiple queues reduce
  // contention)
  virtual std::unique_ptr<CommandEncoder> create_parallel_encoder() {
    return create_encoder();
  }

  virtual size_t num_parallel_queues() const { return 1; }

  virtual void submit_and_wait(CommandEncoder *encoder) = 0;
  virtual void submit(CommandEncoder *encoder) = 0;
  virtual void submit_async(CommandEncoder *encoder,
                            std::function<void()> completion_handler) = 0;
  virtual void synchronize() = 0;

  virtual size_t allocated_memory() const = 0;
  virtual size_t peak_memory() const = 0;
  virtual void reset_peak_memory() = 0;

protected:
  Backend() = default;
};

class ScopedTimer {
public:
  ScopedTimer(const std::string &name,
              std::function<void(double)> callback = nullptr);
  ~ScopedTimer();

  double elapsed_ms() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

inline bool gpu_available() { return Backend::available(); }

inline Backend &gpu() { return Backend::get(); }

// Shutdown the GPU backend - must be called before program exit
// This ensures all GPU resources are released before static destruction
void shutdown_gpu_backend();

bool gpu_backend_shutdown();

} // namespace GPU
} // namespace MetalFish
