/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Metal Backend Implementation

  Implements the GPU backend interface for Apple Metal.
  Takes advantage of unified memory for zero-copy data access.
*/

#ifdef __APPLE__

#include "../gpu_backend.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace MetalFish {
namespace GPU {

static std::atomic<bool> g_backend_shutdown{false};
static std::atomic<bool> g_backend_initialized{false};

class MetalBuffer : public Buffer {
public:
  MetalBuffer(id<MTLBuffer> buffer, size_t size)
      : buffer_(buffer), size_(size) {
    if (buffer_) {
      [buffer_ retain];
    }
  }

  ~MetalBuffer() override {
    if (buffer_) {
      [buffer_ release];
    }
  }

  void *data() override { return buffer_ ? [buffer_ contents] : nullptr; }

  const void *data() const override {
    return buffer_ ? [buffer_ contents] : nullptr;
  }

  size_t size() const override { return size_; }

  bool valid() const override { return buffer_ != nil; }

  id<MTLBuffer> mtl_buffer() const { return buffer_; }

private:
  id<MTLBuffer> buffer_;
  size_t size_;
};

class MetalKernel : public ComputeKernel {
public:
  MetalKernel(const std::string &name, id<MTLComputePipelineState> pipeline)
      : name_(name), pipeline_(pipeline) {
    if (pipeline_) {
      [pipeline_ retain];
    }
  }

  ~MetalKernel() override {
    if (pipeline_) {
      [pipeline_ release];
    }
  }

  const std::string &name() const override { return name_; }

  bool valid() const override { return pipeline_ != nil; }

  size_t max_threads_per_threadgroup() const override {
    return pipeline_ ? [pipeline_ maxTotalThreadsPerThreadgroup] : 0;
  }

  id<MTLComputePipelineState> mtl_pipeline() const { return pipeline_; }

private:
  std::string name_;
  id<MTLComputePipelineState> pipeline_;
};

class MetalCommandEncoder : public CommandEncoder {
public:
  MetalCommandEncoder(id<MTLCommandQueue> queue)
      : queue_(queue), buffer_(nil), encoder_(nil), ended_(false) {
    buffer_ = [queue_ commandBufferWithUnretainedReferences];
    encoder_ = [buffer_ computeCommandEncoder];
  }

  ~MetalCommandEncoder() override {
    if (encoder_ && !ended_) {
      [encoder_ endEncoding];
    }
  }

  void set_kernel(ComputeKernel *kernel) override {
    auto *mtl_kernel = static_cast<MetalKernel *>(kernel);
    if (mtl_kernel && encoder_) {
      [encoder_ setComputePipelineState:mtl_kernel->mtl_pipeline()];
      current_kernel_ = mtl_kernel;
    }
  }

  void set_buffer(Buffer *buffer, int index, size_t offset = 0) override {
    auto *mtl_buffer = static_cast<MetalBuffer *>(buffer);
    if (mtl_buffer && encoder_) {
      [encoder_ setBuffer:mtl_buffer->mtl_buffer() offset:offset atIndex:index];
    }
  }

  void set_bytes(const void *data, size_t size, int index) override {
    if (encoder_ && data) {
      [encoder_ setBytes:data length:size atIndex:index];
    }
  }

  void dispatch_threads(size_t width, size_t height, size_t depth) override {
    if (!encoder_ || !current_kernel_)
      return;

    MTLSize grid_size = MTLSizeMake(width, height, depth);

    // Calculate optimal threadgroup size for the grid dimensions
    NSUInteger max_threads = current_kernel_->max_threads_per_threadgroup();
    NSUInteger thread_width, thread_height, thread_depth;

    if (depth > 1) {
      thread_depth = std::min(depth, (size_t)4);
      thread_height = std::min(height, (size_t)4);
      thread_width =
          std::min(width, max_threads / (thread_height * thread_depth));
    } else if (height > 1) {
      // For feature transform: width=hidden_dim (1024), height=batch_size
      thread_height = std::min(height, (size_t)8);
      thread_width = std::min(width, max_threads / thread_height);
      if (thread_width >= 32) {
        thread_width = (thread_width / 32) * 32;
      }
      thread_depth = 1;
    } else {
      thread_width = std::min(width, max_threads);
      thread_height = 1;
      thread_depth = 1;
    }

    MTLSize threadgroup_size =
        MTLSizeMake(thread_width, thread_height, thread_depth);

    [encoder_ dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
  }

  void dispatch_threadgroups(size_t groups_x, size_t groups_y, size_t groups_z,
                             size_t threads_x, size_t threads_y,
                             size_t threads_z) override {
    if (!encoder_)
      return;

    MTLSize grid_size = MTLSizeMake(groups_x, groups_y, groups_z);
    MTLSize threadgroup_size = MTLSizeMake(threads_x, threads_y, threads_z);

    [encoder_ dispatchThreadgroups:grid_size
             threadsPerThreadgroup:threadgroup_size];
  }

  void barrier() override {
    if (encoder_) {
      [encoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }

  void end_encoding() {
    if (encoder_ && !ended_) {
      [encoder_ endEncoding];
      ended_ = true;
    }
  }

  void commit() {
    if (buffer_) {
      [buffer_ commit];
    }
  }

  void wait_until_completed() {
    if (buffer_) {
      [buffer_ waitUntilCompleted];
    }
  }

  id<MTLCommandBuffer> mtl_buffer() const { return buffer_; }

private:
  id<MTLCommandQueue> queue_;
  id<MTLCommandBuffer> buffer_;
  id<MTLComputeCommandEncoder> encoder_;
  MetalKernel *current_kernel_ = nullptr;
  bool ended_ = false;
};

class MetalBackend : public Backend {
public:
  static MetalBackend &instance() {
    static MetalBackend *instance = [] {
      auto *backend = new MetalBackend();
      g_backend_initialized.store(true, std::memory_order_release);
      return backend;
    }();
    return *instance;
  }

  BackendType type() const override { return BackendType::Metal; }

  std::string device_name() const override {
    return device_ ? std::string([[device_ name] UTF8String]) : "Unknown";
  }

  bool has_unified_memory() const override {
    return device_ ? [device_ hasUnifiedMemory] : false;
  }

  size_t max_buffer_size() const override {
    return device_ ? [device_ maxBufferLength] : 0;
  }

  size_t max_threadgroup_memory() const override {
    return device_ ? [device_ maxThreadgroupMemoryLength] : 0;
  }

  size_t recommended_working_set_size() const override {
    if (!device_)
      return 0;
    return [device_ recommendedMaxWorkingSetSize];
  }

  size_t total_system_memory() const override {
    return [[NSProcessInfo processInfo] physicalMemory];
  }

  int gpu_core_count() const override {
    if (!device_)
      return 0;

    std::string name = device_name();

    // Apple Silicon GPU core counts (as of 2025):
    // M1: 7-8 cores, M1 Pro: 14-16, M1 Max: 24-32, M1 Ultra: 48-64
    // M2: 8-10 cores, M2 Pro: 16-19, M2 Max: 30-38, M2 Ultra: 60-76
    // M3: 8-10 cores, M3 Pro: 14-18, M3 Max: 30-40
    // M4: 10 cores, M4 Pro: 16-20, M4 Max: 32-40

    if (name.find("Ultra") != std::string::npos) {
      if (name.find("M2") != std::string::npos)
        return 76;
      if (name.find("M1") != std::string::npos)
        return 64;
      return 64; // Conservative default for Ultra chips
    }
    if (name.find("Max") != std::string::npos) {
      if (name.find("M4") != std::string::npos)
        return 40;
      if (name.find("M3") != std::string::npos)
        return 40;
      if (name.find("M2") != std::string::npos)
        return 38;
      if (name.find("M1") != std::string::npos)
        return 32;
      return 32; // Conservative default for Max chips
    }
    if (name.find("Pro") != std::string::npos) {
      if (name.find("M4") != std::string::npos)
        return 20;
      if (name.find("M3") != std::string::npos)
        return 18;
      if (name.find("M2") != std::string::npos)
        return 19;
      if (name.find("M1") != std::string::npos)
        return 16;
      return 16; // Conservative default for Pro chips
    }
    // Base models
    if (name.find("M4") != std::string::npos)
      return 10;
    if (name.find("M3") != std::string::npos)
      return 10;
    if (name.find("M2") != std::string::npos)
      return 10;
    if (name.find("M1") != std::string::npos)
      return 8;

    size_t working_set = recommended_working_set_size();
    if (working_set > 0) {
      return static_cast<int>(working_set / (256 * 1024 * 1024));
    }

    return 8;
  }

  int max_threads_per_simd_group() const override { return 32; }

  int recommended_batch_size() const override {
    int cores = gpu_core_count();
    if (cores <= 0)
      return 128;

    int base_batch = cores * 16;

    size_t working_set = recommended_working_set_size();
    size_t memory_per_position = 4 * 1024;
    int memory_limited_batch =
        static_cast<int>(working_set / (4 * memory_per_position));

    int batch = std::min(base_batch, memory_limited_batch);

    batch = ((batch + 31) / 32) * 32;
    return std::max(32, std::min(512, batch));
  }

  id<MTLCommandBuffer> acquire_command_buffer() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!command_buffer_pool_.empty()) {
      id<MTLCommandBuffer> buffer = command_buffer_pool_.back();
      command_buffer_pool_.pop_back();
      return buffer;
    }
    return [queue_ commandBuffer];
  }

  void release_command_buffer(id<MTLCommandBuffer> buffer) { (void)buffer; }

  std::unique_ptr<Buffer> create_buffer(size_t size, MemoryMode mode,
                                        BufferUsage usage) override {
    if (!device_ || size == 0)
      return nullptr;

    MTLResourceOptions options = MTLResourceStorageModeShared;

    switch (mode) {
    case MemoryMode::Shared:
      options = MTLResourceStorageModeShared;
      break;
    case MemoryMode::Private:
      options = MTLResourceStorageModePrivate;
      break;
    case MemoryMode::Managed:
      options = MTLResourceStorageModeShared;
      break;
    }

    if (has_unified_memory()) {
      options |= MTLResourceHazardTrackingModeUntracked;
    } else {
      options |= MTLResourceHazardTrackingModeTracked;
    }

    @autoreleasepool {
      id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:options];
      if (buffer) {
        allocated_memory_ += size;
        peak_memory_ = std::max(peak_memory_.load(), allocated_memory_.load());
        return std::make_unique<MetalBuffer>(buffer, size);
      }
    }

    return nullptr;
  }

  std::unique_ptr<Buffer> create_buffer(const void *data, size_t size,
                                        MemoryMode mode) override {
    if (!device_ || !data || size == 0)
      return nullptr;

    MTLResourceOptions options = MTLResourceStorageModeShared;

    if (has_unified_memory()) {
      options |= MTLResourceHazardTrackingModeUntracked;
    } else {
      options |= MTLResourceHazardTrackingModeTracked;
    }

    @autoreleasepool {
      id<MTLBuffer> buffer = [device_ newBufferWithBytes:data
                                                  length:size
                                                 options:options];
      if (buffer) {
        allocated_memory_ += size;
        peak_memory_ = std::max(peak_memory_.load(), allocated_memory_.load());
        return std::make_unique<MetalBuffer>(buffer, size);
      }
    }

    return nullptr;
  }

  std::unique_ptr<ComputeKernel>
  create_kernel(const std::string &name,
                const std::string &library_name) override {
    if (!device_)
      return nullptr;

    @autoreleasepool {
      id<MTLLibrary> lib = get_library(library_name);
      if (!lib)
        return nullptr;

      NSString *func_name = [NSString stringWithUTF8String:name.c_str()];
      id<MTLFunction> function = [lib newFunctionWithName:func_name];
      if (!function) {
        std::cerr << "[MetalBackend] Kernel not found: " << name << std::endl;
        return nullptr;
      }

      NSError *error = nil;
      id<MTLComputePipelineState> pipeline =
          [device_ newComputePipelineStateWithFunction:function error:&error];
      [function release];

      if (error || !pipeline) {
        std::cerr << "[MetalBackend] Failed to create pipeline: " << name
                  << std::endl;
        return nullptr;
      }

      return std::make_unique<MetalKernel>(name, pipeline);
    }
  }

  std::unique_ptr<CommandEncoder> create_encoder() override {
    if (!queue_)
      return nullptr;
    return std::make_unique<MetalCommandEncoder>(queue_);
  }

  void submit_and_wait(CommandEncoder *encoder) override {
    auto *mtl_encoder = static_cast<MetalCommandEncoder *>(encoder);
    if (mtl_encoder) {
      mtl_encoder->end_encoding();
      mtl_encoder->commit();
      mtl_encoder->wait_until_completed();
    }
  }

  void submit(CommandEncoder *encoder) override {
    auto *mtl_encoder = static_cast<MetalCommandEncoder *>(encoder);
    if (mtl_encoder) {
      mtl_encoder->end_encoding();
      mtl_encoder->commit();
    }
  }

  void submit_async(CommandEncoder *encoder,
                    std::function<void()> completion_handler) override {
    auto *mtl_encoder = static_cast<MetalCommandEncoder *>(encoder);
    if (mtl_encoder) {
      mtl_encoder->end_encoding();

      id<MTLCommandBuffer> buffer = mtl_encoder->mtl_buffer();
      if (buffer && completion_handler) {
        [buffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
          completion_handler();
        }];
      }

      mtl_encoder->commit();
    }
  }

  void synchronize() override {
    if (queue_) {
      id<MTLCommandBuffer> buffer = [queue_ commandBuffer];
      [buffer commit];
      [buffer waitUntilCompleted];
    }
  }

  size_t allocated_memory() const override { return allocated_memory_; }
  size_t peak_memory() const override { return peak_memory_; }
  void reset_peak_memory() override { peak_memory_ = allocated_memory_.load(); }

  bool is_available() const { return device_ != nil && queue_ != nil; }

  bool load_library(const std::string &name, const std::string &path) override {
    if (!device_)
      return false;

    @autoreleasepool {
      NSError *error = nil;
      id<MTLLibrary> lib = nil;

      if (!path.empty()) {
        NSURL *url = [NSURL
            fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        lib = [device_ newLibraryWithURL:url error:&error];
      }

      if (lib) {
        std::lock_guard<std::mutex> lock(library_mutex_);
        if (libraries_[name]) {
          [libraries_[name] release];
        }
        libraries_[name] = lib;
        [lib retain];
        return true;
      }
    }

    return false;
  }

  bool compile_library(const std::string &name,
                       const std::string &source) override {
    if (!device_)
      return false;

    @autoreleasepool {
      NSString *ns_source = [NSString stringWithUTF8String:source.c_str()];
      MTLCompileOptions *options = [[MTLCompileOptions alloc] init];

// Use fastMathEnabled for compatibility with older macOS versions
// mathMode is only available on macOS 15+ but we need to support older
// versions
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      [options setFastMathEnabled:YES];
#pragma clang diagnostic pop

      NSError *error = nil;
      id<MTLLibrary> lib = [device_ newLibraryWithSource:ns_source
                                                 options:options
                                                   error:&error];
      [options release];

      if (error) {
        std::cerr << "[MetalBackend] Shader compile error: " <<
            [[error localizedDescription] UTF8String] << std::endl;
        return false;
      }

      if (lib) {
        std::lock_guard<std::mutex> lock(library_mutex_);
        if (libraries_[name]) {
          [libraries_[name] release];
        }
        libraries_[name] = lib;
        [lib retain];
        return true;
      }
    }

    return false;
  }

private:
  static constexpr int NUM_COMMAND_QUEUES = 4;

  MetalBackend() {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_) {
        queue_ = [device_ newCommandQueue];
        [queue_ retain];

        for (int i = 0; i < NUM_COMMAND_QUEUES; ++i) {
          id<MTLCommandQueue> q = [device_ newCommandQueue];
          if (q) {
            [q retain];
            parallel_queues_.push_back(q);
          }
        }

        std::cerr << "[MetalBackend] Initialized: " << device_name()
                  << std::endl;
        std::cerr << "[MetalBackend] Unified memory: "
                  << (has_unified_memory() ? "Yes" : "No") << std::endl;
        std::cerr << "[MetalBackend] Command queues: "
                  << (1 + parallel_queues_.size()) << std::endl;
      }
    }
  }

  ~MetalBackend() {
    @autoreleasepool {
      if (queue_) {
        id<MTLCommandBuffer> buffer = [queue_ commandBuffer];
        if (buffer) {
          [buffer commit];
          [buffer waitUntilCompleted];
        }
      }

      for (auto q : parallel_queues_) {
        if (q) {
          id<MTLCommandBuffer> buffer = [q commandBuffer];
          if (buffer) {
            [buffer commit];
            [buffer waitUntilCompleted];
          }
        }
      }

      for (auto &[name, lib] : libraries_) {
        if (lib)
          [lib release];
      }
      for (auto q : parallel_queues_) {
        if (q)
          [q release];
      }
      if (queue_)
        [queue_ release];
      if (device_)
        [device_ release];
    }
  }

  id<MTLLibrary> get_library(const std::string &name) {
    std::lock_guard<std::mutex> lock(library_mutex_);

    auto it = libraries_.find(name);
    if (it != libraries_.end()) {
      return it->second;
    }

    if (name.empty()) {
      id<MTLLibrary> lib = [device_ newDefaultLibrary];
      if (lib) {
        libraries_[""] = lib;
        return lib;
      }
    }

    return nil;
  }

  id<MTLCommandQueue> get_parallel_queue() {
    if (parallel_queues_.empty())
      return queue_;
    size_t idx = queue_index_.fetch_add(1, std::memory_order_relaxed) %
                 parallel_queues_.size();
    return parallel_queues_[idx];
  }

public:
  std::unique_ptr<CommandEncoder> create_parallel_encoder() override {
    id<MTLCommandQueue> q = get_parallel_queue();
    if (!q)
      return nullptr;
    return std::make_unique<MetalCommandEncoder>(q);
  }

  size_t num_parallel_queues() const override {
    return parallel_queues_.size();
  }

private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  std::vector<id<MTLCommandQueue>> parallel_queues_;
  std::atomic<size_t> queue_index_{0};
  std::mutex library_mutex_;
  std::mutex pool_mutex_;
  std::unordered_map<std::string, id<MTLLibrary>> libraries_;
  std::vector<id<MTLCommandBuffer>> command_buffer_pool_;
  std::atomic<size_t> allocated_memory_{0};
  std::atomic<size_t> peak_memory_{0};
};

Backend &Backend::get() { return MetalBackend::instance(); }

bool Backend::available() {
  if (g_backend_shutdown.load(std::memory_order_acquire)) {
    return false;
  }
  return MetalBackend::instance().is_available();
}

void shutdown_gpu_backend() {
  g_backend_shutdown.store(true, std::memory_order_release);

  if (!g_backend_initialized.load(std::memory_order_acquire)) {
    return;
  }

  if (MetalBackend::instance().is_available()) {
    MetalBackend::instance().synchronize();
  }
}

bool gpu_backend_shutdown() {
  return g_backend_shutdown.load(std::memory_order_acquire);
}

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

#endif // __APPLE__
