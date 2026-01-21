/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Metal Backend Implementation

  Implements the GPU backend interface for Apple Metal.
  Takes advantage of unified memory for zero-copy data access.
*/

#ifdef __APPLE__

#include "../backend.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace MetalFish {
namespace GPU {

// ============================================================================
// Metal Buffer Implementation
// ============================================================================

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

// ============================================================================
// Metal Compute Kernel Implementation
// ============================================================================

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

// ============================================================================
// Metal Command Encoder Implementation
// ============================================================================

class MetalCommandEncoder : public CommandEncoder {
public:
  MetalCommandEncoder(id<MTLCommandQueue> queue)
      : queue_(queue), buffer_(nil), encoder_(nil), ended_(false) {
    // Use commandBufferWithUnretainedReferences for faster allocation
    // This avoids the overhead of retaining/releasing all referenced objects
    // We manage object lifetimes manually, so this is safe
    buffer_ = [queue_ commandBufferWithUnretainedReferences];
    encoder_ = [buffer_ computeCommandEncoder];
  }

  ~MetalCommandEncoder() override {
    // Ensure encoding is ended before releasing
    if (encoder_ && !ended_) {
      [encoder_ endEncoding];
    }
    // No need to release - we're using unretained references
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
      // 3D dispatch - balance across all dimensions
      thread_depth = std::min(depth, (size_t)4);
      thread_height = std::min(height, (size_t)4);
      thread_width =
          std::min(width, max_threads / (thread_height * thread_depth));
    } else if (height > 1) {
      // 2D dispatch - use square-ish threadgroups for better cache locality
      // For feature transform: width=hidden_dim (1024), height=batch_size
      thread_height = std::min(height, (size_t)8);
      thread_width = std::min(width, max_threads / thread_height);
      // Ensure thread_width is a multiple of 32 (simdgroup size) when possible
      if (thread_width >= 32) {
        thread_width = (thread_width / 32) * 32;
      }
      thread_depth = 1;
    } else {
      // 1D dispatch
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

// ============================================================================
// Metal Backend Implementation
// ============================================================================

class MetalBackend : public Backend {
public:
  static MetalBackend &instance() {
    // Use a pointer to avoid static destruction order issues
    // The backend is intentionally leaked to prevent crashes during static destruction
    static MetalBackend* instance = new MetalBackend();
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

  // Command buffer pool management
  id<MTLCommandBuffer> acquire_command_buffer() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!command_buffer_pool_.empty()) {
      id<MTLCommandBuffer> buffer = command_buffer_pool_.back();
      command_buffer_pool_.pop_back();
      return buffer;
    }
    return [queue_ commandBuffer];
  }

  void release_command_buffer(id<MTLCommandBuffer> buffer) {
    // Command buffers can't be reused after commit, so we just let them go
    // This is here for future optimization with MTLCommandBufferDescriptor
    (void)buffer;
  }

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
      // On Apple Silicon, Managed behaves like Shared
      options = MTLResourceStorageModeShared;
      break;
    }

    // On unified memory systems, disable hazard tracking for better performance
    // We manage synchronization manually via barriers
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

    // On unified memory systems, disable hazard tracking for better performance
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

      // Add completion handler before commit
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
    // Create and immediately complete a command buffer to ensure all work is
    // done
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

  // Load or compile a shader library
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

  // Compile shader from source
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
  static constexpr int NUM_COMMAND_QUEUES =
      4; // Multiple queues for parallelism

  MetalBackend() {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_) {
        // Create primary queue
        queue_ = [device_ newCommandQueue];
        [queue_ retain];

        // Create additional queues for parallel command submission
        for (int i = 0; i < NUM_COMMAND_QUEUES; ++i) {
          id<MTLCommandQueue> q = [device_ newCommandQueue];
          if (q) {
            [q retain];
            parallel_queues_.push_back(q);
          }
        }

        std::cout << "[MetalBackend] Initialized: " << device_name()
                  << std::endl;
        std::cout << "[MetalBackend] Unified memory: "
                  << (has_unified_memory() ? "Yes" : "No") << std::endl;
        std::cout << "[MetalBackend] Command queues: "
                  << (1 + parallel_queues_.size()) << std::endl;
      }
    }
  }

  ~MetalBackend() {
    @autoreleasepool {
      // Synchronize all command queues before releasing resources
      // This ensures all pending GPU operations complete before we destroy the backend
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
      
      // Now release all resources
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

    // Return cached library
    auto it = libraries_.find(name);
    if (it != libraries_.end()) {
      return it->second;
    }

    // Try default library
    if (name.empty()) {
      id<MTLLibrary> lib = [device_ newDefaultLibrary];
      if (lib) {
        libraries_[""] = lib;
        return lib;
      }
    }

    return nil;
  }

  // Get a command queue for parallel work (round-robin selection)
  id<MTLCommandQueue> get_parallel_queue() {
    if (parallel_queues_.empty())
      return queue_;
    size_t idx = queue_index_.fetch_add(1, std::memory_order_relaxed) %
                 parallel_queues_.size();
    return parallel_queues_[idx];
  }

public:
  // Create encoder on a parallel queue for async work
  std::unique_ptr<CommandEncoder> create_parallel_encoder() override {
    id<MTLCommandQueue> q = get_parallel_queue();
    if (!q)
      return nullptr;
    return std::make_unique<MetalCommandEncoder>(q);
  }

  // Get number of parallel queues available
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

// ============================================================================
// Backend Interface Implementation
// ============================================================================

// Global shutdown flag - prevents access to backend after shutdown
static std::atomic<bool> g_backend_shutdown{false};

Backend &Backend::get() { return MetalBackend::instance(); }

bool Backend::available() { 
  if (g_backend_shutdown.load(std::memory_order_acquire)) {
    return false;
  }
  return MetalBackend::instance().is_available(); 
}

void shutdown_gpu_backend() {
  // Set shutdown flag first to prevent any new access
  g_backend_shutdown.store(true, std::memory_order_release);
  
  // Synchronize all pending GPU operations
  if (MetalBackend::instance().is_available()) {
    MetalBackend::instance().synchronize();
  }
}

bool gpu_backend_shutdown() {
  return g_backend_shutdown.load(std::memory_order_acquire);
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
