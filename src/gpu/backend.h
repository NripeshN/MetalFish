/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Backend Abstraction Layer
  
  This provides a backend-agnostic interface for GPU operations.
  Currently supports Metal (Apple Silicon), with CUDA support planned.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace MetalFish {
namespace GPU {

// Forward declarations
class Buffer;
class CommandEncoder;
class ComputeKernel;

// GPU Backend type
enum class BackendType {
    None,   // CPU fallback
    Metal,  // Apple Metal
    CUDA    // NVIDIA CUDA (future)
};

// Buffer usage hints for optimal memory allocation
enum class BufferUsage {
    Default,        // General purpose
    Transient,      // Short-lived, frequently reallocated
    Persistent,     // Long-lived, rarely changed
    Streaming       // Frequently updated from CPU
};

// Memory access mode (relevant for unified memory systems)
enum class MemoryMode {
    Shared,         // CPU and GPU can access (unified memory)
    Private,        // GPU only (fastest for GPU-only data)
    Managed         // System manages CPU/GPU synchronization
};

/**
 * GPU Buffer - Represents memory accessible by the GPU
 * 
 * On Apple Silicon with unified memory, buffers can be accessed
 * directly by both CPU and GPU without explicit copies.
 */
class Buffer {
public:
    virtual ~Buffer() = default;
    
    // Get raw pointer to buffer contents (unified memory only)
    virtual void* data() = 0;
    virtual const void* data() const = 0;
    
    // Buffer size in bytes
    virtual size_t size() const = 0;
    
    // Check if buffer is valid
    virtual bool valid() const = 0;
    
    // Typed access helpers
    template<typename T>
    T* as() { return static_cast<T*>(data()); }
    
    template<typename T>
    const T* as() const { return static_cast<const T*>(data()); }
    
    template<typename T>
    size_t count() const { return size() / sizeof(T); }
};

/**
 * Compute Kernel - Represents a GPU compute shader/kernel
 */
class ComputeKernel {
public:
    virtual ~ComputeKernel() = default;
    
    virtual const std::string& name() const = 0;
    virtual bool valid() const = 0;
    
    // Get optimal threadgroup size for this kernel
    virtual size_t max_threads_per_threadgroup() const = 0;
};

/**
 * Command Encoder - Records GPU commands for execution
 * 
 * Commands are recorded and then submitted as a batch for efficiency.
 */
class CommandEncoder {
public:
    virtual ~CommandEncoder() = default;
    
    // Set compute kernel
    virtual void set_kernel(ComputeKernel* kernel) = 0;
    
    // Set buffer at index
    virtual void set_buffer(Buffer* buffer, int index, size_t offset = 0) = 0;
    
    // Set inline data (small constants)
    virtual void set_bytes(const void* data, size_t size, int index) = 0;
    
    template<typename T>
    void set_value(const T& value, int index) {
        set_bytes(&value, sizeof(T), index);
    }
    
    // Dispatch compute work
    virtual void dispatch_threads(size_t width, size_t height = 1, size_t depth = 1) = 0;
    virtual void dispatch_threadgroups(size_t groups_x, size_t groups_y, size_t groups_z,
                                       size_t threads_x, size_t threads_y, size_t threads_z) = 0;
    
    // Memory barrier
    virtual void barrier() = 0;
};

/**
 * GPU Backend - Main interface for GPU operations
 * 
 * Singleton per backend type. Use Backend::get() to access.
 */
class Backend {
public:
    virtual ~Backend() = default;
    
    // Get the singleton backend instance
    static Backend& get();
    
    // Check if GPU is available
    static bool available();
    
    // Get backend type
    virtual BackendType type() const = 0;
    
    // Device information
    virtual std::string device_name() const = 0;
    virtual bool has_unified_memory() const = 0;
    virtual size_t max_buffer_size() const = 0;
    virtual size_t max_threadgroup_memory() const = 0;
    
    // Buffer management
    virtual std::unique_ptr<Buffer> create_buffer(size_t size, 
                                                   MemoryMode mode = MemoryMode::Shared,
                                                   BufferUsage usage = BufferUsage::Default) = 0;
    
    // Create buffer with initial data
    virtual std::unique_ptr<Buffer> create_buffer(const void* data, size_t size,
                                                   MemoryMode mode = MemoryMode::Shared) = 0;
    
    template<typename T>
    std::unique_ptr<Buffer> create_buffer(const std::vector<T>& data,
                                          MemoryMode mode = MemoryMode::Shared) {
        return create_buffer(data.data(), data.size() * sizeof(T), mode);
    }
    
    // Kernel management
    virtual std::unique_ptr<ComputeKernel> create_kernel(const std::string& name,
                                                          const std::string& library = "") = 0;
    
    // Compile shader from source (returns true if successful)
    virtual bool compile_library(const std::string& name, const std::string& source) = 0;
    
    // Load shader library from file
    virtual bool load_library(const std::string& name, const std::string& path) = 0;
    
    // Command encoding
    virtual std::unique_ptr<CommandEncoder> create_encoder() = 0;
    
    // Submit and wait for completion
    virtual void submit_and_wait(CommandEncoder* encoder) = 0;
    
    // Submit without waiting (for pipelining)
    virtual void submit(CommandEncoder* encoder) = 0;
    
    // Wait for all submitted work to complete
    virtual void synchronize() = 0;
    
    // Memory statistics
    virtual size_t allocated_memory() const = 0;
    virtual size_t peak_memory() const = 0;
    virtual void reset_peak_memory() = 0;
    
protected:
    Backend() = default;
};

/**
 * Scoped GPU timing helper
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, std::function<void(double)> callback = nullptr);
    ~ScopedTimer();
    
    double elapsed_ms() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience function to check if GPU acceleration is available
inline bool gpu_available() { return Backend::available(); }

// Get backend reference (throws if not available)
inline Backend& gpu() { return Backend::get(); }

} // namespace GPU
} // namespace MetalFish
