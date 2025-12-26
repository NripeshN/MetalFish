/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  MetalFish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <map>
#include <vector>

namespace MetalFish {
namespace Metal {

/**
 * Buffer represents a GPU memory allocation with unified memory support.
 * On Apple Silicon, this uses shared memory that is accessible by both CPU and GPU.
 */
struct Buffer {
    MTL::Buffer* ptr{nullptr};
    size_t size{0};
    
    Buffer() = default;
    Buffer(MTL::Buffer* p, size_t s) : ptr(p), size(s) {}
    
    void* contents() const { return ptr ? ptr->contents() : nullptr; }
    bool valid() const { return ptr != nullptr; }
    
    template<typename T>
    T* as() const { return static_cast<T*>(contents()); }
};

/**
 * BufferPool manages a cache of GPU buffers for efficient reuse.
 * Reduces allocation overhead by recycling buffers of similar sizes.
 */
class BufferPool {
public:
    BufferPool() = default;
    ~BufferPool();
    
    // Get a buffer of at least the requested size
    Buffer acquire(size_t size);
    
    // Return a buffer to the pool
    void release(Buffer buffer);
    
    // Clear all cached buffers
    void clear();
    
    // Get total cached memory size
    size_t cache_size() const;

private:
    // Buckets organized by size class (power of 2)
    std::map<size_t, std::vector<MTL::Buffer*>> buckets_;
    mutable std::mutex mutex_;
    size_t cached_size_{0};
};

/**
 * MetalAllocator manages GPU memory allocation with unified memory.
 * 
 * Key features:
 * - Uses MTL::ResourceStorageModeShared for zero-copy CPU-GPU access
 * - Buffer pooling for reduced allocation overhead
 * - Memory tracking and statistics
 */
class MetalAllocator {
public:
    static MetalAllocator& instance();
    
    // Allocate a new buffer
    Buffer allocate(size_t size);
    
    // Allocate with custom options
    Buffer allocate(size_t size, MTL::ResourceOptions options);
    
    // Free a buffer
    void free(Buffer buffer);
    
    // Create buffer from existing memory (no-copy)
    Buffer wrap(void* ptr, size_t size);
    
    // Statistics
    size_t active_memory() const { return active_memory_; }
    size_t peak_memory() const { return peak_memory_; }
    size_t cache_memory() const { return pool_.cache_size(); }
    
    void reset_peak_memory() {
        std::lock_guard<std::mutex> lock(mutex_);
        peak_memory_.store(active_memory_.load());
    }
    
    // Set memory limits
    void set_cache_limit(size_t limit) { cache_limit_ = limit; }
    void set_memory_limit(size_t limit) { memory_limit_ = limit; }
    
    // Clear buffer cache
    void clear_cache() { pool_.clear(); }

private:
    MetalAllocator();
    ~MetalAllocator();
    
    MTL::Device* device_;
    BufferPool pool_;
    
    std::atomic<size_t> active_memory_{0};
    std::atomic<size_t> peak_memory_{0};
    size_t cache_limit_{0};
    size_t memory_limit_{0};
    
    std::mutex mutex_;
};

// Convenience function for allocation
inline Buffer allocate(size_t size) {
    return MetalAllocator::instance().allocate(size);
}

// RAII buffer wrapper
class ScopedBuffer {
public:
    explicit ScopedBuffer(size_t size) : buffer_(allocate(size)) {}
    ~ScopedBuffer() { MetalAllocator::instance().free(buffer_); }
    
    ScopedBuffer(const ScopedBuffer&) = delete;
    ScopedBuffer& operator=(const ScopedBuffer&) = delete;
    
    ScopedBuffer(ScopedBuffer&& other) noexcept : buffer_(other.buffer_) {
        other.buffer_ = Buffer{};
    }
    
    Buffer& get() { return buffer_; }
    const Buffer& get() const { return buffer_; }
    
    void* contents() const { return buffer_.contents(); }
    MTL::Buffer* ptr() const { return buffer_.ptr; }
    size_t size() const { return buffer_.size; }

private:
    Buffer buffer_;
};

} // namespace Metal
} // namespace MetalFish

