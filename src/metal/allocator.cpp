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

#include "metal/allocator.h"
#include "metal/device.h"
#include <algorithm>

namespace MetalFish {
namespace Metal {

// BufferPool implementation
BufferPool::~BufferPool() {
    clear();
}

Buffer BufferPool::acquire(size_t size) {
    // Round up to power of 2 for bucket organization
    size_t bucket_size = 1;
    while (bucket_size < size) {
        bucket_size <<= 1;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& bucket = buckets_[bucket_size];
    if (!bucket.empty()) {
        auto* buf = bucket.back();
        bucket.pop_back();
        cached_size_ -= bucket_size;
        return Buffer{buf, bucket_size};
    }
    
    // No cached buffer, allocate new one
    auto* buf = device().allocate_buffer(bucket_size,
        MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
    
    return Buffer{buf, bucket_size};
}

void BufferPool::release(Buffer buffer) {
    if (!buffer.valid()) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    buckets_[buffer.size].push_back(buffer.ptr);
    cached_size_ += buffer.size;
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [size, bucket] : buckets_) {
        for (auto* buf : bucket) {
            buf->release();
        }
    }
    buckets_.clear();
    cached_size_ = 0;
}

size_t BufferPool::cache_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_size_;
}

// MetalAllocator implementation
MetalAllocator::MetalAllocator() {
    device_ = device().mtl_device();
    
    // Set default limits based on device
    memory_limit_ = device().recommended_working_set_size();
    cache_limit_ = memory_limit_ / 4; // Cache up to 25% of working set
}

MetalAllocator::~MetalAllocator() {
    pool_.clear();
}

MetalAllocator& MetalAllocator::instance() {
    static MetalAllocator allocator;
    return allocator;
}

Buffer MetalAllocator::allocate(size_t size) {
    return allocate(size, 
        MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
}

Buffer MetalAllocator::allocate(size_t size, MTL::ResourceOptions options) {
    // Check memory limit
    if (memory_limit_ > 0 && active_memory_ + size > memory_limit_) {
        // Try to free cached memory first
        pool_.clear();
        
        if (active_memory_ + size > memory_limit_) {
            throw std::runtime_error("[MetalFish] GPU memory limit exceeded");
        }
    }
    
    // Try to get from cache first (only for shared storage mode)
    if ((options & MTL::ResourceStorageModeShared) != 0) {
        Buffer buffer = pool_.acquire(size);
        if (buffer.valid()) {
            active_memory_ += buffer.size;
            if (active_memory_ > peak_memory_) {
                peak_memory_.store(active_memory_.load());
            }
            return buffer;
        }
    }
    
    // Allocate new buffer
    auto* buf = device_->newBuffer(size, options);
    if (!buf) {
        throw std::runtime_error("[MetalFish] Failed to allocate GPU buffer of size " + 
                                std::to_string(size));
    }
    
    active_memory_ += size;
    if (active_memory_ > peak_memory_) {
        peak_memory_.store(active_memory_.load());
    }
    
    return Buffer{buf, size};
}

void MetalAllocator::free(Buffer buffer) {
    if (!buffer.valid()) return;
    
    active_memory_ -= buffer.size;
    
    // Return to pool if under cache limit
    if (pool_.cache_size() + buffer.size <= cache_limit_) {
        pool_.release(buffer);
    } else {
        buffer.ptr->release();
    }
}

Buffer MetalAllocator::wrap(void* ptr, size_t size) {
    // Create a buffer that wraps existing memory (no copy)
    auto* buf = device_->newBuffer(ptr, size,
        MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked,
        nullptr);
    
    if (!buf) {
        throw std::runtime_error("[MetalFish] Failed to wrap memory as GPU buffer");
    }
    
    // Note: This memory is not tracked in active_memory_ since it's external
    return Buffer{buf, size};
}

} // namespace Metal
} // namespace MetalFish

