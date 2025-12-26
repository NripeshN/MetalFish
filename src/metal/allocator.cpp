/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "metal/allocator.h"
#include "metal/device.h"
#include <algorithm>
#include <stdexcept>

namespace MetalFish {
namespace Metal {

MetalAllocator::MetalAllocator() {
    device_ = get_device().mtl_device();
}

MetalAllocator::~MetalAllocator() {
    clear_cache();
}

size_t MetalAllocator::size_class(size_t size) {
    // Round up to nearest power of 2 for caching efficiency
    if (size <= 256) return 256;
    if (size <= 1024) return 1024;
    if (size <= 4096) return 4096;
    if (size <= 16384) return 16384;
    if (size <= 65536) return 65536;
    if (size <= 262144) return 262144;
    if (size <= 1048576) return 1048576;
    // For larger sizes, round up to nearest MB
    return ((size + 1048575) / 1048576) * 1048576;
}

Buffer MetalAllocator::malloc(size_t size) {
    if (size == 0) {
        return {nullptr, 0};
    }
    
    size_t sizeClass = size_class(size);
    
    // Try to find a cached buffer
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = bufferCache_.find(sizeClass);
        if (it != bufferCache_.end() && !it->second.empty()) {
            MTL::Buffer* buffer = it->second.back();
            it->second.pop_back();
            
            cacheMemory_.fetch_sub(sizeClass);
            activeMemory_.fetch_add(size);
            
            size_t current = activeMemory_.load();
            size_t peak = peakMemory_.load();
            while (current > peak && !peakMemory_.compare_exchange_weak(peak, current)) {}
            
            return {buffer, size};
        }
    }
    
    // Allocate new buffer with shared storage mode for unified memory
    MTL::ResourceOptions options = MTL::ResourceStorageModeShared | 
                                   MTL::ResourceHazardTrackingModeTracked;
    
    MTL::Buffer* buffer = device_->newBuffer(sizeClass, options);
    if (!buffer) {
        throw std::runtime_error("[MetalFish::MetalAllocator] Failed to allocate Metal buffer");
    }
    
    activeMemory_.fetch_add(size);
    
    size_t current = activeMemory_.load();
    size_t peak = peakMemory_.load();
    while (current > peak && !peakMemory_.compare_exchange_weak(peak, current)) {}
    
    return {buffer, size};
}

void MetalAllocator::free(Buffer buffer) {
    if (!buffer.ptr) return;
    
    size_t sizeClass = size_class(buffer.size);
    activeMemory_.fetch_sub(buffer.size);
    
    // Add to cache
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        bufferCache_[sizeClass].push_back(buffer.ptr);
        cacheMemory_.fetch_add(sizeClass);
    }
}

size_t MetalAllocator::size(Buffer buffer) const {
    return buffer.size;
}

void MetalAllocator::clear_cache() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    for (auto& [sizeClass, buffers] : bufferCache_) {
        for (MTL::Buffer* buffer : buffers) {
            buffer->release();
        }
    }
    bufferCache_.clear();
    cacheMemory_.store(0);
}

// Global allocator accessor
MetalAllocator& get_allocator() {
    static MetalAllocator allocator;
    return allocator;
}

} // namespace Metal
} // namespace MetalFish
