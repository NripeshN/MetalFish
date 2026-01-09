/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

*/

#pragma once

#ifdef USE_METAL

// Forward declarations only - don't include Metal headers here
namespace MTL {
class Device;
class Buffer;
class CommandQueue;
class CommandBuffer;
class ComputeCommandEncoder;
class ComputePipelineState;
class Library;
struct Size;
} // namespace MTL

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace MetalFish {
namespace Metal {

// Buffer wrapper for Metal buffers
struct Buffer {
  MTL::Buffer *ptr = nullptr;
  size_t size = 0;

  bool is_valid() const { return ptr != nullptr; }
  void *contents() const;

  template <typename T> T *as() const { return static_cast<T *>(contents()); }
};

// Metal memory allocator with buffer caching
class MetalAllocator {
public:
  MetalAllocator();
  ~MetalAllocator();

  // Disable copy
  MetalAllocator(const MetalAllocator &) = delete;
  MetalAllocator &operator=(const MetalAllocator &) = delete;

  // Allocate a new Metal buffer
  Buffer malloc(size_t size);

  // Free a Metal buffer (returns to cache)
  void free(Buffer buffer);

  // Get buffer size
  size_t size(Buffer buffer) const;

  // Memory statistics
  size_t get_active_memory() const { return activeMemory_.load(); }
  size_t get_peak_memory() const { return peakMemory_.load(); }
  size_t get_cache_memory() const { return cacheMemory_.load(); }
  void reset_peak_memory() { peakMemory_.store(activeMemory_.load()); }

  // Clear the buffer cache
  void clear_cache();

private:
  MTL::Device *device_;

  // Memory tracking
  std::atomic<size_t> activeMemory_{0};
  std::atomic<size_t> peakMemory_{0};
  std::atomic<size_t> cacheMemory_{0};

  // Buffer cache (grouped by size class)
  std::mutex cacheMutex_;
  std::unordered_map<size_t, std::vector<MTL::Buffer *>> bufferCache_;

  // Size class calculation
  static size_t size_class(size_t size);
};

// Global allocator accessor
MetalAllocator &get_allocator();

} // namespace Metal
} // namespace MetalFish

#endif // USE_METAL
