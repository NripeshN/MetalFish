/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Metal GPU tests
*/

#include "metal/allocator.h"
#include "metal/device.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace MetalFish;

bool test_metal() {
  try {
    // Test device initialization
    Metal::Device &device = Metal::get_device();

    MTL::Device *mtlDevice = device.mtl_device();
    assert(mtlDevice != nullptr);

    std::cout << "Metal device: " << device.get_architecture() << std::endl;
    std::cout << "Architecture: " << device.get_architecture_gen() << std::endl;

    // Test unified memory (should be true for Apple Silicon)
    bool hasUnifiedMemory = mtlDevice->hasUnifiedMemory();
    std::cout << "Unified memory: " << (hasUnifiedMemory ? "Yes" : "No")
              << std::endl;

    // Test allocator
    Metal::MetalAllocator &allocator = Metal::get_allocator();

    // Allocate a buffer
    size_t testSize = 1024;
    Metal::Buffer buffer = allocator.malloc(testSize);
    assert(buffer.is_valid());
    assert(buffer.size == testSize);

    // Test that we can write to unified memory directly
    if (hasUnifiedMemory) {
      float *ptr = buffer.as<float>();
      assert(ptr != nullptr);

      // Write some test data
      for (size_t i = 0; i < testSize / sizeof(float); ++i) {
        ptr[i] = float(i);
      }

      // Verify data
      for (size_t i = 0; i < testSize / sizeof(float); ++i) {
        assert(ptr[i] == float(i));
      }
    }

    // Track memory
    size_t activeMemory = allocator.get_active_memory();
    assert(activeMemory >= testSize);

    // Free the buffer
    allocator.free(buffer);

    // Memory should be cached, not necessarily reduced
    size_t cacheMemory = allocator.get_cache_memory();
    assert(cacheMemory > 0);

    // Allocate again - should reuse cached buffer
    Metal::Buffer buffer2 = allocator.malloc(testSize);
    assert(buffer2.is_valid());

    // Clean up
    allocator.free(buffer2);
    allocator.clear_cache();

    assert(allocator.get_cache_memory() == 0);

    // Test command queue
    MTL::CommandQueue *queue = device.get_queue(0);
    assert(queue != nullptr);

    // Test multiple buffer allocations
    std::vector<Metal::Buffer> buffers;
    for (int i = 0; i < 10; ++i) {
      buffers.push_back(allocator.malloc(1024 * (i + 1)));
      assert(buffers.back().is_valid());
    }

    // Free all buffers
    for (auto &buf : buffers) {
      allocator.free(buf);
    }

    // Clear cache
    allocator.clear_cache();

    std::cout << "All Metal tests passed!" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Metal test failed: " << e.what() << std::endl;
    return false;
  }
}
