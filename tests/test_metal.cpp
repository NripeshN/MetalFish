/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Metal GPU tests
*/

#include <cassert>
#include <iostream>

#ifdef USE_METAL
#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/batch_ops.h"
#include "gpu/gpu_nnue_integration.h"
#include "gpu/nnue_eval.h"
#include "metal/allocator.h"
#include "metal/device.h"
#include <cstring>
#include <vector>

using namespace MetalFish;

bool test_metal() {
  try {
    std::cout << "=== Testing Legacy Metal Interface ===" << std::endl;

    // Test device initialization
    Metal::Device &device = Metal::get_device();

    assert(device.mtl_device() != nullptr);

    std::cout << "Metal device: " << device.get_architecture() << std::endl;
    std::cout << "Architecture: " << device.get_architecture_gen() << std::endl;

    // Test unified memory (should be true for Apple Silicon)
    bool hasUnifiedMemory = device.has_unified_memory();
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
    assert(device.get_queue(0) != nullptr);

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

    std::cout << "Legacy Metal tests passed!" << std::endl;

    // ========================================
    // Test new GPU Backend abstraction
    // ========================================
    std::cout << "\n=== Testing New GPU Backend ===" << std::endl;

    // Check if GPU is available
    assert(GPU::gpu_available());

    GPU::Backend &gpu = GPU::gpu();

    // Check backend type
    assert(gpu.type() == GPU::BackendType::Metal);

    std::cout << "GPU Backend: Metal" << std::endl;
    std::cout << "Device: " << gpu.device_name() << std::endl;
    std::cout << "Unified Memory: " << (gpu.has_unified_memory() ? "Yes" : "No")
              << std::endl;
    std::cout << "Max Buffer Size: " << (gpu.max_buffer_size() / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "Max Threadgroup Memory: " << gpu.max_threadgroup_memory()
              << " bytes" << std::endl;

    // Test buffer creation
    auto gpu_buffer = gpu.create_buffer(4096);
    assert(gpu_buffer != nullptr);
    assert(gpu_buffer->valid());
    assert(gpu_buffer->size() == 4096);

    // Test unified memory access
    if (gpu.has_unified_memory()) {
      int32_t *data = gpu_buffer->as<int32_t>();
      assert(data != nullptr);

      // Write test pattern
      for (size_t i = 0; i < gpu_buffer->count<int32_t>(); ++i) {
        data[i] = static_cast<int32_t>(i * 7);
      }

      // Verify
      for (size_t i = 0; i < gpu_buffer->count<int32_t>(); ++i) {
        assert(data[i] == static_cast<int32_t>(i * 7));
      }
    }

    // Test buffer with initial data
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto data_buffer = gpu.create_buffer(test_data);
    assert(data_buffer != nullptr);
    assert(data_buffer->valid());

    if (gpu.has_unified_memory()) {
      const float *ptr = data_buffer->as<float>();
      for (size_t i = 0; i < test_data.size(); ++i) {
        assert(ptr[i] == test_data[i]);
      }
    }

    // Test memory tracking
    size_t allocated = gpu.allocated_memory();
    assert(allocated >= 4096 + test_data.size() * sizeof(float));

    std::cout << "Allocated GPU memory: " << allocated << " bytes" << std::endl;

    // Test command encoder creation
    auto encoder = gpu.create_encoder();
    assert(encoder != nullptr);

    // Note: We can't test kernel dispatch without compiled shaders
    // That's tested separately when shaders are available

    std::cout << "GPU Backend tests passed!" << std::endl;

    // ========================================
    // Test GPU Operations (Batch SEE, etc.)
    // ========================================
    std::cout << "\n=== Testing GPU Operations ===" << std::endl;

    // Initialize GPU operations
    GPU::GPUOperations &ops = GPU::gpu_ops();
    if (ops.initialize()) {
      std::cout << "GPU Operations initialized" << std::endl;
      std::cout << "  SEE available: " << (ops.see_available() ? "Yes" : "No")
                << std::endl;
      std::cout << "  Scorer available: "
                << (ops.scorer_available() ? "Yes" : "No") << std::endl;
      std::cout << "  Total GPU memory: " << ops.total_gpu_memory() / 1024
                << " KB" << std::endl;
    } else {
      std::cout << "GPU Operations not available (OK for CI)" << std::endl;
    }

    // Test NNUE GPU evaluator initialization
    std::cout << "\n=== Testing GPU NNUE ===" << std::endl;
    GPU::NNUEEvaluator &nnue = GPU::gpu_nnue();
    // Note: Full initialization requires NNUE weights, which may not be
    // available in tests
    std::cout << "GPU NNUE evaluator created" << std::endl;

    std::cout << "\n=== Testing GPU NNUE Integration ===" << std::endl;
    {
      auto &manager = GPU::gpu_nnue_manager();
      if (manager.initialize()) {
        std::cout << "GPU NNUE Manager: Initialized" << std::endl;

        // Test batch creation
        GPU::GPUEvalBatch batch;
        batch.reserve(16);

        // Create a simple test position
        StateListPtr states(new std::deque<StateInfo>(1));
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                false, &states->back());

        // Add position to batch
        batch.add_position(pos);
        std::cout << "  Batch created with " << batch.count << " position(s)"
                  << std::endl;

        // Status
        std::cout << manager.status_string();
      } else {
        std::cout
            << "GPU NNUE Manager: Not initialized (expected without networks)"
            << std::endl;
      }
    }

    std::cout << "\n=== Testing Shader Compilation ===" << std::endl;
    const char *test_shader = R"(
      #include <metal_stdlib>
      using namespace metal;
      
      kernel void test_kernel(device float* output [[buffer(0)]],
                              constant int& count [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
        if (gid < uint(count)) {
          output[gid] = float(gid) * 2.0f;
        }
      }
    )";

    if (gpu.compile_library("test", test_shader)) {
      std::cout << "Shader compilation: SUCCESS" << std::endl;

      // Try to create kernel from compiled library
      auto test_kernel = gpu.create_kernel("test_kernel", "test");
      if (test_kernel && test_kernel->valid()) {
        std::cout << "Kernel creation: SUCCESS" << std::endl;
        std::cout << "  Max threads per threadgroup: "
                  << test_kernel->max_threads_per_threadgroup() << std::endl;

        // Test kernel execution
        const int count = 256;
        auto output_buf = gpu.create_buffer(count * sizeof(float));

        auto enc = gpu.create_encoder();
        enc->set_kernel(test_kernel.get());
        enc->set_buffer(output_buf.get(), 0);
        enc->set_value(count, 1);
        enc->dispatch_threads(count);

        gpu.submit_and_wait(enc.get());

        // Verify results
        float *results = output_buf->as<float>();
        bool correct = true;
        for (int i = 0; i < count && correct; i++) {
          if (results[i] != float(i) * 2.0f) {
            correct = false;
            std::cerr << "Mismatch at " << i << ": expected " << float(i) * 2.0f
                      << ", got " << results[i] << std::endl;
          }
        }

        if (correct) {
          std::cout << "Kernel execution: SUCCESS (verified " << count
                    << " values)" << std::endl;
        } else {
          std::cerr << "Kernel execution: FAILED" << std::endl;
          return false;
        }
      } else {
        std::cerr << "Kernel creation: FAILED" << std::endl;
        return false;
      }
    } else {
      std::cout << "Shader compilation: SKIPPED (may not be available in CI)"
                << std::endl;
    }

    std::cout << "\nAll Metal tests passed!" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Metal test failed: " << e.what() << std::endl;
    return false;
  }
}

#else

// Stub when Metal is not available
bool test_metal() {
  std::cout << "Metal tests skipped (USE_METAL not defined)" << std::endl;
  return true;
}

#endif // USE_METAL
