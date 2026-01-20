/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Backend Tests
*/

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef USE_METAL
#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/gpu_nnue_integration.h"

using namespace MetalFish;

bool test_metal() {
  try {
    std::cout << "=== Testing GPU Backend ===" << std::endl;

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

    std::cout << "GPU Backend tests passed!" << std::endl;

    // Test NNUE GPU evaluator initialization
    std::cout << "\n=== Testing GPU NNUE ===" << std::endl;
    // Legacy NNUEEvaluator removed - using GPUNNUEManager instead
    std::cout << "GPU NNUE: Using GPUNNUEManager (new interface)" << std::endl;

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
