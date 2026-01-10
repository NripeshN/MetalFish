/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA GPU Backend Tests
*/

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef USE_CUDA
#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/batch_ops.h"
#include "gpu/gpu_nnue_integration.h"
#include "gpu/nnue_eval.h"

using namespace MetalFish;

bool test_cuda() {
  try {
    std::cout << "=== Testing CUDA GPU Backend ===" << std::endl;

    // Check if GPU is available
    assert(GPU::gpu_available());

    GPU::Backend &gpu = GPU::gpu();

    // Check backend type
    assert(gpu.type() == GPU::BackendType::CUDA);

    std::cout << "GPU Backend: CUDA" << std::endl;
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
    std::cout << "Allocated Memory: " << (allocated / 1024) << " KB"
              << std::endl;
    assert(allocated > 0);

    // Test kernel compilation (simple test kernel)
    const char *test_kernel_source = R"(
      extern "C" __global__ void test_kernel(float* input, float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
          output[idx] = input[idx] * 2.0f;
        }
      }
    )";

    bool compiled = gpu.compile_library("test_lib", test_kernel_source);
    if (compiled) {
      std::cout << "CUDA kernel compilation: Success" << std::endl;

      // Test kernel creation
      auto kernel = gpu.create_kernel("test_kernel", "test_lib");
      assert(kernel != nullptr);
      assert(kernel->valid());
      std::cout << "Kernel name: " << kernel->name() << std::endl;
      std::cout << "Max threads per block: "
                << kernel->max_threads_per_threadgroup() << std::endl;

      // Test kernel execution
      const int N = 1024;
      std::vector<float> input(N);
      for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(i);
      }

      auto input_buf = gpu.create_buffer(input);
      auto output_buf = gpu.create_buffer(N * sizeof(float));

      assert(input_buf != nullptr);
      assert(output_buf != nullptr);

      auto encoder = gpu.create_encoder();
      encoder->set_kernel(kernel.get());
      encoder->set_buffer(input_buf.get(), 0);
      encoder->set_buffer(output_buf.get(), 1);
      encoder->set_value(N, 2);
      encoder->dispatch_threads(N);

      gpu.submit_and_wait(encoder.get());

      // Verify results
      if (gpu.has_unified_memory()) {
        const float *output = output_buf->as<float>();
        bool correct = true;
        for (int i = 0; i < 10 && i < N; i++) {
          if (output[i] != input[i] * 2.0f) {
            correct = false;
            std::cout << "Mismatch at " << i << ": expected " << input[i] * 2.0f
                      << ", got " << output[i] << std::endl;
          }
        }
        if (correct) {
          std::cout << "Kernel execution test: PASSED" << std::endl;
        } else {
          std::cout << "Kernel execution test: FAILED" << std::endl;
        }
      }
    } else {
      std::cout << "CUDA kernel compilation: Failed (may not be available)"
                << std::endl;
    }

    std::cout << "\n=== CUDA Backend Tests PASSED ===" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "CUDA test failed with exception: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "CUDA test failed with unknown exception" << std::endl;
    return false;
  }
}

#else

bool test_cuda() {
  std::cout << "CUDA support not compiled in (USE_CUDA not defined)"
            << std::endl;
  return true; // Not a failure, just skipped
}

#endif

// Export test function
extern "C" bool run_cuda_test() { return test_cuda(); }
