/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Backend Tests

  Tests for NVIDIA CUDA GPU acceleration functionality.
*/

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef USE_CUDA
#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/gpu_nnue_integration.h"
#include "gpu/cuda/cuda_backend.h"

using namespace MetalFish;

bool test_cuda() {
  try {
    std::cout << "=== Testing CUDA Backend ===" << std::endl;

    // Check if CUDA is available
    if (!GPU::CUDABackend::is_available()) {
      std::cout << "CUDA not available on this system, skipping CUDA tests"
                << std::endl;
      return true; // Not a failure, just not available
    }

    GPU::CUDABackend &cuda = GPU::CUDABackend::instance();

    // Check backend type
    assert(cuda.type() == GPU::BackendType::CUDA);

    std::cout << "CUDA Backend: NVIDIA CUDA" << std::endl;
    std::cout << "Device: " << cuda.device_name() << std::endl;
    std::cout << "Compute Capability: " << cuda.compute_capability_major()
              << "." << cuda.compute_capability_minor() << std::endl;
    std::cout << "Total Memory: " << (cuda.total_memory() / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "Multiprocessors: " << cuda.multiprocessor_count() << std::endl;
    std::cout << "Unified Memory: " << (cuda.has_unified_memory() ? "Yes" : "No")
              << std::endl;
    std::cout << "Max Buffer Size: " << (cuda.max_buffer_size() / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "Max Threadgroup Memory: " << cuda.max_threadgroup_memory()
              << " bytes" << std::endl;

    // Test buffer creation
    std::cout << "\n=== Testing Buffer Creation ===" << std::endl;
    {
      auto gpu_buffer = cuda.create_buffer(4096);
      assert(gpu_buffer != nullptr);
      assert(gpu_buffer->valid());
      assert(gpu_buffer->size() == 4096);
      std::cout << "Buffer creation (4KB): PASSED" << std::endl;
    }

    {
      auto gpu_buffer = cuda.create_buffer(1024 * 1024); // 1MB
      assert(gpu_buffer != nullptr);
      assert(gpu_buffer->valid());
      std::cout << "Buffer creation (1MB): PASSED" << std::endl;
    }

    // Test unified memory access (if supported)
    std::cout << "\n=== Testing Memory Access ===" << std::endl;
    {
      const size_t count = 1024;
      auto buffer = cuda.create_buffer(count * sizeof(int32_t));
      assert(buffer != nullptr);

      int32_t *data = buffer->as<int32_t>();
      if (data != nullptr) {
        // Write test pattern
        for (size_t i = 0; i < count; ++i) {
          data[i] = static_cast<int32_t>(i * 7);
        }

        // Verify
        bool correct = true;
        for (size_t i = 0; i < count && correct; ++i) {
          if (data[i] != static_cast<int32_t>(i * 7)) {
            correct = false;
          }
        }
        std::cout << "Memory read/write: " << (correct ? "PASSED" : "FAILED")
                  << std::endl;
        assert(correct);
      } else {
        std::cout << "Memory read/write: SKIPPED (non-unified memory)"
                  << std::endl;
      }
    }

    // Test buffer with initial data
    std::cout << "\n=== Testing Buffer with Initial Data ===" << std::endl;
    {
      std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
      auto data_buffer = cuda.create_buffer(test_data.data(),
                                            test_data.size() * sizeof(float),
                                            GPU::MemoryMode::Shared);
      assert(data_buffer != nullptr);
      assert(data_buffer->valid());

      if (cuda.has_unified_memory()) {
        const float *ptr = data_buffer->as<float>();
        bool correct = true;
        for (size_t i = 0; i < test_data.size() && correct; ++i) {
          if (ptr[i] != test_data[i]) {
            correct = false;
          }
        }
        std::cout << "Buffer with initial data: "
                  << (correct ? "PASSED" : "FAILED") << std::endl;
        assert(correct);
      } else {
        std::cout << "Buffer with initial data: PASSED (created successfully)"
                  << std::endl;
      }
    }

    // Test memory tracking
    std::cout << "\n=== Testing Memory Tracking ===" << std::endl;
    {
      size_t initial_memory = cuda.allocated_memory();
      auto buffer1 = cuda.create_buffer(1024);
      auto buffer2 = cuda.create_buffer(2048);
      size_t after_alloc = cuda.allocated_memory();

      assert(after_alloc >= initial_memory + 3072);
      std::cout << "Memory tracking: PASSED" << std::endl;
      std::cout << "  Allocated: " << cuda.allocated_memory() << " bytes"
                << std::endl;
      std::cout << "  Peak: " << cuda.peak_memory() << " bytes" << std::endl;
    }

    // Test command encoder creation
    std::cout << "\n=== Testing Command Encoder ===" << std::endl;
    {
      auto encoder = cuda.create_encoder();
      assert(encoder != nullptr);
      std::cout << "Command encoder creation: PASSED" << std::endl;
    }

    // Test parallel encoders
    std::cout << "\n=== Testing Parallel Queues ===" << std::endl;
    {
      std::cout << "Number of parallel queues: " << cuda.num_parallel_queues()
                << std::endl;
      auto parallel_encoder = cuda.create_parallel_encoder();
      assert(parallel_encoder != nullptr);
      std::cout << "Parallel encoder creation: PASSED" << std::endl;
    }

    // Test synchronization
    std::cout << "\n=== Testing Synchronization ===" << std::endl;
    {
      cuda.synchronize();
      std::cout << "Device synchronization: PASSED" << std::endl;
    }

    // Test GPU NNUE integration
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

    std::cout << "\nAll CUDA tests passed!" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "CUDA test failed: " << e.what() << std::endl;
    return false;
  }
}

// Additional CUDA-specific performance benchmarks
bool test_cuda_performance() {
  std::cout << "\n=== CUDA Performance Benchmarks ===" << std::endl;

  if (!GPU::CUDABackend::is_available()) {
    std::cout << "CUDA not available, skipping performance tests" << std::endl;
    return true;
  }

  GPU::CUDABackend &cuda = GPU::CUDABackend::instance();

  // Memory bandwidth test
  {
    const size_t size = 64 * 1024 * 1024; // 64MB
    auto buffer = cuda.create_buffer(size);

    if (buffer && cuda.has_unified_memory()) {
      float *data = buffer->as<float>();
      const int count = size / sizeof(float);

      // Write test
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < count; i++) {
        data[i] = static_cast<float>(i);
      }
      auto end = std::chrono::high_resolution_clock::now();
      double write_time =
          std::chrono::duration<double, std::milli>(end - start).count();
      double write_bw = (size / (1024.0 * 1024.0 * 1024.0)) / (write_time / 1000.0);
      std::cout << "  Memory write bandwidth: " << write_bw << " GB/s"
                << std::endl;

      // Read test
      start = std::chrono::high_resolution_clock::now();
      volatile float sum = 0;
      for (int i = 0; i < count; i++) {
        sum += data[i];
      }
      end = std::chrono::high_resolution_clock::now();
      double read_time =
          std::chrono::duration<double, std::milli>(end - start).count();
      double read_bw = (size / (1024.0 * 1024.0 * 1024.0)) / (read_time / 1000.0);
      std::cout << "  Memory read bandwidth: " << read_bw << " GB/s"
                << std::endl;
    }
  }

  return true;
}

#else // !USE_CUDA

// Stub when CUDA is not available
bool test_cuda() {
  std::cout << "CUDA tests skipped (USE_CUDA not defined)" << std::endl;
  return true;
}

bool test_cuda_performance() {
  std::cout << "CUDA performance tests skipped (USE_CUDA not defined)"
            << std::endl;
  return true;
}

#endif // USE_CUDA
