/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  ROCm/HIP GPU Backend Tests
*/

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef USE_ROCM
#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/batch_ops.h"
#include "gpu/gpu_nnue_integration.h"
#include "gpu/nnue_eval.h"

using namespace MetalFish;

bool test_rocm() {
  try {
    std::cout << "=== Testing ROCm GPU Backend ===" << std::endl;

    // Check if GPU is available
    assert(GPU::gpu_available());

    GPU::Backend &gpu = GPU::gpu();

    // Check backend type
    assert(gpu.type() == GPU::BackendType::ROCm);

    std::cout << "GPU Backend: ROCm/HIP" << std::endl;
    std::cout << "Device: " << gpu.device_name() << std::endl;
    std::cout << "Unified Memory: " << (gpu.has_unified_memory() ? "Yes" : "No")
              << std::endl;
    std::cout << "Max Buffer Size: " << (gpu.max_buffer_size() / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "Max Threadgroup Memory: " << gpu.max_threadgroup_memory()
              << " bytes" << std::endl;

    // Test buffer creation
    std::cout << "\n=== Testing Buffer Creation ===" << std::endl;
    auto gpu_buffer = gpu.create_buffer(4096);
    assert(gpu_buffer != nullptr);
    assert(gpu_buffer->valid());
    assert(gpu_buffer->size() == 4096);
    std::cout << "Buffer creation: SUCCESS" << std::endl;

    // Test unified memory access
    std::cout << "\n=== Testing Memory Access ===" << std::endl;
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
      std::cout << "Unified memory access: SUCCESS" << std::endl;
    } else {
      std::cout << "Unified memory: Not available (discrete GPU)" << std::endl;
    }

    // Test buffer with initial data
    std::cout << "\n=== Testing Buffer with Initial Data ===" << std::endl;
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto data_buffer = gpu.create_buffer(test_data);
    assert(data_buffer != nullptr);
    assert(data_buffer->valid());
    std::cout << "Buffer with data creation: SUCCESS" << std::endl;

    if (gpu.has_unified_memory()) {
      const float *ptr = data_buffer->as<float>();
      for (size_t i = 0; i < test_data.size(); ++i) {
        assert(ptr[i] == test_data[i]);
      }
      std::cout << "Data verification: SUCCESS" << std::endl;
    }

    // Test different memory modes
    std::cout << "\n=== Testing Memory Modes ===" << std::endl;
    
    // Shared memory
    auto shared_buf = gpu.create_buffer(1024, GPU::MemoryMode::Shared);
    assert(shared_buf != nullptr && shared_buf->valid());
    std::cout << "Shared memory buffer: SUCCESS" << std::endl;
    
    // Private memory
    auto private_buf = gpu.create_buffer(1024, GPU::MemoryMode::Private);
    assert(private_buf != nullptr && private_buf->valid());
    std::cout << "Private memory buffer: SUCCESS" << std::endl;
    
    // Managed memory
    auto managed_buf = gpu.create_buffer(1024, GPU::MemoryMode::Managed);
    assert(managed_buf != nullptr && managed_buf->valid());
    std::cout << "Managed memory buffer: SUCCESS" << std::endl;

    // Test memory tracking
    std::cout << "\n=== Testing Memory Tracking ===" << std::endl;
    size_t allocated = gpu.allocated_memory();
    assert(allocated >= 4096 + test_data.size() * sizeof(float));
    std::cout << "Allocated GPU memory: " << allocated << " bytes" << std::endl;
    
    size_t peak = gpu.peak_memory();
    assert(peak >= allocated);
    std::cout << "Peak GPU memory: " << peak << " bytes" << std::endl;
    
    gpu.reset_peak_memory();
    std::cout << "Memory tracking: SUCCESS" << std::endl;

    // Test command encoder creation
    std::cout << "\n=== Testing Command Encoder ===" << std::endl;
    auto encoder = gpu.create_encoder();
    assert(encoder != nullptr);
    std::cout << "Command encoder creation: SUCCESS" << std::endl;

    // Test synchronization
    std::cout << "\n=== Testing Synchronization ===" << std::endl;
    gpu.synchronize();
    std::cout << "GPU synchronization: SUCCESS" << std::endl;

    std::cout << "\nROCm GPU Backend tests passed!" << std::endl;

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

    // Test kernel loading (if available)
    std::cout << "\n=== Testing Kernel Management ===" << std::endl;
    
    // Note: Runtime compilation with hipRTC is not yet implemented
    // This would test loading pre-compiled kernels
    std::cout << "Kernel loading: SKIPPED (requires pre-compiled kernels)" << std::endl;

    // Test buffer operations
    std::cout << "\n=== Testing Buffer Operations ===" << std::endl;
    {
      const size_t size = 256;
      auto test_buf = gpu.create_buffer(size * sizeof(float), GPU::MemoryMode::Shared);
      assert(test_buf != nullptr && test_buf->valid());
      
      float *data = test_buf->as<float>();
      
      // Write pattern
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i) * 3.14f;
      }
      
      // Verify pattern
      bool correct = true;
      for (size_t i = 0; i < size && correct; ++i) {
        if (data[i] != static_cast<float>(i) * 3.14f) {
          correct = false;
        }
      }
      
      assert(correct);
      std::cout << "Buffer read/write: SUCCESS" << std::endl;
    }

    // Test command submission
    std::cout << "\n=== Testing Command Submission ===" << std::endl;
    {
      auto enc = gpu.create_encoder();
      assert(enc != nullptr);
      
      // Test synchronous submission
      gpu.submit_and_wait(enc.get());
      std::cout << "Synchronous submission: SUCCESS" << std::endl;
      
      // Test asynchronous submission
      auto enc2 = gpu.create_encoder();
      gpu.submit(enc2.get());
      gpu.synchronize();
      std::cout << "Asynchronous submission: SUCCESS" << std::endl;
    }

    // Test barrier functionality
    std::cout << "\n=== Testing Memory Barriers ===" << std::endl;
    {
      auto enc = gpu.create_encoder();
      enc->barrier();
      gpu.submit_and_wait(enc.get());
      std::cout << "Memory barrier: SUCCESS" << std::endl;
    }

    std::cout << "\nAll ROCm tests passed!" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "ROCm test failed: " << e.what() << std::endl;
    return false;
  }
}

#else

// Stub when ROCm is not available
bool test_rocm() {
  std::cout << "ROCm tests skipped (USE_ROCM not defined)" << std::endl;
  return true;
}

#endif // USE_ROCM
