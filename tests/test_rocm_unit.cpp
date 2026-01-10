/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  ROCm Backend Unit Tests
  
  Comprehensive unit tests for ROCm/HIP backend implementation.
  Tests individual components and edge cases.
*/

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>

#ifdef USE_ROCM
#include "gpu/backend.h"

using namespace MetalFish;

namespace ROCmTests {

// Test fixture for ROCm tests
class ROCmTestFixture {
public:
  ROCmTestFixture() : gpu(GPU::gpu()) {
    assert(gpu.type() == GPU::BackendType::ROCm);
  }

  GPU::Backend &gpu;
};

// Test 1: Buffer allocation and deallocation
bool test_buffer_lifecycle() {
  std::cout << "  Testing buffer lifecycle..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Test various buffer sizes
  std::vector<size_t> sizes = {16, 64, 256, 1024, 4096, 1024 * 1024};
  
  for (size_t size : sizes) {
    auto buffer = fixture.gpu.create_buffer(size);
    assert(buffer != nullptr);
    assert(buffer->valid());
    assert(buffer->size() == size);
  }
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 2: Memory mode behavior
bool test_memory_modes() {
  std::cout << "  Testing memory modes..." << std::flush;
  
  ROCmTestFixture fixture;
  const size_t test_size = 1024;
  
  // Test Shared memory
  auto shared = fixture.gpu.create_buffer(test_size, GPU::MemoryMode::Shared);
  assert(shared != nullptr && shared->valid());
  assert(shared->data() != nullptr); // Should be CPU accessible
  
  // Test Private memory
  auto priv = fixture.gpu.create_buffer(test_size, GPU::MemoryMode::Private);
  assert(priv != nullptr && priv->valid());
  
  // Test Managed memory
  auto managed = fixture.gpu.create_buffer(test_size, GPU::MemoryMode::Managed);
  assert(managed != nullptr && managed->valid());
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 3: Buffer data integrity
bool test_buffer_data_integrity() {
  std::cout << "  Testing buffer data integrity..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Test with various data types
  {
    std::vector<int32_t> data(256);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<int32_t>(i * 17);
    }
    
    auto buffer = fixture.gpu.create_buffer(data);
    assert(buffer != nullptr && buffer->valid());
    
    if (fixture.gpu.has_unified_memory()) {
      const int32_t *ptr = buffer->as<int32_t>();
      for (size_t i = 0; i < data.size(); ++i) {
        assert(ptr[i] == data[i]);
      }
    }
  }
  
  {
    std::vector<float> data(128);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i) * 3.14159f;
    }
    
    auto buffer = fixture.gpu.create_buffer(data);
    assert(buffer != nullptr && buffer->valid());
    
    if (fixture.gpu.has_unified_memory()) {
      const float *ptr = buffer->as<float>();
      for (size_t i = 0; i < data.size(); ++i) {
        assert(std::abs(ptr[i] - data[i]) < 1e-6f);
      }
    }
  }
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 4: Memory tracking
bool test_memory_tracking() {
  std::cout << "  Testing memory tracking..." << std::flush;
  
  ROCmTestFixture fixture;
  
  size_t initial_allocated = fixture.gpu.allocated_memory();
  
  // Allocate some buffers
  const size_t buf_size = 1024;
  auto buf1 = fixture.gpu.create_buffer(buf_size);
  auto buf2 = fixture.gpu.create_buffer(buf_size * 2);
  auto buf3 = fixture.gpu.create_buffer(buf_size * 4);
  
  size_t current_allocated = fixture.gpu.allocated_memory();
  assert(current_allocated >= initial_allocated + buf_size * 7);
  
  size_t peak = fixture.gpu.peak_memory();
  assert(peak >= current_allocated);
  
  fixture.gpu.reset_peak_memory();
  size_t peak_after_reset = fixture.gpu.peak_memory();
  assert(peak_after_reset == current_allocated);
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 5: Command encoder creation
bool test_command_encoder() {
  std::cout << "  Testing command encoder..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Create multiple encoders
  auto enc1 = fixture.gpu.create_encoder();
  assert(enc1 != nullptr);
  
  auto enc2 = fixture.gpu.create_encoder();
  assert(enc2 != nullptr);
  
  // Test encoder operations
  enc1->barrier();
  enc2->barrier();
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 6: Synchronization primitives
bool test_synchronization() {
  std::cout << "  Testing synchronization..." << std::flush;
  
  ROCmTestFixture fixture;
  
  auto enc = fixture.gpu.create_encoder();
  
  // Test submit and wait
  fixture.gpu.submit_and_wait(enc.get());
  
  // Test async submit + sync
  auto enc2 = fixture.gpu.create_encoder();
  fixture.gpu.submit(enc2.get());
  fixture.gpu.synchronize();
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 7: Buffer usage patterns
bool test_buffer_usage() {
  std::cout << "  Testing buffer usage patterns..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Test different usage hints
  const size_t size = 512;
  
  auto default_buf = fixture.gpu.create_buffer(size, GPU::MemoryMode::Shared,
                                               GPU::BufferUsage::Default);
  assert(default_buf != nullptr && default_buf->valid());
  
  auto transient_buf = fixture.gpu.create_buffer(size, GPU::MemoryMode::Shared,
                                                 GPU::BufferUsage::Transient);
  assert(transient_buf != nullptr && transient_buf->valid());
  
  auto persistent_buf = fixture.gpu.create_buffer(size, GPU::MemoryMode::Shared,
                                                  GPU::BufferUsage::Persistent);
  assert(persistent_buf != nullptr && persistent_buf->valid());
  
  auto streaming_buf = fixture.gpu.create_buffer(size, GPU::MemoryMode::Shared,
                                                 GPU::BufferUsage::Streaming);
  assert(streaming_buf != nullptr && streaming_buf->valid());
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 8: Edge cases
bool test_edge_cases() {
  std::cout << "  Testing edge cases..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Test zero-size buffer (should still be valid but empty)
  // Note: Some implementations may not support zero-size buffers
  
  // Test large buffer allocation
  size_t large_size = 64 * 1024 * 1024; // 64 MB
  if (fixture.gpu.max_buffer_size() >= large_size) {
    auto large_buf = fixture.gpu.create_buffer(large_size);
    assert(large_buf != nullptr && large_buf->valid());
  }
  
  // Test buffer count helpers
  auto test_buf = fixture.gpu.create_buffer(1024);
  assert(test_buf->count<int32_t>() == 1024 / sizeof(int32_t));
  assert(test_buf->count<float>() == 1024 / sizeof(float));
  assert(test_buf->count<uint8_t>() == 1024);
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 9: Concurrent operations
bool test_concurrent_operations() {
  std::cout << "  Testing concurrent operations..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Create multiple buffers concurrently
  std::vector<std::unique_ptr<GPU::Buffer>> buffers;
  for (int i = 0; i < 10; ++i) {
    buffers.push_back(fixture.gpu.create_buffer(1024 * (i + 1)));
    assert(buffers.back() != nullptr && buffers.back()->valid());
  }
  
  // Create multiple encoders
  std::vector<std::unique_ptr<GPU::CommandEncoder>> encoders;
  for (int i = 0; i < 5; ++i) {
    encoders.push_back(fixture.gpu.create_encoder());
    assert(encoders.back() != nullptr);
  }
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 10: Device information
bool test_device_info() {
  std::cout << "  Testing device information..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Check that device info is available
  std::string device_name = fixture.gpu.device_name();
  assert(!device_name.empty());
  
  size_t max_buf = fixture.gpu.max_buffer_size();
  assert(max_buf > 0);
  
  size_t max_tg_mem = fixture.gpu.max_threadgroup_memory();
  assert(max_tg_mem > 0);
  
  // Unified memory status should be consistent
  bool has_unified = fixture.gpu.has_unified_memory();
  (void)has_unified; // Just query, don't assert specific value
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 11: Buffer write/read patterns
bool test_buffer_rw_patterns() {
  std::cout << "  Testing buffer read/write patterns..." << std::flush;
  
  ROCmTestFixture fixture;
  
  if (!fixture.gpu.has_unified_memory()) {
    std::cout << " SKIPPED (requires unified memory)" << std::endl;
    return true;
  }
  
  const size_t size = 1024;
  auto buffer = fixture.gpu.create_buffer(size * sizeof(float),
                                         GPU::MemoryMode::Shared);
  assert(buffer != nullptr && buffer->valid());
  
  float *data = buffer->as<float>();
  
  // Sequential write
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<float>(i);
  }
  
  // Sequential read and verify
  for (size_t i = 0; i < size; ++i) {
    assert(data[i] == static_cast<float>(i));
  }
  
  // Random access write
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> dist(0, size - 1);
  
  for (int i = 0; i < 100; ++i) {
    size_t idx = dist(rng);
    float val = static_cast<float>(idx) * 2.0f;
    data[idx] = val;
    assert(data[idx] == val);
  }
  
  std::cout << " PASSED" << std::endl;
  return true;
}

// Test 12: Memory barrier behavior
bool test_memory_barriers() {
  std::cout << "  Testing memory barriers..." << std::flush;
  
  ROCmTestFixture fixture;
  
  auto enc = fixture.gpu.create_encoder();
  
  // Multiple barriers
  enc->barrier();
  enc->barrier();
  enc->barrier();
  
  fixture.gpu.submit_and_wait(enc.get());
  
  std::cout << " PASSED" << std::endl;
  return true;
}

} // namespace ROCmTests

// Main test runner
bool run_rocm_unit_tests() {
  std::cout << "\n=== ROCm Backend Unit Tests ===" << std::endl;
  
  try {
    if (!GPU::gpu_available()) {
      std::cout << "ROCm not available, skipping unit tests" << std::endl;
      return true;
    }
    
    GPU::Backend &gpu = GPU::gpu();
    if (gpu.type() != GPU::BackendType::ROCm) {
      std::cout << "Not using ROCm backend, skipping ROCm-specific tests" << std::endl;
      return true;
    }
    
    struct Test {
      const char *name;
      bool (*func)();
    };
    
    Test tests[] = {
      {"Buffer Lifecycle", ROCmTests::test_buffer_lifecycle},
      {"Memory Modes", ROCmTests::test_memory_modes},
      {"Buffer Data Integrity", ROCmTests::test_buffer_data_integrity},
      {"Memory Tracking", ROCmTests::test_memory_tracking},
      {"Command Encoder", ROCmTests::test_command_encoder},
      {"Synchronization", ROCmTests::test_synchronization},
      {"Buffer Usage", ROCmTests::test_buffer_usage},
      {"Edge Cases", ROCmTests::test_edge_cases},
      {"Concurrent Operations", ROCmTests::test_concurrent_operations},
      {"Device Information", ROCmTests::test_device_info},
      {"Buffer R/W Patterns", ROCmTests::test_buffer_rw_patterns},
      {"Memory Barriers", ROCmTests::test_memory_barriers},
    };
    
    int passed = 0;
    int failed = 0;
    
    for (const auto &test : tests) {
      try {
        if (test.func()) {
          passed++;
        } else {
          std::cerr << "  " << test.name << ": FAILED" << std::endl;
          failed++;
        }
      } catch (const std::exception &e) {
        std::cerr << "  " << test.name << ": ERROR - " << e.what() << std::endl;
        failed++;
      }
    }
    
    std::cout << "\nROCm Unit Tests: " << passed << " passed, " << failed << " failed" << std::endl;
    
    return failed == 0;
    
  } catch (const std::exception &e) {
    std::cerr << "ROCm unit test suite failed: " << e.what() << std::endl;
    return false;
  }
}

#else

bool run_rocm_unit_tests() {
  std::cout << "ROCm unit tests skipped (USE_ROCM not defined)" << std::endl;
  return true;
}

#endif // USE_ROCM
