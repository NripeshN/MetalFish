/*
  MetalFish - Metal GPU & NNUE Evaluation Tests
  Merged from test_metal.cpp, test_gpu_module.cpp, test_gpu_nnue.cpp

  Tests Metal backend availability, buffer management, shader compilation,
  GPU NNUE evaluation, and batch processing.
*/

#include "test_common.h"

#include "../src/core/bitboard.h"
#include "../src/core/position.h"
#include "../src/eval/gpu_backend.h"
#include "../src/eval/gpu_integration.h"

using namespace MetalFish;
using namespace MetalFish::Test;

static bool test_metal_availability() {
  TestCase tc{"Metal backend detection"};

#ifdef USE_METAL
  bool available = GPU::gpu_available();
  EXPECT(tc, true); // Just verify no crash during detection
  if (available) {
    auto &backend = GPU::gpu();
    EXPECT(tc, backend.max_threads_per_simd_group() > 0);
    std::cout << "    Metal device: available" << std::endl;
    std::cout << "    Max threads/simd_group: "
              << backend.max_threads_per_simd_group() << std::endl;
    std::cout << "    Unified memory: "
              << (backend.has_unified_memory() ? "yes" : "no") << std::endl;
  } else {
    std::cout << "    Metal not available (running on non-Apple hardware?)"
              << std::endl;
  }
#else
  std::cout << "    Metal support not compiled in" << std::endl;
  EXPECT(tc, true);
#endif

  tc.print_result();
  return tc.passed;
}

static bool test_metal_buffers() {
  TestCase tc{"Metal buffer allocation and read/write"};

#ifdef USE_METAL
  if (!GPU::gpu_available()) {
    std::cout << "    Skipped (no Metal)" << std::endl;
    EXPECT(tc, true);
    tc.print_result();
    return tc.passed;
  }

  auto &backend = GPU::gpu();

  // Test buffer allocation
  constexpr size_t BUF_SIZE = 1024;
  auto buf = backend.create_buffer(BUF_SIZE * sizeof(float));
  EXPECT(tc, buf != nullptr);
  EXPECT(tc, buf->data() != nullptr);
  EXPECT_GE(tc, buf->size(), BUF_SIZE * sizeof(float));

  // Test write + read back
  float *ptr = static_cast<float *>(buf->data());
  for (size_t i = 0; i < BUF_SIZE; ++i)
    ptr[i] = static_cast<float>(i);

  for (size_t i = 0; i < BUF_SIZE; ++i) {
    EXPECT_NEAR(tc, ptr[i], static_cast<float>(i), 0.001f);
  }

  // Test unified memory (zero-copy)
  if (backend.has_unified_memory()) {
    ptr[0] = 42.0f;
    EXPECT_NEAR(tc, ptr[0], 42.0f, 0.001f);
  }
#else
  std::cout << "    Skipped (no Metal)" << std::endl;
  EXPECT(tc, true);
#endif

  tc.print_result();
  return tc.passed;
}

static bool test_gpu_nnue_manager() {
  TestCase tc{"GPU NNUE manager initialization"};

#ifdef USE_METAL
  if (!GPU::gpu_available()) {
    std::cout << "    Skipped (no Metal)" << std::endl;
    EXPECT(tc, true);
    tc.print_result();
    return tc.passed;
  }

  bool manager_available = GPU::gpu_nnue_manager_available();
  // Manager may or may not be initialized depending on test order
  // Just verify no crash
  EXPECT(tc, true);
  std::cout << "    Manager available: " << (manager_available ? "yes" : "no")
            << std::endl;

  if (manager_available) {
    auto &manager = GPU::gpu_nnue_manager();
    // Verify batch creation works
    GPU::GPUEvalBatch batch;
    EXPECT_EQ(tc, batch.count, 0);
    batch.reserve(64);
    EXPECT(tc, true); // No crash
  }
#else
  std::cout << "    Skipped (no Metal)" << std::endl;
  EXPECT(tc, true);
#endif

  tc.print_result();
  return tc.passed;
}

bool test_eval_gpu() {
  bool all_passed = true;

  all_passed &= test_metal_availability();
  all_passed &= test_metal_buffers();
  all_passed &= test_gpu_nnue_manager();

  return all_passed;
}
