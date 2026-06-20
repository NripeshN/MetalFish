/*
  MetalFish - Metal GPU Backend Tests
  Covers Metal backend basics and verifies GPU NNUE stays disabled.
*/

#include "test_common.h"

#include "../src/core/bitboard.h"
#include "../src/core/position.h"
#include "../src/eval/evaluate.h"
#include "../src/eval/gpu_backend.h"

using namespace MetalFish;
using namespace MetalFish::Test;

static bool test_metal_availability() {
  TestCase tc("Metal backend detection");

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
  TestCase tc("Metal buffer allocation and read/write");

#ifdef USE_METAL
  if (!GPU::gpu_available()) {
    std::cout << "    Skipped (no Metal)" << std::endl;
    EXPECT(tc, true);
    tc.print_result();
    return tc.passed;
  }

  auto &backend = GPU::gpu();

  constexpr size_t BUF_SIZE = 1024;
  auto buf = backend.create_buffer(BUF_SIZE * sizeof(float));
  EXPECT(tc, buf != nullptr);
  EXPECT(tc, buf->data() != nullptr);
  EXPECT_GE(tc, buf->size(), BUF_SIZE * sizeof(float));

  float *ptr = static_cast<float *>(buf->data());
  for (size_t i = 0; i < BUF_SIZE; ++i)
    ptr[i] = static_cast<float>(i);

  for (size_t i = 0; i < BUF_SIZE; ++i) {
    EXPECT_NEAR(tc, ptr[i], static_cast<float>(i), 0.001f);
  }

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

static bool test_gpu_nnue_disabled() {
  TestCase tc("GPU NNUE disabled by policy");

  Eval::set_use_apple_silicon_nnue(true);
  EXPECT(tc, !Eval::use_apple_silicon_nnue());
  Eval::set_use_apple_silicon_nnue(false);
  EXPECT(tc, !Eval::use_apple_silicon_nnue());
  std::cout << "    GPU NNUE unavailable: yes" << std::endl;

  tc.print_result();
  return tc.passed;
}

bool test_eval_gpu() {
  bool all_passed = true;

  all_passed &= test_metal_availability();
  all_passed &= test_metal_buffers();
  all_passed &= test_gpu_nnue_disabled();

  return all_passed;
}
