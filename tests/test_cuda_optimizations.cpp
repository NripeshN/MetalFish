/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Optimization Tests

  Tests for tensor cores, warp primitives, and memory optimizations.
*/

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef USE_CUDA

#include "../src/gpu/cuda/cuda_backend.h"
#include "../src/gpu/cuda/cuda_memory.h"
#include "../src/gpu/cuda/cuda_profiling.h"
#include "../src/gpu/cuda/kernels/nnue_simd.h"

#ifdef USE_CUDA_TENSOR_CORES
#include "../src/gpu/cuda/kernels/nnue_tensor_core.h"
#endif

using namespace MetalFish::GPU;

namespace {

// Helper function to compare arrays with tolerance
template <typename T>
bool arrays_equal(const T *a, const T *b, size_t n, float tolerance = 1e-4f) {
  for (size_t i = 0; i < n; i++) {
    float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
    if (diff > tolerance) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                << " (diff: " << diff << ")" << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace

// ============================================================================
// Memory Management Tests
// ============================================================================

bool test_unified_memory() {
  std::cout << "\n[Test] Unified Memory with Hints" << std::endl;
  
  const size_t size = 1024 * 1024; // 1MB
  int device_id = 0;
  
  // Test basic unified memory allocation
  void *ptr = CUDA::UnifiedMemoryManager::allocate_unified(size, device_id);
  if (!ptr) {
    std::cerr << "  Failed to allocate unified memory" << std::endl;
    return false;
  }
  
  // Test read-only allocation
  void *readonly_ptr = CUDA::UnifiedMemoryManager::allocate_unified_readonly(size, device_id);
  if (!readonly_ptr) {
    std::cerr << "  Failed to allocate read-only unified memory" << std::endl;
    CUDA::UnifiedMemoryManager::free_unified(ptr);
    return false;
  }
  
  // Test prefetching
  CUDA::UnifiedMemoryManager::prefetch_to_device(ptr, size, device_id);
  cudaDeviceSynchronize();
  
  CUDA::UnifiedMemoryManager::prefetch_to_host(ptr, size);
  cudaDeviceSynchronize();
  
  // Cleanup
  CUDA::UnifiedMemoryManager::free_unified(ptr);
  CUDA::UnifiedMemoryManager::free_unified(readonly_ptr);
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

bool test_pinned_memory() {
  std::cout << "\n[Test] Pinned Memory" << std::endl;
  
  const size_t size = 1024 * 1024; // 1MB
  
  // Test pinned allocation
  void *ptr = CUDA::PinnedMemoryManager::allocate_pinned(size);
  if (!ptr) {
    std::cerr << "  Failed to allocate pinned memory" << std::endl;
    return false;
  }
  
  // Test memory registration
  std::vector<char> host_mem(size);
  if (!CUDA::PinnedMemoryManager::register_pinned(host_mem.data(), size)) {
    std::cerr << "  Failed to register pinned memory" << std::endl;
    CUDA::PinnedMemoryManager::free_pinned(ptr);
    return false;
  }
  
  // Cleanup
  CUDA::PinnedMemoryManager::unregister_pinned(host_mem.data());
  CUDA::PinnedMemoryManager::free_pinned(ptr);
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

bool test_double_buffer() {
  std::cout << "\n[Test] Double Buffer" << std::endl;
  
  const size_t size = 1024;
  int device_id = 0;
  
  CUDA::DoubleBuffer<int> buffer(size, device_id);
  
  // Fill buffer with test data
  int *host_buf = buffer.get_host_buffer();
  for (size_t i = 0; i < size; i++) {
    host_buf[i] = static_cast<int>(i);
  }
  
  // First, we need to transfer the current buffer to device before swapping
  cudaMemcpy(buffer.get_device_buffer(), host_buf, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  
  // Now swap - this prepares for the next iteration
  buffer.swap_and_transfer();
  buffer.synchronize();
  
  // The current device buffer should still have our data since we just copied it
  int *device_buf = buffer.get_device_buffer();
  std::vector<int> result(size);
  cudaMemcpy(result.data(), device_buf, size * sizeof(int), cudaMemcpyDeviceToHost);
  
  for (size_t i = 0; i < size; i++) {
    if (result[i] != static_cast<int>(i)) {
      std::cerr << "  Data mismatch at index " << i << std::endl;
      return false;
    }
  }
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

bool test_memory_pool() {
  std::cout << "\n[Test] Memory Pool" << std::endl;
  
  const size_t pool_size = 10 * 1024 * 1024; // 10MB
  int device_id = 0;
  
  CUDA::MemoryPool pool(pool_size, device_id);
  
  // Test allocations
  void *ptr1 = pool.allocate(1024);
  void *ptr2 = pool.allocate(2048);
  void *ptr3 = pool.allocate(4096);
  
  if (!ptr1 || !ptr2 || !ptr3) {
    std::cerr << "  Failed to allocate from pool" << std::endl;
    return false;
  }
  
  size_t allocated = pool.get_allocated();
  if (allocated < 7168) { // 1024 + 2048 + 4096
    std::cerr << "  Incorrect allocation size: " << allocated << std::endl;
    return false;
  }
  
  // Test reset
  pool.reset();
  if (pool.get_allocated() != 0) {
    std::cerr << "  Pool reset failed" << std::endl;
    return false;
  }
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// Profiling Tests
// ============================================================================

bool test_kernel_timer() {
  std::cout << "\n[Test] Kernel Timer" << std::endl;
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Allocate a small buffer for the test
  void *test_buffer;
  cudaMalloc(&test_buffer, 1024);
  
  {
    CUDA::KernelTimer timer("test_kernel", stream);
    
    // Simulate some work with actual operation
    cudaMemsetAsync(test_buffer, 0, 1024, stream);
    cudaStreamSynchronize(stream);
  }
  
  float avg_time = CUDA::KernelTimer::get_average_time("test_kernel");
  if (avg_time < 0.0f) {
    std::cerr << "  Invalid timing result" << std::endl;
    cudaFree(test_buffer);
    cudaStreamDestroy(stream);
    return false;
  }
  
  cudaFree(test_buffer);
  cudaStreamDestroy(stream);
  
  std::cout << "  PASSED (avg time: " << avg_time << " ms)" << std::endl;
  return true;
}

bool test_bandwidth_measurement() {
  std::cout << "\n[Test] Bandwidth Measurement" << std::endl;
  
  const size_t test_size = 64 * 1024 * 1024; // 64MB
  
  float h2d_bandwidth = CUDA::BandwidthTester::measure_h2d_bandwidth(test_size);
  float d2h_bandwidth = CUDA::BandwidthTester::measure_d2h_bandwidth(test_size);
  
  std::cout << "  H2D Bandwidth: " << h2d_bandwidth << " GB/s" << std::endl;
  std::cout << "  D2H Bandwidth: " << d2h_bandwidth << " GB/s" << std::endl;
  
  if (h2d_bandwidth <= 0.0f || d2h_bandwidth <= 0.0f) {
    std::cerr << "  Invalid bandwidth measurements" << std::endl;
    return false;
  }
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// Tensor Core Tests
// ============================================================================

#ifdef USE_CUDA_TENSOR_CORES

bool test_tensor_core_availability() {
  std::cout << "\n[Test] Tensor Core Availability" << std::endl;
  
  int device_id = 0;
  bool has_fp16 = cuda_tensor_cores_available(device_id);
  bool has_int8 = cuda_int8_tensor_cores_available(device_id);
  
  std::cout << "  FP16 Tensor Cores: " << (has_fp16 ? "Yes" : "No") << std::endl;
  std::cout << "  INT8 Tensor Cores: " << (has_int8 ? "Yes" : "No") << std::endl;
  
  // Just check that the function runs without error
  std::cout << "  PASSED" << std::endl;
  return true;
}

#endif // USE_CUDA_TENSOR_CORES

// ============================================================================
// Architecture Detection Tests
// ============================================================================

bool test_architecture_detection() {
  std::cout << "\n[Test] Architecture Detection" << std::endl;
  
  auto &backend = CUDABackend::instance();
  
  if (!backend.is_available()) {
    std::cout << "  SKIPPED (no CUDA device)" << std::endl;
    return true;
  }
  
  std::cout << "  Device: " << backend.device_name() << std::endl;
  std::cout << "  Compute Capability: " 
            << backend.compute_capability_major() << "." 
            << backend.compute_capability_minor() << std::endl;
  std::cout << "  Multiprocessors: " << backend.multiprocessor_count() << std::endl;
  std::cout << "  Total Memory: " << (backend.total_memory() / (1024 * 1024)) << " MB" << std::endl;
  std::cout << "  Has Tensor Cores: " << (backend.has_tensor_cores() ? "Yes" : "No") << std::endl;
  std::cout << "  Has INT8 Tensor Cores: " << (backend.has_int8_tensor_cores() ? "Yes" : "No") << std::endl;
  std::cout << "  Has Warp Shuffle: " << (backend.has_warp_shuffle() ? "Yes" : "No") << std::endl;
  std::cout << "  Has Cooperative Groups: " << (backend.has_cooperative_groups() ? "Yes" : "No") << std::endl;
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "CUDA Optimization Tests" << std::endl;
  std::cout << "======================================" << std::endl;
  
  int passed = 0;
  int failed = 0;
  
  // Memory tests
  if (test_unified_memory()) passed++; else failed++;
  if (test_pinned_memory()) passed++; else failed++;
  if (test_double_buffer()) passed++; else failed++;
  if (test_memory_pool()) passed++; else failed++;
  
  // Profiling tests
  if (test_kernel_timer()) passed++; else failed++;
  if (test_bandwidth_measurement()) passed++; else failed++;
  
  // Architecture tests
  if (test_architecture_detection()) passed++; else failed++;
  
#ifdef USE_CUDA_TENSOR_CORES
  // Tensor core tests
  if (test_tensor_core_availability()) passed++; else failed++;
#endif
  
  // Print summary
  std::cout << "\n======================================" << std::endl;
  std::cout << "Test Summary" << std::endl;
  std::cout << "======================================" << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << "Total:  " << (passed + failed) << std::endl;
  
  if (failed == 0) {
    std::cout << "\nAll tests PASSED! ✓" << std::endl;
  } else {
    std::cout << "\nSome tests FAILED! ✗" << std::endl;
  }
  
  return (failed == 0) ? 0 : 1;
}

#else // !USE_CUDA

int main() {
  std::cout << "CUDA support not enabled. Skipping tests." << std::endl;
  return 0;
}

#endif // USE_CUDA
