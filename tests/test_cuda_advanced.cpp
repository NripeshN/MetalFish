/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Advanced CUDA Features Test

  Tests for CUDA graphs, multi-GPU, persistent kernels, and FP16 weights.
*/

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef USE_CUDA

#include "../src/gpu/cuda/cuda_backend.h"
#include "../src/gpu/cuda/cuda_graphs.h"
#include "../src/gpu/cuda/cuda_multi_gpu.h"
#include "../src/gpu/cuda/cuda_fp16_weights.h"

using namespace MetalFish::GPU;
using namespace MetalFish::GPU::CUDA;

// ============================================================================
// CUDA Graphs Tests
// ============================================================================

bool test_cuda_graphs() {
  std::cout << "\n[Test] CUDA Graphs" << std::endl;
  
  GraphManager manager;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Test graph capture
  bool started = manager.begin_capture(stream, "test_graph");
  if (!started) {
    std::cerr << "  Failed to begin capture" << std::endl;
    cudaStreamDestroy(stream);
    return false;
  }
  
  // Simulate some operations (empty kernel for test)
  void *dummy_buffer;
  cudaMalloc(&dummy_buffer, 1024);
  cudaMemsetAsync(dummy_buffer, 0, 1024, stream);
  
  bool ended = manager.end_capture(stream, "test_graph");
  if (!ended) {
    std::cerr << "  Failed to end capture" << std::endl;
    cudaFree(dummy_buffer);
    cudaStreamDestroy(stream);
    return false;
  }
  
  // Test graph replay
  bool launched = manager.launch_graph("test_graph", stream);
  if (!launched) {
    std::cerr << "  Failed to launch graph" << std::endl;
    cudaFree(dummy_buffer);
    cudaStreamDestroy(stream);
    return false;
  }
  
  cudaStreamSynchronize(stream);
  
  // Check statistics
  auto stats = manager.get_stats();
  std::cout << "  Graphs: " << stats.num_graphs 
            << ", Nodes: " << stats.total_nodes << std::endl;
  
  cudaFree(dummy_buffer);
  cudaStreamDestroy(stream);
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// Multi-GPU Tests
// ============================================================================

bool test_multi_gpu() {
  std::cout << "\n[Test] Multi-GPU Support" << std::endl;
  
  MultiGPUManager manager;
  
  // Initialize with all GPUs
  if (!manager.initialize(true)) {
    std::cout << "  SKIPPED (no GPUs available)" << std::endl;
    return true;
  }
  
  int num_gpus = manager.get_num_gpus();
  std::cout << "  Number of GPUs: " << num_gpus << std::endl;
  
  // Test GPU enumeration
  for (int i = 0; i < num_gpus; i++) {
    const auto& info = manager.get_gpu_info(i);
    std::cout << "  GPU " << i << ": " << info.name 
              << " (SM " << info.compute_major << "." << info.compute_minor << ")"
              << std::endl;
  }
  
  // Test batch distribution
  int batch_size = 1024;
  auto distribution = manager.distribute_batch(batch_size);
  
  int total = 0;
  for (size_t i = 0; i < distribution.size(); i++) {
    std::cout << "  GPU " << i << " gets " << distribution[i] << " items" << std::endl;
    total += distribution[i];
  }
  
  if (total != batch_size) {
    std::cerr << "  Batch distribution mismatch: " << total << " vs " << batch_size << std::endl;
    return false;
  }
  
  // Test peer access if multiple GPUs
  if (num_gpus > 1) {
    manager.enable_peer_access();
  }
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// FP16 Weights Tests
// ============================================================================

bool test_fp16_weights() {
  std::cout << "\n[Test] FP16 Weight Storage" << std::endl;
  
  FP16WeightManager manager;
  
  // Create test weights
  const size_t size = 1024;
  std::vector<int16_t> int16_weights(size);
  std::vector<int32_t> int32_biases(32);
  
  for (size_t i = 0; i < size; i++) {
    int16_weights[i] = static_cast<int16_t>(i % 128);
  }
  
  for (size_t i = 0; i < 32; i++) {
    int32_biases[i] = static_cast<int32_t>(i * 64);
  }
  
  // Convert to FP16
  half* fp16_weights = manager.convert_and_store_weights(int16_weights.data(), size);
  if (!fp16_weights) {
    std::cerr << "  Failed to convert weights" << std::endl;
    return false;
  }
  
  half* fp16_biases = manager.convert_and_store_biases(int32_biases.data(), 32);
  if (!fp16_biases) {
    std::cerr << "  Failed to convert biases" << std::endl;
    return false;
  }
  
  // Verify conversion by copying back
  std::vector<half> verify_weights(size);
  cudaMemcpy(verify_weights.data(), fp16_weights, size * sizeof(half), 
             cudaMemcpyDeviceToHost);
  
  // Check a few values
  for (size_t i = 0; i < 10; i++) {
    float expected = static_cast<float>(int16_weights[i]) / 64.0f;
    float actual = __half2float(verify_weights[i]);
    if (std::abs(expected - actual) > 0.01f) {
      std::cerr << "  Conversion mismatch at index " << i << std::endl;
      return false;
    }
  }
  
  size_t mem_usage = manager.get_memory_usage();
  std::cout << "  Memory usage: " << (mem_usage / 1024) << " KB" << std::endl;
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// Backend Integration Test
// ============================================================================

bool test_backend_features() {
  std::cout << "\n[Test] Backend Feature Integration" << std::endl;
  
  auto &backend = CUDABackend::instance();
  
  if (!backend.is_available()) {
    std::cout << "  SKIPPED (no CUDA device)" << std::endl;
    return true;
  }
  
  // Test feature enablement
  backend.enable_cuda_graphs(true);
  backend.enable_multi_gpu(false);  // Keep single GPU for simplicity
  backend.enable_persistent_kernels(false);
  backend.enable_fp16_weights(backend.has_tensor_cores());
  
  std::cout << "  CUDA Graphs: " << (backend.is_cuda_graphs_enabled() ? "ON" : "OFF") << std::endl;
  std::cout << "  Multi-GPU: " << (backend.is_multi_gpu_enabled() ? "ON" : "OFF") << std::endl;
  std::cout << "  Persistent Kernels: " << (backend.is_persistent_kernels_enabled() ? "ON" : "OFF") << std::endl;
  std::cout << "  FP16 Weights: " << (backend.is_fp16_weights_enabled() ? "ON" : "OFF") << std::endl;
  
  std::cout << "  PASSED" << std::endl;
  return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "Advanced CUDA Features Tests" << std::endl;
  std::cout << "======================================" << std::endl;
  
  int passed = 0;
  int failed = 0;
  
  // Run tests
  if (test_cuda_graphs()) passed++; else failed++;
  if (test_multi_gpu()) passed++; else failed++;
  if (test_fp16_weights()) passed++; else failed++;
  if (test_backend_features()) passed++; else failed++;
  
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
