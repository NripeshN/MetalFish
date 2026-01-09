/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Benchmarking utility
*/

#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/batch_ops.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace MetalFish;

// Benchmark shader execution
void benchmark_shader_execution() {
  if (!GPU::gpu_available()) {
    std::cout << "GPU not available, skipping shader benchmark" << std::endl;
    return;
  }

  auto &gpu = GPU::gpu();

  // Compile a simple compute shader
  const char *shader = R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void vector_add(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              constant int& count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
            if (gid < uint(count)) {
                result[gid] = a[gid] + b[gid];
            }
        }
    )";

  if (!gpu.compile_library("bench", shader)) {
    std::cout << "Failed to compile benchmark shader" << std::endl;
    return;
  }

  auto kernel = gpu.create_kernel("vector_add", "bench");
  if (!kernel || !kernel->valid()) {
    std::cout << "Failed to create benchmark kernel" << std::endl;
    return;
  }

  // Test different sizes
  std::vector<int> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};

  std::cout << "\n=== GPU Shader Execution Benchmark ===" << std::endl;
  std::cout << "Size\t\tGPU Time (ms)\tBandwidth (GB/s)" << std::endl;

  for (int size : sizes) {
    // Create buffers
    auto buf_a = gpu.create_buffer(size * sizeof(float));
    auto buf_b = gpu.create_buffer(size * sizeof(float));
    auto buf_result = gpu.create_buffer(size * sizeof(float));

    // Initialize data
    float *a = buf_a->as<float>();
    float *b = buf_b->as<float>();
    for (int i = 0; i < size; i++) {
      a[i] = float(i);
      b[i] = float(size - i);
    }

    // Warm up
    auto enc = gpu.create_encoder();
    enc->set_kernel(kernel.get());
    enc->set_buffer(buf_a.get(), 0);
    enc->set_buffer(buf_b.get(), 1);
    enc->set_buffer(buf_result.get(), 2);
    enc->set_value(size, 3);
    enc->dispatch_threads(size);
    gpu.submit_and_wait(enc.get());

    // Benchmark
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
      auto enc = gpu.create_encoder();
      enc->set_kernel(kernel.get());
      enc->set_buffer(buf_a.get(), 0);
      enc->set_buffer(buf_b.get(), 1);
      enc->set_buffer(buf_result.get(), 2);
      enc->set_value(size, 3);
      enc->dispatch_threads(size);
      gpu.submit_and_wait(enc.get());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;

    // Bandwidth: 3 buffers * size * sizeof(float) / time
    double bytes = 3.0 * size * sizeof(float);
    double bandwidth_gbps =
        (bytes / (avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << size << "\t\t" << avg_ms << "\t\t" << bandwidth_gbps
              << std::endl;
  }
}

// Benchmark unified memory access
void benchmark_unified_memory() {
  if (!GPU::gpu_available())
    return;

  auto &gpu = GPU::gpu();

  if (!gpu.has_unified_memory()) {
    std::cout << "Unified memory not available" << std::endl;
    return;
  }

  std::cout << "\n=== Unified Memory Benchmark ===" << std::endl;

  const int size = 1024 * 1024; // 1M elements
  auto buffer = gpu.create_buffer(size * sizeof(float));

  // CPU write
  auto start = std::chrono::high_resolution_clock::now();
  float *ptr = buffer->as<float>();
  for (int i = 0; i < size; i++) {
    ptr[i] = float(i);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double write_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // CPU read
  start = std::chrono::high_resolution_clock::now();
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += ptr[i];
  }
  end = std::chrono::high_resolution_clock::now();
  double read_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  double bytes = size * sizeof(float);
  std::cout << "CPU Write: " << write_ms << " ms ("
            << (bytes / (write_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0)
            << " GB/s)" << std::endl;
  std::cout << "CPU Read:  " << read_ms << " ms ("
            << (bytes / (read_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0)
            << " GB/s)" << std::endl;
  std::cout << "(Sum: " << sum << " to prevent optimization)" << std::endl;
}

int main() {
  std::cout << "MetalFish GPU Benchmark" << std::endl;
  std::cout << "=======================" << std::endl;

  // Initialize bitboards
  Bitboards::init();

  if (GPU::gpu_available()) {
    auto &gpu = GPU::gpu();
    std::cout << "\nGPU Device: " << gpu.device_name() << std::endl;
    std::cout << "Unified Memory: " << (gpu.has_unified_memory() ? "Yes" : "No")
              << std::endl;
    std::cout << "Max Buffer Size: " << gpu.max_buffer_size() / (1024 * 1024)
              << " MB" << std::endl;
  } else {
    std::cout << "No GPU available" << std::endl;
    return 1;
  }

  benchmark_shader_execution();
  benchmark_unified_memory();

  std::cout << "\nBenchmark complete!" << std::endl;
  return 0;
}
