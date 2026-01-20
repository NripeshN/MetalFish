/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Benchmarking utility
*/

#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/gpu_accumulator.h"
#include "gpu/gpu_nnue_integration.h"
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

// Benchmark GPU NNUE operations
void benchmark_gpu_nnue() {
  if (!GPU::gpu_available()) {
    std::cout << "GPU not available, skipping NNUE benchmark" << std::endl;
    return;
  }

  std::cout << "\n=== GPU NNUE Benchmark ===" << std::endl;

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "GPU NNUE not initialized (networks not loaded)" << std::endl;
    std::cout
        << "Run 'metalfish' and use 'bench' command for full NNUE benchmarks"
        << std::endl;
    return;
  }

  // Create test positions
  std::vector<std::string> test_fens = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
      "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
      "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"};

  // Benchmark batch sizes
  std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64};

  std::cout << "Batch Size\tTime (ms)\tPositions/sec" << std::endl;

  for (int batch_size : batch_sizes) {
    GPU::GPUEvalBatch batch;
    batch.reserve(batch_size);

    // Create positions
    std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
    std::vector<Position> positions(batch_size);

    for (int i = 0; i < batch_size; i++) {
      states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
      positions[i].set(test_fens[i % test_fens.size()], false,
                       &states_vec.back()->back());
      batch.add_position(positions[i]);
    }

    // Warm up
    manager.evaluate_batch(batch, true);

    // Benchmark
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
      batch.clear();
      for (int j = 0; j < batch_size; j++) {
        batch.add_position(positions[j]);
      }
      manager.evaluate_batch(batch, true);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double positions_per_sec = (batch_size * 1000.0) / avg_ms;

    std::cout << batch_size << "\t\t" << avg_ms << "\t\t" << positions_per_sec
              << std::endl;
  }

  // Print statistics
  std::cout << "\nGPU NNUE Statistics:" << std::endl;
  std::cout << "  GPU Evaluations: " << manager.gpu_evaluations() << std::endl;
  std::cout << "  CPU Fallbacks: " << manager.cpu_fallback_evaluations()
            << std::endl;
  std::cout << "  Total Batches: " << manager.total_batches() << std::endl;
  if (manager.total_batches() > 0) {
    std::cout << "  Avg Batch Time: " << manager.avg_batch_time_ms() << " ms"
              << std::endl;
  }
}

// Benchmark GPU accumulator operations
void benchmark_gpu_accumulator() {
  if (!GPU::gpu_available()) {
    std::cout << "GPU not available, skipping accumulator benchmark"
              << std::endl;
    return;
  }

  std::cout << "\n=== GPU Accumulator Benchmark ===" << std::endl;

  GPU::GPUAccumulatorStack stack;
  if (!stack.initialize(GPU::GPU_FT_DIM_BIG, true)) {
    std::cout << "Failed to initialize GPU accumulator stack" << std::endl;
    return;
  }

  std::cout << "GPU Accumulator Stack initialized" << std::endl;
  std::cout << "  Hidden dim: " << GPU::GPU_FT_DIM_BIG << std::endl;
  std::cout << "  Max ply: " << GPU::GPUAccumulatorStack::MAX_PLY << std::endl;

  // Note: Full accumulator benchmarks require network weights
  // For now, just report initialization success
  std::cout << "  (Full benchmarks require loaded networks)" << std::endl;
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
  benchmark_gpu_nnue();
  benchmark_gpu_accumulator();

  std::cout << "\nBenchmark complete!" << std::endl;
  return 0;
}
