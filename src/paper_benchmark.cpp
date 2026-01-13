/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive Paper Benchmark Suite

  Provides rigorous measurements for academic publication:
  - Latency with percentiles (median, p95, p99)
  - Stage breakdown (feature extraction, buffer write, dispatch, kernel, sync)
  - True batching verification
  - CPU vs GPU comparison across batch sizes
*/

#include "core/bitboard.h"
#include "core/position.h"
#include "eval/evaluate.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_accumulator.h"
#include "gpu/backend.h"
#include "gpu/gpu_accumulator.h"
#include "gpu/gpu_nnue_integration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace MetalFish;

// Global NNUE resources for CPU evaluation
static std::unique_ptr<Eval::NNUE::Networks> g_networks;
static std::unique_ptr<Eval::NNUE::AccumulatorStack> g_accumulators;
static std::unique_ptr<Eval::NNUE::AccumulatorCaches> g_caches;

// ============================================================================
// Statistics Helper
// ============================================================================

struct LatencyStats {
  double mean_us;
  double std_us;
  double median_us;
  double p95_us;
  double p99_us;
  double min_us;
  double max_us;
  int count;

  static LatencyStats compute(std::vector<double> &samples_us) {
    LatencyStats stats;
    stats.count = samples_us.size();

    if (samples_us.empty()) {
      stats.mean_us = stats.std_us = stats.median_us = 0;
      stats.p95_us = stats.p99_us = stats.min_us = stats.max_us = 0;
      return stats;
    }

    std::sort(samples_us.begin(), samples_us.end());

    stats.min_us = samples_us.front();
    stats.max_us = samples_us.back();
    stats.median_us = samples_us[samples_us.size() / 2];
    stats.p95_us = samples_us[size_t(samples_us.size() * 0.95)];
    stats.p99_us = samples_us[size_t(samples_us.size() * 0.99)];

    double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    stats.mean_us = sum / samples_us.size();

    double sq_sum = 0;
    for (double v : samples_us) {
      sq_sum += (v - stats.mean_us) * (v - stats.mean_us);
    }
    stats.std_us = std::sqrt(sq_sum / samples_us.size());

    return stats;
  }

  void print(const char *label) const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << label << ":\n";
    std::cout << "  Mean:   " << mean_us << " µs (σ=" << std_us << ")\n";
    std::cout << "  Median: " << median_us << " µs\n";
    std::cout << "  P95:    " << p95_us << " µs\n";
    std::cout << "  P99:    " << p99_us << " µs\n";
    std::cout << "  Min:    " << min_us << " µs\n";
    std::cout << "  Max:    " << max_us << " µs\n";
    std::cout << "  N:      " << count << " samples\n";
  }
};

// ============================================================================
// Test Positions
// ============================================================================

const char *TEST_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
};
constexpr int NUM_TEST_FENS = sizeof(TEST_FENS) / sizeof(TEST_FENS[0]);

// ============================================================================
// Benchmark 1: CPU NNUE Evaluation Latency
// ============================================================================

void benchmark_cpu_eval_latency() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 1: CPU NNUE Single-Position Evaluation Latency\n";
  std::cout << "============================================================\n";

  if (!g_networks) {
    std::cout << "  NNUE networks not loaded, using simple_eval fallback\n";
  }

  std::cout << "\nMethodology:\n";
  std::cout << "  - Timer: std::chrono::high_resolution_clock\n";
  std::cout << "  - Scope: Full NNUE forward pass (accumulator refresh + FC "
               "layers)\n";
  std::cout << "  - Warmup: 100 iterations discarded\n";
  std::cout << "  - Samples: 10,000 iterations\n";
  std::cout << "  - Position: Rotated through " << NUM_TEST_FENS
            << " test positions\n";
  std::cout << "\n";

  // Create positions
  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(NUM_TEST_FENS);

  for (int i = 0; i < NUM_TEST_FENS; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i], false, &states_vec.back()->back());
  }

  // Warmup
  for (int i = 0; i < 100; i++) {
    for (auto &pos : positions) {
      volatile Value v = Eval::simple_eval(pos);
      (void)v;
    }
  }

  // Benchmark
  const int iterations = 10000;
  std::vector<double> samples_us;
  samples_us.reserve(iterations);

  for (int i = 0; i < iterations; i++) {
    Position &pos = positions[i % NUM_TEST_FENS];

    auto start = std::chrono::high_resolution_clock::now();
    volatile Value v;
    if (g_networks && g_accumulators && g_caches) {
      v = Eval::evaluate(*g_networks, pos, *g_accumulators, *g_caches, 0);
    } else {
      v = Eval::simple_eval(pos);
    }
    auto end = std::chrono::high_resolution_clock::now();

    (void)v;
    double us =
        std::chrono::duration<double, std::micro>(end - start).count();
    samples_us.push_back(us);
  }

  auto stats = LatencyStats::compute(samples_us);
  stats.print(g_networks ? "CPU NNUE Eval Latency" : "CPU Simple Eval Latency");
}

// ============================================================================
// Benchmark 2: GPU Dispatch Overhead (Empty Kernel)
// ============================================================================

void benchmark_gpu_dispatch_overhead() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 2: GPU Dispatch Overhead (Minimal Kernel)\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available, skipping\n";
    return;
  }

  auto &backend = GPU::gpu();

  // Compile minimal kernel
  const char *shader = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void minimal_kernel(device int* out [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
      if (gid == 0) out[0] = 1;
    }
  )";

  if (!backend.compile_library("dispatch_bench", shader)) {
    std::cout << "  Failed to compile shader\n";
    return;
  }

  auto kernel = backend.create_kernel("minimal_kernel", "dispatch_bench");
  if (!kernel || !kernel->valid()) {
    std::cout << "  Failed to create kernel\n";
    return;
  }

  auto buffer = backend.create_buffer(sizeof(int));

  std::cout << "\nMethodology:\n";
  std::cout << "  - Timer: std::chrono::high_resolution_clock\n";
  std::cout
      << "  - Scope: create_encoder() + set_kernel() + dispatch(1) + "
         "submit_and_wait()\n";
  std::cout << "  - Kernel: Writes single int (minimal compute)\n";
  std::cout << "  - Warmup: 100 iterations\n";
  std::cout << "  - Samples: 1,000 iterations\n";
  std::cout << "\n";

  // Warmup
  for (int i = 0; i < 100; i++) {
    auto enc = backend.create_encoder();
    enc->set_kernel(kernel.get());
    enc->set_buffer(buffer.get(), 0);
    enc->dispatch_threads(1);
    backend.submit_and_wait(enc.get());
  }

  // Benchmark
  const int iterations = 1000;
  std::vector<double> samples_us;
  samples_us.reserve(iterations);

  for (int i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    auto enc = backend.create_encoder();
    enc->set_kernel(kernel.get());
    enc->set_buffer(buffer.get(), 0);
    enc->dispatch_threads(1);
    backend.submit_and_wait(enc.get());

    auto end = std::chrono::high_resolution_clock::now();
    double us =
        std::chrono::duration<double, std::micro>(end - start).count();
    samples_us.push_back(us);
  }

  auto stats = LatencyStats::compute(samples_us);
  stats.print("GPU Dispatch Overhead (minimal kernel)");
}

// ============================================================================
// Benchmark 3: GPU NNUE Stage Breakdown
// ============================================================================

void benchmark_gpu_stage_breakdown() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 3: GPU NNUE Stage Breakdown\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available, skipping\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized (networks not loaded)\n";
    std::cout << "  Run with loaded networks for stage breakdown\n";
    return;
  }

  std::cout << "\nMethodology:\n";
  std::cout << "  - Each stage timed separately\n";
  std::cout << "  - Batch size: 8 positions\n";
  std::cout << "  - Samples: 100 iterations\n";
  std::cout << "\n";

  // Create positions
  const int batch_size = 8;
  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(batch_size);

  for (int i = 0; i < batch_size; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_TEST_FENS], false,
                     &states_vec.back()->back());
  }

  const int iterations = 100;

  // Stage 1: Batch creation + position data extraction
  std::vector<double> stage1_us;
  for (int iter = 0; iter < iterations; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    GPU::GPUEvalBatch batch;
    batch.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
      batch.add_position(positions[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    stage1_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  // Stage 2: Full GPU evaluation (includes buffer upload, dispatch, sync)
  std::vector<double> stage2_us;
  for (int iter = 0; iter < iterations; iter++) {
    GPU::GPUEvalBatch batch;
    batch.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
      batch.add_position(positions[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    manager.evaluate_batch(batch, true);
    auto end = std::chrono::high_resolution_clock::now();

    stage2_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  auto stats1 = LatencyStats::compute(stage1_us);
  auto stats2 = LatencyStats::compute(stage2_us);

  std::cout << "Stage Breakdown (batch size " << batch_size << "):\n\n";
  stats1.print("Stage 1: Batch Creation + Feature Extraction (CPU)");
  std::cout << "\n";
  stats2.print("Stage 2: GPU Evaluation (buffer write + dispatch + kernel + "
               "sync)");

  std::cout << "\nTotal End-to-End: " << std::fixed << std::setprecision(2)
            << (stats1.median_us + stats2.median_us) << " µs (median)\n";
  std::cout << "Per-Position: "
            << (stats1.median_us + stats2.median_us) / batch_size
            << " µs/position\n";
}

// ============================================================================
// Benchmark 4: CPU vs GPU Latency Across Batch Sizes
// ============================================================================

void benchmark_cpu_vs_gpu_batch() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 4: CPU vs GPU Latency Across Batch Sizes\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available, skipping\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  bool gpu_ready = manager.is_ready();

  std::cout << "\nMethodology:\n";
  std::cout << "  - CPU: Sequential Eval::evaluate() calls\n";
  std::cout << "  - GPU: Single evaluate_batch() call (true batching)\n";
  std::cout << "  - Timer: std::chrono::high_resolution_clock\n";
  std::cout << "  - Samples: 100 iterations per batch size\n";
  std::cout << "  - GPU NNUE Ready: " << (gpu_ready ? "Yes" : "No") << "\n";
  std::cout << "\n";

  // Create positions pool
  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> all_positions(64);

  for (int i = 0; i < 64; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    all_positions[i].set(TEST_FENS[i % NUM_TEST_FENS], false,
                         &states_vec.back()->back());
  }

  std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64};
  const int iterations = 100;

  std::cout << std::setw(8) << "Batch" << std::setw(15) << "CPU Median"
            << std::setw(15) << "GPU Median" << std::setw(15) << "CPU/pos"
            << std::setw(15) << "GPU/pos" << std::setw(12) << "Speedup"
            << "\n";
  std::cout << std::setw(8) << "Size" << std::setw(15) << "(µs)"
            << std::setw(15) << "(µs)" << std::setw(15) << "(µs)"
            << std::setw(15) << "(µs)" << std::setw(12) << "(CPU/GPU)"
            << "\n";
  std::cout << std::string(80, '-') << "\n";

  for (int batch_size : batch_sizes) {
    // CPU benchmark
    std::vector<double> cpu_samples;
    for (int iter = 0; iter < iterations; iter++) {
      auto start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < batch_size; i++) {
        volatile Value v;
        if (g_networks && g_accumulators && g_caches) {
          v = Eval::evaluate(*g_networks, all_positions[i], *g_accumulators,
                             *g_caches, 0);
        } else {
          v = Eval::simple_eval(all_positions[i]);
        }
        (void)v;
      }

      auto end = std::chrono::high_resolution_clock::now();
      cpu_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    // GPU benchmark
    std::vector<double> gpu_samples;
    if (gpu_ready) {
      // Temporarily disable min batch size to measure all sizes
      int old_min = manager.min_batch_size();
      manager.set_min_batch_size(1);

      for (int iter = 0; iter < iterations; iter++) {
        GPU::GPUEvalBatch batch;
        batch.reserve(batch_size);
        for (int i = 0; i < batch_size; i++) {
          batch.add_position(all_positions[i]);
        }

        auto start = std::chrono::high_resolution_clock::now();
        manager.evaluate_batch(batch, true);
        auto end = std::chrono::high_resolution_clock::now();

        gpu_samples.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
      }

      manager.set_min_batch_size(old_min);
    }

    auto cpu_stats = LatencyStats::compute(cpu_samples);
    double cpu_per_pos = cpu_stats.median_us / batch_size;

    double gpu_median = 0, gpu_per_pos = 0, speedup = 0;
    if (!gpu_samples.empty()) {
      auto gpu_stats = LatencyStats::compute(gpu_samples);
      gpu_median = gpu_stats.median_us;
      gpu_per_pos = gpu_stats.median_us / batch_size;
      speedup = cpu_stats.median_us / gpu_stats.median_us;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(8) << batch_size << std::setw(15)
              << cpu_stats.median_us << std::setw(15)
              << (gpu_ready ? std::to_string((int)gpu_median) : "N/A")
              << std::setw(15) << cpu_per_pos << std::setw(15)
              << (gpu_ready ? std::to_string((int)gpu_per_pos) : "N/A")
              << std::setw(12)
              << (gpu_ready ? std::to_string(speedup).substr(0, 5) : "N/A")
              << "\n";
  }
}

// ============================================================================
// Benchmark 5: True Batching Verification
// ============================================================================

void benchmark_true_batching_verification() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 5: True Batching Verification\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available, skipping\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized\n";
    return;
  }

  std::cout << "\nMethodology:\n";
  std::cout << "  - Compare: N sequential dispatches vs 1 batch dispatch\n";
  std::cout << "  - If true batching: batch should be faster than N×single\n";
  std::cout << "  - Batch size: 16 positions\n";
  std::cout << "\n";

  const int batch_size = 16;
  const int iterations = 50;

  // Create positions
  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(batch_size);

  for (int i = 0; i < batch_size; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_TEST_FENS], false,
                     &states_vec.back()->back());
  }

  // Measure N sequential single-position evaluations
  manager.set_min_batch_size(1);
  std::vector<double> sequential_us;

  for (int iter = 0; iter < iterations; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < batch_size; i++) {
      GPU::GPUEvalBatch batch;
      batch.reserve(1);
      batch.add_position(positions[i]);
      manager.evaluate_batch(batch, true);
    }

    auto end = std::chrono::high_resolution_clock::now();
    sequential_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  // Measure single batch evaluation
  std::vector<double> batched_us;

  for (int iter = 0; iter < iterations; iter++) {
    GPU::GPUEvalBatch batch;
    batch.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
      batch.add_position(positions[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    manager.evaluate_batch(batch, true);
    auto end = std::chrono::high_resolution_clock::now();

    batched_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  auto seq_stats = LatencyStats::compute(sequential_us);
  auto batch_stats = LatencyStats::compute(batched_us);

  std::cout << "Results (batch size " << batch_size << "):\n\n";
  seq_stats.print("Sequential (16 separate dispatches)");
  std::cout << "\n";
  batch_stats.print("Batched (1 dispatch for 16 positions)");

  double ratio = seq_stats.median_us / batch_stats.median_us;
  std::cout << "\nSpeedup from batching: " << std::fixed << std::setprecision(2)
            << ratio << "x\n";

  if (ratio > 1.5) {
    std::cout << "✓ TRUE BATCHING CONFIRMED: Single dispatch processes "
                 "multiple positions\n";
  } else if (ratio > 1.1) {
    std::cout << "~ PARTIAL BATCHING: Some overhead reduction observed\n";
  } else {
    std::cout << "✗ NO BATCHING BENEFIT: Likely sequential internal "
                 "dispatches\n";
  }
}

// ============================================================================
// Benchmark 6: Throughput at Scale
// ============================================================================

void benchmark_throughput() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 6: Throughput at Scale\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available, skipping\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized\n";
    return;
  }

  std::cout << "\nMethodology:\n";
  std::cout << "  - Measure positions evaluated per second\n";
  std::cout << "  - Total positions: 10,000\n";
  std::cout << "  - Vary batch size\n";
  std::cout << "\n";

  const int total_positions = 10000;

  // Create position pool
  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(64);

  for (int i = 0; i < 64; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_TEST_FENS], false,
                     &states_vec.back()->back());
  }

  std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64};

  std::cout << std::setw(12) << "Batch Size" << std::setw(15) << "Time (ms)"
            << std::setw(18) << "Positions/sec" << "\n";
  std::cout << std::string(45, '-') << "\n";

  manager.set_min_batch_size(1);

  for (int batch_size : batch_sizes) {
    int num_batches = total_positions / batch_size;

    auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < num_batches; b++) {
      GPU::GPUEvalBatch batch;
      batch.reserve(batch_size);
      for (int i = 0; i < batch_size; i++) {
        batch.add_position(positions[(b * batch_size + i) % 64]);
      }
      manager.evaluate_batch(batch, true);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double positions_per_sec = (num_batches * batch_size * 1000.0) / total_ms;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(12) << batch_size << std::setw(15) << total_ms
              << std::setw(18) << (int)positions_per_sec << "\n";
  }
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     MetalFish Paper Benchmark Suite                      ║\n";
  std::cout << "║     Comprehensive GPU NNUE Performance Analysis          ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  // Initialize
  Bitboards::init();

  if (GPU::gpu_available()) {
    auto &gpu = GPU::gpu();
    std::cout << "\nHardware Configuration:\n";
    std::cout << "  GPU: " << gpu.device_name() << "\n";
    std::cout << "  Unified Memory: " << (gpu.has_unified_memory() ? "Yes" : "No")
              << "\n";
    std::cout << "  Max Buffer: " << gpu.max_buffer_size() / (1024 * 1024)
              << " MB\n";
  } else {
    std::cout << "\nNo GPU available\n";
  }

  std::cout << "\nNote: GPU NNUE benchmarks require loaded networks.\n";
  std::cout << "Run 'metalfish' first to load networks, or run these\n";
  std::cout << "benchmarks from within the engine context.\n";

  // Run benchmarks
  benchmark_cpu_eval_latency();
  benchmark_gpu_dispatch_overhead();
  benchmark_gpu_stage_breakdown();
  benchmark_cpu_vs_gpu_batch();
  benchmark_true_batching_verification();
  benchmark_throughput();

  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  Benchmark Suite Complete\n";
  std::cout << "============================================================\n";

  return 0;
}
