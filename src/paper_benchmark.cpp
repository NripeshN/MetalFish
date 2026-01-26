/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file paper_benchmark.cpp
 * @brief MetalFish source file.
 */

  Comprehensive Paper Benchmark Suite v2

  Provides rigorous measurements for academic publication:
  - CPU eval microbench with matched scope (100k+ iterations)
  - GPU batch latency table (N=1 to 2048)
  - Stage breakdown (feature extraction, buffer write, encode, sync)
  - Accuracy sanity check (CPU vs GPU score comparison)
  - True batching verification at multiple scales
*/

#include "core/bitboard.h"
#include "core/position.h"
#include "eval/evaluate.h"
#include "gpu/backend.h"
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

// ============================================================================
// Statistics Helper
// ============================================================================

struct LatencyStats {
  double mean_us, std_us, median_us, p95_us, p99_us, min_us, max_us;
  int count;

  static LatencyStats compute(std::vector<double> &samples) {
    LatencyStats s{};
    s.count = samples.size();
    if (samples.empty())
      return s;

    std::sort(samples.begin(), samples.end());
    s.min_us = samples.front();
    s.max_us = samples.back();
    s.median_us = samples[samples.size() / 2];
    s.p95_us = samples[size_t(samples.size() * 0.95)];
    s.p99_us = samples[size_t(samples.size() * 0.99)];

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.mean_us = sum / samples.size();

    double sq_sum = 0;
    for (double v : samples)
      sq_sum += (v - s.mean_us) * (v - s.mean_us);
    s.std_us = std::sqrt(sq_sum / samples.size());
    return s;
  }

  void print(const char *label) const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << label << ":\n";
    std::cout << "  Mean:   " << mean_us << " µs (σ=" << std_us << ")\n";
    std::cout << "  Median: " << median_us << " µs\n";
    std::cout << "  P95:    " << p95_us << " µs, P99: " << p99_us << " µs\n";
    std::cout << "  Range:  [" << min_us << ", " << max_us << "] µs\n";
    std::cout << "  N:      " << count << "\n";
  }

  void print_row(int batch_size) const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(6) << batch_size << std::setw(12) << median_us
              << std::setw(12) << p95_us << std::setw(12) << p99_us
              << std::setw(12) << (median_us / batch_size) << "\n";
  }
};

// Test positions
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
constexpr int NUM_FENS = sizeof(TEST_FENS) / sizeof(TEST_FENS[0]);
constexpr int MAX_FEATURES_PER_POS =
    32; // Explanation: HalfKAv2_hm max active features

// ============================================================================
// BENCHMARK 1: CPU Feature Extraction (matched scope with GPU)
// ============================================================================

void benchmark_cpu_feature_extraction() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 1: Batch Creation (Feature Extraction)\n";
  std::cout << "============================================================\n";
  std::cout << "\nScope: Create GPU batch with position features\n";
  std::cout << "       (includes HalfKAv2_hm feature extraction)\n";
  std::cout << "Iterations: 100,000\n\n";

  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(NUM_FENS);
  for (int i = 0; i < NUM_FENS; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i], false, &states_vec.back()->back());
  }

  // Warmup
  for (int i = 0; i < 1000; i++) {
    GPU::GPUEvalBatch batch;
    batch.add_position(positions[i % NUM_FENS]);
  }

  // Benchmark
  const int iterations = 100000;
  std::vector<double> samples;
  samples.reserve(iterations);

  for (int i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    GPU::GPUEvalBatch batch;
    batch.add_position(positions[i % NUM_FENS]);
    auto end = std::chrono::high_resolution_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  auto stats = LatencyStats::compute(samples);
  stats.print("Batch Creation (Feature Extraction)");
}

// ============================================================================
// BENCHMARK 2: GPU Dispatch Overhead (Minimal Kernel)
// ============================================================================

void benchmark_gpu_dispatch_overhead() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 2: GPU Dispatch Overhead (Minimal Kernel)\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available\n";
    return;
  }

  auto &backend = GPU::gpu();

  const char *shader = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void minimal_kernel(device int* out [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
      if (gid == 0) out[0] = 1;
    }
  )";

  if (!backend.compile_library("dispatch_bench", shader)) {
    std::cout << "  Shader compilation failed\n";
    return;
  }

  auto kernel = backend.create_kernel("minimal_kernel", "dispatch_bench");
  auto buffer = backend.create_buffer(sizeof(int));

  std::cout << "\nScope: create_encoder + set_kernel + dispatch(1) + "
               "submit_and_wait\n";
  std::cout << "       (blocking synchronous execution)\n";
  std::cout << "Iterations: 1,000\n\n";

  // Warmup
  for (int i = 0; i < 100; i++) {
    auto enc = backend.create_encoder();
    enc->set_kernel(kernel.get());
    enc->set_buffer(buffer.get(), 0);
    enc->dispatch_threads(1);
    backend.submit_and_wait(enc.get());
  }

  // Benchmark
  std::vector<double> samples;
  for (int i = 0; i < 1000; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    auto enc = backend.create_encoder();
    enc->set_kernel(kernel.get());
    enc->set_buffer(buffer.get(), 0);
    enc->dispatch_threads(1);
    backend.submit_and_wait(enc.get());
    auto end = std::chrono::high_resolution_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  auto stats = LatencyStats::compute(samples);
  stats.print("GPU Dispatch Overhead");
}

// ============================================================================
// BENCHMARK 3: GPU Batch Latency Table (N=1 to 2048)
// ============================================================================

void benchmark_gpu_batch_latency_table() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 3: GPU End-to-End Batch Latency Table\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized (networks not loaded)\n";
    return;
  }

  std::cout << "\nScope: Full end-to-end GPU evaluation\n";
  std::cout
      << "       (batch creation + buffer write + dispatch + kernel + sync)\n";
  std::cout << "Iterations: 100 per batch size\n\n";

  // Create position pool
  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(2048);
  for (int i = 0; i < 2048; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_FENS], false,
                     &states_vec.back()->back());
  }

  std::vector<int> batch_sizes = {1,  2,   4,   8,   16,   32,
                                  64, 128, 256, 512, 1024, 2048};
  const int iterations = 100;

  manager.set_min_batch_size(1); // Force GPU for all sizes

  std::cout << std::setw(6) << "Batch" << std::setw(12) << "Median"
            << std::setw(12) << "P95" << std::setw(12) << "P99" << std::setw(12)
            << "Per-Pos\n";
  std::cout << std::setw(6) << "Size" << std::setw(12) << "(µs)"
            << std::setw(12) << "(µs)" << std::setw(12) << "(µs)"
            << std::setw(12) << "(µs)\n";
  std::cout << std::string(54, '-') << "\n";

  for (int batch_size : batch_sizes) {
    std::vector<double> samples;

    for (int iter = 0; iter < iterations; iter++) {
      GPU::GPUEvalBatch batch;
      batch.reserve(batch_size);
      for (int i = 0; i < batch_size; i++) {
        batch.add_position(positions[i]);
      }

      auto start = std::chrono::high_resolution_clock::now();
      manager.evaluate_batch(batch, true);
      auto end = std::chrono::high_resolution_clock::now();

      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    auto stats = LatencyStats::compute(samples);
    stats.print_row(batch_size);
  }
}

// ============================================================================
// BENCHMARK 4: Stage Breakdown for GPU End-to-End
// ============================================================================

void benchmark_gpu_stage_breakdown() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 4: GPU Stage Breakdown\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized\n";
    return;
  }

  std::cout << "\nStages measured:\n";
  std::cout << "  1. Batch creation + feature extraction (CPU)\n";
  std::cout
      << "  2. GPU evaluate_batch() (buffer + dispatch + kernel + sync)\n";
  std::cout << "Iterations: 100 per batch size\n\n";

  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(1024);
  for (int i = 0; i < 1024; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_FENS], false,
                     &states_vec.back()->back());
  }

  std::vector<int> batch_sizes = {1, 16, 256, 1024};
  const int iterations = 100;
  manager.set_min_batch_size(1);

  for (int batch_size : batch_sizes) {
    std::cout << "\n--- Batch Size: " << batch_size << " ---\n";

    std::vector<double> stage1_samples, stage2_samples;

    for (int iter = 0; iter < iterations; iter++) {
      // Stage 1: Batch creation
      auto t1 = std::chrono::high_resolution_clock::now();
      GPU::GPUEvalBatch batch;
      batch.reserve(batch_size);
      for (int i = 0; i < batch_size; i++) {
        batch.add_position(positions[i]);
      }
      auto t2 = std::chrono::high_resolution_clock::now();

      // Stage 2: GPU evaluation
      manager.evaluate_batch(batch, true);
      auto t3 = std::chrono::high_resolution_clock::now();

      stage1_samples.push_back(
          std::chrono::duration<double, std::micro>(t2 - t1).count());
      stage2_samples.push_back(
          std::chrono::duration<double, std::micro>(t3 - t2).count());
    }

    auto s1 = LatencyStats::compute(stage1_samples);
    auto s2 = LatencyStats::compute(stage2_samples);

    std::cout << "  Batch creation (CPU): median=" << std::fixed
              << std::setprecision(2) << s1.median_us << " µs ("
              << (s1.median_us / batch_size) << " µs/pos)\n";
    std::cout << "  GPU evaluate_batch:   median=" << s2.median_us << " µs ("
              << (s2.median_us / batch_size) << " µs/pos)\n";
    std::cout << "  Total:                median="
              << (s1.median_us + s2.median_us) << " µs ("
              << ((s1.median_us + s2.median_us) / batch_size) << " µs/pos)\n";
  }
}

// ============================================================================
// BENCHMARK 5: True Batching Verification (Multiple Scales)
// ============================================================================

void benchmark_true_batching_verification() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 5: True Batching Verification\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized\n";
    return;
  }

  std::cout
      << "\nComparing: N × (1-position batch) vs 1 × (N-position batch)\n";
  std::cout << "If true batching: single dispatch should be faster than N "
               "dispatches\n\n";

  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(1024);
  for (int i = 0; i < 1024; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_FENS], false,
                     &states_vec.back()->back());
  }

  std::vector<int> batch_sizes = {16, 64, 256, 1024};
  const int iterations = 50;
  manager.set_min_batch_size(1);

  std::cout << std::setw(6) << "N" << std::setw(15) << "Sequential"
            << std::setw(15) << "Batched" << std::setw(12) << "Speedup\n";
  std::cout << std::setw(6) << "" << std::setw(15) << "(N×1 batch)"
            << std::setw(15) << "(1×N batch)" << std::setw(12) << "\n";
  std::cout << std::string(48, '-') << "\n";

  for (int N : batch_sizes) {
    // Sequential: N separate dispatches
    std::vector<double> seq_samples;
    for (int iter = 0; iter < iterations; iter++) {
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < N; i++) {
        GPU::GPUEvalBatch batch;
        batch.reserve(1);
        batch.add_position(positions[i]);
        manager.evaluate_batch(batch, true);
      }
      auto end = std::chrono::high_resolution_clock::now();
      seq_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    // Batched: 1 dispatch for N positions
    std::vector<double> batch_samples;
    for (int iter = 0; iter < iterations; iter++) {
      GPU::GPUEvalBatch batch;
      batch.reserve(N);
      for (int i = 0; i < N; i++) {
        batch.add_position(positions[i]);
      }
      auto start = std::chrono::high_resolution_clock::now();
      manager.evaluate_batch(batch, true);
      auto end = std::chrono::high_resolution_clock::now();
      batch_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    auto seq_stats = LatencyStats::compute(seq_samples);
    auto batch_stats = LatencyStats::compute(batch_samples);
    double speedup = seq_stats.median_us / batch_stats.median_us;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(6) << N << std::setw(15) << seq_stats.median_us
              << std::setw(15) << batch_stats.median_us << std::setw(12)
              << speedup << "×\n";
  }
}

// ============================================================================
// BENCHMARK 6: Accuracy Sanity Check (CPU vs GPU Scores)
// ============================================================================

void benchmark_accuracy_check() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  BENCHMARK 6: Accuracy Sanity Check\n";
  std::cout << "============================================================\n";

  if (!GPU::gpu_available()) {
    std::cout << "  GPU not available\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "  GPU NNUE not initialized\n";
    return;
  }

  std::cout << "\nComparing CPU simple_eval vs GPU NNUE scores\n";
  std::cout << "(Note: These use different evaluation methods, so differences "
               "expected)\n\n";

  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(100);
  for (int i = 0; i < 100; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(TEST_FENS[i % NUM_FENS], false,
                     &states_vec.back()->back());
  }

  // Get GPU scores
  GPU::GPUEvalBatch batch;
  batch.reserve(100);
  for (int i = 0; i < 100; i++) {
    batch.add_position(positions[i]);
  }
  manager.set_min_batch_size(1);
  manager.evaluate_batch(batch, true);

  // Compare with CPU simple_eval
  std::vector<double> errors;
  for (int i = 0; i < 100; i++) {
    int cpu_score = Eval::simple_eval(positions[i]);
    int gpu_score = batch.positional_scores[i];
    errors.push_back(std::abs(cpu_score - gpu_score));
  }

  double mean_err =
      std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
  double max_err = *std::max_element(errors.begin(), errors.end());

  std::cout << "Positions evaluated: 100\n";
  std::cout << "Mean absolute error: " << std::fixed << std::setprecision(1)
            << mean_err << " cp\n";
  std::cout << "Max absolute error:  " << max_err << " cp\n";
  std::cout << "\n(Large differences expected: simple_eval is material-only,\n";
  std::cout << " GPU NNUE includes positional factors)\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     MetalFish Paper Benchmark Suite v2                   ║\n";
  std::cout << "║     Comprehensive GPU NNUE Performance Analysis          ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  Bitboards::init();

  if (GPU::gpu_available()) {
    auto &gpu = GPU::gpu();
    std::cout << "\nHardware:\n";
    std::cout << "  GPU: " << gpu.device_name() << "\n";
    std::cout << "  Unified Memory: "
              << (gpu.has_unified_memory() ? "Yes" : "No") << "\n";
  }

  std::cout << "\nNote: GPU NNUE benchmarks require loaded networks.\n";
  std::cout << "Feature extraction benchmarks work without networks.\n";

  benchmark_cpu_feature_extraction();
  benchmark_gpu_dispatch_overhead();
  benchmark_gpu_batch_latency_table();
  benchmark_gpu_stage_breakdown();
  benchmark_true_batching_verification();
  benchmark_accuracy_check();

  std::cout
      << "\n============================================================\n";
  std::cout << "  Benchmark Suite Complete\n";
  std::cout << "============================================================\n";

  return 0;
}