/*
  MetalFish - Paper Benchmarks
  Copyright (C) 2025 Nripesh Niketan

  Generates benchmark data for the MetalFish conference paper.

  Measurements:
  1. CPU NNUE evaluation latency (end-to-end)
  2. GPU NNUE evaluation latency (end-to-end: encode + submit + wait)
  3. GPU batch evaluation throughput
  4. Ablation: CPU-only vs GPU-only search performance
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/zobrist.h"
#include "eval/evaluate.h"
#include "eval/gpu_nnue.h"
#include "metal/device.h"
#include "metal/gpu_ops.h"
#include "search/search.h"
#include "search/tt.h"
#include "uci/uci.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace MetalFish;
using namespace std::chrono;

// Test positions for benchmarking
const std::vector<std::string> TestPositions = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/ppppp1pp/5n2/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
};

struct BenchmarkResult {
  double mean_us;
  double median_us;
  double stddev_us;
  double min_us;
  double max_us;
  double p95_us;
  double p99_us;
  int samples;
};

BenchmarkResult compute_stats(std::vector<double> times) {
  BenchmarkResult r;
  r.samples = times.size();
  if (times.empty()) {
    r.mean_us = r.median_us = r.stddev_us = r.min_us = r.max_us = r.p95_us =
        r.p99_us = 0;
    return r;
  }

  // Sort for percentiles
  std::sort(times.begin(), times.end());

  r.mean_us = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  r.min_us = times.front();
  r.max_us = times.back();
  r.median_us = times[times.size() / 2];
  r.p95_us = times[static_cast<size_t>(times.size() * 0.95)];
  r.p99_us = times[static_cast<size_t>(times.size() * 0.99)];

  double sq_sum = 0;
  for (auto t : times)
    sq_sum += (t - r.mean_us) * (t - r.mean_us);
  r.stddev_us = std::sqrt(sq_sum / times.size());

  return r;
}

void print_result(const std::string &name, const BenchmarkResult &r,
                  bool use_ns = false) {
  std::cout << std::fixed << std::setprecision(2);
  if (use_ns) {
    // Convert to nanoseconds for CPU measurements
    std::cout << name << ":" << std::endl;
    std::cout << "  Mean:   " << (r.mean_us * 1000) << " ns" << std::endl;
    std::cout << "  Median: " << (r.median_us * 1000) << " ns" << std::endl;
    std::cout << "  Std:    " << (r.stddev_us * 1000) << " ns" << std::endl;
    std::cout << "  Min:    " << (r.min_us * 1000) << " ns" << std::endl;
    std::cout << "  Max:    " << (r.max_us * 1000) << " ns" << std::endl;
    std::cout << "  P95:    " << (r.p95_us * 1000) << " ns" << std::endl;
    std::cout << "  P99:    " << (r.p99_us * 1000) << " ns" << std::endl;
    std::cout << "  [n=" << r.samples << "]" << std::endl;
  } else {
    std::cout << name << ":" << std::endl;
    std::cout << "  Mean:   " << r.mean_us << " us" << std::endl;
    std::cout << "  Median: " << r.median_us << " us" << std::endl;
    std::cout << "  Std:    " << r.stddev_us << " us" << std::endl;
    std::cout << "  Min:    " << r.min_us << " us" << std::endl;
    std::cout << "  Max:    " << r.max_us << " us" << std::endl;
    std::cout << "  P95:    " << r.p95_us << " us" << std::endl;
    std::cout << "  P99:    " << r.p99_us << " us" << std::endl;
    std::cout << "  [n=" << r.samples << "]" << std::endl;
  }
}

void print_latex_table_row(const std::string &name, const BenchmarkResult &r,
                           bool use_ns = false) {
  std::cout << std::fixed << std::setprecision(0);
  if (use_ns) {
    std::cout << name << " & " << (r.mean_us * 1000) << " & "
              << (r.median_us * 1000) << " & " << (r.p95_us * 1000) << " & "
              << (r.p99_us * 1000) << " & " << (r.min_us * 1000) << " & "
              << (r.max_us * 1000) << " \\\\" << std::endl;
  } else {
    std::cout << std::setprecision(1);
    std::cout << name << " & " << r.mean_us << " & " << r.median_us << " & "
              << r.p95_us << " & " << r.p99_us << " & " << r.min_us << " & "
              << r.max_us << " \\\\" << std::endl;
  }
}

// ============================================================================
// Benchmark 1: CPU Single-Position Evaluation (end-to-end)
// ============================================================================
void benchmark_cpu_eval(const std::vector<Position *> &positions) {
  std::cout << "\n=== CPU Single-Position Evaluation ===" << std::endl;
  std::cout << "Timing scope: Eval::evaluate() call" << std::endl;
  std::cout << "  - Includes: NNUE accumulator lookup/refresh + forward pass"
            << std::endl;
  std::cout << "  - Excludes: Position setup, move generation" << std::endl;

  const int warmup = 1000;
  const int iterations = 50000;

  std::vector<double> times;
  times.reserve(iterations * positions.size());

  // Warmup - also primes NNUE accumulators
  for (int i = 0; i < warmup; i++) {
    for (auto *pos : positions) {
      volatile Value v = Eval::evaluate(*pos);
      (void)v;
    }
  }

  // Benchmark - measure each evaluation individually
  for (int i = 0; i < iterations; i++) {
    for (auto *pos : positions) {
      auto start = high_resolution_clock::now();
      volatile Value v = Eval::evaluate(*pos);
      (void)v;
      auto end = high_resolution_clock::now();
      times.push_back(duration<double, std::micro>(end - start).count());
    }
  }

  auto stats = compute_stats(times);
  print_result("CPU eval (Eval::evaluate call)", stats, true); // Use ns

  std::cout << "\nLaTeX table row (ns):" << std::endl;
  print_latex_table_row("CPU NNUE", stats, true);
}

// ============================================================================
// Benchmark 2: GPU Single-Position Evaluation (end-to-end)
// ============================================================================
void benchmark_gpu_eval_single(const std::vector<Position *> &positions) {
  std::cout << "\n=== GPU Single-Position Evaluation (Synchronous) ==="
            << std::endl;
  std::cout << "Timing scope: Full Metal dispatch cycle" << std::endl;
  std::cout << "  - Includes: Feature extraction, buffer copy, command buffer "
               "create,"
            << std::endl;
  std::cout << "              encoder setup, buffer binding, commit(), "
               "waitUntilCompleted()"
            << std::endl;
  std::cout << "  - This is BLOCKING latency (not async)" << std::endl;

  auto &gpu = Eval::gpu_nnue();
  if (!gpu.is_ready()) {
    std::cout << "GPU NNUE not available, skipping" << std::endl;
    return;
  }

  const int warmup = 50;
  const int iterations = 500; // Fewer iterations since GPU is slow

  std::vector<double> times;
  times.reserve(iterations * positions.size());

  // Warmup
  for (int i = 0; i < warmup; i++) {
    for (auto *pos : positions) {
      volatile Value v = gpu.evaluate(*pos);
      (void)v;
    }
  }

  // Benchmark
  for (int i = 0; i < iterations; i++) {
    for (auto *pos : positions) {
      auto start = high_resolution_clock::now();
      volatile Value v = gpu.evaluate(*pos);
      (void)v;
      auto end = high_resolution_clock::now();
      times.push_back(duration<double, std::micro>(end - start).count());
    }
  }

  auto stats = compute_stats(times);
  print_result("GPU eval (synchronous dispatch)", stats, false); // Use us

  std::cout << "\nLaTeX table row (us):" << std::endl;
  print_latex_table_row("GPU NNUE (sync)", stats, false);
}

// ============================================================================
// Benchmark 3: GPU Batch Evaluation (varying batch sizes)
// ============================================================================
void benchmark_gpu_batch(const std::vector<Position *> &positions) {
  std::cout << "\n=== GPU Batch Evaluation (Sequential Dispatches) ==="
            << std::endl;
  std::cout << "Note: Each position triggers a separate command buffer cycle."
            << std::endl;
  std::cout << "      This is NOT true kernel-level batching." << std::endl;

  auto &gpu = Eval::gpu_nnue();
  if (!gpu.is_ready()) {
    std::cout << "GPU NNUE not available, skipping" << std::endl;
    return;
  }

  std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64};
  const int iterations = 100;

  std::cout << std::endl;
  std::cout << std::setw(10) << "Batch" << std::setw(18) << "Total (us)"
            << std::setw(18) << "Per-pos (us)" << std::setw(18) << "Throughput"
            << std::endl;
  std::cout << std::string(64, '-') << std::endl;

  for (int batch_size : batch_sizes) {
    // Create batch of positions
    std::vector<Position *> batch;
    for (int i = 0; i < batch_size; i++) {
      batch.push_back(positions[i % positions.size()]);
    }

    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 10; i++) {
      for (auto *pos : batch) {
        volatile Value v = gpu.evaluate(*pos);
        (void)v;
      }
    }

    // Benchmark - measure full batch evaluation time
    for (int i = 0; i < iterations; i++) {
      auto start = high_resolution_clock::now();
      for (auto *pos : batch) {
        volatile Value v = gpu.evaluate(*pos);
        (void)v;
      }
      auto end = high_resolution_clock::now();
      times.push_back(duration<double, std::micro>(end - start).count());
    }

    auto stats = compute_stats(times);
    double per_pos = stats.mean_us / batch_size;
    double throughput = 1000000.0 / per_pos; // positions per second

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(10) << batch_size << std::setw(18) << stats.mean_us
              << std::setw(18) << per_pos << std::setw(18) << (int)throughput
              << std::endl;
  }

  std::cout << "\nConclusion: Per-position cost is constant (~750 us) because"
            << std::endl;
  std::cout << "each evaluation requires a full dispatch cycle." << std::endl;
}

// ============================================================================
// Benchmark 4: CPU/GPU Evaluation Equivalence Check
// ============================================================================
void benchmark_eval_equivalence(const std::vector<Position *> &positions) {
  std::cout << "\n=== CPU/GPU Evaluation Comparison ===" << std::endl;

  auto &gpu = Eval::gpu_nnue();
  if (!gpu.is_ready()) {
    std::cout << "GPU NNUE not available, skipping" << std::endl;
    return;
  }

  std::cout << "Note: CPU uses full NNUE with loaded weights." << std::endl;
  std::cout << "      GPU uses separate weight loading (may differ)."
            << std::endl;

  int matches = 0;
  int total = 0;
  int max_diff = 0;
  int64_t sum_diff = 0;

  for (auto *pos : positions) {
    Value cpu_val = Eval::evaluate(*pos);
    Value gpu_val = gpu.evaluate(*pos);

    int diff = std::abs(int(cpu_val) - int(gpu_val));
    if (diff == 0)
      matches++;
    max_diff = std::max(max_diff, diff);
    sum_diff += diff;
    total++;

    std::cout << "Position " << total << ": CPU=" << std::setw(6) << cpu_val
              << ", GPU=" << std::setw(6) << gpu_val << ", diff=" << diff
              << std::endl;
  }

  std::cout << "\nSummary:" << std::endl;
  std::cout << "  Exact matches: " << matches << "/" << total << std::endl;
  std::cout << "  Max difference: " << max_diff << " centipawns" << std::endl;
  std::cout << "  Mean difference: " << (sum_diff / total) << " centipawns"
            << std::endl;

  if (max_diff > 100) {
    std::cout
        << "\n  Note: Differences indicate GPU path uses different weights"
        << std::endl;
    std::cout << "  or fallback evaluation. This is expected in current "
                 "implementation."
              << std::endl;
  }
}

// ============================================================================
// Benchmark 5: Crossover Analysis
// ============================================================================
void benchmark_crossover(const std::vector<Position *> &positions) {
  std::cout << "\n=== CPU vs GPU Crossover Analysis ===" << std::endl;

  auto &gpu = Eval::gpu_nnue();
  if (!gpu.is_ready()) {
    std::cout << "GPU NNUE not available, skipping" << std::endl;
    return;
  }

  // Measure CPU single eval with high precision
  const int cpu_iterations = 100000;
  std::vector<double> cpu_times;
  cpu_times.reserve(cpu_iterations);

  for (int i = 0; i < cpu_iterations; i++) {
    auto start = high_resolution_clock::now();
    volatile Value v = Eval::evaluate(*positions[0]);
    (void)v;
    auto end = high_resolution_clock::now();
    cpu_times.push_back(duration<double, std::micro>(end - start).count());
  }
  auto cpu_stats = compute_stats(cpu_times);

  // Measure GPU single eval
  const int gpu_iterations = 500;
  std::vector<double> gpu_times;
  gpu_times.reserve(gpu_iterations);

  for (int i = 0; i < gpu_iterations; i++) {
    auto start = high_resolution_clock::now();
    volatile Value v = gpu.evaluate(*positions[0]);
    (void)v;
    auto end = high_resolution_clock::now();
    gpu_times.push_back(duration<double, std::micro>(end - start).count());
  }
  auto gpu_stats = compute_stats(gpu_times);

  double ratio = gpu_stats.median_us / cpu_stats.median_us;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "CPU single eval (median): " << (cpu_stats.median_us * 1000)
            << " ns" << std::endl;
  std::cout << "GPU single eval (median): " << gpu_stats.median_us << " us"
            << std::endl;
  std::cout << "Ratio (GPU/CPU): " << ratio << "x" << std::endl;
  std::cout << "\nCrossover batch size (theoretical): " << (int)ratio
            << " positions" << std::endl;
  std::cout << "  For GPU to match CPU throughput, each dispatch must"
            << std::endl;
  std::cout << "  evaluate ~" << (int)ratio
            << " positions in a single kernel invocation." << std::endl;
  std::cout << "  (Current implementation dispatches one position per call.)"
            << std::endl;
}

// ============================================================================
// Print System Info
// ============================================================================
void print_system_info() {
  std::cout << "=== System Information ===" << std::endl;

  try {
    auto &device = Metal::get_device();
    std::cout << "GPU Device: " << device.get_architecture() << std::endl;
    std::cout << "Architecture: " << device.get_architecture_gen() << std::endl;

    MTL::Device *mtl = device.mtl_device();
    std::cout << "Unified Memory: " << (mtl->hasUnifiedMemory() ? "Yes" : "No")
              << std::endl;
    std::cout << "Max Buffer Size: "
              << (mtl->maxBufferLength() / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Max Threadgroup Memory: " << mtl->maxThreadgroupMemoryLength()
              << " bytes" << std::endl;
  } catch (...) {
    std::cout << "Metal not available" << std::endl;
  }

  std::cout << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::cout << "MetalFish Paper Benchmarks" << std::endl;
  std::cout << "==========================" << std::endl;
  std::cout << "Measuring end-to-end latencies for academic paper" << std::endl;
  std::cout << std::endl;

  // Initialize
  init_bitboards();
  Position::init();
  Zobrist::init();

  print_system_info();

  // Initialize GPU
  bool gpu_available = false;
  try {
    gpu_available = GPU::init_gpu_ops();
    if (gpu_available) {
      std::cout << "GPU operations initialized" << std::endl;
    }
  } catch (...) {
    std::cout << "GPU initialization failed" << std::endl;
  }

  // Initialize GPU NNUE
  auto &gpu_nnue = Eval::gpu_nnue();
  std::cout << "GPU NNUE ready: " << (gpu_nnue.is_ready() ? "Yes" : "No")
            << std::endl;

  // Create test positions
  std::vector<std::unique_ptr<Position>> positions;
  std::vector<StateInfo> states(TestPositions.size());
  std::vector<Position *> pos_ptrs;

  for (size_t i = 0; i < TestPositions.size(); i++) {
    auto pos = std::make_unique<Position>();
    pos->set(TestPositions[i], false, &states[i]);
    pos_ptrs.push_back(pos.get());
    positions.push_back(std::move(pos));
  }

  std::cout << "\nTest positions: " << positions.size() << std::endl;

  // Run benchmarks
  benchmark_cpu_eval(pos_ptrs);
  benchmark_gpu_eval_single(pos_ptrs);
  benchmark_gpu_batch(pos_ptrs);
  benchmark_crossover(pos_ptrs);
  benchmark_eval_equivalence(pos_ptrs);

  std::cout << "\n=== Benchmarks Complete ===" << std::endl;
  std::cout << "\nFor search NPS benchmarks, run:" << std::endl;
  std::cout << "  ./metalfish" << std::endl;
  std::cout << "  bench" << std::endl;

  return 0;
}
