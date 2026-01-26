/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file apple_silicon_mcts.cpp
 * @brief MetalFish source file.
 */

  Apple Silicon MCTS Optimizations - Implementation

  Licensed under GPL-3.0
*/

#include "apple_silicon_mcts.h"
#include "../core/position.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // Apple's SIMD/BLAS framework
#include <sys/sysctl.h>
#endif

namespace MetalFish {
namespace MCTS {

// ============================================================================
// Metal Shader Source for MCTS Operations
// ============================================================================

const char *MCTS_METAL_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// Score transformation kernel: NNUE centipawns to MCTS Q value
// Q = tanh(score / 300)
kernel void score_to_q(
    device const int32_t* scores [[buffer(0)]],
    device float* q_values [[buffer(1)]],
    device const uint8_t* side_to_move [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    float score = float(scores[tid]);
    score = clamp(score, -3000.0f, 3000.0f);
    
    float q = tanh(score / 300.0f);
    
    // Negate for black to move (NNUE returns from white's perspective)
    if (side_to_move[tid] == 1) {
        q = -q;
    }
    
    q_values[tid] = q;
}

// Policy softmax kernel with temperature
kernel void policy_softmax(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant float& temperature [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    device float* max_score [[buffer(4)]],  // Shared max for numerical stability
    device float* sum_exp [[buffer(5)]],    // Shared sum of exponentials
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tid >= count) return;
    
    // Phase 1: Find max score (parallel reduction would be better for large N)
    threadgroup_barrier(mem_flags::mem_device);
    
    float local_score = scores[tid] / temperature;
    
    // Compute exp(score - max) for numerical stability
    // For simplicity, we assume max_score is pre-computed
    float max_s = *max_score;
    float exp_val = exp(local_score - max_s);
    probs[tid] = exp_val;
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Normalize by sum
    float sum = *sum_exp;
    if (sum > 0.0f) {
        probs[tid] = exp_val / sum;
    }
}

// PUCT score computation kernel
// Score = Q + U + M
// U = cpuct * sqrt(parent_N) * P / (1 + child_N)
kernel void compute_puct_scores(
    device const float* child_q [[buffer(0)]],      // Child Q values (negated)
    device const float* child_n [[buffer(1)]],      // Child visit counts
    device const float* policy [[buffer(2)]],       // Policy priors
    device float* scores [[buffer(3)]],             // Output PUCT scores
    constant float& cpuct [[buffer(4)]],            // PUCT constant
    constant float& cpuct_sqrt_n [[buffer(5)]],     // cpuct * sqrt(parent_N)
    constant float& fpu [[buffer(6)]],              // First Play Urgency
    constant uint& count [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    float n = child_n[tid];
    float q = (n > 0.0f) ? child_q[tid] : fpu;
    float p = policy[tid];
    
    // U = cpuct * sqrt(parent_N) * P / (1 + N)
    float u = cpuct_sqrt_n * p / (1.0f + n);
    
    scores[tid] = q + u;
}

// Batch backpropagation kernel
// Updates W and computes new Q for multiple nodes
kernel void batch_backprop(
    device float* node_w [[buffer(0)]],       // Node W values (in-place update)
    device uint* node_n [[buffer(1)]],        // Node N values (in-place update)
    device float* node_q [[buffer(2)]],       // Node Q values (output)
    device const float* values [[buffer(3)]], // Values to backprop
    device const int* virtual_loss [[buffer(4)]], // Virtual loss to remove
    constant uint& count [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    
    // Remove virtual loss
    node_n[tid] -= virtual_loss[tid];
    
    // Update W and N
    float old_w = node_w[tid];
    float value = values[tid];
    
    node_w[tid] = old_w + value;
    uint new_n = node_n[tid] + 1;
    node_n[tid] = new_n;
    
    // Compute new Q
    node_q[tid] = (old_w + value) / float(new_n);
}
)";

// ============================================================================
// Utility Functions Implementation
// ============================================================================

bool is_apple_silicon() {
#ifdef __APPLE__
  char brand[256];
  size_t size = sizeof(brand);
  if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) {
    std::string brand_str(brand);
    return brand_str.find("Apple") != std::string::npos;
  }

  // Fallback: check for ARM64
#ifdef __aarch64__
  return true;
#endif
#endif
  return false;
}

int get_optimal_thread_count() {
#ifdef __APPLE__
  // On Apple Silicon, use performance cores only for MCTS
  // M1: 4 performance + 4 efficiency
  // M2: 4 performance + 4 efficiency (or 8+2 for Pro/Max)
  // M3: 4 performance + 4 efficiency (or more for Pro/Max)

  int perf_cores = 0;
  size_t size = sizeof(perf_cores);

  // Try to get performance core count
  if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &size, nullptr,
                   0) == 0) {
    return std::max(2, perf_cores);
  }

  // Fallback: use half of total cores (assume half are performance)
  int total_cores = static_cast<int>(std::thread::hardware_concurrency());
  return std::max(2, total_cores / 2);
#else
  return std::max(2, static_cast<int>(std::thread::hardware_concurrency()));
#endif
}

int get_optimal_gpu_batch_size() {
  if (!GPU::gpu_available()) {
    return 1;
  }

  // Use the backend's recommended batch size which accounts for:
  // - GPU core count
  // - Available GPU memory (recommendedMaxWorkingSetSize)
  // - SIMD group width (32 for Apple Silicon)
  return GPU::gpu().recommended_batch_size();
}

bool has_unified_memory() {
#ifdef __APPLE__
  return GPU::gpu_available() && GPU::gpu().has_unified_memory();
#else
  return false;
#endif
}

// ============================================================================
// GPUPositionData Implementation
// ============================================================================

void GPUPositionData::from_position(const Position &pos) {
  // Clear all bitboards
  std::memset(pieces, 0, sizeof(pieces));

  // Fill piece bitboards
  for (Color c : {WHITE, BLACK}) {
    for (PieceType pt = PAWN; pt <= KING; ++pt) {
      pieces[c][pt] = pos.pieces(c, pt);
    }
  }

  // King squares
  king_sq[WHITE] = static_cast<uint8_t>(pos.square<KING>(WHITE));
  king_sq[BLACK] = static_cast<uint8_t>(pos.square<KING>(BLACK));

  // Side to move
  stm = pos.side_to_move() == WHITE ? 0 : 1;

  // Piece count for bucket selection
  piece_count = static_cast<uint8_t>(popcount(pos.pieces()));

  // Clear padding
  std::memset(padding, 0, sizeof(padding));
}

// ============================================================================
// GPUResidentEvalBatch Implementation
// ============================================================================

bool GPUResidentEvalBatch::initialize(int batch_capacity) {
  if (!GPU::gpu_available()) {
    return false;
  }

  auto &backend = GPU::gpu();

  // Allocate position buffer in unified memory (Shared mode)
  // This allows zero-copy access from both CPU and GPU
  size_t positions_size = batch_capacity * sizeof(GPUPositionData);
  positions_buffer_ =
      backend.create_buffer(positions_size,
                            GPU::MemoryMode::Shared, // Unified memory!
                            GPU::BufferUsage::Streaming);

  if (!positions_buffer_) {
    return false;
  }

  // Allocate results buffer: psqt + positional scores per position
  size_t results_size = batch_capacity * 2 * sizeof(int32_t);
  results_buffer_ = backend.create_buffer(
      results_size,
      GPU::MemoryMode::Shared, // Results read by CPU after GPU writes
      GPU::BufferUsage::Streaming);

  if (!results_buffer_) {
    positions_buffer_.reset();
    return false;
  }

  capacity_ = batch_capacity;
  current_size_.store(0, std::memory_order_relaxed);
  initialized_ = true;

  return true;
}

int GPUResidentEvalBatch::add_position(const GPUPositionData &pos_data) {
  int index = current_size_.fetch_add(1, std::memory_order_acq_rel);

  if (index >= capacity_) {
    current_size_.fetch_sub(1, std::memory_order_relaxed);
    return -1;
  }

  // Direct write to unified memory - no copy needed!
  GPUPositionData *positions = positions_data();
  if (!positions) {
    current_size_.fetch_sub(1, std::memory_order_relaxed);
    return -1;
  }
  positions[index] = pos_data;

  return index;
}

void GPUResidentEvalBatch::clear() {
  current_size_.store(0, std::memory_order_release);
}

const int32_t *GPUResidentEvalBatch::psqt_scores() const {
  if (!results_buffer_)
    return nullptr;
  return results_buffer_->as<int32_t>();
}

const int32_t *GPUResidentEvalBatch::positional_scores() const {
  if (!results_buffer_)
    return nullptr;
  return results_buffer_->as<int32_t>() + capacity_;
}

GPUPositionData *GPUResidentEvalBatch::positions_data() {
  if (!positions_buffer_)
    return nullptr;
  return positions_buffer_->as<GPUPositionData>();
}

const GPUPositionData *GPUResidentEvalBatch::positions_data() const {
  if (!positions_buffer_)
    return nullptr;
  return positions_buffer_->as<GPUPositionData>();
}

// ============================================================================
// AppleSiliconMCTSConfig Implementation
// ============================================================================

void AppleSiliconMCTSConfig::auto_tune_for_apple_silicon() {
  if (!is_apple_silicon()) {
    return;
  }

  // Set Lc0 defaults
  lc0_params = Lc0SearchParams();

  // Apple Silicon specific tuning
  gpu_batch_size = get_optimal_gpu_batch_size();
  num_search_threads = get_optimal_thread_count();

  // Use unified memory features
  use_unified_memory = has_unified_memory();
  use_async_evaluation = true;

  // Tune virtual loss based on thread count
  // More threads = higher virtual loss to reduce contention
  virtual_loss = std::min(5, 2 + num_search_threads / 2);

  // Memory pool sizing based on available system memory
  // Apple Silicon Macs have unified memory, so we can use a portion for MCTS
  if (GPU::gpu_available()) {
    size_t total_mem = GPU::gpu().total_system_memory();
    size_t recommended_gpu_mem = GPU::gpu().recommended_working_set_size();

    // Use ~5% of total memory for node pool, capped at 256MB
    // Each node is ~64 bytes, so 256MB = ~4M nodes
    size_t node_pool_bytes =
        std::min(total_mem / 20, size_t(256) * 1024 * 1024);
    node_pool_size = node_pool_bytes / 64; // 64 bytes per node

    // TT size: ~2% of total memory, capped at 128MB
    // Each TT entry is ~8 bytes
    size_t tt_bytes = std::min(total_mem / 50, size_t(128) * 1024 * 1024);
    tt_size = tt_bytes / 8;

    // Log the auto-tuned configuration
    std::cerr << "[AppleSiliconMCTS] Auto-tuned configuration:" << std::endl;
    std::cerr << "  GPU batch size: " << gpu_batch_size << std::endl;
    std::cerr << "  Search threads: " << num_search_threads << std::endl;
    std::cerr << "  Node pool: " << (node_pool_size * 64 / (1024 * 1024))
              << " MB (" << node_pool_size << " nodes)" << std::endl;
    std::cerr << "  TT size: " << (tt_size * 8 / (1024 * 1024)) << " MB ("
              << tt_size << " entries)" << std::endl;
    std::cerr << "  Total system memory: " << (total_mem / (1024 * 1024 * 1024))
              << " GB" << std::endl;
    std::cerr << "  Recommended GPU working set: "
              << (recommended_gpu_mem / (1024 * 1024)) << " MB" << std::endl;
  } else {
    // Fallback defaults when GPU is not available
    node_pool_size = 1 << 20; // 1M nodes
    tt_size = 1 << 22;        // 4M TT entries
  }
}

// ============================================================================
// AppleSiliconMCTSEvaluator Implementation
// ============================================================================

AppleSiliconMCTSEvaluator::AppleSiliconMCTSEvaluator() = default;

AppleSiliconMCTSEvaluator::~AppleSiliconMCTSEvaluator() = default;

bool AppleSiliconMCTSEvaluator::initialize(
    GPU::GPUNNUEManager *gpu_manager, const AppleSiliconMCTSConfig &config) {
  if (!gpu_manager) {
    return false;
  }

  gpu_manager_ = gpu_manager;
  config_ = config;

  // Initialize single evaluation batch
  single_eval_batch_ = std::make_unique<GPUResidentEvalBatch>();
  if (!single_eval_batch_->initialize(1)) {
    return false;
  }

  // Compile MCTS-specific Metal kernels
  if (!compile_mcts_kernels()) {
    // Non-fatal: we can still use NNUE evaluation without custom kernels
  }

  return true;
}

bool AppleSiliconMCTSEvaluator::compile_mcts_kernels() {
  if (!GPU::gpu_available()) {
    return false;
  }

  auto &backend = GPU::gpu();

  // Compile the MCTS shader library
  if (!backend.compile_library("mcts_kernels", MCTS_METAL_SHADER_SOURCE)) {
    return false;
  }

  // Create kernel objects
  score_transform_kernel_ = backend.create_kernel("score_to_q", "mcts_kernels");

  return score_transform_kernel_ && score_transform_kernel_->valid();
}

float AppleSiliconMCTSEvaluator::evaluate_position(const Position &pos) {
  if (!gpu_manager_) {
    return 0.0f;
  }

  // Use GPU NNUE evaluation
  auto [psqt, score] = gpu_manager_->evaluate_single(pos, true);

  // Convert to Q value using Lc0-style transformation
  float q = NnueScoreToQ(score);

  // Adjust for side to move
  if (pos.side_to_move() == BLACK) {
    q = -q;
  }

  total_evals_.fetch_add(1, std::memory_order_relaxed);

  return q;
}

void AppleSiliconMCTSEvaluator::evaluate_batch_async(
    GPUResidentEvalBatch &batch, std::function<void()> completion_handler) {

  if (!gpu_manager_ || batch.size() == 0) {
    if (completion_handler) {
      completion_handler();
    }
    return;
  }

  // Create GPU evaluation batch from our unified memory batch
  GPU::GPUEvalBatch gpu_batch;
  gpu_batch.reserve(batch.size());

  const GPUPositionData *positions = batch.positions_data();
  for (int i = 0; i < batch.size(); ++i) {
    GPU::GPUPositionData pos_data;
    std::memcpy(&pos_data, &positions[i], sizeof(GPU::GPUPositionData));
    gpu_batch.add_position_data(pos_data);
  }

  // Evaluate using GPU NNUE (this uses Metal internally)
  gpu_manager_->evaluate_batch(gpu_batch, true);

  // Copy results to our unified memory buffer
  // On Apple Silicon, this is a unified memory copy (very fast)
  int32_t *psqt_out = const_cast<int32_t *>(batch.psqt_scores());
  int32_t *pos_out = const_cast<int32_t *>(batch.positional_scores());

  if (!psqt_out || !pos_out) {
    if (completion_handler) {
      completion_handler();
    }
    return;
  }

  for (int i = 0; i < batch.size(); ++i) {
    psqt_out[i] = gpu_batch.psqt_scores.size() > static_cast<size_t>(i)
                      ? gpu_batch.psqt_scores[i]
                      : 0;
    pos_out[i] = gpu_batch.positional_scores.size() > static_cast<size_t>(i)
                     ? gpu_batch.positional_scores[i]
                     : 0;
  }

  total_evals_.fetch_add(batch.size(), std::memory_order_relaxed);
  batch_count_.fetch_add(1, std::memory_order_relaxed);

  if (completion_handler) {
    completion_handler();
  }
}

void AppleSiliconMCTSEvaluator::evaluate_batch_sync(
    GPUResidentEvalBatch &batch) {
  evaluate_batch_async(batch, nullptr);
}

float AppleSiliconMCTSEvaluator::get_batch_result(
    const GPUResidentEvalBatch &batch, int index, Color side_to_move) const {

  if (index < 0 || index >= batch.size()) {
    return 0.0f;
  }

  const int32_t *psqt = batch.psqt_scores();
  const int32_t *pos = batch.positional_scores();

  if (!psqt || !pos) {
    return 0.0f;
  }

  int32_t score = psqt[index] + pos[index];
  float q = NnueScoreToQ(score);

  // Adjust for side to move
  if (side_to_move == BLACK) {
    q = -q;
  }

  return q;
}

double AppleSiliconMCTSEvaluator::avg_batch_size() const {
  uint64_t batches = batch_count_.load();
  if (batches == 0)
    return 0.0;
  return static_cast<double>(total_evals_.load()) / batches;
}

// ============================================================================
// AppleSiliconPolicySoftmax Implementation
// ============================================================================

void AppleSiliconPolicySoftmax::compute_softmax_simd(const float *scores,
                                                     int count,
                                                     float temperature,
                                                     float *probs_out) {

  if (count <= 0)
    return;

  // Handle temperature == 0 as argmax (deterministic selection)
  if (temperature == 0.0f) {
    float max_score = *std::max_element(scores, scores + count);
    // Count how many moves tie for max score
    int max_count = 0;
    for (int i = 0; i < count; ++i) {
      if (scores[i] == max_score)
        ++max_count;
    }
    float normalized_prob = 1.0f / max_count;
    for (int i = 0; i < count; ++i) {
      probs_out[i] = (scores[i] == max_score) ? normalized_prob : 0.0f;
    }
    return;
  }

#ifdef __APPLE__
  // Use Accelerate framework for SIMD operations

  // Step 1: Divide by temperature
  std::vector<float> scaled(count);
  if (temperature != 1.0f) {
    float inv_temp = 1.0f / temperature;
    vDSP_vsmul(scores, 1, &inv_temp, scaled.data(), 1, count);
  } else {
    std::copy(scores, scores + count, scaled.begin());
  }

  // Step 2: Find max for numerical stability
  float max_val;
  vDSP_maxv(scaled.data(), 1, &max_val, count);

  // Step 3: Subtract max and compute exp
  float neg_max = -max_val;
  vDSP_vsadd(scaled.data(), 1, &neg_max, probs_out, 1, count);

  // Step 4: Compute exp (using vvexpf for SIMD exp)
  int count_i = count;
  vvexpf(probs_out, probs_out, &count_i);

  // Step 5: Sum and normalize
  float sum;
  vDSP_sve(probs_out, 1, &sum, count);

  if (sum > 0.0f) {
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(probs_out, 1, &inv_sum, probs_out, 1, count);
  }

#else
  // Fallback implementation
  float max_score = *std::max_element(scores, scores + count);

  float sum = 0.0f;
  for (int i = 0; i < count; ++i) {
    probs_out[i] = std::exp((scores[i] - max_score) / temperature);
    sum += probs_out[i];
  }

  if (sum > 0.0f) {
    for (int i = 0; i < count; ++i) {
      probs_out[i] /= sum;
    }
  }
#endif
}

void AppleSiliconPolicySoftmax::compute_softmax_gpu(
    const std::vector<float> &scores, float temperature,
    std::vector<float> &probs_out) {

  // For small batches, SIMD is faster than GPU dispatch overhead
  if (scores.size() < 256 || !GPU::gpu_available()) {
    probs_out.resize(scores.size());
    compute_softmax_simd(scores.data(), static_cast<int>(scores.size()),
                         temperature, probs_out.data());
    return;
  }

  // For larger batches, use GPU
  // TODO: Implement GPU softmax using Metal kernel
  // For now, fall back to SIMD
  probs_out.resize(scores.size());
  compute_softmax_simd(scores.data(), static_cast<int>(scores.size()),
                       temperature, probs_out.data());
}

// ============================================================================
// AppleSiliconNodePool Implementation
// ============================================================================

AppleSiliconNodePool::AppleSiliconNodePool(size_t capacity)
    : capacity_(capacity), node_size_(64) { // 64 bytes per node

  // Allocate aligned memory for the pool
  // Use 128-byte alignment for Apple Silicon cache lines
  size_t total_size = capacity_ * node_size_;

#ifdef __APPLE__
  // Use posix_memalign for aligned allocation
  void *ptr = nullptr;
  if (posix_memalign(&ptr, APPLE_CACHE_LINE_SIZE, total_size) == 0) {
    memory_.reset(static_cast<char *>(ptr));
  }
#else
  memory_ = std::make_unique<char[]>(total_size);
#endif
}

AppleSiliconNodePool::~AppleSiliconNodePool() = default;

void *AppleSiliconNodePool::allocate() {
  // Check if memory allocation succeeded in constructor
  if (!memory_) {
    return nullptr;
  }

  size_t index = next_free_.fetch_add(1, std::memory_order_acq_rel);

  if (index >= capacity_) {
    next_free_.fetch_sub(1, std::memory_order_relaxed);
    return nullptr;
  }

  allocated_.fetch_add(1, std::memory_order_relaxed);
  return memory_.get() + index * node_size_;
}

void AppleSiliconNodePool::deallocate(void *ptr) {
  // Simple pool doesn't support individual deallocation
  // Nodes are freed when pool is reset
  (void)ptr;
}

void AppleSiliconNodePool::reset() {
#ifdef __APPLE__
  os_unfair_lock_lock(&lock_);
#else
  std::lock_guard<std::mutex> lock(lock_);
#endif

  next_free_.store(0, std::memory_order_release);
  allocated_.store(0, std::memory_order_release);

#ifdef __APPLE__
  os_unfair_lock_unlock(&lock_);
#endif
}

} // namespace MCTS
} // namespace MetalFish