/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Thread-Safe MCTS Implementation - Optimized for Apple Silicon

  This implementation incorporates state-of-the-art MCTS algorithms,
  including PUCT with logarithmic growth, FPU reduction strategy, and
  moves left head (MLH) utility.

  Apple Silicon Optimizations:
  - Unified memory zero-copy evaluation batches
  - SIMD-accelerated policy softmax
  - Cache-line aligned node structures
  - Lock-free atomic operations

  Licensed under GPL-3.0
*/

#include "thread_safe_mcts.h"
#include "apple_silicon_mcts.h"
#include "mcts_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __aarch64__
// ARM64 (Apple Silicon) - use yield instruction
#define CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
// x86 - use _mm_pause
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#endif

// Cross-platform prefetch macro
#if defined(_MSC_VER)
#include <intrin.h>
#define PREFETCH(addr)                                                         \
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#else
#define PREFETCH(addr) (void)(addr)
#endif

#include "../core/movegen.h"
#include "../eval/evaluate.h"
#include "../uci/uci.h"

namespace MetalFish {
namespace MCTS {

// ============================================================================
// BatchedGPUEvaluator - High-Performance Implementation
// ============================================================================

BatchedGPUEvaluator::BatchedGPUEvaluator(GPU::GPUNNUEManager *gpu_manager,
                                         ThreadSafeMCTSStats *stats,
                                         int min_batch_size, int max_batch_size,
                                         int batch_timeout_us)
    : gpu_manager_(gpu_manager), stats_(stats), min_batch_size_(min_batch_size),
      max_batch_size_(max_batch_size), batch_timeout_us_(batch_timeout_us) {
  tt_.resize(TT_SIZE);
  pending_requests_.reserve(max_batch_size);
}

BatchedGPUEvaluator::~BatchedGPUEvaluator() { stop(); }

void BatchedGPUEvaluator::start() {
  if (running_.load(std::memory_order_acquire))
    return;

  running_.store(true, std::memory_order_release);
  eval_thread_ = std::thread(&BatchedGPUEvaluator::eval_thread_main, this);
}

void BatchedGPUEvaluator::stop() {
  running_.store(false, std::memory_order_release);
  pending_cv_.notify_all();

  if (eval_thread_.joinable()) {
    eval_thread_.join();
  }
}

void BatchedGPUEvaluator::eval_thread_main() {
  std::vector<EvalRequest *> batch;
  std::vector<EvalRequest *> next_batch;
  batch.reserve(max_batch_size_);
  next_batch.reserve(max_batch_size_);

  // Adaptive timeout: start with configured value, adjust based on queue
  // pressure
  int adaptive_timeout_us = batch_timeout_us_;

  auto collect_batch = [&](std::vector<EvalRequest *> &target, int timeout_us) {
    target.clear();
    std::unique_lock<std::mutex> lock(pending_mutex_);

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(timeout_us);

    // Wait for minimum batch size OR timeout
    while (pending_requests_.size() < static_cast<size_t>(min_batch_size_) &&
           running_.load(std::memory_order_acquire)) {
      if (pending_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        break;
      }
    }

    // Collect batch - take all available up to max
    size_t count = std::min(pending_requests_.size(),
                            static_cast<size_t>(max_batch_size_));
    if (count > 0) {
      target.insert(target.end(), pending_requests_.begin(),
                    pending_requests_.begin() + count);
      pending_requests_.erase(pending_requests_.begin(),
                              pending_requests_.begin() + count);
    }

    // Adaptive timeout based on queue pressure
    size_t remaining = pending_requests_.size();
    if (remaining > 8) {
      adaptive_timeout_us = std::max(10, adaptive_timeout_us / 2);
    } else if (remaining == 0 && target.size() < 4) {
      adaptive_timeout_us =
          std::min(batch_timeout_us_ * 2, adaptive_timeout_us + 10);
    }
  };

  // Initial batch collection
  collect_batch(batch, adaptive_timeout_us);

  while (running_.load(std::memory_order_acquire)) {
    if (!batch.empty()) {
      // Start collecting next batch in parallel with processing
      std::thread prefetch_thread([&]() {
        if (running_.load(std::memory_order_acquire)) {
          collect_batch(next_batch, adaptive_timeout_us / 2);
        }
      });

      // Process current batch - use async if enabled and enough in-flight
      // capacity
      if (use_async_mode_ && inflight_batches_.load(std::memory_order_acquire) <
                                 MAX_INFLIGHT_BATCHES) {
        process_batch_async(batch);
      } else {
        process_batch(batch);
      }

      if (stats_) {
        stats_->nn_batches.fetch_add(1, std::memory_order_relaxed);
        stats_->total_batch_size.fetch_add(batch.size(),
                                           std::memory_order_relaxed);
        stats_->batch_count.fetch_add(1, std::memory_order_relaxed);
      }

      // Wait for prefetch to complete
      prefetch_thread.join();

      // Swap buffers
      std::swap(batch, next_batch);
    } else {
      // No batch to process, just collect
      collect_batch(batch, adaptive_timeout_us);
    }
  }

  // Wait for any in-flight async batches to complete
  while (inflight_batches_.load(std::memory_order_acquire) > 0) {
    std::this_thread::yield();
  }

  // Process any remaining requests
  {
    std::unique_lock<std::mutex> lock(pending_mutex_);
    if (!pending_requests_.empty()) {
      batch.clear();
      batch.insert(batch.end(), pending_requests_.begin(),
                   pending_requests_.end());
      pending_requests_.clear();
    }
  }
  if (!batch.empty()) {
    process_batch(batch);
  }
}

void BatchedGPUEvaluator::process_batch(std::vector<EvalRequest *> &batch) {
  if (!gpu_manager_ || batch.empty())
    return;

  const size_t batch_size = batch.size();

  // Deduplication: group requests by position key to avoid redundant GPU evals
  // Use unordered_map for O(1) lookup (faster than sorting for typical batch
  // sizes)
  std::unordered_map<uint64_t, std::vector<size_t>> key_to_indices;
  key_to_indices.reserve(batch_size);
  std::vector<size_t> unique_indices;
  unique_indices.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    uint64_t key = batch[i]->position_key;
    auto it = key_to_indices.find(key);
    if (it == key_to_indices.end()) {
      key_to_indices[key] = {i};
      unique_indices.push_back(i);
    } else {
      it->second.push_back(i);
    }
  }

  const size_t unique_count = unique_indices.size();

  // Only send unique positions to GPU
  GPU::GPUEvalBatch gpu_batch;
  gpu_batch.reserve(static_cast<int>(unique_count));

  for (size_t idx : unique_indices) {
    gpu_batch.add_position_data(batch[idx]->pos_data);
  }

  gpu_manager_->evaluate_batch(gpu_batch, true);

  // Distribute results to all requests (including duplicates)
  for (size_t i = 0; i < unique_count; ++i) {
    size_t orig_idx = unique_indices[i];
    EvalRequest *req = batch[orig_idx];

    int32_t psqt =
        gpu_batch.psqt_scores.size() > i ? gpu_batch.psqt_scores[i] : 0;
    int32_t pos_score = gpu_batch.positional_scores.size() > i
                            ? gpu_batch.positional_scores[i]
                            : 0;
    int32_t raw_score = psqt + pos_score;

    // Fast tanh using standard library (well-optimized on modern CPUs)
    float x = static_cast<float>(raw_score) / 400.0f;
    float value = std::tanh(x);

    if (req->side_to_move == BLACK) {
      value = -value;
    }

    // Store in TT with age
    uint32_t age = current_age_.load(std::memory_order_relaxed);
    size_t tt_idx = req->position_key % TT_SIZE;
    tt_[tt_idx].value = value;
    tt_[tt_idx].key = req->position_key;
    tt_[tt_idx].age = age;

    // Complete all requests with same key
    for (size_t dup_idx : key_to_indices[req->position_key]) {
      batch[dup_idx]->result = value;
      batch[dup_idx]->completed.store(true, std::memory_order_release);
    }
  }

  if (stats_) {
    stats_->nn_evaluations.fetch_add(unique_count, std::memory_order_relaxed);
  }
}

void BatchedGPUEvaluator::evaluate_async(const Position &pos, uint64_t key,
                                         Color stm, EvalCallback callback) {
  // Check TT first
  size_t tt_idx = key % TT_SIZE;
  if (tt_[tt_idx].key == key) {
    // Cache hit - invoke callback immediately
    callback(tt_[tt_idx].value);
    return;
  }

  // Add to async queue
  pending_async_count_.fetch_add(1, std::memory_order_relaxed);

  AsyncEvalRequest async_req;
  async_req.pos_data.from_position(pos);
  async_req.position_key = key;
  async_req.side_to_move = stm;
  async_req.callback = std::move(callback);

  {
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_requests_.push_back(std::move(async_req));
  }
  pending_cv_.notify_one();
}

void BatchedGPUEvaluator::process_batch_async(
    std::vector<EvalRequest *> &batch) {
  if (!gpu_manager_ || batch.empty())
    return;

  // Wait if too many batches in flight
  while (inflight_batches_.load(std::memory_order_acquire) >=
         MAX_INFLIGHT_BATCHES) {
    std::this_thread::yield();
  }

  const size_t batch_size = batch.size();

  // Deduplication
  std::unordered_map<uint64_t, std::vector<size_t>> key_to_indices;
  key_to_indices.reserve(batch_size);
  std::vector<size_t> unique_indices;
  unique_indices.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    uint64_t key = batch[i]->position_key;
    auto it = key_to_indices.find(key);
    if (it == key_to_indices.end()) {
      key_to_indices[key] = {i};
      unique_indices.push_back(i);
    } else {
      it->second.push_back(i);
    }
  }

  const size_t unique_count = unique_indices.size();

  // Build GPU batch
  auto gpu_batch = std::make_shared<GPU::GPUEvalBatch>();
  gpu_batch->reserve(static_cast<int>(unique_count));

  for (size_t idx : unique_indices) {
    gpu_batch->add_position_data(batch[idx]->pos_data);
  }

  // Copy data needed for completion handler
  auto batch_copy = std::make_shared<std::vector<EvalRequest *>>(batch);
  auto unique_indices_copy =
      std::make_shared<std::vector<size_t>>(std::move(unique_indices));
  auto key_map_copy =
      std::make_shared<std::unordered_map<uint64_t, std::vector<size_t>>>(
          std::move(key_to_indices));

  inflight_batches_.fetch_add(1, std::memory_order_release);

  // Submit async with completion handler
  gpu_manager_->evaluate_batch_async(*gpu_batch, [this, gpu_batch, batch_copy,
                                                  unique_indices_copy,
                                                  key_map_copy,
                                                  unique_count](bool success) {
    // Completion handler - runs on GPU completion thread
    if (success) {
      for (size_t i = 0; i < unique_count; ++i) {
        size_t orig_idx = (*unique_indices_copy)[i];
        EvalRequest *req = (*batch_copy)[orig_idx];

        int32_t psqt =
            gpu_batch->psqt_scores.size() > i ? gpu_batch->psqt_scores[i] : 0;
        int32_t pos_score = gpu_batch->positional_scores.size() > i
                                ? gpu_batch->positional_scores[i]
                                : 0;
        int32_t raw_score = psqt + pos_score;

        float x = static_cast<float>(raw_score) / 400.0f;
        float value = std::tanh(x);

        if (req->side_to_move == BLACK) {
          value = -value;
        }

        // Store in TT
        uint32_t age = current_age_.load(std::memory_order_relaxed);
        size_t tt_idx = req->position_key % TT_SIZE;
        tt_[tt_idx].value = value;
        tt_[tt_idx].key = req->position_key;
        tt_[tt_idx].age = age;

        // Complete all requests with same key
        for (size_t dup_idx : (*key_map_copy)[req->position_key]) {
          (*batch_copy)[dup_idx]->result = value;
          (*batch_copy)[dup_idx]->completed.store(true,
                                                  std::memory_order_release);
        }
      }
    } else {
      // On failure, complete with default value
      for (auto *req : *batch_copy) {
        req->result = 0.0f;
        req->completed.store(true, std::memory_order_release);
      }
    }

    inflight_batches_.fetch_sub(1, std::memory_order_release);

    if (stats_) {
      stats_->nn_evaluations.fetch_add(unique_count, std::memory_order_relaxed);
    }
  });

  if (stats_) {
    stats_->nn_batches.fetch_add(1, std::memory_order_relaxed);
    stats_->total_batch_size.fetch_add(batch_size, std::memory_order_relaxed);
    stats_->batch_count.fetch_add(1, std::memory_order_relaxed);
  }
}

float BatchedGPUEvaluator::evaluate(const Position &pos, WorkerContext &ctx) {
  uint64_t key = pos.key();
  size_t tt_idx = key % TT_SIZE;

  // Prefetch TT entry for cache efficiency
  PREFETCH(&tt_[tt_idx]);

  if (tt_[tt_idx].key == key) {
    ctx.cache_hits++;
    return tt_[tt_idx].value;
  }

  ctx.cache_misses++;

  auto wait_start = std::chrono::steady_clock::now();

  // Acquire request from pool
  EvalRequest *req = request_pool_.acquire();
  req->pos_data.from_position(pos);
  req->position_key = key;
  req->side_to_move = pos.side_to_move();
  req->ready.store(true, std::memory_order_release);

  // Submit request
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_requests_.push_back(req);
  }
  pending_cv_.notify_one();

  // Exponential backoff spin-wait (more efficient than constant yield)
  int spin_count = 0;
  int backoff = 1;
  while (!req->completed.load(std::memory_order_acquire)) {
    if (++spin_count > backoff) {
      if (backoff < 1024) {
        // Exponential backoff
        backoff *= 2;
        CPU_PAUSE(); // CPU hint for spin-wait
      } else {
        // After enough spins, yield
        std::this_thread::yield();
      }
      spin_count = 0;
    }
  }

  float result = req->result;
  request_pool_.release(req);

  auto wait_end = std::chrono::steady_clock::now();
  if (stats_) {
    stats_->batch_wait_time_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(wait_end -
                                                              wait_start)
            .count(),
        std::memory_order_relaxed);
  }

  return result;
}

// ============================================================================
// ThreadSafeNode Implementation
// ============================================================================

ThreadSafeNode::ThreadSafeNode(ThreadSafeNode *parent, int edge_idx)
    : parent_(parent), edge_index_(edge_idx) {}

void ThreadSafeNode::create_edges(const MoveList<LEGAL> &moves) {
  if (moves.size() == 0)
    return;

  int count = static_cast<int>(moves.size());
  edges_ = std::make_unique<TSEdge[]>(count);

  // Initialize with uniform policy
  float uniform = 1.0f / count;
  int idx = 0;
  for (const auto &m : moves) {
    edges_[idx].move = m;
    edges_[idx].SetPolicy(uniform);
    edges_[idx].child.store(nullptr, std::memory_order_relaxed);
    idx++;
  }

  // Publish edges (release ensures edges are visible before count)
  num_edges_.store(count, std::memory_order_release);
}

// MCTS FinalizeScoreUpdate implementation
// Updates statistics using running average: Q = (Q * N + V * multivisit) / (N +
// multivisit) This is the core algorithm from Lc0 node.cc
void ThreadSafeNode::FinalizeScoreUpdate(float v, float d_val, float m_val,
                                         int multivisit) {
  // Get current N before update
  uint32_t old_n = n_.load(std::memory_order_acquire);
  uint32_t new_n = old_n + multivisit;

  if (new_n == 0)
    return; // Safety check

  // Lc0's running average formula:
  // Q_new = Q_old + multivisit * (V - Q_old) / N_new
  // This is algebraically equivalent to: (Q_old * N_old + V * multivisit) /
  // N_new

  float mult = static_cast<float>(multivisit);
  float inv_new_n = 1.0f / static_cast<float>(new_n);

  // Update WL (Win-Loss)
  float old_wl = wl_.load(std::memory_order_relaxed);
  float new_wl = old_wl + mult * (v - old_wl) * inv_new_n;
  wl_.store(new_wl, std::memory_order_relaxed);

  // Update D (Draw probability)
  float old_d = d_.load(std::memory_order_relaxed);
  float new_d = old_d + mult * (d_val - old_d) * inv_new_n;
  d_.store(new_d, std::memory_order_relaxed);

  // Update M (Moves left)
  float old_m = m_.load(std::memory_order_relaxed);
  float new_m = old_m + mult * (m_val - old_m) * inv_new_n;
  m_.store(new_m, std::memory_order_relaxed);

  // Update W (total value sum) for debugging/verification
  float old_w = w_.load(std::memory_order_relaxed);
  w_.store(old_w + v * mult, std::memory_order_relaxed);

  // Increment N (this is the synchronization point)
  n_.store(new_n, std::memory_order_release);

  // Decrement virtual loss
  n_in_flight_.fetch_sub(multivisit, std::memory_order_acq_rel);
}

// Calculate visited policy (sum of policy for children with N > 0)
float ThreadSafeNode::GetVisitedPolicy() const {
  float sum = 0.0f;
  int num = num_edges_.load(std::memory_order_acquire);
  const TSEdge *e = edges_.get();

  for (int i = 0; i < num; ++i) {
    ThreadSafeNode *child = e[i].child.load(std::memory_order_acquire);
    if (child && child->GetN() > 0) {
      sum += e[i].GetPolicy();
    }
  }
  return sum;
}

// Get the edge from parent that points to this node
TSEdge *ThreadSafeNode::GetOwnEdge() const {
  if (!parent_ || edge_index_ < 0)
    return nullptr;
  return &parent_->edges()[edge_index_];
}

// Make node terminal with specific values
void ThreadSafeNode::MakeTerminal(Terminal type, float wl, float d_val,
                                  float m_val) {
  terminal_type_.store(type, std::memory_order_release);
  wl_.store(wl, std::memory_order_relaxed);
  d_.store(d_val, std::memory_order_relaxed);
  m_.store(m_val, std::memory_order_relaxed);

  // Lc0 behavior: terminal losses have no uncertainty
  // Clear policy to prevent U value from being comparable to non-loss choices
  if (wl < -0.99f && parent_ && edge_index_ >= 0) {
    parent_->edges()[edge_index_].SetPolicy(0.0f);
  }
}

// MCTS solid tree optimization
// Converts sparse child nodes to a contiguous array for better cache locality
// This improves performance for frequently visited subtrees
bool ThreadSafeNode::MakeSolid() {
  // Check if already solid or no children
  if (is_solid_.load(std::memory_order_acquire) ||
      num_edges_.load(std::memory_order_acquire) == 0 || IsTerminal()) {
    return false;
  }

  // Lock to prevent concurrent solidification
  std::lock_guard<std::mutex> lock(mutex_);

  // Double-check after acquiring lock
  if (is_solid_.load(std::memory_order_acquire)) {
    return false;
  }

  int num = num_edges_.load(std::memory_order_acquire);
  if (num == 0)
    return false;

  // Check if any children have in-flight visits
  // Can't solidify if children might be modified
  TSEdge *e = edges_.get();
  uint32_t total_in_flight = 0;

  for (int i = 0; i < num; ++i) {
    ThreadSafeNode *child = e[i].child.load(std::memory_order_acquire);
    if (child) {
      // Can't solidify if child has only 1 visit and is in-flight
      if (child->GetN() <= 1 && child->GetNInFlight() > 0) {
        return false;
      }
      if (child->IsTerminal() && child->GetNInFlight() > 0) {
        return false;
      }
      total_in_flight += child->GetNInFlight();
    }
  }

  // If total in-flight doesn't match our in-flight, there are collisions
  if (total_in_flight != GetNInFlight()) {
    return false;
  }

  // Mark as solid - this is a one-way transition
  // The edges array is already contiguous, we just need to mark it
  // In a full Lc0 implementation, this would reallocate children to a
  // contiguous block For our simpler implementation, we just mark it to enable
  // optimized iteration
  is_solid_.store(true, std::memory_order_release);

  return true;
}

void ThreadSafeNode::set_terminal(Terminal type, float value) {
  // Map legacy Terminal types to Lc0-style
  float wl_val = 0.0f;
  float d_val = 0.0f;

  // Determine WL and D based on terminal type
  // Note: Legacy types Win/Draw/Loss need to be mapped
  switch (type) {
  case Terminal::EndOfGame:
    // EndOfGame with value determines win/draw/loss
    if (value > 0.5f) {
      wl_val = 1.0f; // Win
      d_val = 0.0f;
    } else if (value < -0.5f) {
      wl_val = -1.0f; // Loss
      d_val = 0.0f;
    } else {
      wl_val = 0.0f; // Draw
      d_val = 1.0f;
    }
    break;
  case Terminal::Tablebase:
    wl_val = value;
    d_val = (std::abs(value) < 0.01f) ? 1.0f : 0.0f;
    break;
  case Terminal::TwoFold:
    wl_val = 0.0f;
    d_val = 1.0f;
    break;
  default:
    wl_val = value;
    d_val = 0.0f;
    break;
  }

  MakeTerminal(type, wl_val, d_val, 0.0f);
  n_.store(1, std::memory_order_release);
}

// ============================================================================
// ThreadSafeTree Implementation with Arena Allocation
// ============================================================================

ThreadSafeTree::ThreadSafeTree() {
  root_ = std::make_unique<ThreadSafeNode>();
  // Pre-allocate first arena
  arenas_.push_back(std::make_unique<NodeArena>());
}

ThreadSafeTree::~ThreadSafeTree() = default;

void ThreadSafeTree::reset(const std::string &fen) {
  {
    std::unique_lock<std::shared_mutex> lock(fen_mutex_);
    root_fen_ = fen;
  }

  // Reset arenas
  {
    std::lock_guard<std::mutex> lock(arena_mutex_);
    arenas_.clear();
    arenas_.push_back(std::make_unique<NodeArena>());
    current_arena_.store(0, std::memory_order_relaxed);
  }

  root_ = std::make_unique<ThreadSafeNode>();
  node_count_.store(1, std::memory_order_relaxed);
}

ThreadSafeNode *ThreadSafeTree::allocate_node(ThreadSafeNode *parent,
                                              int edge_idx) {
  // Try current arena first (lock-free fast path)
  size_t arena_idx = current_arena_.load(std::memory_order_acquire);

  if (arena_idx < arenas_.size()) {
    NodeArena *arena = arenas_[arena_idx].get();
    size_t slot = arena->next.fetch_add(1, std::memory_order_relaxed);

    if (slot < ARENA_SIZE) {
      ThreadSafeNode *node = &arena->nodes[slot];
      new (node) ThreadSafeNode(parent, edge_idx);
      node_count_.fetch_add(1, std::memory_order_relaxed);
      return node;
    }
  }

  // Need new arena (slow path with lock)
  std::lock_guard<std::mutex> lock(arena_mutex_);

  // Double-check after acquiring lock
  arena_idx = current_arena_.load(std::memory_order_acquire);
  if (arena_idx < arenas_.size()) {
    NodeArena *arena = arenas_[arena_idx].get();
    if (arena->next.load(std::memory_order_relaxed) < ARENA_SIZE) {
      size_t slot = arena->next.fetch_add(1, std::memory_order_relaxed);
      if (slot < ARENA_SIZE) {
        ThreadSafeNode *node = &arena->nodes[slot];
        new (node) ThreadSafeNode(parent, edge_idx);
        node_count_.fetch_add(1, std::memory_order_relaxed);
        return node;
      }
    }
  }

  // Create new arena
  arenas_.push_back(std::make_unique<NodeArena>());
  current_arena_.store(arenas_.size() - 1, std::memory_order_release);

  NodeArena *arena = arenas_.back().get();
  size_t slot = arena->next.fetch_add(1, std::memory_order_relaxed);
  ThreadSafeNode *node = &arena->nodes[slot];
  new (node) ThreadSafeNode(parent, edge_idx);
  node_count_.fetch_add(1, std::memory_order_relaxed);
  return node;
}

// ============================================================================
// ThreadSafeMCTS Implementation
// ============================================================================

ThreadSafeMCTS::ThreadSafeMCTS(const ThreadSafeMCTSConfig &config)
    : config_(config), tree_(std::make_unique<ThreadSafeTree>()) {
  // Initialize simple TT for direct evaluation mode
  simple_tt_.resize(SIMPLE_TT_SIZE);
}

ThreadSafeMCTS::~ThreadSafeMCTS() {
  stop();
  wait();

  // Stop batched evaluator if running
  if (batched_evaluator_) {
    batched_evaluator_->stop();
  }
}

void ThreadSafeMCTS::start_search(const std::string &fen,
                                  const Search::LimitsType &limits,
                                  BestMoveCallback best_move_cb,
                                  InfoCallback info_cb) {
  // Stop any existing search
  stop();
  wait();

  // Reset state
  stats_.reset();
  stop_flag_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_release);
  limits_ = limits;
  best_move_callback_ = best_move_cb;
  info_callback_ = info_cb;
  search_start_ = std::chrono::steady_clock::now();

  // Calculate time budget
  time_budget_ms_ = calculate_time_budget();

  // Initialize tree
  tree_->reset(fen);

  // Get actual thread count and auto-tune
  int actual_threads = config_.get_num_threads();
  config_.auto_tune(actual_threads);

  // Initialize batched evaluator if enabled
  if (config_.use_batched_eval && gpu_manager_) {
    batched_evaluator_ = std::make_unique<BatchedGPUEvaluator>(
        gpu_manager_, &stats_, config_.min_batch_size, config_.max_batch_size,
        config_.batch_timeout_us);
    // Note: Async mode disabled by default - synchronous batching is more
    // efficient for MCTS because workers need to wait for evaluation results
    // anyway. Multiple command queues are still available for future async
    // workloads.
    batched_evaluator_->set_async_mode(false);
    batched_evaluator_->start();
  }

  // Create worker contexts
  worker_contexts_.clear();
  for (int i = 0; i < actual_threads; ++i) {
    worker_contexts_.push_back(std::make_unique<WorkerContext>());
  }

  // Start worker threads
  workers_.clear();
  for (int i = 0; i < actual_threads; ++i) {
    workers_.emplace_back(&ThreadSafeMCTS::worker_thread, this, i);
  }
}

void ThreadSafeMCTS::stop() {
  stop_flag_.store(true, std::memory_order_release);
}

void ThreadSafeMCTS::wait() {
  for (auto &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();

  // Stop batched evaluator
  if (batched_evaluator_) {
    batched_evaluator_->stop();
    batched_evaluator_.reset();
  }

  running_.store(false, std::memory_order_release);

  // Report best move only if callback is valid
  // Make a copy to prevent race conditions
  auto callback_copy = best_move_callback_;
  best_move_callback_ = nullptr; // Clear to prevent double-call

  if (callback_copy) {
    Move best = get_best_move();
    std::vector<Move> pv = get_pv();
    Move ponder = pv.size() > 1 ? pv[1] : Move::none();
    callback_copy(best, ponder);
  }
}

bool ThreadSafeMCTS::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire))
    return true;

  // Check time limit
  if (time_budget_ms_ > 0) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - search_start_)
                       .count();
    if (elapsed >= time_budget_ms_)
      return true;
  }

  // Check node limit
  if (limits_.nodes > 0 && stats_.total_nodes >= limits_.nodes)
    return true;

  return false;
}

int64_t ThreadSafeMCTS::calculate_time_budget() const {
  if (limits_.movetime > 0)
    return limits_.movetime;
  if (limits_.infinite)
    return 0;

  // Get time for our side
  Color us = WHITE; // Will be determined from position
  int64_t time_left = limits_.time[us];
  int64_t inc = limits_.inc[us];

  if (time_left <= 0)
    return 1000; // Default 1 second

  // IMPROVED: Use more aggressive time allocation for MCTS
  // MCTS needs more time than alpha-beta to build a good tree
  // Use ~8% of remaining time + most of increment for better tree building
  int64_t base_time = time_left / 12; // ~8% of remaining time
  int64_t inc_time = inc * 4 / 5;     // Use 80% of increment

  // Minimum 1000ms for MCTS to be effective (was 500ms)
  int64_t budget = std::max(int64_t(1000), base_time + inc_time);

  // Cap at 25% of remaining time to avoid time trouble (was 20%)
  int64_t max_time = time_left / 4;

  return std::min(budget, max_time);
}

void ThreadSafeMCTS::worker_thread(int thread_id) {
  WorkerContext &ctx = *worker_contexts_[thread_id];

  // Cache root FEN once per search (avoid repeated string copies)
  ctx.set_root_fen(tree_->root_fen());

  // Expand root node if needed (only one thread should do this)
  ThreadSafeNode *root = tree_->root();
  if (!root->has_children()) {
    std::lock_guard<std::mutex> lock(root->mutex());
    if (!root->has_children()) {
      MoveList<LEGAL> moves(ctx.pos);
      root->create_edges(moves);

      // Add Dirichlet noise at root
      if (config_.add_dirichlet_noise) {
        add_dirichlet_noise(root);
      }

      // Set heuristic policy priors
      expand_node(root, ctx);
    }
  }

  // Main search loop with batched stop checks
  constexpr int STOP_CHECK_INTERVAL = 64;
  int iterations_since_check = 0;

  while (true) {
    // Batch stop checks to reduce overhead
    if (++iterations_since_check >= STOP_CHECK_INTERVAL) {
      iterations_since_check = 0;
      if (should_stop())
        break;
    }

    run_iteration(ctx);
  }

  // Flush accumulated profiling stats
  stats_.selection_time_us.fetch_add(ctx.selection_time_acc,
                                     std::memory_order_relaxed);
  stats_.expansion_time_us.fetch_add(ctx.expansion_time_acc,
                                     std::memory_order_relaxed);
  stats_.evaluation_time_us.fetch_add(ctx.evaluation_time_acc,
                                      std::memory_order_relaxed);
  stats_.backprop_time_us.fetch_add(ctx.backprop_time_acc,
                                    std::memory_order_relaxed);

  // Aggregate worker stats
  stats_.cache_hits.fetch_add(ctx.cache_hits, std::memory_order_relaxed);
  stats_.cache_misses.fetch_add(ctx.cache_misses, std::memory_order_relaxed);
}

void ThreadSafeMCTS::run_iteration(WorkerContext &ctx) {
  // Use cached root FEN instead of fetching every time
  ctx.reset_to_cached_root();

  // Profile only every N iterations to reduce chrono overhead
  bool do_profile = (ctx.iterations % WorkerContext::PROFILE_SAMPLE_RATE) == 0;
  auto iter_start = do_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  // 1. Selection - traverse to leaf
  auto select_start = iter_start;
  ThreadSafeNode *leaf = select_leaf(ctx);
  auto select_end = do_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  if (!leaf)
    return;

  // 2. Check for terminal - generate moves once and reuse
  MoveList<LEGAL> moves(ctx.pos);
  bool is_in_check = ctx.pos.checkers() != 0;

  if (moves.size() == 0) {
    if (is_in_check) {
      // Checkmate - terminal loss (from perspective of side to move)
      leaf->set_terminal(ThreadSafeNode::Terminal::EndOfGame, -1.0f);
      backpropagate(leaf, -1.0f, 0.0f, 0.0f);
    } else {
      // Stalemate - terminal draw
      leaf->set_terminal(ThreadSafeNode::Terminal::EndOfGame, 0.0f);
      backpropagate(leaf, 0.0f, 1.0f, 0.0f);
    }
    return;
  }

  // 3. Expansion - add children if not expanded (reuse moves list)
  auto expand_start = do_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  if (!leaf->has_children()) {
    std::lock_guard<std::mutex> lock(leaf->mutex());
    if (!leaf->has_children()) {
      leaf->create_edges(moves);
      expand_node(leaf, ctx);
    }
  }
  auto expand_end = do_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

  // 4. Evaluation
  auto eval_start = do_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  float value = evaluate_position(ctx);
  auto eval_end = do_profile ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};

  // 5. Backpropagation
  auto backprop_start = do_profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
  float draw = std::max(0.0f, 0.4f - std::abs(value) * 0.3f);
  backpropagate(leaf, value, draw, 30.0f);
  auto backprop_end = do_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};

  // Update profiling stats (sampled)
  if (do_profile) {
    ctx.selection_time_acc +=
        std::chrono::duration_cast<std::chrono::microseconds>(select_end -
                                                              select_start)
            .count() *
        WorkerContext::PROFILE_SAMPLE_RATE;
    ctx.expansion_time_acc +=
        std::chrono::duration_cast<std::chrono::microseconds>(expand_end -
                                                              expand_start)
            .count() *
        WorkerContext::PROFILE_SAMPLE_RATE;
    ctx.evaluation_time_acc +=
        std::chrono::duration_cast<std::chrono::microseconds>(eval_end -
                                                              eval_start)
            .count() *
        WorkerContext::PROFILE_SAMPLE_RATE;
    ctx.backprop_time_acc +=
        std::chrono::duration_cast<std::chrono::microseconds>(backprop_end -
                                                              backprop_start)
            .count() *
        WorkerContext::PROFILE_SAMPLE_RATE;
  }

  stats_.total_nodes.fetch_add(1, std::memory_order_relaxed);
  stats_.total_iterations.fetch_add(1, std::memory_order_relaxed);
  ctx.iterations++;
}

ThreadSafeNode *ThreadSafeMCTS::select_leaf(WorkerContext &ctx) {
  ThreadSafeNode *node = tree_->root();

  while (node->has_children() && !node->is_terminal()) {
    // Select best child using PUCT
    int best_idx = select_child_puct(node, config_.cpuct, ctx);
    if (best_idx < 0)
      break;

    TSEdge &edge = node->edges()[best_idx];

    // Get or create child node - use lock-free CAS when possible
    ThreadSafeNode *child = edge.child.load(std::memory_order_acquire);

    if (!child) {
      // Try to create child using CAS (lock-free fast path)
      ThreadSafeNode *new_child = tree_->allocate_node(node, best_idx);
      ThreadSafeNode *expected = nullptr;

      if (edge.child.compare_exchange_strong(expected, new_child,
                                             std::memory_order_release,
                                             std::memory_order_acquire)) {
        // We won the race, use our new child
        child = new_child;
      } else {
        // Another thread created it first, use theirs and recycle ours
        child = expected;
        // Note: new_child is "lost" but arena allocation makes this cheap
      }
    }

    // Add virtual loss
    child->add_virtual_loss(config_.virtual_loss);

    // Make move on position
    ctx.do_move(edge.move);

    node = child;
  }

  return node;
}

int ThreadSafeMCTS::select_child_puct(ThreadSafeNode *node, float cpuct,
                                      WorkerContext &ctx) {
  int num_edges = node->num_edges();
  if (num_edges == 0)
    return -1;

  // Get parent statistics using MCTS accessors
  uint32_t parent_n = node->GetN() + node->GetNInFlight();
  float parent_q = node->GetN() > 0 ? node->GetQ(0.0f) : 0.0f;
  float draw_score = 0.0f; // Can be configured for contempt

  // MCTS PUCT with logarithmic growth
  // Formula: cpuct_init + cpuct_factor * log((N + cpuct_base) / cpuct_base)
  const float cpuct_base = config_.cpuct_base;
  const float cpuct_factor = config_.cpuct_factor;
  float effective_cpuct =
      cpuct +
      cpuct_factor *
          std::log((static_cast<float>(parent_n) + cpuct_base) / cpuct_base);

  // Compute U coefficient: cpuct * sqrt(children_visits)
  // Use GetChildrenVisits() which returns N-1 for non-root (Lc0 style)
  uint32_t children_visits = node->GetChildrenVisits();
  float cpuct_sqrt_n =
      effective_cpuct *
      std::sqrt(static_cast<float>(std::max(children_visits, 1u)));

  // MCTS FPU with reduction strategy
  // FPU = parent_Q - fpu_value * sqrt(visited_policy)
  // This encourages exploration of unvisited nodes while being pessimistic
  float visited_policy = node->GetVisitedPolicy();

  // FPU reduction: unvisited nodes get parent Q minus a reduction
  // The reduction is proportional to sqrt of visited policy
  float fpu = parent_q - config_.fpu_reduction * std::sqrt(visited_policy);

  // Set up moves left evaluator for MLH utility (Lc0 feature)
  MCTSSearchParams lc0_params;
  lc0_params.moves_left_max_effect = 0.0345f;
  lc0_params.moves_left_threshold = 0.8f;
  lc0_params.moves_left_slope = 0.0027f;
  lc0_params.moves_left_scaled_factor = 1.6521f;
  lc0_params.moves_left_quadratic_factor = -0.6521f;

  MovesLeftEvaluator m_eval(lc0_params, node->GetM(), parent_q);

  // Single-pass selection with SIMD-friendly layout
  const TSEdge *edges = node->edges();
  int best_idx = 0;
  float best_score = -1e9f;

#ifdef __APPLE__
  // Apple Silicon optimization: prefetch edge data
  for (int i = 0; i < std::min(4, num_edges); ++i) {
    PREFETCH(&edges[i]);
  }
#endif

  for (int i = 0; i < num_edges; ++i) {
    const TSEdge &edge = edges[i];
    ThreadSafeNode *child = edge.child.load(std::memory_order_acquire);

    float q, m_utility = 0.0f;
    float policy = edge.GetPolicy(); // Use MCTS compressed policy
    int n_started = 0;

    if (child) {
      uint32_t n = child->GetN();
      uint32_t n_in_flight = child->GetNInFlight();
      n_started = static_cast<int>(n + n_in_flight);

      // CRITICAL: Negate child Q value because it's from opponent's perspective
      // This is the core of minimax in MCTS
      q = (n > 0) ? -child->GetQ(draw_score) : fpu;

      // Add moves left utility if enabled and child has visits
      // MLH prefers shorter wins and longer losses
      if (n > 0 && m_eval.IsEnabled()) {
        m_utility = m_eval.GetMUtility(child->GetM(), q);
      }
    } else {
      q = fpu;
      m_utility = m_eval.GetDefaultMUtility();
    }

    // MCTS PUCT score: Q + U + M
    // U = cpuct * sqrt(parent_N) * P / (1 + child_N_started)
    // This balances exploitation (Q) with exploration (U)
    float u = cpuct_sqrt_n * policy / (1.0f + static_cast<float>(n_started));
    float score = q + u + m_utility;

    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  return best_idx;
}

void ThreadSafeMCTS::expand_node(ThreadSafeNode *node, WorkerContext &ctx) {
  int num_edges = node->num_edges();
  if (num_edges == 0)
    return;

  TSEdge *edges = node->edges();
  std::vector<float> scores(num_edges);
  float max_score = -1e9f;

  // Score each move using improved heuristics (closer to Stockfish move
  // ordering)
  for (int i = 0; i < num_edges; ++i) {
    Move m = edges[i].move;
    float score = 0.0f;

    // Captures scored by MVV-LVA and SEE (most important for tactics)
    if (ctx.pos.capture(m)) {
      PieceType captured = m.type_of() == EN_PASSANT
                               ? PAWN
                               : type_of(ctx.pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(ctx.pos.piece_on(m.from_sq()));

      // Improved piece values matching Stockfish
      static const float piece_values[] = {0, 100, 320, 330, 500, 1000, 0};

      // MVV-LVA: Prioritize capturing valuable pieces with less valuable
      // attackers
      score += piece_values[captured] * 8.0f - piece_values[attacker] * 0.5f;

      // SEE bonus: Good captures get significant boost
      if (ctx.pos.see_ge(m, Value(0))) {
        score += 500.0f; // Increased from 300
      } else {
        // Bad captures (losing material) get penalty
        score -= 200.0f;
      }
    }

    // Promotions - queens are almost always best
    if (m.type_of() == PROMOTION) {
      PieceType promo = m.promotion_type();
      if (promo == QUEEN)
        score += 5000.0f; // Very high priority
      else if (promo == KNIGHT)
        score += 1000.0f; // Knight promotions for discovered attacks
      else
        score -= 500.0f; // Underpromotions rarely good
    }

    // Checks - very important tactically
    if (ctx.pos.gives_check(m)) {
      score += 600.0f; // Increased from 400

      // Discovered checks are even more valuable
      Bitboard blockers = ctx.pos.blockers_for_king(~ctx.pos.side_to_move());
      if (blockers & m.from_sq()) {
        score += 300.0f;
      }
    }

    // Piece development in opening/middlegame
    int game_phase = ctx.pos.count<ALL_PIECES>() > 24   ? 0
                     : ctx.pos.count<ALL_PIECES>() > 10 ? 1
                                                        : 2;

    if (game_phase < 2) { // Not endgame
      PieceType pt = type_of(ctx.pos.piece_on(m.from_sq()));

      // Knights and bishops should be developed
      if (pt == KNIGHT || pt == BISHOP) {
        Rank from_rank =
            relative_rank(ctx.pos.side_to_move(), rank_of(m.from_sq()));
        if (from_rank == RANK_1) { // Moving from back rank
          score += 150.0f;
        }
      }

      // Don't move queen too early
      if (pt == QUEEN && game_phase == 0) {
        score -= 100.0f;
      }
    }

    // Center control (important in all phases)
    int to_file = file_of(m.to_sq());
    int to_rank = rank_of(m.to_sq());
    float center_dist = std::abs(to_file - 3.5f) + std::abs(to_rank - 3.5f);
    score += (7.0f - center_dist) * 20.0f; // Increased from 15

    // Castling bonus - king safety is important
    if (m.type_of() == CASTLING) {
      score += 400.0f; // Increased from 200
    }

    // Pawn advances (especially passed pawns)
    if (type_of(ctx.pos.piece_on(m.from_sq())) == PAWN) {
      Rank to_rank_rel =
          relative_rank(ctx.pos.side_to_move(), rank_of(m.to_sq()));
      if (to_rank_rel >= RANK_6) {
        score += 200.0f * (to_rank_rel - RANK_5); // Bonus for advanced pawns
      }
    }

    // Avoid moving pieces that are well-placed (unless capturing)
    if (!ctx.pos.capture(m)) {
      // Small penalty for moving pieces multiple times in opening
      if (game_phase == 0) {
        score -= 30.0f;
      }
    }

    scores[i] = score;
    max_score = std::max(max_score, score);
  }

  // Softmax normalization with temperature
  float sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    // Temperature controls exploration: lower = more exploitation
    float temp = config_.policy_softmax_temp * 300.0f; // Adjusted divisor
    scores[i] = std::exp((scores[i] - max_score) / temp);
    sum += scores[i];
  }

  // Set policy priors using MCTS compressed storage
  for (int i = 0; i < num_edges; ++i) {
    edges[i].SetPolicy(scores[i] / sum);
  }
}

void ThreadSafeMCTS::add_dirichlet_noise(ThreadSafeNode *root) {
  int num_edges = root->num_edges();
  if (num_edges == 0)
    return;

  TSEdge *edges = root->edges();

  // MCTS Dirichlet noise
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);

  std::vector<float> noise(num_edges);
  float noise_sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    noise[i] = gamma(gen);
    noise_sum += noise[i];
  }

  // Avoid division by zero
  if (noise_sum < std::numeric_limits<float>::min())
    return;

  // Mix noise with existing policy: P' = (1 - epsilon) * P + epsilon * noise
  // This is Lc0's Dirichlet noise implementation for exploration
  for (int i = 0; i < num_edges; ++i) {
    float current = edges[i].GetPolicy();
    float noisy = (1.0f - config_.dirichlet_epsilon) * current +
                  config_.dirichlet_epsilon * (noise[i] / noise_sum);
    edges[i].SetPolicy(noisy);
  }
}

float ThreadSafeMCTS::evaluate_position(WorkerContext &ctx) {
  if (config_.use_batched_eval && batched_evaluator_) {
    return evaluate_position_batched(ctx);
  } else {
    return evaluate_position_direct(ctx);
  }
}

float ThreadSafeMCTS::evaluate_position_batched(WorkerContext &ctx) {
  return batched_evaluator_->evaluate(ctx.pos, ctx);
}

float ThreadSafeMCTS::evaluate_position_direct(WorkerContext &ctx) {
  // Check TT first - lock-free read (may get stale data, but that's OK for
  // MCTS)
  uint64_t key = ctx.pos.key();
  size_t tt_idx = key % SIMPLE_TT_SIZE;

  SimpleTTEntry &entry = simple_tt_[tt_idx];

  // Relaxed read - may see torn write but value will still be valid float
  if (entry.key == key) {
    ctx.cache_hits++;
    return entry.value;
  }

  ctx.cache_misses++;

  // Evaluate using GPU NNUE
  float value = 0.0f;

  if (gpu_manager_) {
    // Thread-safe GPU evaluation
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    auto [psqt, score] = gpu_manager_->evaluate_single(ctx.pos, true);

    // Use MCTS score transformation
    // This converts NNUE centipawn scores to MCTS Q values in [-1, 1]
    value = NnueScoreToQ(score);
  } else {
    // Fallback to simple eval
    int simple = Eval::simple_eval(ctx.pos);
    value = NnueScoreToQ(simple);
  }

  // CRITICAL: Adjust for side to move
  // NNUE returns score from white's perspective, so negate for black
  if (ctx.pos.side_to_move() == BLACK) {
    value = -value;
  }

  // Store in TT - lock-free write (benign race - last writer wins)
  entry.value = value;
  entry.key = key; // Write key last to serve as a release

  stats_.nn_evaluations.fetch_add(1, std::memory_order_relaxed);

  return value;
}

void ThreadSafeMCTS::backpropagate(ThreadSafeNode *node, float value,
                                   float draw, float moves_left) {
  // MCTS backpropagation with proper value negation
  // The value is from the perspective of the player who just moved
  // As we go up the tree, we negate it for the opponent

  int multivisit = config_.virtual_loss; // Match virtual loss count

  while (node) {
    // MCTS FinalizeScoreUpdate handles:
    // 1. Removing virtual loss (n_in_flight -= multivisit)
    // 2. Incrementing N
    // 3. Updating WL, D, M using running average
    node->FinalizeScoreUpdate(value, draw, moves_left, multivisit);

    // Flip value for parent (opponent's perspective)
    // This is the core of minimax in MCTS
    value = -value;

    // Increment moves left as we go up the tree
    // This helps with MLH (Moves Left Head) utility
    moves_left += 1.0f;

    node = node->parent();
  }
}

Move ThreadSafeMCTS::get_best_move() const {
  const ThreadSafeNode *root = tree_->root();
  if (!root->has_children())
    return Move::none();

  int num_edges = root->num_edges();
  const TSEdge *edges = root->edges();

  // MCTS best move selection
  // Priority: Terminal wins > Tablebase wins > Most visits > Best Q > Highest
  // policy Also prefers shorter wins and longer losses

  struct EdgeInfo {
    int idx;
    uint32_t visits;
    float q;
    float policy;
    float m; // Moves left estimate
    bool is_terminal;
    bool is_win;
    bool is_loss;
  };

  std::vector<EdgeInfo> candidates;
  candidates.reserve(num_edges);

  for (int i = 0; i < num_edges; ++i) {
    ThreadSafeNode *child = edges[i].child.load(std::memory_order_acquire);
    if (child && child->n() > 0) {
      EdgeInfo info;
      info.idx = i;
      info.visits = child->n();
      info.q =
          -child->q(); // Negate because child Q is from opponent's perspective
      info.policy =
          edges[i].GetPolicy(); // Use MCTS compressed policy accessor
      info.m = child->m();
      info.is_terminal = child->is_terminal();
      info.is_win = info.is_terminal && info.q > 0.5f;
      info.is_loss = info.is_terminal && info.q < -0.5f;
      candidates.push_back(info);
    }
  }

  if (candidates.empty()) {
    // No visits, return first move
    return edges[0].move;
  }

  // Sort by Lc0 criteria
  std::sort(candidates.begin(), candidates.end(),
            [](const EdgeInfo &a, const EdgeInfo &b) {
              // Terminal wins first (prefer shorter)
              if (a.is_win && !b.is_win)
                return true;
              if (!a.is_win && b.is_win)
                return false;
              if (a.is_win && b.is_win)
                return a.m < b.m; // Shorter win

              // Terminal losses last (prefer longer)
              if (a.is_loss && !b.is_loss)
                return false;
              if (!a.is_loss && b.is_loss)
                return true;
              if (a.is_loss && b.is_loss)
                return a.m > b.m; // Longer loss

              // Non-terminal: prefer more visits
              if (a.visits != b.visits)
                return a.visits > b.visits;

              // Then prefer better Q
              if (std::abs(a.q - b.q) > 0.001f)
                return a.q > b.q;

              // Then prefer higher policy
              return a.policy > b.policy;
            });

  return edges[candidates[0].idx].move;
}

std::vector<Move> ThreadSafeMCTS::get_pv() const {
  std::vector<Move> pv;
  const ThreadSafeNode *node = tree_->root();

  while (node && node->has_children()) {
    int num_edges = node->num_edges();
    const TSEdge *edges = node->edges();

    int best_idx = -1;
    uint32_t best_n = 0;

    for (int i = 0; i < num_edges; ++i) {
      ThreadSafeNode *child = edges[i].child.load(std::memory_order_acquire);
      if (child && child->n() > best_n) {
        best_n = child->n();
        best_idx = i;
      }
    }

    if (best_idx < 0)
      break;

    pv.push_back(edges[best_idx].move);
    node = edges[best_idx].child.load(std::memory_order_acquire);
  }

  return pv;
}

float ThreadSafeMCTS::get_best_q() const {
  const ThreadSafeNode *root = tree_->root();
  if (!root->has_children())
    return 0.0f;

  int num_edges = root->num_edges();
  const TSEdge *edges = root->edges();

  int best_idx = -1;
  uint32_t best_n = 0;

  for (int i = 0; i < num_edges; ++i) {
    ThreadSafeNode *child = edges[i].child.load(std::memory_order_acquire);
    if (child && child->n() > best_n) {
      best_n = child->n();
      best_idx = i;
    }
  }

  if (best_idx < 0)
    return 0.0f;

  ThreadSafeNode *best_child =
      edges[best_idx].child.load(std::memory_order_acquire);
  return best_child ? -best_child->q() : 0.0f;
}

void ThreadSafeMCTS::send_info() {
  if (!info_callback_)
    return;

  auto now = std::chrono::steady_clock::now();
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start_)
          .count();

  uint64_t nodes = stats_.total_nodes.load(std::memory_order_relaxed);
  uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

  std::ostringstream ss;
  ss << "info depth " << 1;
  ss << " nodes " << nodes;
  ss << " nps " << nps;
  ss << " time " << elapsed_ms;

  float q = get_best_q();
  int cp = static_cast<int>(q * 100);
  ss << " score cp " << cp;

  // PV
  std::vector<Move> pv = get_pv();
  if (!pv.empty()) {
    ss << " pv";
    for (const Move &m : pv) {
      ss << " " << UCIEngine::move(m, false);
    }
  }

  info_callback_(ss.str());
}

std::unique_ptr<ThreadSafeMCTS>
create_thread_safe_mcts(GPU::GPUNNUEManager *gpu_manager,
                        const ThreadSafeMCTSConfig &config) {
  auto mcts = std::make_unique<ThreadSafeMCTS>(config);
  mcts->set_gpu_manager(gpu_manager);
  return mcts;
}

} // namespace MCTS
} // namespace MetalFish
