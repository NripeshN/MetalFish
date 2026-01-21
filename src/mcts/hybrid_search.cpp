/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Implementation of the Hybrid MCTS + Alpha-Beta search.
  Optimized for Apple Silicon unified memory and Metal GPU.
  
  This implementation incorporates algorithms from Leela Chess Zero (Lc0),
  including PUCT with logarithmic growth, FPU reduction strategy, and
  moves left head (MLH) utility.

  Licensed under GPL-3.0
*/

#include "hybrid_search.h"
#include "lc0_mcts_core.h"
#include "../eval/evaluate.h"
#include "../gpu/backend.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <unordered_map>

#ifdef __aarch64__
// ARM64 (Apple Silicon) - use yield instruction
#define CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
// x86 - use _mm_pause
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#endif

using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// ============================================================================
// HybridBatchedEvaluator Implementation
// ============================================================================

HybridBatchedEvaluator::HybridBatchedEvaluator(GPU::GPUNNUEManager *gpu_manager,
                                               HybridSearchStats *stats,
                                               int min_batch_size,
                                               int max_batch_size,
                                               int batch_timeout_us)
    : gpu_manager_(gpu_manager), stats_(stats), min_batch_size_(min_batch_size),
      max_batch_size_(max_batch_size), batch_timeout_us_(batch_timeout_us) {
  tt_.resize(TT_SIZE);
  pending_requests_.reserve(max_batch_size);
}

HybridBatchedEvaluator::~HybridBatchedEvaluator() { stop(); }

void HybridBatchedEvaluator::start() {
  if (running_.load(std::memory_order_acquire))
    return;
  running_.store(true, std::memory_order_release);
  eval_thread_ = std::thread(&HybridBatchedEvaluator::eval_thread_main, this);
}

void HybridBatchedEvaluator::stop() {
  running_.store(false, std::memory_order_release);
  pending_cv_.notify_all();
  if (eval_thread_.joinable()) {
    eval_thread_.join();
  }
}

void HybridBatchedEvaluator::eval_thread_main() {
  std::vector<HybridEvalRequest *> batch;
  std::vector<HybridEvalRequest *> next_batch;
  batch.reserve(max_batch_size_);
  next_batch.reserve(max_batch_size_);

  int adaptive_timeout_us = batch_timeout_us_;

  auto collect_batch = [&](std::vector<HybridEvalRequest *> &target,
                           int timeout_us) {
    target.clear();
    std::unique_lock<std::mutex> lock(pending_mutex_);

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(timeout_us);

    while (pending_requests_.size() < static_cast<size_t>(min_batch_size_) &&
           running_.load(std::memory_order_acquire)) {
      if (pending_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        break;
      }
    }

    size_t count = std::min(pending_requests_.size(),
                            static_cast<size_t>(max_batch_size_));
    if (count > 0) {
      target.insert(target.end(), pending_requests_.begin(),
                    pending_requests_.begin() + count);
      pending_requests_.erase(pending_requests_.begin(),
                              pending_requests_.begin() + count);
    }

    size_t remaining = pending_requests_.size();
    if (remaining > 8) {
      adaptive_timeout_us = std::max(10, adaptive_timeout_us / 2);
    } else if (remaining == 0 && target.size() < 4) {
      adaptive_timeout_us =
          std::min(batch_timeout_us_ * 2, adaptive_timeout_us + 10);
    }
  };

  collect_batch(batch, adaptive_timeout_us);

  while (running_.load(std::memory_order_acquire)) {
    if (!batch.empty()) {
      std::thread prefetch_thread([&]() {
        if (running_.load(std::memory_order_acquire)) {
          collect_batch(next_batch, adaptive_timeout_us / 2);
        }
      });

      process_batch(batch);

      if (stats_) {
        stats_->nn_batches.fetch_add(1, std::memory_order_relaxed);
        stats_->total_batch_size.fetch_add(batch.size(),
                                           std::memory_order_relaxed);
        stats_->batch_count.fetch_add(1, std::memory_order_relaxed);
      }

      prefetch_thread.join();
      std::swap(batch, next_batch);
    } else {
      collect_batch(batch, adaptive_timeout_us);
    }
  }

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

void HybridBatchedEvaluator::process_batch(
    std::vector<HybridEvalRequest *> &batch) {
  // Check if GPU manager is available and not in shutdown
  if (!gpu_manager_ || batch.empty() || !GPU::gpu_nnue_manager_available())
    return;

  const size_t batch_size = batch.size();

  // Deduplication: group requests by position key
  std::unordered_map<uint64_t, std::vector<size_t>> key_to_indices;
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

  GPU::GPUEvalBatch gpu_batch;
  gpu_batch.reserve(static_cast<int>(unique_count));

  for (size_t idx : unique_indices) {
    gpu_batch.add_position_data(batch[idx]->pos_data);
  }

  gpu_manager_->evaluate_batch(gpu_batch, true);

  for (size_t i = 0; i < unique_count; ++i) {
    size_t orig_idx = unique_indices[i];
    HybridEvalRequest *req = batch[orig_idx];

    int32_t psqt =
        gpu_batch.psqt_scores.size() > i ? gpu_batch.psqt_scores[i] : 0;
    int32_t pos_score = gpu_batch.positional_scores.size() > i
                            ? gpu_batch.positional_scores[i]
                            : 0;
    int32_t raw_score = psqt + pos_score;

    float value = std::tanh(static_cast<float>(raw_score) / 400.0f);

    if (req->side_to_move == BLACK) {
      value = -value;
    }

    // Store in TT with age
    uint32_t age = current_age_.load(std::memory_order_relaxed);
    size_t tt_idx = req->position_key % TT_SIZE;
    tt_[tt_idx].value = value;
    tt_[tt_idx].key = req->position_key;
    tt_[tt_idx].age = age;

    // Complete all requests with same key (including duplicates)
    for (size_t dup_idx : key_to_indices[req->position_key]) {
      batch[dup_idx]->result = value;
      batch[dup_idx]->completed.store(true, std::memory_order_release);
    }
  }

  if (stats_) {
    stats_->nn_evaluations.fetch_add(unique_count, std::memory_order_relaxed);
  }
}

float HybridBatchedEvaluator::evaluate(const MCTSPosition &pos,
                                       uint64_t &cache_hits,
                                       uint64_t &cache_misses) {
  const Position &sf_pos = pos.stockfish_position();
  uint64_t key = sf_pos.key();
  size_t tt_idx = key % TT_SIZE;

  if (tt_[tt_idx].key == key) {
    cache_hits++;
    return tt_[tt_idx].value;
  }

  cache_misses++;

  HybridEvalRequest *req = request_pool_.acquire();
  req->pos_data.from_position(sf_pos);
  req->position_key = key;
  req->side_to_move = sf_pos.side_to_move();
  req->ready.store(true, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_requests_.push_back(req);
  }
  pending_cv_.notify_one();

  // Spin-wait for result
  int spin_count = 0;
  while (!req->completed.load(std::memory_order_acquire)) {
    if (++spin_count > 1000) {
      std::this_thread::yield();
      spin_count = 0;
    }
  }

  float result = req->result;
  request_pool_.release(req);

  return result;
}

// ============================================================================
// HybridNode Implementation
// ============================================================================

HybridNode::HybridNode(HybridNode *parent, int edge_index)
    : parent_(parent), edge_index_(edge_index) {
  n_.store(0, std::memory_order_relaxed);
  n_in_flight_.store(0, std::memory_order_relaxed);
  w_.store(0.0f, std::memory_order_relaxed);
  q_.store(0.0f, std::memory_order_relaxed);
  d_.store(0.0f, std::memory_order_relaxed);
  m_.store(0.0f, std::memory_order_relaxed);
  terminal_type_.store(Terminal::NonTerminal, std::memory_order_relaxed);
  has_ab_score_.store(false, std::memory_order_relaxed);
  ab_score_.store(0, std::memory_order_relaxed);
  ab_depth_.store(0, std::memory_order_relaxed);
}

void HybridNode::create_edges(const MCTSMoveList &moves) {
  if (moves.empty())
    return;

  int count = static_cast<int>(moves.size());
  edges_ = std::make_unique<HybridEdge[]>(count);

  float uniform = 1.0f / count;
  int idx = 0;
  for (const auto &m : moves) {
    edges_[idx].init(m, uniform);
    idx++;
  }

  num_edges_.store(count, std::memory_order_release);
}

void HybridNode::update_stats(float value, float draw_prob, float moves_left) {
  // Increment visit count first
  uint32_t new_n = n_.fetch_add(1, std::memory_order_acq_rel) + 1;

  // Update W using CAS
  float old_w = w_.load(std::memory_order_relaxed);
  while (!w_.compare_exchange_weak(old_w, old_w + value,
                                   std::memory_order_release,
                                   std::memory_order_relaxed)) {
  }

  // Update Q (relaxed is fine for MCTS)
  float new_q = (old_w + value) / new_n;
  q_.store(new_q, std::memory_order_relaxed);

  // Update D and M
  float old_d = d_.load(std::memory_order_relaxed);
  d_.store(old_d + (draw_prob - old_d) / new_n, std::memory_order_relaxed);

  float old_m = m_.load(std::memory_order_relaxed);
  m_.store(old_m + (moves_left - old_m) / new_n, std::memory_order_relaxed);
}

void HybridNode::set_terminal(Terminal type, float value) {
  terminal_type_.store(type, std::memory_order_release);
  w_.store(value, std::memory_order_release);
  q_.store(value, std::memory_order_release);
  n_.store(1, std::memory_order_release);
  d_.store((type == Terminal::Draw) ? 1.0f : 0.0f, std::memory_order_release);
  m_.store(0.0f, std::memory_order_release);
}

void HybridNode::set_ab_score(int score, int depth) {
  has_ab_score_.store(true, std::memory_order_release);
  ab_score_.store(score, std::memory_order_release);
  ab_depth_.store(depth, std::memory_order_release);
}

float HybridNode::get_u(float cpuct, float parent_n_sqrt) const {
  uint32_t n = n_.load(std::memory_order_relaxed);
  uint32_t n_in_flight = n_in_flight_.load(std::memory_order_relaxed);
  return cpuct * parent_n_sqrt / (1.0f + n + n_in_flight);
}

float HybridNode::get_puct(float cpuct, float parent_n_sqrt, float fpu) const {
  uint32_t n = n_.load(std::memory_order_relaxed);
  float q_val = (n > 0) ? q_.load(std::memory_order_relaxed) : fpu;
  float u = get_u(cpuct, parent_n_sqrt);

  float p = 1.0f;
  if (parent_ && edge_index_ >= 0 && edge_index_ < parent_->num_edges()) {
    p = parent_->edges()[edge_index_].policy();
  }

  return q_val + u * p;
}

// ============================================================================
// HybridTree Implementation with Arena Allocation
// ============================================================================

HybridTree::HybridTree() {
  arenas_.push_back(std::make_unique<NodeArena>());
  root_ = allocate_node(nullptr, -1);
}

HybridTree::~HybridTree() = default;

void HybridTree::reset(const MCTSPositionHistory &history) {
  {
    std::unique_lock<std::shared_mutex> lock(fen_mutex_);
    root_fen_ = history.current().fen();
  }

  history_ = history;

  {
    std::lock_guard<std::mutex> lock(arena_mutex_);
    arenas_.clear();
    arenas_.push_back(std::make_unique<NodeArena>());
    current_arena_.store(0, std::memory_order_relaxed);
  }

  node_count_.store(0, std::memory_order_relaxed);
  root_ = allocate_node(nullptr, -1);
}

bool HybridTree::apply_move(MCTSMove move) {
  history_.do_move(move);

  {
    std::unique_lock<std::shared_mutex> lock(fen_mutex_);
    root_fen_ = history_.current().fen();
  }

  {
    std::lock_guard<std::mutex> lock(arena_mutex_);
    arenas_.clear();
    arenas_.push_back(std::make_unique<NodeArena>());
    current_arena_.store(0, std::memory_order_relaxed);
  }

  node_count_.store(0, std::memory_order_relaxed);
  root_ = allocate_node(nullptr, -1);
  return false;
}

HybridNode *HybridTree::allocate_node(HybridNode *parent, int edge_index) {
  // Try current arena first (lock-free fast path)
  size_t arena_idx = current_arena_.load(std::memory_order_acquire);

  if (arena_idx < arenas_.size()) {
    NodeArena *arena = arenas_[arena_idx].get();
    size_t slot = arena->next.fetch_add(1, std::memory_order_relaxed);

    if (slot < ARENA_SIZE) {
      HybridNode *node = &arena->nodes[slot];
      node->init(parent, edge_index);
      node_count_.fetch_add(1, std::memory_order_relaxed);
      return node;
    }
  }

  // Need new arena (slow path)
  std::lock_guard<std::mutex> lock(arena_mutex_);

  arena_idx = current_arena_.load(std::memory_order_acquire);
  if (arena_idx < arenas_.size()) {
    NodeArena *arena = arenas_[arena_idx].get();
    if (arena->next.load(std::memory_order_relaxed) < ARENA_SIZE) {
      size_t slot = arena->next.fetch_add(1, std::memory_order_relaxed);
      if (slot < ARENA_SIZE) {
        HybridNode *node = &arena->nodes[slot];
        node->init(parent, edge_index);
        node_count_.fetch_add(1, std::memory_order_relaxed);
        return node;
      }
    }
  }

  arenas_.push_back(std::make_unique<NodeArena>());
  current_arena_.store(arenas_.size() - 1, std::memory_order_release);

  NodeArena *arena = arenas_.back().get();
  size_t slot = arena->next.fetch_add(1, std::memory_order_relaxed);
  HybridNode *node = &arena->nodes[slot];
  node->init(parent, edge_index);
  node_count_.fetch_add(1, std::memory_order_relaxed);
  return node;
}

// ============================================================================
// HybridSearch Implementation
// ============================================================================

HybridSearch::HybridSearch(const HybridSearchConfig &config)
    : config_(config), tree_(std::make_unique<HybridTree>()) {
  simple_tt_.resize(SIMPLE_TT_SIZE);
}

HybridSearch::~HybridSearch() {
  stop();
  wait();

  if (batched_evaluator_) {
    batched_evaluator_->stop();
  }
  
  // Synchronize GPU to ensure all pending operations complete
  // This prevents crashes from async callbacks accessing destroyed objects
  // Only synchronize if the backend is still available and not shut down
  if (GPU::gpu_available() && !GPU::gpu_backend_shutdown()) {
    GPU::gpu().synchronize();
  }
}

void HybridSearch::set_neural_network(std::shared_ptr<MCTSNeuralNetwork> nn) {
  neural_network_ = nn;
}

void HybridSearch::set_gpu_nnue(GPU::GPUNNUEManager *gpu_nnue) {
  gpu_nnue_ = gpu_nnue;
}

void HybridSearch::start_search(const MCTSPositionHistory &history,
                                const Search::LimitsType &limits,
                                BestMoveCallback best_move_cb,
                                InfoCallback info_cb) {
  stop();
  wait();

  stats_.reset();
  stop_flag_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_release);
  limits_ = limits;
  best_move_callback_ = best_move_cb;
  info_callback_ = info_cb;
  search_start_ = std::chrono::steady_clock::now();

  // Calculate time budget
  time_budget_ms_ = get_time_budget_ms();

  // Initialize tree
  tree_->reset(history);

  // Get actual thread count and auto-tune
  int actual_threads = config_.get_num_threads();
  config_.auto_tune(actual_threads);

  // Initialize batched evaluator if enabled
  if (config_.use_batched_eval && gpu_nnue_) {
    batched_evaluator_ = std::make_unique<HybridBatchedEvaluator>(
        gpu_nnue_, &stats_, config_.min_batch_size, config_.max_batch_size,
        config_.batch_timeout_us);
    batched_evaluator_->start();
  }

  // Create worker contexts
  worker_contexts_.clear();
  for (int i = 0; i < actual_threads; ++i) {
    auto ctx = std::make_unique<HybridWorkerContext>();
    ctx->set_root_fen(tree_->root_fen());
    worker_contexts_.push_back(std::move(ctx));
  }

  // Start worker threads
  search_threads_.clear();
  for (int i = 0; i < actual_threads; ++i) {
    search_threads_.emplace_back(&HybridSearch::search_thread_main, this, i);
  }
}

void HybridSearch::stop() { stop_flag_.store(true, std::memory_order_release); }

void HybridSearch::wait() {
  for (auto &thread : search_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  search_threads_.clear();

  if (batched_evaluator_) {
    batched_evaluator_->stop();
    batched_evaluator_.reset();
  }

  running_.store(false, std::memory_order_release);

  // Report best move
  if (best_move_callback_) {
    MCTSMove best = select_best_move();
    std::vector<MCTSMove> pv = get_pv();
    MCTSMove ponder = (pv.size() > 1) ? pv[1] : MCTSMove();
    best_move_callback_(best, ponder);
  }
}

MCTSMove HybridSearch::get_best_move() const { return select_best_move(); }

float HybridSearch::get_best_move_q() const {
  const HybridNode *root = tree_->root();
  if (!root || !root->has_children())
    return 0.0f;

  // Find child with most visits (same logic as select_best_move)
  int best_idx = -1;
  uint32_t best_n = 0;

  for (int i = 0; i < root->num_edges(); ++i) {
    const HybridEdge &edge = root->edges()[i];
    if (edge.child() && edge.child()->n() > best_n) {
      best_n = edge.child()->n();
      best_idx = i;
    }
  }

  if (best_idx < 0 || !root->edges()[best_idx].child())
    return 0.0f;

  // Return the Q value of the best child (negated since it's from opponent's
  // perspective)
  return -root->edges()[best_idx].child()->q();
}

std::vector<MCTSMove> HybridSearch::get_pv() const {
  std::vector<MCTSMove> pv;
  const HybridNode *node = tree_->root();

  while (node && node->has_children()) {
    // Find child with most visits
    int best_idx = -1;
    uint32_t best_n = 0;

    for (int i = 0; i < node->num_edges(); ++i) {
      const HybridEdge &edge = node->edges()[i];
      if (edge.child() && edge.child()->n() > best_n) {
        best_n = edge.child()->n();
        best_idx = i;
      }
    }

    if (best_idx < 0)
      break;

    pv.push_back(node->edges()[best_idx].move());
    node = node->edges()[best_idx].child();
  }

  return pv;
}

void HybridSearch::search_thread_main(int thread_id) {
  HybridWorkerContext &ctx = *worker_contexts_[thread_id];

  // Thread-local position
  MCTSPosition thread_pos;
  thread_pos.set_from_fen(ctx.cached_root_fen);

  // Expand root node first
  HybridNode *root = tree_->root();
  if (!root) {
    return;
  }

  // Synchronize root expansion across threads
  static std::mutex root_expand_mutex;
  {
    std::lock_guard<std::mutex> lock(root_expand_mutex);
    if (!root->has_children()) {
      expand_node(root, thread_pos, ctx);
    }
  }

  int iterations_since_stop_check = 0;

  while (true) {
    // Batched stop checks
    if (++iterations_since_stop_check >=
        HybridSearchConfig::STOP_CHECK_INTERVAL) {
      iterations_since_stop_check = 0;
      if (should_stop())
        break;
    }

    bool do_profile =
        (ctx.iterations % HybridSearchConfig::PROFILE_SAMPLE_RATE) == 0;

    // Reset position
    MCTSPosition search_pos;
    search_pos.set_from_fen(ctx.cached_root_fen);

    // Selection
    auto select_start = do_profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    HybridNode *node = select_node(tree_->root(), search_pos, ctx);
    auto select_end = do_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};

    if (!node) {
      ctx.iterations++;
      if (ctx.iterations > 100)
        break;
      continue;
    }

    // Check if terminal
    if (search_pos.is_terminal()) {
      GameResult result = search_pos.get_game_result();
      float value = 0.0f;
      HybridNode::Terminal term_type = HybridNode::Terminal::Draw;

      if (result == GameResult::WHITE_WON) {
        value = search_pos.is_black_to_move() ? -1.0f : 1.0f;
        term_type = search_pos.is_black_to_move() ? HybridNode::Terminal::Loss
                                                  : HybridNode::Terminal::Win;
      } else if (result == GameResult::BLACK_WON) {
        value = search_pos.is_black_to_move() ? 1.0f : -1.0f;
        term_type = search_pos.is_black_to_move() ? HybridNode::Terminal::Win
                                                  : HybridNode::Terminal::Loss;
      }

      node->set_terminal(term_type, value);
      backpropagate(node, value, (result == GameResult::DRAW) ? 1.0f : 0.0f,
                    0.0f);
      ctx.iterations++;
      continue;
    }

    // Expansion
    auto expand_start = do_profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    if (!node->has_children()) {
      expand_node(node, search_pos, ctx);
    }
    auto expand_end = do_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};

    // Evaluation
    auto eval_start = do_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    float value = evaluate_position(search_pos, ctx);
    auto eval_end = do_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};

    // Backpropagation
    auto backprop_start = do_profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
    float draw = std::max(0.0f, 0.4f - std::abs(value) * 0.3f);
    backpropagate(node, value, draw, 30.0f);
    auto backprop_end = do_profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};

    // Update profiling (sampled)
    if (do_profile) {
      ctx.selection_time_acc +=
          std::chrono::duration_cast<std::chrono::microseconds>(select_end -
                                                                select_start)
              .count() *
          HybridSearchConfig::PROFILE_SAMPLE_RATE;
      ctx.expansion_time_acc +=
          std::chrono::duration_cast<std::chrono::microseconds>(expand_end -
                                                                expand_start)
              .count() *
          HybridSearchConfig::PROFILE_SAMPLE_RATE;
      ctx.evaluation_time_acc +=
          std::chrono::duration_cast<std::chrono::microseconds>(eval_end -
                                                                eval_start)
              .count() *
          HybridSearchConfig::PROFILE_SAMPLE_RATE;
      ctx.backprop_time_acc +=
          std::chrono::duration_cast<std::chrono::microseconds>(backprop_end -
                                                                backprop_start)
              .count() *
          HybridSearchConfig::PROFILE_SAMPLE_RATE;
    }

    stats_.mcts_nodes.fetch_add(1, std::memory_order_relaxed);
    stats_.total_iterations.fetch_add(1, std::memory_order_relaxed);
    ctx.iterations++;
  }

  // Flush accumulated stats
  stats_.selection_time_us.fetch_add(ctx.selection_time_acc,
                                     std::memory_order_relaxed);
  stats_.expansion_time_us.fetch_add(ctx.expansion_time_acc,
                                     std::memory_order_relaxed);
  stats_.evaluation_time_us.fetch_add(ctx.evaluation_time_acc,
                                      std::memory_order_relaxed);
  stats_.backprop_time_us.fetch_add(ctx.backprop_time_acc,
                                    std::memory_order_relaxed);
  stats_.cache_hits.fetch_add(ctx.cache_hits, std::memory_order_relaxed);
  stats_.cache_misses.fetch_add(ctx.cache_misses, std::memory_order_relaxed);
}

// Evaluation methods
float HybridSearch::evaluate_position(const MCTSPosition &pos,
                                      HybridWorkerContext &ctx) {
  if (config_.use_batched_eval && batched_evaluator_) {
    return evaluate_position_batched(pos, ctx);
  }
  return evaluate_position_direct(pos, ctx);
}

float HybridSearch::evaluate_position_batched(const MCTSPosition &pos,
                                              HybridWorkerContext &ctx) {
  return batched_evaluator_->evaluate(pos, ctx.cache_hits, ctx.cache_misses);
}

float HybridSearch::evaluate_position_direct(const MCTSPosition &pos,
                                             HybridWorkerContext &ctx) {
  uint64_t key = pos.stockfish_position().key();
  size_t tt_idx = key % SIMPLE_TT_SIZE;

  if (simple_tt_[tt_idx].key == key) {
    ctx.cache_hits++;
    return simple_tt_[tt_idx].value;
  }

  ctx.cache_misses++;

  float value = 0.0f;
  if (gpu_nnue_) {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    auto [psqt, score] =
        gpu_nnue_->evaluate_single(pos.stockfish_position(), true);
    
    // Use Lc0-style score transformation
    // This converts NNUE centipawn scores to MCTS Q values in [-1, 1]
    value = NnueScoreToQ(score);
    
    // CRITICAL: Ensure value is from side-to-move's perspective
    // NNUE returns score from white's perspective, so negate for black
    if (pos.is_black_to_move()) {
      value = -value;
    }
  }

  simple_tt_[tt_idx].key = key;
  simple_tt_[tt_idx].value = value;

  return value;
}

HybridNode *HybridSearch::select_node(HybridNode *node, MCTSPosition &pos,
                                      HybridWorkerContext &ctx) {
  while (node->has_children() && !node->is_terminal()) {
    int best_idx = select_child_puct(node, config_.cpuct, ctx);

    if (best_idx < 0)
      break;

    HybridEdge &edge = node->edges()[best_idx];
    HybridNode *child = edge.child();

    if (!child) {
      child = tree_->allocate_node(node, best_idx);
      edge.set_child(child);
    }

    // Add virtual loss
    child->add_virtual_loss(config_.virtual_loss);

    // Make the move
    pos.do_move(edge.move());
    node = child;
  }

  return node;
}

int HybridSearch::select_child_puct(HybridNode *node, float cpuct,
                                    HybridWorkerContext &ctx) {
  int num_edges = node->num_edges();
  if (num_edges == 0)
    return -1;

  ctx.puct_scores.resize(num_edges);

  // Get parent statistics
  uint32_t parent_n = node->n() + node->n_in_flight();
  float parent_q = node->n() > 0 ? node->q() : 0.0f;
  
  // Lc0-style PUCT with logarithmic growth
  // Formula: cpuct_init + cpuct_factor * log((N + cpuct_base) / cpuct_base)
  float effective_cpuct = cpuct + config_.cpuct_factor * 
      std::log((static_cast<float>(parent_n) + config_.cpuct_base) / config_.cpuct_base);
  
  // Compute U coefficient: cpuct * sqrt(children_visits)
  uint32_t children_visits = node->n() > 0 ? node->n() - 1 : 0;
  float cpuct_sqrt_n = effective_cpuct * std::sqrt(static_cast<float>(
      std::max(children_visits, 1u)));
  
  // Lc0-style FPU with reduction strategy
  // FPU = -parent_Q - fpu_value * sqrt(visited_policy)
  float visited_policy = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    const HybridEdge &edge = node->edges()[i];
    HybridNode *child = edge.child();
    if (child && child->n() > 0) {
      visited_policy += edge.policy();
    }
  }
  
  // FPU reduction: unvisited nodes get parent Q minus a reduction
  float fpu = parent_q - config_.fpu_reduction * std::sqrt(visited_policy);
  
  // Set up moves left evaluator for MLH utility
  Lc0SearchParams lc0_params;
  lc0_params.moves_left_max_effect = 0.0345f;
  lc0_params.moves_left_threshold = 0.8f;
  lc0_params.moves_left_slope = 0.0027f;
  lc0_params.moves_left_scaled_factor = 1.6521f;
  lc0_params.moves_left_quadratic_factor = -0.6521f;
  
  MovesLeftEvaluator m_eval(lc0_params, node->m(), parent_q);

  int best_idx = -1;
  float best_puct = -std::numeric_limits<float>::infinity();

  for (int i = 0; i < num_edges; ++i) {
    const HybridEdge &edge = node->edges()[i];
    HybridNode *child = edge.child();

    float q, m_utility = 0.0f;
    float policy = edge.policy();
    int n_started = 0;
    
    if (child) {
      uint32_t n = child->n();
      uint32_t n_in_flight = child->n_in_flight();
      n_started = static_cast<int>(n + n_in_flight);
      
      // CRITICAL: Negate child Q value because it's from opponent's perspective
      q = (n > 0) ? -child->q() : fpu;
      
      // Add moves left utility if enabled and child has visits
      if (n > 0 && m_eval.IsEnabled()) {
        m_utility = m_eval.GetMUtility(child->m(), q);
      }
    } else {
      q = fpu;
      m_utility = m_eval.GetDefaultMUtility();
    }
    
    // Lc0-style PUCT score: Q + U + M
    // U = cpuct * sqrt(parent_N) * P / (1 + child_N)
    float u = cpuct_sqrt_n * policy / (1.0f + static_cast<float>(n_started));
    float puct = q + u + m_utility;

    ctx.puct_scores[i] = puct;
    if (puct > best_puct) {
      best_puct = puct;
      best_idx = i;
    }
  }

  return best_idx;
}

void HybridSearch::expand_node(HybridNode *node, const MCTSPosition &pos,
                               HybridWorkerContext &ctx) {
  if (node->has_children())
    return;

  MCTSMoveList moves = pos.generate_legal_moves();
  if (moves.empty()) {
    if (pos.is_check()) {
      node->set_terminal(HybridNode::Terminal::Loss, -1.0f);
    } else {
      node->set_terminal(HybridNode::Terminal::Draw, 0.0f);
    }
    return;
  }

  node->create_edges(moves);
  
  // CRITICAL FIX: Set heuristic policy priors (like thread_safe_mcts.cpp)
  // Without this, all moves have equal probability which hurts search quality
  int num_edges = node->num_edges();
  if (num_edges > 0) {
    std::vector<float> scores(num_edges);
    float max_score = -1e9f;
    
    const Position &sf_pos = pos.stockfish_position();
    
    for (int i = 0; i < num_edges; ++i) {
      Move m = node->edges()[i].move().to_stockfish();
      float score = 0.0f;
      
      // Captures scored by MVV-LVA and SEE
      if (sf_pos.capture(m)) {
        PieceType captured = m.type_of() == EN_PASSANT
                               ? PAWN
                               : type_of(sf_pos.piece_on(m.to_sq()));
        PieceType attacker = type_of(sf_pos.piece_on(m.from_sq()));
        static const float piece_values[] = {0, 100, 320, 330, 500, 900, 0};
        score += piece_values[captured] * 6.0f - piece_values[attacker];
        
        if (sf_pos.see_ge(m, Value(0))) {
          score += 300.0f;
        }
      }
      
      // Promotions
      if (m.type_of() == PROMOTION) {
        PieceType promo = m.promotion_type();
        if (promo == QUEEN)
          score += 4000.0f;
        else if (promo == KNIGHT)
          score += 800.0f;
      }
      
      // Checks
      if (sf_pos.gives_check(m)) {
        score += 400.0f;
      }
      
      // Center control
      int to_file = file_of(m.to_sq());
      int to_rank = rank_of(m.to_sq());
      float center_dist = std::abs(to_file - 3.5f) + std::abs(to_rank - 3.5f);
      score += (7.0f - center_dist) * 15.0f;
      
      // Castling bonus
      if (m.type_of() == CASTLING) {
        score += 200.0f;
      }
      
      scores[i] = score;
      max_score = std::max(max_score, score);
    }
    
    // Softmax normalization with temperature
    // Handle temperature == 0 as argmax (deterministic selection)
    if (config_.policy_softmax_temp == 0.0f) {
      for (int i = 0; i < num_edges; ++i) {
        node->edges()[i].set_policy((scores[i] == max_score) ? 1.0f : 0.0f);
      }
    } else {
      float sum = 0.0f;
      for (int i = 0; i < num_edges; ++i) {
        scores[i] = std::exp((scores[i] - max_score) / 
                             (config_.policy_softmax_temp * 400.0f));
        sum += scores[i];
      }
      
      // Set policy priors
      for (int i = 0; i < num_edges; ++i) {
        node->edges()[i].set_policy(scores[i] / sum);
      }
    }
  }

  // Add Dirichlet noise at root
  if (config_.add_dirichlet_noise && node == tree_->root()) {
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    int num_edges = node->num_edges();
    std::vector<float> noise(num_edges);
    float noise_sum = 0.0f;

    for (int i = 0; i < num_edges; ++i) {
      noise[i] = gamma(ctx.rng);
      noise_sum += noise[i];
    }

    if (noise_sum > 0) {
      float eps = config_.dirichlet_epsilon;
      for (int i = 0; i < num_edges; ++i) {
        float p = node->edges()[i].policy();
        float noisy_p = (1.0f - eps) * p + eps * (noise[i] / noise_sum);
        node->edges()[i].set_policy(noisy_p);
      }
    }
  }
}

void HybridSearch::backpropagate(HybridNode *node, float value, float draw_prob,
                                 float moves_left) {
  // CRITICAL: Start with the value from the leaf node's perspective
  // As we go up the tree, we flip the value because each level is opponent's turn
  while (node) {
    // Remove virtual loss first
    node->remove_virtual_loss(config_.virtual_loss);

    // Update statistics with current value
    node->update_stats(value, draw_prob, moves_left);

    // Flip value for parent (opponent's perspective)
    // This is essential for minimax-style MCTS
    value = -value;
    moves_left += 1.0f;

    node = node->parent();
  }
}

bool HybridSearch::should_use_alphabeta(HybridNode *node,
                                        const MCTSPosition &pos) {
  if (!config_.use_ab_for_tactics)
    return false;

  if (pos.is_check()) {
    stats_.tactical_positions.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  if (node->n() < static_cast<uint32_t>(config_.ab_node_threshold)) {
    return false;
  }

  return false;
}

int HybridSearch::alphabeta_verify(const MCTSPosition &pos, int depth,
                                   int alpha, int beta) {
  stats_.ab_nodes.fetch_add(1, std::memory_order_relaxed);

  if (depth <= 0 || pos.is_terminal()) {
    Value v = Eval::simple_eval(pos.stockfish_position());
    return static_cast<int>(v);
  }

  MCTSMoveList moves = pos.generate_legal_moves();
  if (moves.empty()) {
    if (pos.is_check()) {
      return -32000 + depth;
    }
    return 0; // Stalemate
  }

  int best_score = -32001;
  MCTSPosition child_pos = pos;

  for (const auto &move : moves) {
    child_pos.do_move(move);
    int score = -alphabeta_verify(child_pos, depth - 1, -beta, -alpha);
    child_pos.undo_move();

    if (score > best_score) {
      best_score = score;
      if (score > alpha) {
        alpha = score;
        if (alpha >= beta) {
          break; // Beta cutoff
        }
      }
    }
  }

  return best_score;
}

bool HybridSearch::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire))
    return true;

  // Check node limit
  if (limits_.nodes > 0 && stats_.mcts_nodes >= limits_.nodes) {
    return true;
  }

  // Check time limit
  if (time_budget_ms_ > 0) {
    auto elapsed = std::chrono::steady_clock::now() - search_start_;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (ms >= time_budget_ms_) {
      return true;
    }
  }

  if (limits_.movetime > 0) {
    auto elapsed = std::chrono::steady_clock::now() - search_start_;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (ms >= limits_.movetime) {
      return true;
    }
  }

  return false;
}

int64_t HybridSearch::get_time_budget_ms() const {
  if (limits_.movetime > 0) {
    return limits_.movetime;
  }

  // Simple time management
  Color us = tree_->history().current().side_to_move();
  int64_t time_left = (us == WHITE) ? limits_.time[WHITE] : limits_.time[BLACK];
  int64_t inc = (us == WHITE) ? limits_.inc[WHITE] : limits_.inc[BLACK];

  if (time_left <= 0)
    return 1000; // Default 1 second

  // CRITICAL FIX: Use more aggressive time allocation for MCTS
  // MCTS needs more time than alpha-beta to build a good tree
  // Use ~5% of remaining time + most of increment (was 2.5% + increment)
  int64_t base_time = time_left / 20;  // 5% of remaining time
  int64_t inc_time = inc * 3 / 4;       // Use 75% of increment
  
  // Minimum 500ms for MCTS to be effective
  int64_t budget = std::max(int64_t(500), base_time + inc_time);
  
  // Cap at 20% of remaining time to avoid time trouble
  int64_t max_time = time_left / 5;
  
  return std::min(budget, max_time);
}

MCTSMove HybridSearch::select_best_move() const {
  const HybridNode *root = tree_->root();
  if (!root->has_children())
    return MCTSMove();

  int num_edges = root->num_edges();
  
  // Lc0-style best move selection
  // Priority: Terminal wins > Most visits > Best Q > Highest policy
  // Also prefers shorter wins and longer losses
  
  struct EdgeInfo {
    int idx;
    uint32_t visits;
    float q;
    float policy;
    float m;  // Moves left estimate
    bool is_terminal;
    bool is_win;
    bool is_loss;
  };
  
  std::vector<EdgeInfo> candidates;
  candidates.reserve(num_edges);
  
  for (int i = 0; i < num_edges; ++i) {
    const HybridEdge &edge = root->edges()[i];
    HybridNode *child = edge.child();
    
    if (child && child->n() > 0) {
      EdgeInfo info;
      info.idx = i;
      info.visits = child->n();
      info.q = -child->q();  // Negate because child Q is from opponent's perspective
      info.policy = edge.policy();
      info.m = child->m();
      info.is_terminal = child->is_terminal();
      info.is_win = info.is_terminal && info.q > 0.5f;
      info.is_loss = info.is_terminal && info.q < -0.5f;
      candidates.push_back(info);
    }
  }
  
  if (candidates.empty()) {
    // No visits, return first move
    return root->edges()[0].move();
  }
  
  // Sort by Lc0 criteria
  std::sort(candidates.begin(), candidates.end(), [](const EdgeInfo& a, const EdgeInfo& b) {
    // Terminal wins first (prefer shorter)
    if (a.is_win && !b.is_win) return true;
    if (!a.is_win && b.is_win) return false;
    if (a.is_win && b.is_win) return a.m < b.m;  // Shorter win
    
    // Terminal losses last (prefer longer)
    if (a.is_loss && !b.is_loss) return false;
    if (!a.is_loss && b.is_loss) return true;
    if (a.is_loss && b.is_loss) return a.m > b.m;  // Longer loss
    
    // Non-terminal: prefer more visits
    if (a.visits != b.visits) return a.visits > b.visits;
    
    // Then prefer better Q
    if (std::abs(a.q - b.q) > 0.001f) return a.q > b.q;
    
    // Then prefer higher policy
    return a.policy > b.policy;
  });

  return root->edges()[candidates[0].idx].move();
}

void HybridSearch::update_info() {
  if (!info_callback_)
    return;

  auto elapsed = std::chrono::steady_clock::now() - search_start_;
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

  std::ostringstream ss;
  ss << "info";
  ss << " nodes " << stats_.mcts_nodes;
  ss << " time " << ms;

  if (ms > 0) {
    uint64_t nps = stats_.mcts_nodes * 1000 / ms;
    ss << " nps " << nps;
  }

  // PV
  auto pv = get_pv();
  if (!pv.empty()) {
    ss << " pv";
    for (const auto &move : pv) {
      ss << " " << move.to_string();
    }
  }

  // Score
  const HybridNode *root = tree_->root();
  if (root->n() > 0) {
    int cp = static_cast<int>(root->q() * 100);
    ss << " score cp " << cp;
  }

  info_callback_(ss.str());
}

std::unique_ptr<HybridSearch>
create_hybrid_search(GPU::GPUNNUEManager *gpu_nnue,
                     const HybridSearchConfig &config) {
  auto search = std::make_unique<HybridSearch>(config);
  search->set_gpu_nnue(gpu_nnue);
  return search;
}

} // namespace MCTS
} // namespace MetalFish
