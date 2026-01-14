/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Thread-Safe MCTS Implementation

  Licensed under GPL-3.0
*/

#include "thread_safe_mcts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "../core/movegen.h"
#include "../eval/evaluate.h"
#include "../uci/uci.h"

namespace MetalFish {
namespace MCTS {

// ============================================================================
// ThreadSafeNode Implementation
// ============================================================================

ThreadSafeNode::ThreadSafeNode(ThreadSafeNode* parent, int edge_idx)
    : parent_(parent), edge_index_(edge_idx) {}

void ThreadSafeNode::create_edges(const MoveList<LEGAL>& moves) {
  if (moves.size() == 0) return;
  
  int count = static_cast<int>(moves.size());
  edges_ = std::make_unique<TSEdge[]>(count);
  
  // Initialize with uniform policy
  float uniform = 1.0f / count;
  int idx = 0;
  for (const auto& m : moves) {
    edges_[idx].move = m;
    edges_[idx].policy.store(uniform, std::memory_order_relaxed);
    edges_[idx].child.store(nullptr, std::memory_order_relaxed);
    idx++;
  }
  
  // Publish edges (release ensures edges are visible before count)
  num_edges_.store(count, std::memory_order_release);
}

void ThreadSafeNode::update_stats(float value, float draw_prob, float moves_left) {
  // Lock-free update using fetch_add for w_ and n_
  // Then compute q_ = w_ / n_
  
  // Add value to sum
  float old_w = w_.load(std::memory_order_relaxed);
  float new_w;
  do {
    new_w = old_w + value;
  } while (!w_.compare_exchange_weak(old_w, new_w, 
                                      std::memory_order_release,
                                      std::memory_order_relaxed));
  
  // Increment visit count
  uint32_t new_n = n_.fetch_add(1, std::memory_order_acq_rel) + 1;
  
  // Update Q value (average)
  float new_q = new_w / new_n;
  q_.store(new_q, std::memory_order_release);
  
  // Update draw probability (exponential moving average)
  float old_d = d_.load(std::memory_order_relaxed);
  float new_d = old_d + (draw_prob - old_d) / new_n;
  d_.store(new_d, std::memory_order_release);
  
  // Update moves left estimate
  float old_m = m_.load(std::memory_order_relaxed);
  float new_m = old_m + (moves_left - old_m) / new_n;
  m_.store(new_m, std::memory_order_release);
}

void ThreadSafeNode::set_terminal(Terminal type, float value) {
  terminal_type_.store(type, std::memory_order_release);
  w_.store(value, std::memory_order_release);
  q_.store(value, std::memory_order_release);
  n_.store(1, std::memory_order_release);
}

// ============================================================================
// ThreadSafeTree Implementation
// ============================================================================

ThreadSafeTree::ThreadSafeTree() {
  root_ = std::make_unique<ThreadSafeNode>();
}

ThreadSafeTree::~ThreadSafeTree() = default;

void ThreadSafeTree::reset(const std::string& fen) {
  {
    std::unique_lock<std::shared_mutex> lock(fen_mutex_);
    root_fen_ = fen;
  }
  
  // Clear node pool
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    node_pool_.clear();
  }
  
  // Reset root
  root_ = std::make_unique<ThreadSafeNode>();
  node_count_.store(1, std::memory_order_relaxed);
}

ThreadSafeNode* ThreadSafeTree::allocate_node(ThreadSafeNode* parent, int edge_idx) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  node_pool_.push_back(std::make_unique<ThreadSafeNode>(parent, edge_idx));
  node_count_.fetch_add(1, std::memory_order_relaxed);
  return node_pool_.back().get();
}

// ============================================================================
// ThreadSafeMCTS Implementation
// ============================================================================

ThreadSafeMCTS::ThreadSafeMCTS(const ThreadSafeMCTSConfig& config)
    : config_(config), tree_(std::make_unique<ThreadSafeTree>()) {
  // Initialize TT
  tt_.resize(TT_SIZE);
}

ThreadSafeMCTS::~ThreadSafeMCTS() {
  stop();
  wait();
}

void ThreadSafeMCTS::start_search(const std::string& fen,
                                   const Search::LimitsType& limits,
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
  
  // Create worker contexts
  worker_contexts_.clear();
  for (int i = 0; i < config_.num_threads; ++i) {
    worker_contexts_.push_back(std::make_unique<WorkerContext>());
  }
  
  // Start worker threads
  workers_.clear();
  for (int i = 0; i < config_.num_threads; ++i) {
    workers_.emplace_back(&ThreadSafeMCTS::worker_thread, this, i);
  }
}

void ThreadSafeMCTS::stop() {
  stop_flag_.store(true, std::memory_order_release);
}

void ThreadSafeMCTS::wait() {
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
  running_.store(false, std::memory_order_release);
  
  // Report best move
  if (best_move_callback_) {
    Move best = get_best_move();
    std::vector<Move> pv = get_pv();
    Move ponder = pv.size() > 1 ? pv[1] : Move::none();
    best_move_callback_(best, ponder);
  }
}

bool ThreadSafeMCTS::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire)) return true;
  
  // Check time limit
  if (time_budget_ms_ > 0) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - search_start_).count();
    if (elapsed >= time_budget_ms_) return true;
  }
  
  // Check node limit
  if (limits_.nodes > 0 && stats_.total_nodes >= limits_.nodes) return true;
  
  return false;
}

int64_t ThreadSafeMCTS::calculate_time_budget() const {
  if (limits_.movetime > 0) return limits_.movetime;
  if (limits_.infinite) return 0;
  
  // Get time for our side
  Color us = WHITE;  // Will be determined from position
  int64_t time_left = limits_.time[us];
  int64_t inc = limits_.inc[us];
  
  if (time_left <= 0) return 1000;  // Default 1 second
  
  // Use ~2.5% of remaining time + increment
  return time_left / 40 + inc;
}

void ThreadSafeMCTS::worker_thread(int thread_id) {
  WorkerContext& ctx = *worker_contexts_[thread_id];
  std::string root_fen = tree_->root_fen();
  
  // Expand root node if needed (only one thread should do this)
  ThreadSafeNode* root = tree_->root();
  if (!root->has_children()) {
    std::lock_guard<std::mutex> lock(root->mutex());
    if (!root->has_children()) {
      ctx.reset_position(root_fen);
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
  
  // Main search loop
  while (!should_stop()) {
    run_iteration(ctx);
  }
  
  // Aggregate worker stats
  stats_.cache_hits.fetch_add(ctx.cache_hits, std::memory_order_relaxed);
  stats_.cache_misses.fetch_add(ctx.cache_misses, std::memory_order_relaxed);
}

void ThreadSafeMCTS::run_iteration(WorkerContext& ctx) {
  std::string root_fen = tree_->root_fen();
  ctx.reset_position(root_fen);
  
  auto iter_start = std::chrono::steady_clock::now();
  
  // 1. Selection - traverse to leaf
  auto select_start = std::chrono::steady_clock::now();
  ThreadSafeNode* leaf = select_leaf(ctx);
  auto select_end = std::chrono::steady_clock::now();
  
  if (!leaf) return;
  
  // 2. Check for terminal
  if (ctx.pos.checkers() == 0) {
    MoveList<LEGAL> moves(ctx.pos);
    if (moves.size() == 0) {
      // Stalemate
      leaf->set_terminal(ThreadSafeNode::Terminal::Draw, 0.0f);
      backpropagate(leaf, 0.0f, 1.0f, 0.0f);
      return;
    }
  } else {
    MoveList<LEGAL> moves(ctx.pos);
    if (moves.size() == 0) {
      // Checkmate
      leaf->set_terminal(ThreadSafeNode::Terminal::Loss, -1.0f);
      backpropagate(leaf, -1.0f, 0.0f, 0.0f);
      return;
    }
  }
  
  // 3. Expansion - add children if not expanded
  auto expand_start = std::chrono::steady_clock::now();
  if (!leaf->has_children()) {
    std::lock_guard<std::mutex> lock(leaf->mutex());
    if (!leaf->has_children()) {
      MoveList<LEGAL> moves(ctx.pos);
      leaf->create_edges(moves);
      expand_node(leaf, ctx);
    }
  }
  auto expand_end = std::chrono::steady_clock::now();
  
  // 4. Evaluation
  auto eval_start = std::chrono::steady_clock::now();
  float value = evaluate_position(ctx);
  auto eval_end = std::chrono::steady_clock::now();
  
  // 5. Backpropagation
  auto backprop_start = std::chrono::steady_clock::now();
  float draw = std::max(0.0f, 0.4f - std::abs(value) * 0.3f);
  backpropagate(leaf, value, draw, 30.0f);
  auto backprop_end = std::chrono::steady_clock::now();
  
  // Update profiling stats
  stats_.selection_time_us.fetch_add(
      std::chrono::duration_cast<std::chrono::microseconds>(select_end - select_start).count(),
      std::memory_order_relaxed);
  stats_.expansion_time_us.fetch_add(
      std::chrono::duration_cast<std::chrono::microseconds>(expand_end - expand_start).count(),
      std::memory_order_relaxed);
  stats_.evaluation_time_us.fetch_add(
      std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count(),
      std::memory_order_relaxed);
  stats_.backprop_time_us.fetch_add(
      std::chrono::duration_cast<std::chrono::microseconds>(backprop_end - backprop_start).count(),
      std::memory_order_relaxed);
  
  stats_.total_nodes.fetch_add(1, std::memory_order_relaxed);
  stats_.total_iterations.fetch_add(1, std::memory_order_relaxed);
  ctx.iterations++;
}

ThreadSafeNode* ThreadSafeMCTS::select_leaf(WorkerContext& ctx) {
  ThreadSafeNode* node = tree_->root();
  
  while (node->has_children() && !node->is_terminal()) {
    // Select best child using PUCT
    int best_idx = select_child_puct(node, config_.cpuct);
    if (best_idx < 0) break;
    
    TSEdge& edge = node->edges()[best_idx];
    
    // Get or create child node
    ThreadSafeNode* child = edge.child.load(std::memory_order_acquire);
    
    if (!child) {
      // Try to create child (use CAS to avoid race)
      std::lock_guard<std::mutex> lock(node->mutex());
      child = edge.child.load(std::memory_order_acquire);
      if (!child) {
        child = tree_->allocate_node(node, best_idx);
        edge.child.store(child, std::memory_order_release);
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

int ThreadSafeMCTS::select_child_puct(ThreadSafeNode* node, float cpuct) {
  int num_edges = node->num_edges();
  if (num_edges == 0) return -1;
  
  float parent_n = static_cast<float>(node->n() + node->n_in_flight());
  float parent_sqrt = std::sqrt(parent_n + 1.0f);
  
  float best_score = -1e9f;
  int best_idx = -1;
  
  const TSEdge* edges = node->edges();
  
  for (int i = 0; i < num_edges; ++i) {
    const TSEdge& edge = edges[i];
    ThreadSafeNode* child = edge.child.load(std::memory_order_acquire);
    
    float q, u;
    float policy = edge.policy.load(std::memory_order_relaxed);
    
    if (child) {
      uint32_t n = child->n();
      uint32_t n_in_flight = child->n_in_flight();
      float total_n = static_cast<float>(n + n_in_flight);
      
      if (n > 0) {
        q = -child->q();  // Negate for opponent's perspective
      } else {
        q = config_.fpu_value;  // First play urgency
      }
      
      u = cpuct * policy * parent_sqrt / (1.0f + total_n);
    } else {
      // Unvisited node
      q = config_.fpu_value;
      u = cpuct * policy * parent_sqrt;
    }
    
    float score = q + u;
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }
  
  return best_idx;
}

void ThreadSafeMCTS::expand_node(ThreadSafeNode* node, WorkerContext& ctx) {
  int num_edges = node->num_edges();
  if (num_edges == 0) return;
  
  TSEdge* edges = node->edges();
  std::vector<float> scores(num_edges);
  float max_score = -1e9f;
  
  // Score each move using heuristics
  for (int i = 0; i < num_edges; ++i) {
    Move m = edges[i].move;
    float score = 0.0f;
    
    // Captures scored by MVV-LVA and SEE
    if (ctx.pos.capture(m)) {
      PieceType captured = m.type_of() == EN_PASSANT 
          ? PAWN : type_of(ctx.pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(ctx.pos.piece_on(m.from_sq()));
      static const float piece_values[] = {0, 100, 320, 330, 500, 900, 0};
      score += piece_values[captured] * 6.0f - piece_values[attacker];
      
      if (ctx.pos.see_ge(m, Value(0))) {
        score += 300.0f;
      }
    }
    
    // Promotions
    if (m.type_of() == PROMOTION) {
      PieceType promo = m.promotion_type();
      if (promo == QUEEN) score += 4000.0f;
      else if (promo == KNIGHT) score += 800.0f;
    }
    
    // Checks
    if (ctx.pos.gives_check(m)) {
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
  
  // Softmax normalization
  float sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    scores[i] = std::exp((scores[i] - max_score) / (config_.policy_softmax_temp * 400.0f));
    sum += scores[i];
  }
  
  // Set policy priors
  for (int i = 0; i < num_edges; ++i) {
    edges[i].policy.store(scores[i] / sum, std::memory_order_release);
  }
}

void ThreadSafeMCTS::add_dirichlet_noise(ThreadSafeNode* root) {
  int num_edges = root->num_edges();
  if (num_edges == 0) return;
  
  TSEdge* edges = root->edges();
  
  // Generate Dirichlet noise
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
  
  std::vector<float> noise(num_edges);
  float noise_sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    noise[i] = gamma(gen);
    noise_sum += noise[i];
  }
  
  // Mix with existing policy
  for (int i = 0; i < num_edges; ++i) {
    float current = edges[i].policy.load(std::memory_order_relaxed);
    float noisy = (1.0f - config_.dirichlet_epsilon) * current +
                  config_.dirichlet_epsilon * (noise[i] / noise_sum);
    edges[i].policy.store(noisy, std::memory_order_release);
  }
}

float ThreadSafeMCTS::evaluate_position(WorkerContext& ctx) {
  // Check TT first
  uint64_t key = ctx.pos.key();
  size_t tt_idx = key % TT_SIZE;
  
  TTEntry& entry = tt_[tt_idx];
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
    value = std::tanh(score / 400.0f);
  } else {
    // Fallback to simple eval
    int simple = Eval::simple_eval(ctx.pos);
    value = std::tanh(simple / 400.0f);
  }
  
  // Adjust for side to move
  if (ctx.pos.side_to_move() == BLACK) {
    value = -value;
  }
  
  // Store in TT
  entry.key = key;
  entry.value = value;
  entry.draw = 0.0f;
  entry.moves_left = 30.0f;
  
  stats_.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
  
  return value;
}

void ThreadSafeMCTS::backpropagate(ThreadSafeNode* node, float value, 
                                    float draw, float moves_left) {
  while (node) {
    // Remove virtual loss
    node->remove_virtual_loss(config_.virtual_loss);
    
    // Update statistics
    node->update_stats(value, draw, moves_left);
    
    // Flip value for parent (opponent's perspective)
    value = -value;
    moves_left += 1.0f;
    
    node = node->parent();
  }
}

Move ThreadSafeMCTS::get_best_move() const {
  const ThreadSafeNode* root = tree_->root();
  if (!root->has_children()) return Move::none();
  
  int num_edges = root->num_edges();
  const TSEdge* edges = root->edges();
  
  int best_idx = -1;
  uint32_t best_n = 0;
  
  for (int i = 0; i < num_edges; ++i) {
    ThreadSafeNode* child = edges[i].child.load(std::memory_order_acquire);
    if (child && child->n() > best_n) {
      best_n = child->n();
      best_idx = i;
    }
  }
  
  if (best_idx < 0) {
    // No visits, return first move
    return edges[0].move;
  }
  
  return edges[best_idx].move;
}

std::vector<Move> ThreadSafeMCTS::get_pv() const {
  std::vector<Move> pv;
  const ThreadSafeNode* node = tree_->root();
  
  while (node && node->has_children()) {
    int num_edges = node->num_edges();
    const TSEdge* edges = node->edges();
    
    int best_idx = -1;
    uint32_t best_n = 0;
    
    for (int i = 0; i < num_edges; ++i) {
      ThreadSafeNode* child = edges[i].child.load(std::memory_order_acquire);
      if (child && child->n() > best_n) {
        best_n = child->n();
        best_idx = i;
      }
    }
    
    if (best_idx < 0) break;
    
    pv.push_back(edges[best_idx].move);
    node = edges[best_idx].child.load(std::memory_order_acquire);
  }
  
  return pv;
}

float ThreadSafeMCTS::get_best_q() const {
  const ThreadSafeNode* root = tree_->root();
  if (!root->has_children()) return 0.0f;
  
  int num_edges = root->num_edges();
  const TSEdge* edges = root->edges();
  
  int best_idx = -1;
  uint32_t best_n = 0;
  
  for (int i = 0; i < num_edges; ++i) {
    ThreadSafeNode* child = edges[i].child.load(std::memory_order_acquire);
    if (child && child->n() > best_n) {
      best_n = child->n();
      best_idx = i;
    }
  }
  
  if (best_idx < 0) return 0.0f;
  
  ThreadSafeNode* best_child = edges[best_idx].child.load(std::memory_order_acquire);
  return best_child ? -best_child->q() : 0.0f;
}

void ThreadSafeMCTS::send_info() {
  if (!info_callback_) return;
  
  auto now = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - search_start_).count();
  
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
    for (const Move& m : pv) {
      ss << " " << UCIEngine::move(m, false);
    }
  }
  
  info_callback_(ss.str());
}

std::unique_ptr<ThreadSafeMCTS> create_thread_safe_mcts(
    GPU::GPUNNUEManager* gpu_manager,
    const ThreadSafeMCTSConfig& config) {
  auto mcts = std::make_unique<ThreadSafeMCTS>(config);
  mcts->set_gpu_manager(gpu_manager);
  return mcts;
}

} // namespace MCTS
} // namespace MetalFish
