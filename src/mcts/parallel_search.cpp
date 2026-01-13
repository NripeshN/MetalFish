/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Multi-threaded MCTS Search Worker - Implementation

  Licensed under GPL-3.0
*/

#include "parallel_search.h"
#include "hybrid_search.h"
#include "../core/movegen.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// ParallelGPUEvaluator Implementation
// ============================================================================

ParallelGPUEvaluator::ParallelGPUEvaluator() = default;

ParallelGPUEvaluator::~ParallelGPUEvaluator() {
  stop();
}

bool ParallelGPUEvaluator::initialize(GPU::GPUNNUEManager* gpu_manager,
                                      const ParallelEvalConfig& config) {
  if (!gpu_manager || !gpu_manager->is_ready()) {
    return false;
  }
  
  gpu_manager_ = gpu_manager;
  config_ = config;
  
  // Allocate persistent batches
  persistent_batches_.resize(config.num_eval_threads);
  for (int i = 0; i < config.num_eval_threads; ++i) {
    persistent_batches_[i] = std::make_unique<PersistentBatch>();
    persistent_batches_[i]->reserve(config.max_batch_size);
  }
  
  // Start evaluation threads
  stop_flag_ = false;
  running_ = true;
  for (int i = 0; i < config.num_eval_threads; ++i) {
    eval_threads_.emplace_back(&ParallelGPUEvaluator::eval_thread_main, this, i);
  }
  
  initialized_ = true;
  return true;
}

void ParallelGPUEvaluator::submit(const EvalRequest& request) {
  if (!initialized_) return;
  
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    request_queue_.push(request);
  }
  request_cv_.notify_one();
}

bool ParallelGPUEvaluator::get_result(EvalResult& result) {
  std::lock_guard<std::mutex> lock(result_mutex_);
  if (result_queue_.empty()) {
    return false;
  }
  result = std::move(result_queue_.front());
  result_queue_.pop();
  return true;
}

void ParallelGPUEvaluator::flush() {
  // Wait for all pending requests to be processed
  std::unique_lock<std::mutex> lock(request_mutex_);
  while (!request_queue_.empty() && running_) {
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    lock.lock();
  }
}

void ParallelGPUEvaluator::stop() {
  stop_flag_ = true;
  request_cv_.notify_all();
  
  for (auto& thread : eval_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  eval_threads_.clear();
  
  running_ = false;
  initialized_ = false;
}

void ParallelGPUEvaluator::eval_thread_main(int thread_id) {
  auto& batch = *persistent_batches_[thread_id];
  
  while (!stop_flag_) {
    // Collect batch
    {
      std::unique_lock<std::mutex> lock(request_mutex_);
      
      // Wait for requests or timeout
      auto timeout = std::chrono::microseconds(config_.batch_timeout_us);
      request_cv_.wait_for(lock, timeout, [this] {
        return !request_queue_.empty() || stop_flag_;
      });
      
      if (stop_flag_) break;
      
      // Collect up to max_batch_size requests
      batch.clear();
      while (!request_queue_.empty() && batch.size() < config_.max_batch_size) {
        batch.requests.push_back(std::move(request_queue_.front()));
        request_queue_.pop();
      }
    }
    
    // Process batch if we have enough requests
    if (batch.size() >= config_.min_batch_size) {
      process_batch(batch);
    } else if (batch.size() > 0) {
      // Put requests back if batch is too small
      std::lock_guard<std::mutex> lock(request_mutex_);
      for (auto& req : batch.requests) {
        request_queue_.push(std::move(req));
      }
    }
  }
}

void ParallelGPUEvaluator::process_batch(PersistentBatch& batch) {
  if (batch.size() == 0 || !gpu_manager_) return;
  
  // Prepare GPU batch
  batch.gpu_batch.clear();
  batch.gpu_batch.count = batch.size();
  
  // Set up positions from FEN strings
  std::deque<StateInfo> states(batch.size());
  std::vector<Position> positions(batch.size());
  
  for (int i = 0; i < batch.size(); ++i) {
    positions[i].set(batch.requests[i].fen, false, &states[i]);
    batch.gpu_batch.add_position(positions[i]);
  }
  
  // Evaluate on GPU
  bool success = gpu_manager_->evaluate_batch(batch.gpu_batch, true);
  
  if (success) {
    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_evals_.fetch_add(batch.size(), std::memory_order_relaxed);
    
    // Create results
    std::lock_guard<std::mutex> lock(result_mutex_);
    for (int i = 0; i < batch.size(); ++i) {
      EvalResult result;
      result.node = batch.requests[i].node;
      
      // Convert NNUE score to value in [-1, 1]
      int score = batch.gpu_batch.positional_scores[i];
      result.value = std::tanh(static_cast<float>(score) / 400.0f);
      
      // Adjust for side to move
      if (positions[i].side_to_move() == BLACK) {
        result.value = -result.value;
      }
      
      // Estimate draw probability
      result.draw_prob = std::max(0.0f, 0.4f - std::abs(result.value) * 0.3f);
      result.moves_left = 30.0f;
      
      // Generate policy from move ordering heuristics
      MoveList<LEGAL> moves(positions[i]);
      float total_score = 0.0f;
      std::vector<std::pair<Move, float>> scored_moves;
      
      for (const auto& m : moves) {
        float move_score = 1.0f;
        
        // Bonus for captures (MVV-LVA style)
        if (positions[i].capture(m)) {
          PieceType captured = type_of(positions[i].piece_on(m.to_sq()));
          PieceType attacker = type_of(positions[i].piece_on(m.from_sq()));
          static const float piece_values[] = {0, 1, 3, 3, 5, 9, 0};
          move_score += piece_values[captured] * 10.0f - piece_values[attacker];
        }
        
        // Bonus for promotions
        if (m.type_of() == PROMOTION) {
          move_score += 50.0f;
        }
        
        // Bonus for checks
        if (positions[i].gives_check(m)) {
          move_score += 20.0f;
        }
        
        // Center control bonus
        int to_file = file_of(m.to_sq());
        int to_rank = rank_of(m.to_sq());
        float center_dist = std::abs(to_file - 3.5f) + std::abs(to_rank - 3.5f);
        move_score += (7.0f - center_dist) * 0.5f;
        
        scored_moves.emplace_back(m, move_score);
        total_score += move_score;
      }
      
      // Normalize to probabilities
      if (total_score > 0) {
        for (auto& [move, score] : scored_moves) {
          result.policy.emplace_back(move, score / total_score);
        }
      }
      
      result_queue_.push(std::move(result));
    }
  }
  
  result_cv_.notify_all();
}

// ============================================================================
// SearchWorker Implementation
// ============================================================================

SearchWorker::SearchWorker(int id, HybridTree* tree, ParallelGPUEvaluator* evaluator,
                           MCTSTranspositionTable* tt, const HybridSearchConfig& config)
    : id_(id), tree_(tree), evaluator_(evaluator), tt_(tt), config_(config) {}

SearchWorker::~SearchWorker() {
  stop();
  wait();
}

void SearchWorker::start(const std::string& root_fen) {
  context_.reset(root_fen);
  stop_flag_ = false;
  running_ = true;
  thread_ = std::thread(&SearchWorker::worker_main, this);
}

void SearchWorker::stop() {
  stop_flag_ = true;
}

void SearchWorker::wait() {
  if (thread_.joinable()) {
    thread_.join();
  }
  running_ = false;
}

void SearchWorker::worker_main() {
  // Thread-local position and state stack
  std::vector<StateInfo> states(256);
  Position pos;
  
  while (!stop_flag_) {
    // Reset position to root
    pos.set(context_.root_fen, false, &states[0]);
    
    // Select and expand
    HybridNode* leaf = select_and_expand(pos, states.data());
    
    if (!leaf) {
      // No valid node found, yield
      std::this_thread::yield();
      continue;
    }
    
    // Evaluate
    float value = evaluate_position(pos);
    
    // Backpropagate
    backpropagate(leaf, value, 0.3f, 30.0f);
    
    context_.nodes_searched++;
  }
}

HybridNode* SearchWorker::select_and_expand(Position& pos, StateInfo* states) {
  HybridNode* node = tree_->root();
  if (!node) return nullptr;
  
  int depth = 0;
  
  while (node->has_children() && !node->is_terminal()) {
    // Calculate parent N sqrt for PUCT
    float parent_n_sqrt = std::sqrt(static_cast<float>(node->n() + 1));
    
    // Find best child according to PUCT
    int best_idx = -1;
    float best_puct = -std::numeric_limits<float>::infinity();
    
    float fpu = config_.fpu_value;
    if (node->n() > 0) {
      fpu = node->q() - config_.fpu_reduction * std::sqrt(static_cast<float>(node->n()));
    }
    
    const auto& edges = node->edges();
    for (size_t i = 0; i < edges.size(); ++i) {
      const auto& edge = edges[i];
      HybridNode* child = edge.child();
      
      float puct;
      if (child) {
        // UCB formula with virtual loss
        uint32_t n = child->n() + child->n_in_flight();
        float q = (n > 0) ? child->q() : fpu;
        float u = config_.cpuct * edge.policy() * parent_n_sqrt / (1.0f + n);
        puct = q + u;
      } else {
        // Unexpanded node - use FPU
        float u = config_.cpuct * edge.policy() * parent_n_sqrt;
        puct = fpu + u;
      }
      
      if (puct > best_puct) {
        best_puct = puct;
        best_idx = static_cast<int>(i);
      }
    }
    
    if (best_idx < 0) break;
    
    // Get or create child node
    auto& edge = node->edges()[best_idx];
    HybridNode* child = edge.child();
    
    if (!child) {
      // Create new node (with synchronization)
      std::lock_guard<std::mutex> lock(node->mutex());
      child = edge.child();  // Double-check
      if (!child) {
        child = tree_->allocate_node(node, best_idx);
        edge.set_child(child);
        
        // Expand the new node
        MoveList<LEGAL> moves(pos);
        std::vector<MCTSMove> mcts_moves;
        for (const auto& m : moves) {
          mcts_moves.push_back(MCTSMove::FromStockfish(m));
        }
        child->create_edges(mcts_moves);
        
        // Generate policy
        generate_policy(child, pos);
      }
    }
    
    // Apply move
    Move m = edge.move().to_stockfish();
    pos.do_move(m, states[depth + 1]);
    depth++;
    
    // Add virtual loss
    child->add_virtual_loss();
    
    node = child;
  }
  
  return node;
}

void SearchWorker::backpropagate(HybridNode* node, float value, float draw_prob, float moves_left) {
  while (node) {
    node->remove_virtual_loss();
    node->update_stats(value, draw_prob, moves_left);
    node->add_visit();
    
    // Flip value for opponent
    value = -value;
    moves_left += 1.0f;
    
    // Store in TT
    if (tt_) {
      // Use node address as a simple hash (not ideal but functional)
      uint64_t hash = reinterpret_cast<uint64_t>(node);
      tt_->update_mcts(hash, value, draw_prob, moves_left);
    }
    
    node = node->parent();
  }
}

float SearchWorker::evaluate_position(const Position& pos) {
  // Check TT first
  if (tt_) {
    uint64_t hash = pos.key();
    MCTSStats stats;
    if (tt_->probe_mcts(hash, stats) && stats.n > 0) {
      context_.cache_hits++;
      return stats.q;
    }
  }
  
  context_.cache_misses++;
  
  // Use GPU evaluator if available
  if (evaluator_ && evaluator_->is_running()) {
    EvalRequest request;
    request.node = nullptr;  // We'll handle result inline
    request.fen = pos.fen();
    request.thread_id = id_;
    
    evaluator_->submit(request);
    
    // Wait for result (blocking for now)
    EvalResult result;
    while (!evaluator_->get_result(result)) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    // Store in TT
    if (tt_) {
      tt_->update_mcts(pos.key(), result.value, result.draw_prob, result.moves_left);
    }
    
    return result.value;
  }
  
  // Fallback: simple material evaluation
  int material = 0;
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece p = pos.piece_on(s);
    if (p != NO_PIECE) {
      static const int piece_values[] = {0, 100, 320, 330, 500, 900, 0};
      int value = piece_values[type_of(p)];
      material += (color_of(p) == WHITE) ? value : -value;
    }
  }
  
  float value = std::tanh(material / 1000.0f);
  if (pos.side_to_move() == BLACK) {
    value = -value;
  }
  
  return value;
}

void SearchWorker::generate_policy(HybridNode* node, const Position& pos) {
  if (!node->has_children()) return;
  
  auto& edges = node->edges();
  float total_score = 0.0f;
  std::vector<float> scores(edges.size());
  
  for (size_t i = 0; i < edges.size(); ++i) {
    Move m = edges[i].move().to_stockfish();
    float score = 1.0f;
    
    // Capture bonus
    if (pos.capture(m)) {
      PieceType captured = type_of(pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(pos.piece_on(m.from_sq()));
      static const float piece_values[] = {0, 1, 3, 3, 5, 9, 0};
      score += piece_values[captured] * 10.0f - piece_values[attacker];
    }
    
    // Promotion bonus
    if (m.type_of() == PROMOTION) {
      score += 50.0f;
    }
    
    // Check bonus
    if (pos.gives_check(m)) {
      score += 20.0f;
    }
    
    scores[i] = score;
    total_score += score;
  }
  
  // Normalize and set policies
  if (total_score > 0) {
    for (size_t i = 0; i < edges.size(); ++i) {
      edges[i].set_policy(scores[i] / total_score);
    }
  }
}

// ============================================================================
// ParallelSearchManager Implementation
// ============================================================================

ParallelSearchManager::ParallelSearchManager() = default;

ParallelSearchManager::~ParallelSearchManager() {
  stop();
}

bool ParallelSearchManager::initialize(GPU::GPUNNUEManager* gpu_manager, int num_threads) {
  gpu_manager_ = gpu_manager;
  num_threads_ = num_threads;
  
  // Initialize GPU evaluator
  ParallelEvalConfig eval_config;
  eval_config.batch_size = 64;
  eval_config.max_batch_size = 256;
  eval_config.min_batch_size = 1;
  eval_config.batch_timeout_us = 500;
  eval_config.num_eval_threads = 1;
  
  evaluator_ = std::make_unique<ParallelGPUEvaluator>();
  if (!evaluator_->initialize(gpu_manager, eval_config)) {
    return false;
  }
  
  initialized_ = true;
  return true;
}

void ParallelSearchManager::start_search(HybridTree* tree, const std::string& root_fen,
                                         const HybridSearchConfig& config) {
  if (!initialized_) return;
  
  // Stop any existing search
  stop();
  
  // Create workers
  workers_.clear();
  for (int i = 0; i < num_threads_; ++i) {
    workers_.push_back(std::make_unique<SearchWorker>(
        i, tree, evaluator_.get(), &mcts_tt(), config));
    workers_.back()->start(root_fen);
  }
}

void ParallelSearchManager::stop() {
  for (auto& worker : workers_) {
    worker->stop();
  }
}

void ParallelSearchManager::wait() {
  for (auto& worker : workers_) {
    worker->wait();
  }
  workers_.clear();
}

uint64_t ParallelSearchManager::total_nodes() const {
  uint64_t total = 0;
  for (const auto& worker : workers_) {
    total += worker->nodes_searched();
  }
  return total;
}

ParallelSearchManager::Stats ParallelSearchManager::get_stats() const {
  Stats stats;
  for (const auto& worker : workers_) {
    stats.total_nodes += worker->nodes_searched();
    stats.cache_hits += worker->cache_hits();
    stats.cache_misses += worker->cache_misses();
  }
  if (evaluator_) {
    stats.gpu_evals = evaluator_->total_evaluations();
    stats.gpu_batches = evaluator_->total_batches();
    stats.avg_batch_size = evaluator_->avg_batch_size();
  }
  return stats;
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<ParallelSearchManager> g_parallel_search_manager;

ParallelSearchManager& parallel_search_manager() {
  if (!g_parallel_search_manager) {
    g_parallel_search_manager = std::make_unique<ParallelSearchManager>();
  }
  return *g_parallel_search_manager;
}

bool initialize_parallel_search(GPU::GPUNNUEManager* gpu_manager, int num_threads) {
  return parallel_search_manager().initialize(gpu_manager, num_threads);
}

}  // namespace MCTS
}  // namespace MetalFish
