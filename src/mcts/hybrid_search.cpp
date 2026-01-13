/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Implementation of the Hybrid MCTS + Alpha-Beta search.

  Licensed under GPL-3.0
*/

#include "hybrid_search.h"
#include "../eval/evaluate.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// ============================================================================
// HybridNode Implementation
// ============================================================================

HybridNode::HybridNode(HybridNode *parent, int edge_index)
    : parent_(parent), edge_index_(edge_index) {}

HybridNode::~HybridNode() {
  // Children are NOT deleted here - they are owned by node_pool_ in HybridTree.
  // The node_pool_ uses unique_ptr to manage all allocated nodes, so manual
  // deletion here would cause double-free when both the destructor and the
  // unique_ptr try to delete the same node.
}

void HybridNode::create_edges(const MCTSMoveList &moves) {
  edges_.clear();
  edges_.reserve(moves.size());
  for (const auto &move : moves) {
    edges_.emplace_back(move);
  }
}

void HybridNode::add_visit() { n_.fetch_add(1, std::memory_order_relaxed); }

void HybridNode::add_virtual_loss() {
  n_in_flight_.fetch_add(1, std::memory_order_relaxed);
}

void HybridNode::remove_virtual_loss() {
  n_in_flight_.fetch_sub(1, std::memory_order_relaxed);
}

void HybridNode::update_stats(float value, float draw_prob, float moves_left) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t n = n_.load(std::memory_order_relaxed);
  if (n == 0) {
    q_ = value;
    d_ = draw_prob;
    m_ = moves_left;
  } else {
    // Incremental update
    float n_f = static_cast<float>(n);
    q_ = (q_ * n_f + value) / (n_f + 1.0f);
    d_ = (d_ * n_f + draw_prob) / (n_f + 1.0f);
    m_ = (m_ * n_f + moves_left) / (n_f + 1.0f);
  }
  // Increment visit count atomically while still holding the mutex.
  // This ensures that n_ and the statistics (q_, d_, m_) are always consistent
  // when read by other threads, preventing race conditions where another thread
  // could read updated statistics but a stale visit count.
  n_.fetch_add(1, std::memory_order_relaxed);
}

void HybridNode::set_terminal(Terminal type, float value) {
  terminal_type_ = type;
  q_ = value;
  d_ = (type == Terminal::Draw) ? 1.0f : 0.0f;
  m_ = 0.0f;
}

void HybridNode::set_ab_score(int score, int depth) {
  std::lock_guard<std::mutex> lock(mutex_);
  has_ab_score_ = true;
  ab_score_ = score;
  ab_depth_ = depth;
}

float HybridNode::get_u(float cpuct, float parent_n_sqrt) const {
  uint32_t n = n_.load(std::memory_order_relaxed);
  uint32_t n_in_flight = n_in_flight_.load(std::memory_order_relaxed);
  return cpuct * parent_n_sqrt / (1.0f + n + n_in_flight);
}

float HybridNode::get_puct(float cpuct, float parent_n_sqrt, float fpu) const {
  uint32_t n = n_.load(std::memory_order_relaxed);
  float q = (n > 0) ? q_ : fpu;
  float u = get_u(cpuct, parent_n_sqrt);

  // Get policy from parent edge
  float p = 1.0f; // Default uniform
  if (parent_ && edge_index_ >= 0 &&
      edge_index_ < static_cast<int>(parent_->edges_.size())) {
    p = parent_->edges_[edge_index_].policy();
  }

  return q + u * p;
}

// ============================================================================
// HybridTree Implementation
// ============================================================================

HybridTree::HybridTree() : root_(std::make_unique<HybridNode>()) {}

HybridTree::~HybridTree() = default;

void HybridTree::reset(const MCTSPositionHistory &history) {
  // Clear the node pool first (this deletes all allocated nodes)
  node_pool_.clear();
  root_ = std::make_unique<HybridNode>();
  history_ = history;
  node_count_ = 1;
}

bool HybridTree::apply_move(MCTSMove move) {
  // Tree reuse is disabled because child nodes are owned by node_pool_,
  // not by their parent nodes. Extracting a subtree would require complex
  // ownership transfer from node_pool_. For simplicity, we always reset.
  // This is a common trade-off in MCTS implementations.
  history_.do_move(move);
  node_pool_.clear();
  root_ = std::make_unique<HybridNode>();
  node_count_ = 1;
  return false;
}

HybridNode *HybridTree::allocate_node(HybridNode *parent, int edge_index) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  node_pool_.push_back(std::make_unique<HybridNode>(parent, edge_index));
  node_count_.fetch_add(1, std::memory_order_relaxed);
  return node_pool_.back().get();
}

// ============================================================================
// HybridSearch Implementation
// ============================================================================

HybridSearch::HybridSearch(const HybridSearchConfig &config)
    : config_(config), tree_(std::make_unique<HybridTree>()) {}

HybridSearch::~HybridSearch() {
  stop();
  wait();
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
  // Stop any existing search
  stop();
  wait();

  // Reset state
  stats_.reset();
  stop_flag_ = false;
  running_ = true;
  limits_ = limits;
  best_move_callback_ = best_move_cb;
  info_callback_ = info_cb;
  search_start_ = std::chrono::steady_clock::now();

  // Initialize tree
  tree_->reset(history);

  // Start search threads
  for (int i = 0; i < config_.num_search_threads; ++i) {
    search_threads_.emplace_back(&HybridSearch::search_thread_main, this);
  }

  // Don't start eval thread - we're doing direct evaluation now
  // eval_thread_ = std::thread(&HybridSearch::eval_thread_main, this);
}

void HybridSearch::stop() {
  stop_flag_ = true;
  batch_cv_.notify_all();
  results_cv_.notify_all();
}

void HybridSearch::wait() {
  for (auto &thread : search_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  search_threads_.clear();

  if (eval_thread_.joinable()) {
    eval_thread_.join();
  }

  running_ = false;
}

MCTSMove HybridSearch::get_best_move() const { return select_best_move(); }

float HybridSearch::get_best_move_q() const {
  const HybridNode *root = tree_->root();
  if (!root || !root->has_children())
    return 0.0f;

  // Find child with most visits (same logic as select_best_move)
  int best_idx = -1;
  uint32_t best_n = 0;

  for (size_t i = 0; i < root->edges().size(); ++i) {
    const auto &edge = root->edges()[i];
    if (edge.child() && edge.child()->n() > best_n) {
      best_n = edge.child()->n();
      best_idx = static_cast<int>(i);
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

    for (size_t i = 0; i < node->edges().size(); ++i) {
      const auto &edge = node->edges()[i];
      if (edge.child() && edge.child()->n() > best_n) {
        best_n = edge.child()->n();
        best_idx = static_cast<int>(i);
      }
    }

    if (best_idx < 0)
      break;

    pv.push_back(node->edges()[best_idx].move());
    node = node->edges()[best_idx].child();
  }

  return pv;
}

void HybridSearch::search_thread_main() {
  MCTSPosition pos = tree_->history().current();

  // Expand root node first (with synchronization)
  HybridNode *root = tree_->root();
  if (!root) {
    if (best_move_callback_) {
      best_move_callback_(MCTSMove(), MCTSMove());
    }
    return;
  }

  // Synchronize root expansion across threads
  {
    std::lock_guard<std::mutex> lock(root->mutex());
    if (!root->has_children()) {
      expand_node(root, pos);
      // Uniform policy for initial expansion (fast path)
      int num_edges = static_cast<int>(root->edges().size());
      float uniform_policy = 1.0f / std::max(1, num_edges);
      for (auto &edge : root->edges()) {
        edge.set_policy(uniform_policy);
      }
    }
  }

  int iterations = 0;
  while (!should_stop()) {
    // Selection: traverse tree to find leaf
    HybridNode *node = tree_->root();
    MCTSPosition search_pos = pos;

    node = select_node(node, search_pos);

    if (!node) {
      iterations++;
      if (iterations > 100)
        break; // Safety break
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
      continue;
    }

    // Expansion: add children (with per-node locking)
    if (!node->has_children()) {
      std::lock_guard<std::mutex> lock(node->mutex());
      // Double check after acquiring lock
      if (!node->has_children()) {
        expand_node(node, search_pos);
        // Uniform policy for fast expansion
        int num_edges = static_cast<int>(node->edges().size());
        float uniform_policy = 1.0f / std::max(1, num_edges);
        for (auto &edge : node->edges()) {
          edge.set_policy(uniform_policy);
        }
      }
    }

    // Check TT for cached evaluation (fast path)
    float value = 0.0f;
    float draw = 0.0f;
    float moves_left = 30.0f;

    uint64_t hash = search_pos.hash();
    MCTSStats cached_stats;
    bool found_in_tt = mcts_tt().probe_mcts(hash, cached_stats);

    if (found_in_tt && cached_stats.n > 0) {
      // Use cached value
      value = cached_stats.q;
      draw = cached_stats.d;
      moves_left = cached_stats.m;
      stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      // Evaluate using GPU NNUE
      if (gpu_nnue_) {
        auto [psqt, score] =
            gpu_nnue_->evaluate_single(search_pos.stockfish_position(), true);
        // Convert centipawn score to value in [-1, 1]
        value = std::tanh(score / 400.0f);
        if (search_pos.is_black_to_move()) {
          value = -value;
        }
        // Estimate draw probability from score magnitude
        draw = std::max(0.0f, 0.4f - std::abs(value) * 0.3f);
      }

      // Store in TT
      mcts_tt().update_mcts(hash, value, draw, moves_left);
      stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
    }

    backpropagate(node, value, draw, moves_left);
    stats_.mcts_nodes.fetch_add(1, std::memory_order_relaxed);
    iterations++;
  }

  // Report best move when done
  if (best_move_callback_) {
    MCTSMove best = select_best_move();
    std::vector<MCTSMove> pv = get_pv();
    MCTSMove ponder = (pv.size() > 1) ? pv[1] : MCTSMove();
    best_move_callback_(best, ponder);
  }
}

void HybridSearch::eval_thread_main() {
  while (!stop_flag_) {
    EvalBatch batch;

    // Wait for batch or timeout
    {
      std::unique_lock<std::mutex> lock(batch_mutex_);
      batch_cv_.wait_for(
          lock, std::chrono::microseconds(config_.batch_timeout_us),
          [this] { return !current_batch_.empty() || stop_flag_; });

      if (stop_flag_ && current_batch_.empty())
        break;

      // Take the batch
      batch = std::move(current_batch_);
      current_batch_.clear();
    }

    if (batch.empty())
      continue;

    // Evaluate batch using GPU NNUE
    if (gpu_nnue_ && !batch.positions.empty()) {
      // Create GPU eval batch
      GPU::GPUEvalBatch gpu_batch;
      gpu_batch.count = static_cast<int>(batch.positions.size());
      gpu_batch.positions.resize(gpu_batch.count);
      gpu_batch.buckets.resize(gpu_batch.count, 0);

      // Convert positions to GPU format
      for (size_t i = 0; i < batch.positions.size(); ++i) {
        const auto &pos = batch.positions[i].stockfish_position();
        auto &gpu_pos = gpu_batch.positions[i];

        // Fill piece bitboards
        for (int c = 0; c < 2; ++c) {
          for (int pt = 0; pt < 7; ++pt) {
            gpu_pos.pieces[c][pt] = pos.pieces(Color(c), PieceType(pt));
          }
        }
        gpu_pos.king_sq[0] = pos.square<KING>(WHITE);
        gpu_pos.king_sq[1] = pos.square<KING>(BLACK);
        gpu_pos.stm = pos.side_to_move();
        gpu_pos.piece_count = popcount(pos.pieces());
      }

      // Evaluate on GPU
      bool success = gpu_nnue_->evaluate_batch(gpu_batch, true);

      if (success) {
        stats_.nn_batches.fetch_add(1, std::memory_order_relaxed);
        stats_.nn_evaluations.fetch_add(batch.size(),
                                        std::memory_order_relaxed);

        // Process results
        for (size_t i = 0; i < batch.nodes.size(); ++i) {
          HybridNode *node = batch.nodes[i];

          // Convert NNUE score to probability
          int score = gpu_batch.positional_scores[i];
          float value =
              std::tanh(static_cast<float>(score) / 400.0f); // Normalize

          // Set uniform policy for now (GPU NNUE doesn't provide policy)
          if (node->has_children()) {
            float uniform_p = 1.0f / node->edges().size();
            for (auto &edge : node->edges()) {
              edge.set_policy(uniform_p);
            }
          }

          // Backpropagate
          backpropagate(node, value, 0.3f,
                        30.0f); // Default draw prob and moves left
        }
      }
    }
    // Fallback to neural network if available
    else if (neural_network_) {
      std::vector<const MCTSPosition *> pos_ptrs;
      for (const auto &pos : batch.positions) {
        pos_ptrs.push_back(&pos);
      }

      auto results = neural_network_->evaluate_batch(pos_ptrs);

      stats_.nn_batches.fetch_add(1, std::memory_order_relaxed);
      stats_.nn_evaluations.fetch_add(batch.size(), std::memory_order_relaxed);

      // Process results
      for (size_t i = 0; i < batch.nodes.size(); ++i) {
        HybridNode *node = batch.nodes[i];
        const auto &eval = results[i];

        // Set policy
        if (node->has_children() && !eval.policy.empty()) {
          for (auto &edge : node->edges()) {
            for (const auto &[move, prob] : eval.policy) {
              if (edge.move() == move) {
                edge.set_policy(prob);
                break;
              }
            }
          }
        }

        // Backpropagate
        backpropagate(node, eval.q, eval.wdl[1], eval.m);
      }
    }
    // Last resort: use simple material evaluation
    else {
      for (size_t i = 0; i < batch.nodes.size(); ++i) {
        HybridNode *node = batch.nodes[i];
        const auto &pos = batch.positions[i];

        // Simple material evaluation
        Value v = Eval::simple_eval(pos.stockfish_position());
        float value = std::tanh(static_cast<float>(v) / 400.0f);

        // Uniform policy
        if (node->has_children()) {
          float uniform_p = 1.0f / node->edges().size();
          for (auto &edge : node->edges()) {
            edge.set_policy(uniform_p);
          }
        }

        backpropagate(node, value, 0.3f, 30.0f);
      }
    }
  }
}

HybridNode *HybridSearch::select_node(HybridNode *node, MCTSPosition &pos) {
  while (node->has_children() && !node->is_terminal()) {
    // Calculate parent N sqrt for PUCT
    float parent_n_sqrt = std::sqrt(static_cast<float>(node->n() + 1));

    // Find best child according to PUCT
    int best_idx = -1;
    float best_puct = -std::numeric_limits<float>::infinity();

    float fpu = config_.fpu_value;
    if (node->n() > 0) {
      fpu = node->q() - config_.fpu_reduction * std::sqrt(node->n());
    }

    for (size_t i = 0; i < node->edges().size(); ++i) {
      const auto &edge = node->edges()[i];
      HybridNode *child = edge.child();

      float puct;
      if (child) {
        puct = child->get_puct(config_.cpuct, parent_n_sqrt, fpu);
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

    if (best_idx < 0)
      break;

    // Get or create child node (with synchronization)
    auto &edge = node->edges()[best_idx];
    HybridNode *child = edge.child();

    if (!child) {
      // Use parent's mutex to protect child creation
      std::lock_guard<std::mutex> lock(node->mutex());
      child = edge.child(); // Double-check after lock
      if (!child) {
        child = tree_->allocate_node(node, best_idx);
        edge.set_child(child);
      }
    }

    // Apply move
    pos.do_move(edge.move());

    // Add virtual loss
    child->add_virtual_loss();

    node = child;
  }

  return node;
}

void HybridSearch::expand_node(HybridNode *node, const MCTSPosition &pos) {
  MCTSMoveList moves = pos.generate_legal_moves();
  node->create_edges(moves);

  // Add Dirichlet noise at root
  if (node == tree_->root() && config_.add_dirichlet_noise && !moves.empty()) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);

    std::vector<float> noise(moves.size());
    float noise_sum = 0.0f;
    for (auto &n : noise) {
      n = gamma(gen);
      noise_sum += n;
    }

    // Normalize and mix with uniform prior
    float uniform = 1.0f / moves.size();
    for (size_t i = 0; i < moves.size(); ++i) {
      float policy = (1.0f - config_.dirichlet_epsilon) * uniform +
                     config_.dirichlet_epsilon * (noise[i] / noise_sum);
      node->edges()[i].set_policy(policy);
    }
  } else {
    // Uniform policy until NN evaluation
    float uniform = moves.empty() ? 0.0f : 1.0f / moves.size();
    for (auto &edge : node->edges()) {
      edge.set_policy(uniform);
    }
  }
}

void HybridSearch::backpropagate(HybridNode *node, float value, float draw_prob,
                                 float moves_left) {
  while (node) {
    node->remove_virtual_loss();
    // update_stats now atomically updates statistics and increments visit count
    // under the same mutex lock to prevent race conditions
    node->update_stats(value, draw_prob, moves_left);

    // Flip value for opponent
    value = -value;
    moves_left += 1.0f;

    node = node->parent();
  }
}

bool HybridSearch::should_use_alphabeta(HybridNode *node,
                                        const MCTSPosition &pos) {
  if (!config_.use_ab_for_tactics)
    return false;

  // Check if position is tactical (in check, captures available, etc.)
  if (pos.is_check()) {
    stats_.tactical_positions.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Check node visit threshold
  if (node->n() < static_cast<uint32_t>(config_.ab_node_threshold)) {
    return false;
  }

  return false;
}

int HybridSearch::alphabeta_verify(const MCTSPosition &pos, int depth,
                                   int alpha, int beta) {
  // Use Stockfish's search for alpha-beta verification
  // Placeholder for alpha-beta search integration
  stats_.ab_nodes.fetch_add(1, std::memory_order_relaxed);

  if (depth <= 0 || pos.is_terminal()) {
    Value v = Eval::simple_eval(pos.stockfish_position());
    return static_cast<int>(v);
  }

  MCTSMoveList moves = pos.generate_legal_moves();
  if (moves.empty()) {
    if (pos.is_check()) {
      return -32000 + depth; // Checkmate
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

void HybridSearch::add_to_batch(HybridNode *node, const MCTSPosition &pos) {
  std::lock_guard<std::mutex> lock(batch_mutex_);

  current_batch_.nodes.push_back(node);
  current_batch_.positions.push_back(pos);
  current_batch_.legal_moves.push_back(pos.generate_legal_moves());

  // Always notify - let eval thread decide when to process
  batch_cv_.notify_one();
}

void HybridSearch::process_batch() {
  // Handled by eval_thread_main
}

bool HybridSearch::should_stop() const {
  if (stop_flag_)
    return true;

  // Check node limit
  if (limits_.nodes > 0 && stats_.mcts_nodes >= limits_.nodes) {
    return true;
  }

  // Check time limit
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

  // Use about 2.5% of remaining time + increment
  return time_left / 40 + inc;
}

float HybridSearch::get_q_value(HybridNode *node) const {
  if (node->n() == 0)
    return 0.0f;
  return node->q();
}

MCTSMove HybridSearch::select_best_move() const {
  const HybridNode *root = tree_->root();
  if (!root->has_children())
    return MCTSMove();

  // Find child with most visits
  int best_idx = -1;
  uint32_t best_n = 0;

  for (size_t i = 0; i < root->edges().size(); ++i) {
    const auto &edge = root->edges()[i];
    if (edge.child() && edge.child()->n() > best_n) {
      best_n = edge.child()->n();
      best_idx = static_cast<int>(i);
    }
  }

  if (best_idx < 0) {
    // No visits, return first move
    return root->edges()[0].move();
  }

  return root->edges()[best_idx].move();
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
