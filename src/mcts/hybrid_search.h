/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Hybrid Search: Combines MCTS tree search with alpha-beta pruning.

  The key insight is that MCTS excels at strategic evaluation with neural networks,
  while alpha-beta excels at tactical calculation with fast evaluation.
  
  This hybrid approach:
  1. Uses MCTS for the root and early tree exploration (strategic planning)
  2. Uses alpha-beta for leaf evaluation when depth is sufficient (tactical verification)
  3. Batches neural network evaluations on GPU for throughput
  4. Falls back to alpha-beta for time-critical situations

  Licensed under GPL-3.0
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "stockfish_adapter.h"
#include "mcts_batch_evaluator.h"
#include "mcts_tt.h"
#include "../search/search.h"
#include "../gpu/gpu_nnue_integration.h"

// Use MetalFish namespace
using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// Forward declarations
class HybridNode;
class HybridTree;
class HybridSearch;

// Configuration for hybrid search
struct HybridSearchConfig {
  // MCTS parameters
  float cpuct = 2.5f;                    // Exploration constant
  float fpu_value = -1.0f;               // First play urgency value
  float fpu_reduction = 0.0f;            // FPU reduction based on parent visits
  float policy_softmax_temp = 1.0f;      // Policy temperature
  bool add_dirichlet_noise = true;       // Add exploration noise at root
  float dirichlet_alpha = 0.3f;          // Dirichlet noise alpha
  float dirichlet_epsilon = 0.25f;       // Dirichlet noise weight
  
  // Alpha-beta integration parameters
  int ab_depth_threshold = 6;            // Min depth for alpha-beta verification
  int ab_node_threshold = 1000;          // Min nodes for alpha-beta use
  float ab_confidence_threshold = 0.9f;  // Confidence needed to skip AB
  bool use_ab_for_tactics = true;        // Use AB for tactical positions
  
  // Batching parameters
  int min_batch_size = 8;                // Minimum batch size for GPU
  int max_batch_size = 256;              // Maximum batch size
  int batch_timeout_us = 1000;           // Timeout for batch collection (microseconds)
  
  // Threading
  int num_search_threads = 1;            // Number of search threads
  int num_eval_threads = 1;              // Number of evaluation threads
  
  // Time management
  bool use_smart_pruning = true;         // Prune obviously bad moves early
  float time_curve_peak = 0.3f;          // Peak of time allocation curve
  float time_curve_left_width = 0.2f;    // Left width of time curve
  float time_curve_right_width = 0.3f;   // Right width of time curve
};

// Statistics for hybrid search
struct HybridSearchStats {
  std::atomic<uint64_t> mcts_nodes{0};
  std::atomic<uint64_t> ab_nodes{0};
  std::atomic<uint64_t> nn_evaluations{0};
  std::atomic<uint64_t> nn_batches{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> cache_misses{0};
  std::atomic<uint64_t> tactical_positions{0};
  std::atomic<uint64_t> ab_overrides{0};
  
  void reset() {
    mcts_nodes = 0;
    ab_nodes = 0;
    nn_evaluations = 0;
    nn_batches = 0;
    cache_hits = 0;
    cache_misses = 0;
    tactical_positions = 0;
    ab_overrides = 0;
  }
};

// Edge in the MCTS tree
class HybridEdge {
public:
  HybridEdge() = default;
  explicit HybridEdge(MCTSMove move) : move_(move) {}
  
  MCTSMove move() const { return move_; }
  float policy() const { return policy_; }
  void set_policy(float p) { policy_ = p; }
  
  HybridNode* child() const { return child_; }
  void set_child(HybridNode* node) { child_ = node; }
  
private:
  MCTSMove move_;
  float policy_ = 0.0f;
  HybridNode* child_ = nullptr;
};

// Node in the MCTS tree
class HybridNode {
public:
  enum class Terminal : uint8_t {
    NonTerminal,
    Win,
    Draw,
    Loss,
    Tablebase
  };
  
  HybridNode(HybridNode* parent = nullptr, int edge_index = -1);
  ~HybridNode();
  
  // Tree structure
  HybridNode* parent() const { return parent_; }
  int edge_index() const { return edge_index_; }
  bool has_children() const { return !edges_.empty(); }
  
  // Edge access
  const std::vector<HybridEdge>& edges() const { return edges_; }
  std::vector<HybridEdge>& edges() { return edges_; }
  void create_edges(const MCTSMoveList& moves);
  
  // Statistics
  uint32_t n() const { return n_.load(std::memory_order_relaxed); }
  uint32_t n_in_flight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  float q() const { return q_; }
  float d() const { return d_; }
  float m() const { return m_; }
  
  // Update methods
  void add_visit();
  void add_virtual_loss();
  void remove_virtual_loss();
  void update_stats(float value, float draw_prob, float moves_left);
  
  // Terminal state
  Terminal terminal_type() const { return terminal_type_; }
  bool is_terminal() const { return terminal_type_ != Terminal::NonTerminal; }
  void set_terminal(Terminal type, float value);
  
  // Alpha-beta integration
  bool has_ab_score() const { return has_ab_score_; }
  int ab_score() const { return ab_score_; }
  int ab_depth() const { return ab_depth_; }
  void set_ab_score(int score, int depth);
  
  // For selection
  float get_u(float cpuct, float parent_n_sqrt) const;
  float get_puct(float cpuct, float parent_n_sqrt, float fpu) const;
  
  // Reset parent (for tree reuse)
  void reset_parent() { parent_ = nullptr; edge_index_ = -1; }
  
private:
  HybridNode* parent_;
  int edge_index_;
  
  std::vector<HybridEdge> edges_;
  
  std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};
  float q_ = 0.0f;
  float d_ = 0.0f;
  float m_ = 0.0f;
  
  Terminal terminal_type_ = Terminal::NonTerminal;
  
  // Alpha-beta verification
  bool has_ab_score_ = false;
  int ab_score_ = 0;
  int ab_depth_ = 0;
  
  mutable std::mutex mutex_;
};

// MCTS tree structure
class HybridTree {
public:
  HybridTree();
  ~HybridTree();
  
  // Initialize tree with position
  void reset(const MCTSPositionHistory& history);
  
  // Apply move (reuse subtree if possible)
  bool apply_move(MCTSMove move);
  
  // Get root node
  HybridNode* root() { return root_.get(); }
  const HybridNode* root() const { return root_.get(); }
  
  // Get position history
  const MCTSPositionHistory& history() const { return history_; }
  MCTSPositionHistory& history() { return history_; }
  
  // Node allocation
  HybridNode* allocate_node(HybridNode* parent, int edge_index);
  
  // Statistics
  size_t node_count() const { return node_count_.load(); }
  
private:
  std::unique_ptr<HybridNode> root_;
  MCTSPositionHistory history_;
  std::atomic<size_t> node_count_{0};
  
  // Node pool for efficient allocation
  std::vector<std::unique_ptr<HybridNode>> node_pool_;
  std::mutex pool_mutex_;
};

// Batch of positions for GPU evaluation
struct EvalBatch {
  std::vector<HybridNode*> nodes;
  std::vector<MCTSPosition> positions;
  std::vector<std::vector<MCTSMove>> legal_moves;
  
  void clear() {
    nodes.clear();
    positions.clear();
    legal_moves.clear();
  }
  
  size_t size() const { return nodes.size(); }
  bool empty() const { return nodes.empty(); }
};

// Main hybrid search class
class HybridSearch {
public:
  using BestMoveCallback = std::function<void(MCTSMove, MCTSMove)>;  // bestmove, ponder
  using InfoCallback = std::function<void(const std::string&)>;
  
  HybridSearch(const HybridSearchConfig& config = HybridSearchConfig());
  ~HybridSearch();
  
  // Initialize with neural network backend
  void set_neural_network(std::shared_ptr<MCTSNeuralNetwork> nn);
  
  // Set GPU NNUE manager for batch evaluation
  void set_gpu_nnue(GPU::GPUNNUEManager* gpu_nnue);
  
  // Start search
  void start_search(const MCTSPositionHistory& history,
                    const Search::LimitsType& limits,
                    BestMoveCallback best_move_cb,
                    InfoCallback info_cb = nullptr);
  
  // Stop search
  void stop();
  
  // Wait for search to complete
  void wait();
  
  // Get current best move
  MCTSMove get_best_move() const;
  
  // Get PV (principal variation)
  std::vector<MCTSMove> get_pv() const;
  
  // Get statistics
  const HybridSearchStats& stats() const { return stats_; }
  
  // Configuration
  const HybridSearchConfig& config() const { return config_; }
  void set_config(const HybridSearchConfig& config) { config_ = config; }
  
private:
  // Search threads
  void search_thread_main();
  void eval_thread_main();
  
  // MCTS operations
  HybridNode* select_node(HybridNode* node, MCTSPosition& pos);
  void expand_node(HybridNode* node, const MCTSPosition& pos);
  void backpropagate(HybridNode* node, float value, float draw_prob, float moves_left);
  
  // Alpha-beta integration
  bool should_use_alphabeta(HybridNode* node, const MCTSPosition& pos);
  int alphabeta_verify(const MCTSPosition& pos, int depth, int alpha, int beta);
  
  // Batch evaluation
  void add_to_batch(HybridNode* node, const MCTSPosition& pos);
  void process_batch();
  
  // Time management
  bool should_stop() const;
  int64_t get_time_budget_ms() const;
  
  // Utility
  float get_q_value(HybridNode* node) const;
  MCTSMove select_best_move() const;
  void update_info();
  
  HybridSearchConfig config_;
  HybridSearchStats stats_;
  
  std::unique_ptr<HybridTree> tree_;
  std::shared_ptr<MCTSNeuralNetwork> neural_network_;
  GPU::GPUNNUEManager* gpu_nnue_ = nullptr;
  
  // Threading
  std::vector<std::thread> search_threads_;
  std::thread eval_thread_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};
  
  // Batch management
  EvalBatch current_batch_;
  std::mutex batch_mutex_;
  std::condition_variable batch_cv_;
  
  // Results
  std::queue<std::pair<HybridNode*, MCTSEvaluation>> eval_results_;
  std::mutex results_mutex_;
  std::condition_variable results_cv_;
  
  // Callbacks
  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;
  
  // Time management
  Search::LimitsType limits_;
  std::chrono::steady_clock::time_point search_start_;
};

// Factory function for creating hybrid search with GPU backend
std::unique_ptr<HybridSearch> create_hybrid_search(
    GPU::GPUNNUEManager* gpu_nnue,
    const HybridSearchConfig& config = HybridSearchConfig());

}  // namespace MCTS
}  // namespace MetalFish
