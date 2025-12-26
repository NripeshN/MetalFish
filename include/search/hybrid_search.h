/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  HYBRID SEARCH ARCHITECTURE
  ==========================
  
  MetalFish combines the best aspects of both Stockfish (alpha-beta + NNUE) 
  and Leela Chess Zero (MCTS + Deep NN) with Apple Silicon's unique unified
  memory architecture for maximum performance.
  
  Key innovations:
  
  1. UNIFIED MEMORY ADVANTAGE (vs LC0)
     - LC0 was designed for discrete GPUs with separate memory
     - MetalFish leverages Apple's unified memory for zero-copy data sharing
     - CPU and GPU work on the same memory without transfers
     - Enables fine-grained CPU/GPU collaboration not possible on discrete GPUs
  
  2. HYBRID ALPHA-BETA + MCTS
     - Use alpha-beta as primary search (proven, highly efficient)
     - MCTS-style tree expansion for difficult positions
     - GPU batch evaluation guides move ordering and pruning
     - Policy network provides move probabilities for search prioritization
  
  3. GPU-ACCELERATED NNUE
     - Stockfish's NNUE runs on CPU (optimized for AVX/SIMD)
     - MetalFish's NNUE runs on GPU with batch evaluation
     - Unified memory means positions evaluated by CPU are instantly available to GPU
     - Parallel evaluation of multiple positions in one GPU dispatch
  
  4. DUAL NETWORK ARCHITECTURE
     - Small "policy" network: Fast, guides move ordering (~100K params)
     - Large "value" network: Accurate evaluation (~10M params)
     - Both run on GPU, policy network runs during search, value at leaves
  
  5. ASYNCHRONOUS SEARCH PIPELINE
     - CPU: Tree expansion, alpha-beta, move generation
     - GPU: Batch NNUE evaluation, policy network
     - CPU/GPU work in parallel on shared memory
*/

#pragma once

#include "core/types.h"
#include "core/position.h"
#include <Metal/Metal.hpp>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace MetalFish {

// Forward declarations
class TranspositionTable;
struct SearchThread;

// Node in the hybrid search tree
struct SearchNode {
    Move move;                          // Move that led to this node
    Value eval = VALUE_NONE;            // Evaluation score
    float policy = 0.0f;                // Policy network probability
    int visits = 0;                     // Visit count (for MCTS portion)
    int depth = 0;                      // Search depth
    
    // Statistics for adaptive selection
    float Q = 0.0f;                     // Average value (MCTS)
    float U = 0.0f;                     // Upper confidence bound
    
    std::vector<SearchNode> children;
    
    bool is_terminal() const { return children.empty() && visits > 0; }
    
    // UCB1 selection formula (combines alpha-beta and MCTS)
    float ucb(float exploration = 1.41f) const {
        if (visits == 0) return std::numeric_limits<float>::max();
        return Q + exploration * policy * std::sqrt(std::log(visits + 1) / (visits + 1));
    }
};

// Position snapshot for evaluation (copyable)
struct PositionSnapshot {
    std::array<int8_t, 64> pieces;
    Color side_to_move;
    CastlingRights castling;
    Square ep_square;
    int halfmove_clock;
    int game_ply;
    
    PositionSnapshot() = default;
    explicit PositionSnapshot(const Position& pos);
};

// Batch evaluation request
struct EvalRequest {
    PositionSnapshot pos;
    int node_id;
    int depth;
};

// Batch evaluation result
struct EvalResult {
    int node_id;
    Value eval;
    std::array<float, 256> policy;  // Move probabilities
    int num_moves;
};

// Search limits (matching UCI go command)
struct SearchLimits {
    int depth = 0;                      // Max depth to search
    uint64_t nodes = 0;                 // Max nodes to search
    int64_t movetime = 0;               // Time per move in ms
    int64_t wtime = 0;                  // White time remaining
    int64_t btime = 0;                  // Black time remaining
    int64_t winc = 0;                   // White increment
    int64_t binc = 0;                   // Black increment
    int movestogo = 0;                  // Moves until time control
    bool infinite = false;              // Search until stopped
    bool ponder = false;                // Pondering mode
};

// Configuration for hybrid search
struct HybridSearchConfig {
    // Search mode balance
    float mcts_ratio = 0.3f;            // Fraction of time using MCTS-style expansion
    int min_batch_size = 16;            // Minimum positions for GPU batch
    int max_batch_size = 256;           // Maximum GPU batch size
    int batch_wait_ms = 1;              // Max time to wait for batch to fill
    
    // Network configuration
    bool use_policy_network = true;     // Use policy network for move ordering
    bool use_value_network = true;      // Use value network for evaluation
    int policy_temperature = 100;       // Temperature for policy (higher = more exploration)
    
    // GPU configuration
    bool async_gpu = true;              // Asynchronous GPU evaluation
    int gpu_queue_depth = 4;            // Number of concurrent GPU batches
    
    // Tree configuration
    int max_tree_nodes = 1000000;       // Maximum nodes in search tree
    float tree_reuse_fraction = 0.5f;   // Fraction of tree to reuse between moves
    
    // Hybrid balance
    int ab_depth_threshold = 8;         // Use alpha-beta for depth < threshold
    int mcts_simulation_limit = 10000;  // Max MCTS simulations per move
};

// GPU Evaluation Pipeline
class GPUEvaluationPipeline {
public:
    GPUEvaluationPipeline();
    ~GPUEvaluationPipeline();
    
    bool init();
    
    // Submit positions for GPU evaluation
    void submit(const std::vector<EvalRequest>& requests);
    
    // Get completed evaluations (non-blocking)
    bool get_results(std::vector<EvalResult>& results);
    
    // Wait for all pending evaluations
    void flush();
    
    // Statistics
    uint64_t positions_evaluated() const { return positions_evaluated_; }
    uint64_t batches_processed() const { return batches_processed_; }
    double avg_batch_size() const { 
        return batches_processed_ > 0 ? 
               double(positions_evaluated_) / batches_processed_ : 0; 
    }
    
private:
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::Library* library_ = nullptr;
    
    // Compute pipelines
    MTL::ComputePipelineState* policy_kernel_ = nullptr;
    MTL::ComputePipelineState* value_kernel_ = nullptr;
    MTL::ComputePipelineState* feature_kernel_ = nullptr;
    
    // Shared memory buffers (zero-copy on unified memory)
    MTL::Buffer* position_buffer_ = nullptr;
    MTL::Buffer* feature_buffer_ = nullptr;
    MTL::Buffer* policy_output_ = nullptr;
    MTL::Buffer* value_output_ = nullptr;
    MTL::Buffer* weight_buffer_ = nullptr;
    
    // Request/result queues
    std::queue<std::vector<EvalRequest>> pending_requests_;
    std::queue<std::vector<EvalResult>> completed_results_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker thread
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    
    // Statistics
    std::atomic<uint64_t> positions_evaluated_{0};
    std::atomic<uint64_t> batches_processed_{0};
    
    void worker_loop();
    void process_batch(const std::vector<EvalRequest>& batch);
};

// Main hybrid search engine
class HybridSearchEngine {
public:
    HybridSearchEngine();
    ~HybridSearchEngine();
    
    // Initialize GPU resources
    bool init();
    
    // Main search function
    Move search(Position& pos, const SearchLimits& limits);
    
    // Configuration
    void set_config(const HybridSearchConfig& config) { config_ = config; }
    HybridSearchConfig& config() { return config_; }
    
    // Clear search state
    void clear();
    
    // Stop search
    void stop() { stop_flag_ = true; }
    
    // Statistics
    struct SearchStats {
        uint64_t nodes_searched = 0;
        uint64_t gpu_evaluations = 0;
        uint64_t cpu_evaluations = 0;
        uint64_t tt_hits = 0;
        int seldepth = 0;
        int ab_nodes = 0;
        int mcts_nodes = 0;
        double search_time_ms = 0;
        double nps = 0;
    };
    
    const SearchStats& stats() const { return stats_; }
    
private:
    // Configuration
    HybridSearchConfig config_;
    
    // GPU pipeline
    std::unique_ptr<GPUEvaluationPipeline> gpu_pipeline_;
    
    // Search tree
    std::unique_ptr<SearchNode> root_;
    
    // State
    std::atomic<bool> stop_flag_{false};
    SearchStats stats_;
    
    // Search methods
    Value alpha_beta(Position& pos, int depth, Value alpha, Value beta, 
                     SearchNode* node, bool is_pv);
    
    Value quiescence(Position& pos, Value alpha, Value beta);
    
    void mcts_expand(Position& pos, SearchNode* node);
    
    SearchNode* select_best_child(SearchNode* node, bool exploration);
    
    void backpropagate(SearchNode* node, Value value);
    
    // Move ordering using GPU policy
    void order_moves_with_policy(Position& pos,
                                  std::vector<Move>& moves,
                                  std::vector<float>& scores);
    
    // Batch evaluation
    void batch_evaluate_leaves(std::vector<SearchNode*>& leaves,
                               std::vector<EvalResult>& results);
    
    // Tree management
    void prune_tree(int max_nodes);
    void reuse_subtree(Move best_move);
};

// Global hybrid search instance
HybridSearchEngine& get_hybrid_search();

} // namespace MetalFish

