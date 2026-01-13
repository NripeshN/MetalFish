/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Multi-threaded MCTS Search Worker

  This module provides thread-local search workers for parallel MCTS.
  Each worker maintains its own position state to avoid synchronization
  issues with MetalFish::Position.

  Key features:
  1. Thread-local position management
  2. Lock-free batch submission to GPU
  3. Virtual loss for exploration diversity
  4. Efficient backpropagation

  Licensed under GPL-3.0
*/

#pragma once

#include "stockfish_adapter.h"
#include "mcts_tt.h"
#include "hybrid_search.h"
#include "../gpu/gpu_nnue_integration.h"
#include "../core/position.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace MetalFish {
namespace MCTS {

// Forward declarations
class HybridNode;
class HybridTree;
class HybridSearch;

// ============================================================================
// Thread-Local Search Context
// ============================================================================

// Each search thread maintains its own context to avoid contention
struct SearchContext {
  // Thread-local position (rebuilt from FEN for each selection)
  std::string root_fen;
  std::vector<Move> move_path;  // Path from root to current node
  
  // Statistics for this thread
  uint64_t nodes_searched = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  
  // Batch accumulator
  std::vector<HybridNode*> pending_nodes;
  std::vector<std::string> pending_fens;
  
  // Random generator for this thread
  std::mt19937 rng;
  
  SearchContext() : rng(std::random_device{}()) {}
  
  void reset(const std::string& fen) {
    root_fen = fen;
    move_path.clear();
    nodes_searched = 0;
    cache_hits = 0;
    cache_misses = 0;
    pending_nodes.clear();
    pending_fens.clear();
  }
};

// ============================================================================
// GPU Evaluation Request
// ============================================================================

struct EvalRequest {
  HybridNode* node;
  std::string fen;
  int thread_id;
  
  EvalRequest() : node(nullptr), thread_id(-1) {}
  EvalRequest(HybridNode* n, const std::string& f, int tid)
      : node(n), fen(f), thread_id(tid) {}
};

// ============================================================================
// GPU Evaluation Result
// ============================================================================

struct EvalResult {
  HybridNode* node;
  float value;
  float draw_prob;
  float moves_left;
  std::vector<std::pair<Move, float>> policy;  // Move -> prior probability
  
  EvalResult() : node(nullptr), value(0), draw_prob(0), moves_left(30) {}
};

// ============================================================================
// Parallel GPU Evaluator
// ============================================================================

// Configuration for the parallel evaluator
struct ParallelEvalConfig {
  int batch_size = 64;           // Target batch size
  int max_batch_size = 256;      // Maximum batch size
  int min_batch_size = 1;        // Minimum before forcing evaluation
  int batch_timeout_us = 500;    // Timeout for batch collection
  int num_eval_threads = 1;      // Number of GPU evaluation threads
  bool use_persistent_buffers = true;  // Reuse Metal buffers
};

class ParallelGPUEvaluator {
public:
  ParallelGPUEvaluator();
  ~ParallelGPUEvaluator();
  
  // Initialize with GPU manager
  bool initialize(GPU::GPUNNUEManager* gpu_manager, const ParallelEvalConfig& config);
  
  // Submit evaluation request (thread-safe)
  void submit(const EvalRequest& request);
  
  // Get completed results (thread-safe)
  bool get_result(EvalResult& result);
  
  // Wait for all pending evaluations
  void flush();
  
  // Stop the evaluator
  void stop();
  
  // Check if running
  bool is_running() const { return running_; }
  
  // Statistics
  uint64_t total_evaluations() const { return total_evals_.load(); }
  uint64_t total_batches() const { return total_batches_.load(); }
  double avg_batch_size() const {
    uint64_t batches = total_batches_.load();
    return batches > 0 ? double(total_evals_.load()) / batches : 0;
  }
  
private:
  bool initialized_ = false;
  bool running_ = false;
  ParallelEvalConfig config_;
  
  GPU::GPUNNUEManager* gpu_manager_ = nullptr;
  
  // Request queue
  std::queue<EvalRequest> request_queue_;
  std::mutex request_mutex_;
  std::condition_variable request_cv_;
  
  // Result queue
  std::queue<EvalResult> result_queue_;
  std::mutex result_mutex_;
  std::condition_variable result_cv_;
  
  // Evaluation thread
  std::vector<std::thread> eval_threads_;
  std::atomic<bool> stop_flag_{false};
  
  // Statistics
  std::atomic<uint64_t> total_evals_{0};
  std::atomic<uint64_t> total_batches_{0};
  
  // Persistent buffers for reduced allocation overhead
  struct PersistentBatch {
    GPU::GPUEvalBatch gpu_batch;
    std::vector<EvalRequest> requests;
    
    void reserve(int size) {
      gpu_batch.reserve(size);
      requests.reserve(size);
    }
    
    void clear() {
      gpu_batch.clear();
      requests.clear();
    }
    
    int size() const { return static_cast<int>(requests.size()); }
  };
  std::vector<std::unique_ptr<PersistentBatch>> persistent_batches_;
  
  void eval_thread_main(int thread_id);
  void process_batch(PersistentBatch& batch);
};

// ============================================================================
// Search Worker
// ============================================================================

class SearchWorker {
public:
  SearchWorker(int id, HybridTree* tree, ParallelGPUEvaluator* evaluator,
               MCTSTranspositionTable* tt, const HybridSearchConfig& config);
  ~SearchWorker();
  
  // Start/stop worker
  void start(const std::string& root_fen);
  void stop();
  void wait();
  
  // Check if running
  bool is_running() const { return running_; }
  
  // Get statistics
  uint64_t nodes_searched() const { return context_.nodes_searched; }
  uint64_t cache_hits() const { return context_.cache_hits; }
  uint64_t cache_misses() const { return context_.cache_misses; }
  
private:
  int id_;
  HybridTree* tree_;
  ParallelGPUEvaluator* evaluator_;
  MCTSTranspositionTable* tt_;
  HybridSearchConfig config_;
  
  SearchContext context_;
  std::thread thread_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};
  
  void worker_main();
  
  // MCTS operations
  HybridNode* select_and_expand(Position& pos, StateInfo* states);
  void backpropagate(HybridNode* node, float value, float draw_prob, float moves_left);
  float evaluate_position(const Position& pos);
  
  // Policy generation
  void generate_policy(HybridNode* node, const Position& pos);
};

// ============================================================================
// Multi-threaded Search Manager
// ============================================================================

class ParallelSearchManager {
public:
  ParallelSearchManager();
  ~ParallelSearchManager();
  
  // Initialize with GPU manager
  bool initialize(GPU::GPUNNUEManager* gpu_manager, int num_threads = 4);
  
  // Start search
  void start_search(HybridTree* tree, const std::string& root_fen,
                    const HybridSearchConfig& config);
  
  // Stop search
  void stop();
  
  // Wait for completion
  void wait();
  
  // Get total nodes searched
  uint64_t total_nodes() const;
  
  // Get statistics
  struct Stats {
    uint64_t total_nodes = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t gpu_evals = 0;
    uint64_t gpu_batches = 0;
    double avg_batch_size = 0;
  };
  Stats get_stats() const;
  
private:
  bool initialized_ = false;
  int num_threads_ = 4;
  
  std::unique_ptr<ParallelGPUEvaluator> evaluator_;
  std::vector<std::unique_ptr<SearchWorker>> workers_;
  
  GPU::GPUNNUEManager* gpu_manager_ = nullptr;
};

// ============================================================================
// Global Parallel Search Manager
// ============================================================================

ParallelSearchManager& parallel_search_manager();
bool initialize_parallel_search(GPU::GPUNNUEManager* gpu_manager, int num_threads = 4);

}  // namespace MCTS
}  // namespace MetalFish
