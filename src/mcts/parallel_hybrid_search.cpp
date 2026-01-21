/*
  MetalFish - Parallel Hybrid Search Implementation
  Copyright (C) 2025 Nripesh Niketan
  
  Optimized for Apple Silicon with unified memory architecture.
  
  This implementation incorporates algorithms from Leela Chess Zero (Lc0),
  including PUCT with logarithmic growth, FPU reduction strategy, and
  moves left head (MLH) utility.
  
  Licensed under GPL-3.0
*/

#include "parallel_hybrid_search.h"
#include "lc0_mcts_core.h"
#include "../core/misc.h"
#include "../eval/evaluate.h"
#include "../uci/engine.h"
#include "../uci/uci.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// GPU-Resident Batch Implementation
// ============================================================================

bool GPUResidentBatch::initialize(int batch_capacity) {
  if (!GPU::gpu_available()) {
    return false;
  }
  
  auto& backend = GPU::gpu();
  
  // Calculate buffer sizes
  // Each position needs: GPUPositionData (aligned struct)
  size_t position_size = sizeof(GPU::GPUPositionData);
  size_t positions_buffer_size = batch_capacity * position_size;
  
  // Results: psqt score + positional score per position
  size_t results_buffer_size = batch_capacity * 2 * sizeof(int32_t);
  
  // Create GPU-accessible buffers using unified memory (Shared mode)
  // On Apple Silicon, this means zero-copy access from both CPU and GPU
  positions_buffer = backend.create_buffer(
      positions_buffer_size, 
      GPU::MemoryMode::Shared,  // Unified memory - zero copy!
      GPU::BufferUsage::Streaming  // Frequently updated from CPU
  );
  
  results_buffer = backend.create_buffer(
      results_buffer_size,
      GPU::MemoryMode::Shared,  // Results read by CPU after GPU writes
      GPU::BufferUsage::Streaming
  );
  
  if (!positions_buffer || !results_buffer) {
    return false;
  }
  
  capacity = batch_capacity;
  position_indices.reserve(batch_capacity);
  initialized = true;
  
  return true;
}

// ============================================================================
// ParallelHybridSearch Implementation
// ============================================================================

ParallelHybridSearch::ParallelHybridSearch() {
  // Lc0-style MCTS parameters
  config_.mcts_config.cpuct = 1.745f;       // Lc0 default
  config_.mcts_config.fpu_reduction = 0.330f; // Lc0 default
  config_.mcts_config.cpuct_base = 38739.0f;
  config_.mcts_config.cpuct_factor = 3.894f;
  
  config_.ab_min_depth = 8;
  config_.agreement_threshold = 0.3f;
  config_.override_threshold = 1.0f;
  
  // Apple Silicon GPU defaults
  config_.gpu_batch_size = 128;
  config_.use_async_gpu_eval = true;
  config_.use_gpu_resident_batches = true;
  
  // Initialize thread state
  mcts_thread_state_.store(ThreadState::IDLE, std::memory_order_relaxed);
  ab_thread_state_.store(ThreadState::IDLE, std::memory_order_relaxed);
  coordinator_thread_state_.store(ThreadState::IDLE, std::memory_order_relaxed);
  mcts_thread_done_.store(true, std::memory_order_relaxed);
  ab_thread_done_.store(true, std::memory_order_relaxed);
  coordinator_thread_done_.store(true, std::memory_order_relaxed);
}

ParallelHybridSearch::~ParallelHybridSearch() {
  // Mark that we're shutting down - this prevents any new searches
  shutdown_requested_.store(true, std::memory_order_release);
  
  // Signal stop to all threads
  stop_flag_.store(true, std::memory_order_release);
  searching_.store(false, std::memory_order_release);
  
  // Stop MCTS search if running
  if (mcts_search_) {
    mcts_search_->stop();
  }
  
  // Give threads a moment to notice the stop flag
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  
  // Wait for MCTS to fully stop
  if (mcts_search_) {
    mcts_search_->wait();
  }
  
  // Join all threads safely
  join_all_threads();
  
  // Wait for any pending GPU evaluations to complete
  // This is critical to prevent crashes from async callbacks accessing destroyed objects
  if (pending_evaluations_.load(std::memory_order_relaxed) > 0) {
    wait_gpu_evaluations();
  }
  
  // Synchronize GPU only if backend is still available
  if (GPU::gpu_available() && !GPU::gpu_backend_shutdown()) {
    GPU::gpu().synchronize();
  }
  
  // Clear callbacks to prevent any dangling references
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = nullptr;
    info_callback_ = nullptr;
  }
}

void ParallelHybridSearch::join_all_threads() {
  // This function safely joins all threads, handling the case where
  // threads may have already been joined or may not have been started.
  
  std::unique_lock<std::mutex> lock(thread_mutex_);
  
  // Wait for all threads to signal completion (with timeout)
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  thread_cv_.wait_until(lock, deadline, [this]() {
    return all_threads_done();
  });
  
  // Now join the threads - they should all be finished
  // We need to release the lock to avoid deadlock if threads are trying
  // to signal completion
  lock.unlock();
  
  // Join coordinator first (it's the last to finish in normal operation)
  if (coordinator_thread_.joinable()) {
    coordinator_thread_.join();
  }
  
  // Then join worker threads
  if (mcts_thread_.joinable()) {
    mcts_thread_.join();
  }
  
  if (ab_thread_.joinable()) {
    ab_thread_.join();
  }
}

void ParallelHybridSearch::signal_thread_done(std::atomic<bool>& done_flag) {
  done_flag.store(true, std::memory_order_release);
  thread_cv_.notify_all();
}

bool ParallelHybridSearch::all_threads_done() const {
  return mcts_thread_done_.load(std::memory_order_acquire) &&
         ab_thread_done_.load(std::memory_order_acquire) &&
         coordinator_thread_done_.load(std::memory_order_acquire);
}

bool ParallelHybridSearch::initialize(GPU::GPUNNUEManager *gpu_manager,
                                      Engine *engine) {
  if (!gpu_manager || !gpu_manager->is_ready()) {
    return false;
  }
  if (!engine) {
    return false;
  }

  gpu_manager_ = gpu_manager;
  engine_ = engine;

  // Check for unified memory (Apple Silicon)
  if (GPU::gpu_available()) {
    has_unified_memory_ = GPU::gpu().has_unified_memory();
  }

  // Create GPU MCTS backend
  gpu_backend_ = GPU::create_gpu_mcts_backend(gpu_manager);
  if (!gpu_backend_) {
    return false;
  }
  
  // Set optimal batch size for Apple Silicon
  if (has_unified_memory_) {
    gpu_backend_->set_optimal_batch_size(config_.gpu_batch_size);
  }

  // Initialize GPU-resident batches for zero-copy evaluation
  if (config_.use_gpu_resident_batches && has_unified_memory_) {
    if (!initialize_gpu_batches()) {
      // Fall back to regular batches if GPU batches fail
      config_.use_gpu_resident_batches = false;
    }
  }

  // Create MCTS search
  mcts_search_ = std::make_unique<HybridSearch>(config_.mcts_config);
  mcts_search_->set_gpu_nnue(gpu_manager);
  mcts_search_->set_neural_network(std::move(gpu_backend_));

  // Initialize shared state
  ab_state_.reset();
  mcts_state_.reset();

  initialized_ = true;
  return true;
}

void ParallelHybridSearch::start_search(const Position &pos,
                                        const Search::LimitsType &limits,
                                        BestMoveCallback best_move_cb,
                                        InfoCallback info_cb) {
  // Check if shutdown was requested
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    if (best_move_cb) {
      best_move_cb(Move::none(), Move::none());
    }
    return;
  }
  
  if (!initialized_) {
    // Cannot start search without initialization - mcts_search_ would be nullptr
    if (best_move_cb) {
      best_move_cb(Move::none(), Move::none());
    }
    return;
  }

  // Stop any existing search and wait for threads to finish
  stop();
  wait();

  // Reset state
  stats_.reset();
  ab_state_.reset();
  mcts_state_.reset();
  stop_flag_.store(false, std::memory_order_release);
  searching_.store(true, std::memory_order_release);
  final_best_move_.store(0, std::memory_order_relaxed);
  final_ponder_move_.store(0, std::memory_order_relaxed);
  callback_invoked_.store(false, std::memory_order_relaxed);

  // Store callbacks safely
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = best_move_cb;
    info_callback_ = info_cb;
  }

  // Store parameters
  root_fen_ = pos.fen();
  limits_ = limits;
  search_start_ = std::chrono::steady_clock::now();

  // Calculate time budget
  time_budget_ms_ = calculate_time_budget();

  // Analyze position for strategy
  if (config_.use_position_classifier) {
    PositionFeatures features = classifier_.analyze(pos);
    current_strategy_ = strategy_selector_.get_strategy(features);
    
    Color us = pos.side_to_move();
    int time_left = (us == WHITE) ? limits.time[WHITE] : limits.time[BLACK];
    int increment = (us == WHITE) ? limits.inc[WHITE] : limits.inc[BLACK];
    strategy_selector_.adjust_for_time(current_strategy_, time_left, increment);

    send_info_string("Parallel search - Position: " +
        std::string(current_strategy_.position_type == PositionType::HIGHLY_TACTICAL
                        ? "TACTICAL"
                    : current_strategy_.position_type == PositionType::TACTICAL
                        ? "TACTICAL"
                    : current_strategy_.position_type == PositionType::BALANCED
                        ? "BALANCED"
                        : "STRATEGIC"));
  }

  send_info_string("Time budget: " + std::to_string(time_budget_ms_) + "ms");

  // Mark threads as not done before starting
  mcts_thread_done_.store(false, std::memory_order_release);
  ab_thread_done_.store(false, std::memory_order_release);
  coordinator_thread_done_.store(false, std::memory_order_release);

  // Start all threads with proper state tracking
  mcts_state_.mcts_running.store(true, std::memory_order_release);
  ab_state_.ab_running.store(true, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(thread_mutex_);
    
    mcts_thread_state_.store(ThreadState::RUNNING, std::memory_order_release);
    ab_thread_state_.store(ThreadState::RUNNING, std::memory_order_release);
    coordinator_thread_state_.store(ThreadState::RUNNING, std::memory_order_release);
    
    mcts_thread_ = std::thread(&ParallelHybridSearch::mcts_thread_main, this);
    ab_thread_ = std::thread(&ParallelHybridSearch::ab_thread_main, this);
    coordinator_thread_ = std::thread(&ParallelHybridSearch::coordinator_thread_main, this);
  }
}

void ParallelHybridSearch::stop() {
  // Signal stop
  stop_flag_.store(true, std::memory_order_release);
  
  // Stop MCTS search
  if (mcts_search_) {
    mcts_search_->stop();
  }
  
  // NOTE: We don't call engine_->stop() because:
  // 1. The engine is not owned by us
  // 2. Calling stop() while search_sync is running can cause issues
  // 3. search_sync will return naturally when the AB thread sees should_stop()
}

void ParallelHybridSearch::wait() {
  // Wait for all threads to complete
  // Use a loop with timeout to avoid infinite hangs
  
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  
  while (!all_threads_done()) {
    if (std::chrono::steady_clock::now() > deadline) {
      // Timeout - threads are stuck, force stop
      stop_flag_.store(true, std::memory_order_release);
      if (mcts_search_) {
        mcts_search_->stop();
      }
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  // Now join the threads
  {
    std::lock_guard<std::mutex> lock(thread_mutex_);
    
    if (coordinator_thread_.joinable()) {
      coordinator_thread_.join();
    }
    if (mcts_thread_.joinable()) {
      mcts_thread_.join();
    }
    if (ab_thread_.joinable()) {
      ab_thread_.join();
    }
    
    // Reset thread states
    mcts_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
    ab_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
    coordinator_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
  }
  
  searching_.store(false, std::memory_order_release);
}

Move ParallelHybridSearch::get_best_move() const {
  return Move(final_best_move_.load(std::memory_order_acquire));
}

Move ParallelHybridSearch::get_ponder_move() const {
  return Move(final_ponder_move_.load(std::memory_order_acquire));
}

void ParallelHybridSearch::new_game() {
  stop();
  wait();
  
  // Reset state
  stats_.reset();
  ab_state_.reset();
  mcts_state_.reset();
  callback_invoked_.store(false, std::memory_order_relaxed);
  
  // Clear callbacks
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = nullptr;
    info_callback_ = nullptr;
  }
  
  if (mcts_search_) {
    mcts_search_ = std::make_unique<HybridSearch>(config_.mcts_config);
    mcts_search_->set_gpu_nnue(gpu_manager_);
    if (gpu_manager_) {
      auto backend = GPU::create_gpu_mcts_backend(gpu_manager_);
      if (backend) {
        mcts_search_->set_neural_network(std::move(backend));
      }
    }
  }
  mcts_tt().clear();
}

void ParallelHybridSearch::apply_move(Move move) {
  if (mcts_search_ && mcts_search_->tree()) {
    mcts_search_->tree()->apply_move(MCTSMove::FromStockfish(move));
  }
}

int ParallelHybridSearch::calculate_time_budget() const {
  if (limits_.movetime > 0) {
    return limits_.movetime;
  }

  Position pos;
  StateInfo st;
  pos.set(root_fen_, false, &st);
  Color us = pos.side_to_move();
  
  int time_left = (us == WHITE) ? limits_.time[WHITE] : limits_.time[BLACK];
  int increment = (us == WHITE) ? limits_.inc[WHITE] : limits_.inc[BLACK];

  if (time_left <= 0) {
    return 1000;
  }

  int base_time = static_cast<int>(time_left * config_.time_fraction);
  base_time += static_cast<int>(increment * config_.increment_usage);
  base_time = std::max(500, base_time);
  int max_time = static_cast<int>(time_left * config_.max_time_fraction);

  return std::min(base_time, max_time);
}

bool ParallelHybridSearch::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire)) return true;

  if (limits_.nodes > 0) {
    uint64_t total = stats_.mcts_nodes + stats_.ab_nodes;
    if (total >= limits_.nodes) return true;
  }

  if (time_budget_ms_ > 0) {
    auto elapsed = std::chrono::steady_clock::now() - search_start_;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (ms >= time_budget_ms_) return true;
  }

  return false;
}

// MCTS thread - runs GPU-accelerated MCTS
void ParallelHybridSearch::mcts_thread_main() {
  // RAII guard to ensure we always signal completion
  struct ThreadGuard {
    ParallelHybridSearch* self;
    ~ThreadGuard() {
      self->mcts_state_.mcts_running.store(false, std::memory_order_release);
      self->mcts_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
      self->signal_thread_done(self->mcts_thread_done_);
    }
  } guard{this};
  
  auto start = std::chrono::steady_clock::now();

  MCTSPositionHistory history;
  history.reset(root_fen_);

  // Set up MCTS limits - run for full time budget
  Search::LimitsType mcts_limits;
  mcts_limits.movetime = time_budget_ms_;
  mcts_limits.startTime = now();

  MCTSMove best_move;
  std::atomic<bool> mcts_done{false};

  auto mcts_callback = [&](MCTSMove move, MCTSMove ponder) {
    best_move = move;
    mcts_done = true;
  };

  // Start MCTS search
  mcts_search_->start_search(history, mcts_limits, mcts_callback, nullptr);

  // Periodically update shared state and check for AB policy updates
  int update_interval_ms = config_.policy_update_interval_ms;
  auto last_update = std::chrono::steady_clock::now();
  uint64_t last_ab_counter = 0;

  while (!mcts_done && !should_stop()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto now_time = std::chrono::steady_clock::now();
    auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now_time - last_update).count();

    if (since_update >= update_interval_ms) {
      // Publish MCTS state
      publish_mcts_state();

      // Check for AB updates and apply to MCTS policy
      uint64_t ab_counter = ab_state_.update_counter.load(std::memory_order_acquire);
      if (ab_counter > last_ab_counter) {
        update_mcts_policy_from_ab();
        last_ab_counter = ab_counter;
      }

      last_update = now_time;
    }
  }

  // Wait for MCTS to finish
  mcts_search_->wait();

  // Final state update
  publish_mcts_state();

  auto end = std::chrono::steady_clock::now();
  stats_.mcts_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  stats_.mcts_nodes = mcts_search_->stats().mcts_nodes.load();
  stats_.gpu_evaluations = mcts_search_->stats().nn_evaluations.load();
  stats_.gpu_batches = mcts_search_->stats().nn_batches.load();
  
  // ThreadGuard destructor will signal completion
}

void ParallelHybridSearch::publish_mcts_state() {
  if (!mcts_search_ || !mcts_search_->tree()) return;

  const HybridNode *root = mcts_search_->tree()->root();
  if (!root || !root->has_children()) return;

  // Find best move by visits
  int best_idx = -1;
  uint32_t best_visits = 0;
  float best_q = -2.0f;

  for (int i = 0; i < root->num_edges(); ++i) {
    const HybridEdge &edge = root->edges()[i];
    HybridNode *child = edge.child();
    if (child && child->n() > best_visits) {
      best_visits = child->n();
      best_idx = i;
      best_q = -child->q();
    }
  }

  if (best_idx >= 0) {
    Move best = root->edges()[best_idx].move().to_stockfish();
    mcts_state_.best_move_raw.store(best.raw(), std::memory_order_relaxed);
    mcts_state_.best_q.store(best_q, std::memory_order_relaxed);
    mcts_state_.best_visits.store(best_visits, std::memory_order_relaxed);
    mcts_state_.total_nodes.store(mcts_search_->stats().mcts_nodes.load(),
                                  std::memory_order_relaxed);
    mcts_state_.has_result.store(true, std::memory_order_release);

    // Update top moves
    std::vector<std::tuple<int, uint32_t, float, float>> moves;
    for (int i = 0; i < root->num_edges(); ++i) {
      const HybridEdge &edge = root->edges()[i];
      HybridNode *child = edge.child();
      if (child && child->n() > 0) {
        moves.emplace_back(i, child->n(), edge.policy(), -child->q());
      }
    }

    std::sort(moves.begin(), moves.end(),
              [](const auto &a, const auto &b) { return std::get<1>(a) > std::get<1>(b); });

    int num_top = std::min(static_cast<int>(moves.size()),
                           MCTSSharedState::MAX_TOP_MOVES);
    for (int i = 0; i < num_top; ++i) {
      auto [idx, visits, policy, q] = moves[i];
      Move m = root->edges()[idx].move().to_stockfish();
      mcts_state_.top_moves[i].move_raw.store(m.raw(), std::memory_order_relaxed);
      mcts_state_.top_moves[i].visits.store(visits, std::memory_order_relaxed);
      mcts_state_.top_moves[i].policy.store(policy, std::memory_order_relaxed);
      mcts_state_.top_moves[i].q.store(q, std::memory_order_relaxed);
    }
    mcts_state_.num_top_moves.store(num_top, std::memory_order_release);
    mcts_state_.update_counter.fetch_add(1, std::memory_order_release);
  }
}

void ParallelHybridSearch::update_mcts_policy_from_ab() {
  // This updates MCTS policy priors based on AB scores
  // The idea: AB provides accurate tactical evaluations that can guide MCTS
  
  if (!mcts_search_ || !mcts_search_->tree()) return;
  HybridNode *root = mcts_search_->tree()->root();
  if (!root || !root->has_children()) return;

  int num_scored = ab_state_.num_scored_moves.load(std::memory_order_acquire);
  if (num_scored == 0) return;

  // Collect AB scores
  std::vector<std::pair<Move, int>> ab_scores;
  int max_score = -32001;
  for (int i = 0; i < num_scored && i < ABSharedState::MAX_MOVES; ++i) {
    Move m(ab_state_.move_scores[i].move_raw.load(std::memory_order_relaxed));
    int score = ab_state_.move_scores[i].score.load(std::memory_order_relaxed);
    if (score > -32000) {
      ab_scores.emplace_back(m, score);
      max_score = std::max(max_score, score);
    }
  }

  if (ab_scores.empty()) return;

  // Update MCTS policy priors
  float weight = config_.ab_policy_weight;
  
  for (int i = 0; i < root->num_edges(); ++i) {
    HybridEdge &edge = root->edges()[i];
    Move m = edge.move().to_stockfish();
    
    // Find AB score for this move
    auto it = std::find_if(ab_scores.begin(), ab_scores.end(),
                           [m](const auto &p) { return p.first == m; });
    
    if (it != ab_scores.end()) {
      // Convert AB score to policy adjustment
      float ab_advantage = (it->second - max_score + 200) / 400.0f;
      ab_advantage = std::clamp(ab_advantage, 0.0f, 1.0f);
      
      float old_policy = edge.policy();
      float new_policy = (1.0f - weight) * old_policy + weight * ab_advantage;
      edge.set_policy(new_policy);
    }
  }

  // Renormalize policies
  float sum = 0.0f;
  for (int i = 0; i < root->num_edges(); ++i) {
    sum += root->edges()[i].policy();
  }
  if (sum > 0) {
    for (int i = 0; i < root->num_edges(); ++i) {
      root->edges()[i].set_policy(root->edges()[i].policy() / sum);
    }
  }

  stats_.policy_updates.fetch_add(1, std::memory_order_relaxed);
}

// AB thread - runs full Stockfish iterative deepening
void ParallelHybridSearch::ab_thread_main() {
  // RAII guard to ensure we always signal completion
  struct ThreadGuard {
    ParallelHybridSearch* self;
    ~ThreadGuard() {
      self->ab_state_.ab_running.store(false, std::memory_order_release);
      self->ab_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
      self->signal_thread_done(self->ab_thread_done_);
    }
  } guard{this};
  
  auto start = std::chrono::steady_clock::now();

  run_ab_search();

  auto end = std::chrono::steady_clock::now();
  stats_.ab_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  
  // ThreadGuard destructor will signal completion
}

void ParallelHybridSearch::run_ab_search() {
  if (!engine_) return;

  // Run iterative deepening search
  int depth = config_.ab_use_time ? 0 : config_.ab_min_depth;
  int time_ms = config_.ab_use_time ? time_budget_ms_ : 0;

  auto result = engine_->search_sync(root_fen_, depth, time_ms);

  if (result.best_move != Move::none()) {
    publish_ab_state(result.best_move, result.score, result.depth, result.nodes);
    
    // Update move scores for policy guidance
    // The PV gives us the best line
    for (size_t i = 0; i < result.pv.size() && i < 1; ++i) {
      ab_state_.update_move_score(result.pv[i], result.score, result.depth);
    }
  }

  stats_.ab_nodes = result.nodes;
  stats_.ab_depth = result.depth;
}

void ParallelHybridSearch::publish_ab_state(Move best, int score, int depth,
                                            uint64_t nodes) {
  ab_state_.set_best_move(best, score, depth, nodes);
}

// Coordinator thread - monitors both searches and makes final decision
void ParallelHybridSearch::coordinator_thread_main() {
  // RAII guard to ensure we always signal completion
  struct ThreadGuard {
    ParallelHybridSearch* self;
    ~ThreadGuard() {
      self->coordinator_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
      self->searching_.store(false, std::memory_order_release);
      self->signal_thread_done(self->coordinator_thread_done_);
    }
  } guard{this};
  
  auto start = std::chrono::steady_clock::now();

  // Wait for search to complete or time to expire
  while (!should_stop()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Check if both searches have results
    bool mcts_done = !mcts_state_.mcts_running.load(std::memory_order_acquire);
    bool ab_done = !ab_state_.ab_running.load(std::memory_order_acquire);

    // Send periodic info updates
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    
    if (ms > 0 && (ms % 500) < 15) {
      // Send combined info
      uint64_t total_nodes = stats_.mcts_nodes + stats_.ab_nodes;
      int ab_depth = ab_state_.completed_depth.load(std::memory_order_relaxed);
      int ab_score = ab_state_.best_score.load(std::memory_order_relaxed);
      
      if (ab_state_.has_result.load(std::memory_order_acquire)) {
        Move ab_best = ab_state_.get_best_move();
        std::vector<Move> pv;
        pv.push_back(ab_best);
        send_info(ab_depth, ab_score, total_nodes, static_cast<int>(ms), pv, "hybrid");
      }
    }

    if (mcts_done && ab_done) {
      break;
    }
  }

  // Signal stop to all threads
  stop_flag_.store(true, std::memory_order_release);
  
  // Stop MCTS search safely
  if (mcts_search_) {
    mcts_search_->stop();
  }
  
  // NOTE: Do NOT call engine_->stop() here - it causes race conditions
  // The AB thread will naturally finish when search_sync completes
  // Just wait for it to finish

  // Wait for MCTS and AB threads to finish before making decision
  // This ensures we have their final results
  int wait_count = 0;
  while ((!mcts_thread_done_.load(std::memory_order_acquire) ||
          !ab_thread_done_.load(std::memory_order_acquire)) && wait_count < 500) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    wait_count++;
  }

  // Make final decision
  Move final_move = make_final_decision();
  final_best_move_.store(final_move.raw(), std::memory_order_release);

  // Get ponder move
  Move ponder_move = Move::none();
  {
    std::lock_guard<std::mutex> lock(pv_mutex_);
    if (final_pv_.size() > 1) {
      ponder_move = final_pv_[1];
      final_ponder_move_.store(ponder_move.raw(), std::memory_order_release);
    }
  }

  auto end = std::chrono::steady_clock::now();
  stats_.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

  // Report final stats
  send_info_string("Final: MCTS=" + std::to_string(stats_.mcts_nodes.load()) +
                   " AB=" + std::to_string(stats_.ab_nodes.load()) +
                   " agreements=" + std::to_string(stats_.move_agreements.load()) +
                   " overrides=" + std::to_string(stats_.ab_overrides.load()));

  // Invoke callback safely (exactly once)
  invoke_best_move_callback(final_move, ponder_move);
  
  // ThreadGuard destructor will signal completion
}

void ParallelHybridSearch::invoke_best_move_callback(Move best, Move ponder) {
  // Ensure callback is invoked exactly once
  bool expected = false;
  if (!callback_invoked_.compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel)) {
    return;  // Already invoked
  }
  
  // Get callback under lock, then invoke outside lock to avoid deadlock
  BestMoveCallback callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = best_move_callback_;
    // Clear callback after capturing to prevent double invocation
    best_move_callback_ = nullptr;
  }
  
  if (callback) {
    callback(best, ponder);
  }
}

Move ParallelHybridSearch::make_final_decision() {
  Move mcts_best = mcts_state_.get_best_move();
  Move ab_best = ab_state_.get_best_move();
  int ab_score = ab_state_.best_score.load(std::memory_order_relaxed);
  float mcts_q = mcts_state_.best_q.load(std::memory_order_relaxed);
  int ab_depth = ab_state_.completed_depth.load(std::memory_order_relaxed);
  uint32_t mcts_visits = mcts_state_.best_visits.load(std::memory_order_relaxed);

  // If only one has a result, use it
  if (mcts_best == Move::none() && ab_best != Move::none()) {
    stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(pv_mutex_);
      final_pv_.clear();
      final_pv_.push_back(ab_best);
    }
    return ab_best;
  }
  if (ab_best == Move::none() && mcts_best != Move::none()) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(pv_mutex_);
      final_pv_.clear();
      final_pv_.push_back(mcts_best);
    }
    return mcts_best;
  }
  if (mcts_best == Move::none() && ab_best == Move::none()) {
    // Fallback: find any legal move
    Position pos;
    StateInfo st;
    pos.set(root_fen_, false, &st);
    MoveList<LEGAL> moves(pos);
    if (moves.size() > 0) {
      return *moves.begin();
    }
    return Move::none();
  }

  // Both have results - decide based on mode
  if (mcts_best == ab_best) {
    stats_.move_agreements.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(pv_mutex_);
      final_pv_.clear();
      final_pv_.push_back(ab_best);
    }
    return ab_best;
  }

  // Moves disagree - need to decide
  // Use Lc0-style Q to centipawn conversion
  int mcts_cp = QToNnueScore(mcts_q);
  
  float diff_pawns = std::abs(ab_score - mcts_cp) / 100.0f;

  // Calculate confidence metrics
  float ab_confidence = std::min(1.0f, static_cast<float>(ab_depth) / 20.0f);
  float mcts_confidence = std::min(1.0f, static_cast<float>(mcts_visits) / 50000.0f);

  Move chosen;
  
  switch (config_.decision_mode) {
  case ParallelHybridConfig::DecisionMode::AB_PRIMARY:
    // Always trust AB unless MCTS strongly disagrees
    if (mcts_cp > ab_score + 150 && mcts_confidence > 0.5f) {
      chosen = mcts_best;
      stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    } else {
      chosen = ab_best;
      stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
    }
    break;

  case ParallelHybridConfig::DecisionMode::MCTS_PRIMARY:
    // Trust MCTS unless AB strongly disagrees
    if (ab_score > mcts_cp + 100 && ab_confidence > 0.5f) {
      chosen = ab_best;
      stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
    } else {
      chosen = mcts_best;
      stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    }
    break;

  case ParallelHybridConfig::DecisionMode::DYNAMIC:
    // Use position type to decide - IMPROVED thresholds
    if (current_strategy_.position_type == PositionType::HIGHLY_TACTICAL ||
        current_strategy_.position_type == PositionType::TACTICAL) {
      // Tactical positions: trust AB more (it's better at tactics)
      // But only override if AB is significantly better
      if (ab_score > mcts_cp + 50) {  // 0.5 pawn advantage
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      } else if (mcts_cp > ab_score + 100 && mcts_confidence > 0.3f) {
        // MCTS found something AB missed
        chosen = mcts_best;
        stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Close - trust AB in tactical positions
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      // Strategic positions: trust MCTS more (it's better at long-term planning)
      // But respect AB's tactical corrections
      if (ab_score > mcts_cp + 150) {  // 1.5 pawn - AB found a tactic
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Trust MCTS for strategic decisions
        chosen = mcts_best;
        stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      }
    }
    break;

  case ParallelHybridConfig::DecisionMode::VOTE_WEIGHTED:
  default:
    // Weighted decision based on confidence
    // For losing positions, we want to choose the LEAST BAD option
    // with proper confidence weighting.
    //
    // The key insight: compare confidence-weighted PREFERENCES, not raw scores.
    // Higher confidence should make us more likely to trust that assessment,
    // regardless of whether the position is winning or losing.
    
    // Compute weighted scores using absolute values to handle negative scores correctly
    // We compare which engine is MORE CONFIDENT that their move is BETTER (or less bad)
    float ab_conf = ab_confidence;
    float mcts_conf = mcts_confidence;
    
    // Normalize confidences
    float conf_sum = ab_conf + mcts_conf;
    if (conf_sum > 0) {
      ab_conf /= conf_sum;
      mcts_conf /= conf_sum;
    } else {
      ab_conf = 0.5f;
      mcts_conf = 0.5f;
    }
    
    // Decision: trust the engine with higher confidence, but only if scores
    // are reasonably close. If scores differ significantly, trust the one
    // claiming a better position (accounting for confidence).
    //
    // For close scores (within ~50cp), prefer higher confidence.
    // For divergent scores, weight both confidence AND score difference.
    int score_diff = ab_score - mcts_cp;
    float score_diff_pawns = std::abs(score_diff) / 100.0f;
    
    if (score_diff_pawns < 0.5f) {
      // Scores are close - trust higher confidence
      if (ab_conf > mcts_conf + 0.1f) {
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      } else if (mcts_conf > ab_conf + 0.1f) {
        chosen = mcts_best;
        stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Very close confidence - default to AB (more reliable for tactics)
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      // Scores diverge significantly - weight confidence against score difference
      // Use confidence as a "discount factor" on the claimed advantage
      // Higher confidence means we trust the score more
      float ab_effective = ab_score * ab_conf;
      float mcts_effective = mcts_cp * mcts_conf;
      
      if (ab_effective > mcts_effective) {
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      } else if (mcts_effective > ab_effective) {
        chosen = mcts_best;
        stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Equal - default to AB
        chosen = ab_best;
        stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      }
    }
    break;
  }

  {
    std::lock_guard<std::mutex> lock(pv_mutex_);
    final_pv_.clear();
    final_pv_.push_back(chosen);
  }

  return chosen;
}

void ParallelHybridSearch::send_info(int depth, int score, uint64_t nodes,
                                     int time_ms, const std::vector<Move> &pv,
                                     const std::string &source) {
  // Get callback under lock
  InfoCallback callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = info_callback_;
  }
  
  if (!callback) return;

  std::ostringstream ss;
  ss << "info depth " << depth;
  ss << " score cp " << score;
  ss << " nodes " << nodes;
  ss << " time " << time_ms;

  if (time_ms > 0) {
    uint64_t nps = nodes * 1000 / time_ms;
    ss << " nps " << nps;
  }

  if (!pv.empty()) {
    ss << " pv";
    for (const auto &m : pv) {
      ss << " " << UCIEngine::move(m, false);
    }
  }

  ss << " string " << source;

  callback(ss.str());
}

void ParallelHybridSearch::send_info_string(const std::string &msg) {
  // Get callback under lock
  InfoCallback callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = info_callback_;
  }
  
  if (callback) {
    callback("info string " + msg);
  }
}

// ============================================================================
// GPU Optimization Helpers (Apple Silicon)
// ============================================================================

bool ParallelHybridSearch::initialize_gpu_batches() {
  if (!GPU::gpu_available()) {
    return false;
  }
  
  // Initialize double-buffered GPU-resident batches
  // Double buffering allows us to fill one batch while the other is being evaluated
  bool success = true;
  for (int i = 0; i < 2; ++i) {
    if (!gpu_batch_[i].initialize(config_.gpu_batch_size)) {
      success = false;
      break;
    }
  }
  
  if (success && has_unified_memory_) {
    send_info_string("GPU unified memory: enabled (zero-copy batches)");
  }
  
  return success;
}

void ParallelHybridSearch::submit_gpu_batch_async(int batch_idx,
    std::function<void(bool)> completion_handler) {
  if (!gpu_manager_ || batch_idx < 0 || batch_idx > 1) {
    if (completion_handler) completion_handler(false);
    return;
  }
  
  GPUResidentBatch& batch = gpu_batch_[batch_idx];
  if (batch.count == 0) {
    if (completion_handler) completion_handler(true);
    return;
  }
  
  // Create evaluation batch from GPU-resident data
  GPU::GPUEvalBatch eval_batch;
  eval_batch.reserve(batch.count);
  
  // Since we're using unified memory, the positions are already in GPU-accessible memory
  // We just need to set up the batch metadata
  if (batch.positions_buffer && batch.positions_buffer->valid()) {
    auto* positions = batch.positions_buffer->as<GPU::GPUPositionData>();
    for (int i = 0; i < batch.count; ++i) {
      eval_batch.add_position_data(positions[i]);
    }
  }
  
  // Track pending evaluations
  pending_evaluations_.fetch_add(1, std::memory_order_relaxed);
  
  // Capture batch count before moving eval_batch (batch is a member reference, safe to capture)
  const int batch_count = batch.count;
  
  // Submit async evaluation - move eval_batch into the lambda to avoid use-after-free
  // (eval_batch is a local variable that would go out of scope before the async callback)
  bool started = gpu_manager_->evaluate_batch_async(eval_batch,
      [this, batch_idx, completion_handler, &batch, batch_count,
       eval_batch = std::move(eval_batch)](bool success) mutable {
        // Copy results back (unified memory = no actual copy needed)
        if (success && batch.results_buffer && batch.results_buffer->valid()) {
          auto* results = batch.results_buffer->as<int32_t>();
          for (int i = 0; i < batch_count; ++i) {
            results[i * 2] = eval_batch.psqt_scores[i];
            results[i * 2 + 1] = eval_batch.positional_scores[i];
          }
        }
        
        // Decrement pending count
        pending_evaluations_.fetch_sub(1, std::memory_order_relaxed);
        
        // Notify completion
        {
          std::lock_guard<std::mutex> lock(async_mutex_);
          async_cv_.notify_all();
        }
        
        if (completion_handler) {
          completion_handler(success);
        }
      });
  
  if (!started) {
    pending_evaluations_.fetch_sub(1, std::memory_order_relaxed);
    if (completion_handler) completion_handler(false);
  }
}

int ParallelHybridSearch::swap_batch() {
  int old_batch = current_batch_.load(std::memory_order_relaxed);
  int new_batch = 1 - old_batch;
  current_batch_.store(new_batch, std::memory_order_release);
  
  // Clear the new batch for filling
  gpu_batch_[new_batch].clear();
  
  return old_batch;
}

void ParallelHybridSearch::wait_gpu_evaluations() {
  std::unique_lock<std::mutex> lock(async_mutex_);
  async_cv_.wait(lock, [this]() {
    return pending_evaluations_.load(std::memory_order_relaxed) == 0;
  });
}

// Factory function
std::unique_ptr<ParallelHybridSearch>
create_parallel_hybrid_search(GPU::GPUNNUEManager *gpu_manager, Engine *engine,
                              const ParallelHybridConfig &config) {
  auto search = std::make_unique<ParallelHybridSearch>();
  search->set_config(config);
  if (search->initialize(gpu_manager, engine)) {
    return search;
  }
  return nullptr;
}

} // namespace MCTS
} // namespace MetalFish
