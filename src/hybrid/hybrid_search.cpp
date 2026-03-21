/*
  MetalFish - Parallel Hybrid Search Implementation
  Copyright (C) 2025 Nripesh Niketan

  Optimized for Apple Silicon with unified memory architecture.

  This implementation incorporates state-of-the-art MCTS algorithms,
  including PUCT with logarithmic growth, FPU reduction strategy, and
  moves left head (MLH) utility.

  Licensed under GPL-3.0
*/

#include "hybrid_search.h"
#include "../core/misc.h"
#include "../eval/evaluate.h"
#include "../mcts/core.h"
#include "../uci/engine.h"
#include "../uci/uci.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <type_traits>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

namespace MetalFish {
namespace MCTS {

// ============================================================================
// ParallelHybridSearch Implementation
// ============================================================================

ParallelHybridSearch::ParallelHybridSearch() {
  // MCTS parameters
  config_.mcts_config.cpuct = 1.745f;
  config_.mcts_config.cpuct_at_root = 2.15f;
  config_.mcts_config.fpu_reduction = 0.330f;
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

  // Join all our threads FIRST before touching MCTS
  // This ensures our threads aren't using MCTS while we destroy it
  join_all_threads();

  // Now stop MCTS search if running
  if (mcts_search_) {
    // Clear callbacks BEFORE calling wait to prevent crash from invalid
    // callback
      mcts_search_->ClearCallbacks();
    mcts_search_->Stop();
    mcts_search_->Wait();
  }

  // Apple Silicon Optimization: Only synchronize if absolutely necessary
  // On unified memory systems, we don't need heavy synchronization since
  // CPU and GPU share the same memory. We only sync if there were pending
  // GPU operations that might still be in flight.
  if (GPU::gpu_available() && !GPU::gpu_backend_shutdown()) {
    // Light synchronization - just ensure command buffers are committed
    // This is faster than full synchronization on Apple Silicon
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
  thread_cv_.wait_until(lock, deadline,
                        [this]() { return all_threads_done(); });

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

void ParallelHybridSearch::signal_thread_done(std::atomic<bool> &done_flag) {
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
  if (!engine) {
    return false;
  }

  gpu_manager_ =
      gpu_manager; // May be nullptr -- that's OK if transformer is loaded
  engine_ = engine;

  // Create GPU MCTS backend (optional -- only if GPU NNUE is available)
  if (gpu_manager && gpu_manager->is_ready()) {
    gpu_backend_ = GPU::create_gpu_mcts_backend(gpu_manager);
    if (gpu_backend_) {
      gpu_backend_->set_optimal_batch_size(config_.gpu_batch_size);
    }
  }

  // Create MCTS search engine
  // This loads the transformer network from config_.mcts_config.nn_weights_path
  mcts_search_ = std::make_unique<Search>(config_.mcts_config,
      std::make_unique<Backend>(config_.mcts_config.nn_weights_path));

  if (engine_) {
    shared_tt_reader_ = std::make_unique<SharedTTReader>(&engine_->get_tt());
    mcts_search_->SetSharedTT(shared_tt_reader_.get());
  }

  // Initialize shared state
  ab_state_.reset();
  mcts_state_.reset();

  initialized_ = true;
  return true;
}

void ParallelHybridSearch::start_search(const Position &pos,
                                        const ::MetalFish::Search::LimitsType &limits,
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
    // Cannot start search without initialization - mcts_search_ would be
    // nullptr
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

  nn_policy_hints_.clear();

  // Analyze position for strategy
  if (config_.use_position_classifier) {
    PositionFeatures features = classifier_.analyze(pos);
    current_strategy_ = strategy_selector_.get_strategy(features);

    Color us = pos.side_to_move();
    int time_left = (us == WHITE) ? limits.time[WHITE] : limits.time[BLACK];
    int increment = (us == WHITE) ? limits.inc[WHITE] : limits.inc[BLACK];
    strategy_selector_.adjust_for_time(current_strategy_, time_left, increment);

    send_info_string(
        "Parallel search - Position: " +
        std::string(current_strategy_.position_type ==
                            PositionType::HIGHLY_TACTICAL
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
    coordinator_thread_state_.store(ThreadState::RUNNING,
                                    std::memory_order_release);

    mcts_thread_ = std::thread(&ParallelHybridSearch::mcts_thread_main, this);
    ab_thread_ = std::thread(&ParallelHybridSearch::ab_thread_main, this);
    coordinator_thread_ =
        std::thread(&ParallelHybridSearch::coordinator_thread_main, this);
  }
}

void ParallelHybridSearch::stop() {
  // Signal stop
  stop_flag_.store(true, std::memory_order_release);

  // Stop MCTS search
  if (mcts_search_) {
    mcts_search_->Stop();
  }

  // Stop AB search immediately -- engine_->stop() sets threads.stop = true
  // which the AB search checks at every node. This ensures the AB thread
  // winds down in <1ms rather than waiting for the polling loop.
  if (engine_) {
    engine_->stop();
  }
}

void ParallelHybridSearch::wait() {
  // Wait for threads to complete with a short timeout.
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);

  while (!all_threads_done()) {
    if (std::chrono::steady_clock::now() > deadline) {
      std::cerr << "[HYB] wait() TIMEOUT - coord_done="
                << coordinator_thread_done_.load()
                << " mcts_done=" << mcts_thread_done_.load()
                << " ab_done=" << ab_thread_done_.load() << std::endl;
      stop_flag_.store(true, std::memory_order_release);
      if (mcts_search_)
        mcts_search_->Stop();
      if (engine_)
        engine_->stop();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  // Join all threads. If they're not done, wait for them.
  // Never detach -- detached threads can access destroyed objects.
  {
    std::lock_guard<std::mutex> lock(thread_mutex_);

    if (coordinator_thread_.joinable())
      coordinator_thread_.join();
    if (mcts_thread_.joinable())
      mcts_thread_.join();
    if (ab_thread_.joinable())
      ab_thread_.join();

    // Reset thread states
    mcts_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
    ab_thread_state_.store(ThreadState::IDLE, std::memory_order_release);
    coordinator_thread_state_.store(ThreadState::IDLE,
                                    std::memory_order_release);
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
    mcts_search_ = std::make_unique<Search>(config_.mcts_config,
      std::make_unique<Backend>(config_.mcts_config.nn_weights_path));
    if (engine_) {
      shared_tt_reader_ = std::make_unique<SharedTTReader>(&engine_->get_tt());
      mcts_search_->SetSharedTT(shared_tt_reader_.get());
    }
  }
  // Search manages its own TT internally
}

int ParallelHybridSearch::calculate_time_budget() const {
  if (limits_.movetime > 0) {
    return limits_.movetime;
  }
  if (limits_.infinite || limits_.nodes > 0) {
    return 0;
  }

  Position pos;
  StateInfo st;
  pos.set(root_fen_, false, &st);
  Color us = pos.side_to_move();

  int time_left = (us == WHITE) ? limits_.time[WHITE] : limits_.time[BLACK];
  int increment = (us == WHITE) ? limits_.inc[WHITE] : limits_.inc[BLACK];

  if (time_left <= 0) {
    return std::max(1000, increment);
  }

  // Hard safety cap: 42% of remaining time (Lc0's max-move-budget).
  // The individual engines (AB via Stockfish TM, MCTS via Lc0 smooth TM)
  // handle their own time allocation — this is just a safety backstop
  // so the hybrid coordinator can force-stop if both engines overrun.
  int hard_cap = static_cast<int>(time_left * 0.42f);
  return std::max(500, hard_cap);
}

bool ParallelHybridSearch::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire))
    return true;

  if (limits_.nodes > 0) {
    uint64_t mcts_n = mcts_search_ ?
        mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed) : 0;
    uint64_t total = mcts_n + ab_state_.nodes_searched.load(std::memory_order_relaxed);
    if (total >= limits_.nodes)
      return true;
  }

  if (time_budget_ms_ > 0) {
    auto elapsed = std::chrono::steady_clock::now() - search_start_;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (ms >= time_budget_ms_)
      return true;
  }

  return false;
}

// MCTS thread - runs GPU-accelerated MCTS
void ParallelHybridSearch::mcts_thread_main() {
  // RAII guard to ensure we always signal completion
  struct ThreadGuard {
    ParallelHybridSearch *self;
    ~ThreadGuard() {
      self->mcts_state_.mcts_running.store(false, std::memory_order_release);
      self->mcts_thread_state_.store(ThreadState::IDLE,
                                     std::memory_order_release);
      self->signal_thread_done(self->mcts_thread_done_);
    }
  } guard{this};

  auto start = std::chrono::steady_clock::now();

  // Pass the real time controls so MCTS's Lc0-style smooth time manager
  // handles allocation (with piggybank, NPS estimation, tree reuse, etc.)
  ::MetalFish::Search::LimitsType mcts_limits = limits_;
  mcts_limits.startTime = now();

  Move best_move = Move::none();
  std::atomic<bool> mcts_done{false};

  // Search uses Move natively
  auto mcts_callback = [&](Move move, Move ponder) {
    best_move = move;
    mcts_done = true;
  };

  // Start MCTS search with FEN string.
  // Note: don't call start_search() which internally does stop()+wait() --
  // that would block if the previous eval thread is in a GPU call.
  // Instead, the hybrid's own start_search() already stopped everything.
  mcts_search_->Stop();
  mcts_search_->StartSearch(root_fen_, mcts_limits, mcts_callback, nullptr);

  // Periodically update shared state and check for AB policy updates
  int update_interval_ms = config_.policy_update_interval_ms;
  auto last_update = std::chrono::steady_clock::now();
  uint64_t last_ab_counter = 0;

  while (!mcts_done && !should_stop()) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    auto now_time = std::chrono::steady_clock::now();
    auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now_time - last_update)
                            .count();

    if (since_update >= update_interval_ms) {
      // Publish MCTS state
      publish_mcts_state();

      // Check for AB updates and apply to MCTS policy
      uint64_t ab_counter =
          ab_state_.update_counter.load(std::memory_order_acquire);
      if (ab_counter > last_ab_counter) {
        update_mcts_policy_from_ab();
        last_ab_counter = ab_counter;
      }

      last_update = now_time;
    }
  }

  // Wait for MCTS to finish
  mcts_search_->Wait();

  // Final state update
  publish_mcts_state();

  auto end = std::chrono::steady_clock::now();
  stats_.mcts_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  stats_.mcts_nodes = mcts_search_->Stats().total_nodes.load();
  stats_.gpu_evaluations = mcts_search_->Stats().nn_evaluations.load();
  stats_.gpu_batches = mcts_search_->Stats().nn_evaluations.load();

  // ThreadGuard destructor will signal completion
}

void ParallelHybridSearch::publish_mcts_state() {
  // Use GetBestMove() for the MCTS result
  if (!mcts_search_)
    return;

  Move best = mcts_search_->GetBestMove();
  if (best == Move::none())
    return;

  float best_q = mcts_search_->GetBestQ();

  mcts_state_.best_move_raw.store(best.raw(), std::memory_order_relaxed);
  mcts_state_.best_q.store(best_q, std::memory_order_relaxed);
  mcts_state_.best_visits.store(mcts_search_->Stats().total_nodes.load(),
                                std::memory_order_relaxed);
  mcts_state_.total_nodes.store(mcts_search_->Stats().total_nodes.load(),
                                std::memory_order_relaxed);
  mcts_state_.has_result.store(true, std::memory_order_release);
  mcts_state_.update_counter.fetch_add(1, std::memory_order_release);
}

void ParallelHybridSearch::update_mcts_policy_from_ab() {
  int depth = ab_state_.pv_depth.load(std::memory_order_acquire);
  if (depth < 10) return;

  uint64_t mcts_nodes = mcts_search_ ? mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed) : 0;
  if (mcts_nodes < 100) return;

  int pv_len = ab_state_.pv_length.load(std::memory_order_acquire);
  if (pv_len <= 0 || !mcts_search_)
    return;

  Move pv[ABSharedState::MAX_PV];
  for (int i = 0; i < pv_len; ++i) {
    pv[i] = Move(ab_state_.pv_moves[i].load(std::memory_order_relaxed));
  }
  mcts_search_->InjectPVBoost(pv, pv_len, depth);
  stats_.policy_updates.fetch_add(1, std::memory_order_relaxed);
}

// AB thread - runs full alpha-beta iterative deepening
void ParallelHybridSearch::ab_thread_main() {
  // RAII guard to ensure we always signal completion
  struct ThreadGuard {
    ParallelHybridSearch *self;
    ~ThreadGuard() {
      self->ab_state_.ab_running.store(false, std::memory_order_release);
      self->ab_thread_state_.store(ThreadState::IDLE,
                                   std::memory_order_release);
      self->signal_thread_done(self->ab_thread_done_);
    }
  } guard{this};

  auto start = std::chrono::steady_clock::now();

  run_ab_search();

  auto end = std::chrono::steady_clock::now();
  stats_.ab_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // ThreadGuard destructor will signal completion
}

void ParallelHybridSearch::run_ab_search() {
  if (!engine_)
    return;

  // Set up position using the standard Engine interface
  engine_->set_position(root_fen_, {});

  // Pass the real time controls so Stockfish's own time manager handles
  // move allocation (with move overhead, ponder support, instamove, etc.)
  // instead of forcing a flat movetime budget.
  ::MetalFish::Search::LimitsType ab_limits = limits_;
  ab_limits.startTime = now();

  // Suppress bestmove output -- the coordinator handles it
  auto saved_bestmove = engine_->get_on_bestmove();
  auto saved_update_full = engine_->get_on_update_full();
  Move ab_best_move = Move::none();
  int ab_score = 0;
  int ab_depth = 0;

  // Hook into the per-depth update to capture the actual score.
  // InfoFull inherits InfoShort which carries `depth` and `score`.
  engine_->set_on_update_full(
      [this, &ab_score, &ab_depth](const Engine::InfoFull &info) {
        ab_score = info.score.visit([](auto &&val) -> int {
          using T = std::decay_t<decltype(val)>;
          if constexpr (std::is_same_v<T, Score::InternalUnits>)
            return val.value;
          else if constexpr (std::is_same_v<T, Score::Mate>)
            return val.plies > 0 ? 30000 - val.plies : -30000 - val.plies;
          else if constexpr (std::is_same_v<T, Score::Tablebase>)
            return val.win ? 20000 - val.plies : -20000 + val.plies;
          else
            return 0;
        });
        ab_depth = info.depth;
        publish_ab_state(ab_state_.get_best_move(), ab_score, ab_depth,
                         engine_->threads_nodes_searched());
      });

  engine_->set_on_bestmove([this, &ab_best_move, &ab_score, &ab_depth](
                               std::string_view bestmove, std::string_view) {
    Position pos;
    StateInfo st;
    pos.set(root_fen_, false, &st);
    ab_best_move = UCIEngine::to_move(pos, std::string(bestmove));
    if (ab_best_move != Move::none()) {
      publish_ab_state(ab_best_move, ab_score, ab_depth,
                       engine_->threads_nodes_searched());
    }
  });

  // Standard search path -- no state corruption
  engine_->go(ab_limits);
  engine_->wait_for_search_finished();

  // Restore callbacks
  engine_->set_on_update_full(std::move(saved_update_full));
  engine_->set_on_bestmove(std::move(saved_bestmove));
}

void ParallelHybridSearch::publish_ab_state(Move best, int score, int depth,
                                            uint64_t nodes) {
  ab_state_.set_best_move(best, score, depth, nodes);
}

// Coordinator thread - monitors both searches and makes final decision
void ParallelHybridSearch::coordinator_thread_main() {
  // RAII guard to ensure we always signal completion
  struct ThreadGuard {
    ParallelHybridSearch *self;
    ~ThreadGuard() {
      self->coordinator_thread_state_.store(ThreadState::IDLE,
                                            std::memory_order_release);
      self->searching_.store(false, std::memory_order_release);
      self->signal_thread_done(self->coordinator_thread_done_);
    }
  } guard{this};

  auto start = std::chrono::steady_clock::now();
  int agreement_count = 0;

  uint32_t last_ab_move_raw = 0;
  uint32_t last_mcts_move_raw = 0;
  int64_t last_info_ms = 0;

  // Wait for search to complete or time to expire
  while (!should_stop()) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    // Check if both searches have results
    bool mcts_done = !mcts_state_.mcts_running.load(std::memory_order_acquire);
    bool ab_done = !ab_state_.ab_running.load(std::memory_order_acquire);

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Agreement-based early stopping: if both engines agree on the same
    // move for several consecutive checks, we can stop early and save time.
    uint32_t ab_move = ab_state_.best_move_raw.load(std::memory_order_relaxed);
    uint32_t mcts_move =
        mcts_state_.best_move_raw.load(std::memory_order_relaxed);

    if (ab_move != 0 && mcts_move != 0) {
      if (ab_move == mcts_move) {
        agreement_count++;
        // Both agree for 3+ checks AND we've used at least 25% of time
        if (agreement_count >= 3 && ms > time_budget_ms_ / 4) {
          send_info_string("Hybrid: engines agree, stopping early at " +
                           std::to_string(ms) + "ms");
          break;
        }
      } else {
        agreement_count = 0;
      }
    }

    // Send combined info every ~500ms (fixed timing)
    if (ms - last_info_ms >= 500) {
      last_info_ms = ms;
      uint64_t mcts_nodes = mcts_search_ ?
          mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed) : 0;
      uint64_t ab_nodes = ab_state_.nodes_searched.load(std::memory_order_relaxed);
      uint64_t total_nodes = mcts_nodes + ab_nodes;
      int ab_depth = ab_state_.completed_depth.load(std::memory_order_relaxed);
      int ab_score = ab_state_.best_score.load(std::memory_order_relaxed);

      if (ab_state_.has_result.load(std::memory_order_acquire)) {
        Move ab_best = ab_state_.get_best_move();
        std::vector<Move> pv;
        pv.push_back(ab_best);
        send_info(ab_depth, ab_score, total_nodes, static_cast<int>(ms), pv,
                  "hybrid");
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
    mcts_search_->Stop();
  }

  // Stop AB search immediately so it winds down before we read its result
  if (engine_) {
    engine_->stop();
  }

  // Wait for MCTS and AB threads to finish before making decision.
  // After external stop: emit bestmove immediately using AB result.
  // Don't wait for MCTS -- GPU inference can't be interrupted.
  int max_wait = stop_flag_.load(std::memory_order_acquire) ? 100 : 4000;
  int wait_count = 0;
  // Only wait for AB thread (MCTS might be stuck in GPU).
  while (!ab_thread_done_.load(std::memory_order_acquire) &&
         wait_count < max_wait) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    wait_count++;
  }

  // Make final decision
  Move final_move = make_final_decision();

  // Guard: if no valid move, try to find any legal move
  if (final_move == Move::none()) {
    Position pos;
    StateInfo st;
    pos.set(root_fen_, false, &st);
    MoveList<LEGAL> moves(pos);
    if (moves.size() > 0) {
      final_move = *moves.begin();
    }
  }

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
  stats_.total_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Report final stats
  send_info_string(
      "Final: MCTS=" + std::to_string(stats_.mcts_nodes.load()) +
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
    return; // Already invoked
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
  Move ab_best = ab_state_.get_best_move();
  Move mcts_best = mcts_state_.get_best_move();
  int ab_score = ab_state_.best_score.load(std::memory_order_relaxed);
  float mcts_q = mcts_state_.best_q.load(std::memory_order_relaxed);
  uint32_t mcts_visits = mcts_state_.best_visits.load(std::memory_order_relaxed);

  if (ab_best == Move::none() && mcts_best != Move::none()) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    return mcts_best;
  }
  if (mcts_best == Move::none() || ab_best == Move::none()) {
    if (ab_best != Move::none()) {
      stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      return ab_best;
    }
    Position pos;
    StateInfo st;
    pos.set(root_fen_, false, &st);
    MoveList<LEGAL> moves(pos);
    return moves.size() > 0 ? *moves.begin() : Move::none();
  }

  if (ab_best == mcts_best) {
    stats_.move_agreements.fetch_add(1, std::memory_order_relaxed);
    return ab_best;
  }

  int mcts_cp = QToNnueScore(mcts_q);
  bool mcts_reliable = mcts_visits > 5000;
  bool mcts_much_better = mcts_cp > ab_score + 200;
  bool is_tactical = (current_strategy_.position_type == PositionType::HIGHLY_TACTICAL ||
                      current_strategy_.position_type == PositionType::TACTICAL);

  if (mcts_reliable && mcts_much_better && !is_tactical) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    return mcts_best;
  }

  stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
  return ab_best;
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

  if (!callback)
    return;

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
    bool has_valid = false;
    for (const auto &m : pv) {
      if (m != Move::none()) { has_valid = true; break; }
    }
    if (has_valid) {
      ss << " pv";
      for (const auto &m : pv) {
        if (m != Move::none())
          ss << " " << UCIEngine::move(m, false);
      }
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
