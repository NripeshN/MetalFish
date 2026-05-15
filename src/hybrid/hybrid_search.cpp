/*
  MetalFish - Parallel Hybrid Search Implementation
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "hybrid_search.h"
#include "../core/misc.h"
#include "../eval/evaluate.h"
#include "../mcts/core.h"
#include "../uci/engine.h"
#include "../uci/uci.h"
#include <algorithm>
#include <cassert>
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

#ifdef __APPLE__
namespace {
inline void set_thread_qos(qos_class_t qos) {
  pthread_set_qos_class_self_np(qos, 0);
}
} // namespace
#endif

ParallelHybridSearch::ParallelHybridSearch() {
  config_.mcts_config.cpuct = 1.745f;
  config_.mcts_config.cpuct_at_root = 1.745f;
  config_.mcts_config.fpu_reduction = 0.330f;
  config_.mcts_config.cpuct_base = 38739.0f;
  config_.mcts_config.cpuct_factor = 3.894f;

  config_.ab_min_depth = 8;
  config_.agreement_threshold = 0.3f;
  config_.override_threshold = 1.0f;

  config_.transformer_batch_size = 128;
  config_.use_transformer_prefetch = true;

  mcts_thread_state_.store(ThreadState::IDLE, std::memory_order_relaxed);
  ab_thread_state_.store(ThreadState::IDLE, std::memory_order_relaxed);
  coordinator_thread_state_.store(ThreadState::IDLE, std::memory_order_relaxed);
  mcts_thread_done_.store(true, std::memory_order_relaxed);
  ab_thread_done_.store(true, std::memory_order_relaxed);
  coordinator_thread_done_.store(true, std::memory_order_relaxed);
}

ParallelHybridSearch::~ParallelHybridSearch() {
  shutdown_requested_.store(true, std::memory_order_release);
  stop_flag_.store(true, std::memory_order_release);
  searching_.store(false, std::memory_order_release);

  // Join our threads before touching MCTS to avoid use-after-free
  join_all_threads();

  if (mcts_search_) {
    mcts_search_->ClearCallbacks();
    mcts_search_->Stop();
    mcts_search_->Wait();
  }

  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = nullptr;
    info_callback_ = nullptr;
  }
}

void ParallelHybridSearch::join_all_threads() {
  std::unique_lock<std::mutex> lock(thread_mutex_);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  thread_cv_.wait_until(lock, deadline,
                        [this]() { return all_threads_done(); });

  lock.unlock();

  if (coordinator_thread_.joinable()) {
    coordinator_thread_.join();
  }
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

bool ParallelHybridSearch::initialize(Engine *engine) {
  if (!engine)
    return false;

  engine_ = engine;

  mcts_search_ = std::make_unique<Search>(
      config_.mcts_config,
      std::make_unique<Backend>(
          config_.mcts_config.nn_weights_path,
          static_cast<size_t>(std::max(1, config_.mcts_config.nn_cache_size))));

  if (engine_) {
    shared_tt_reader_ = std::make_unique<SharedTTReader>(&engine_->get_tt());
    mcts_search_->SetSharedTT(shared_tt_reader_.get());
  }

  ab_state_.reset();
  mcts_state_.reset();

  initialized_ = true;
  return true;
}

void ParallelHybridSearch::start_search(
    const Position &pos, const ::MetalFish::Search::LimitsType &limits,
    BestMoveCallback best_move_cb, InfoCallback info_cb) {
  if (shutdown_requested_.load(std::memory_order_acquire)) {
    if (best_move_cb)
      best_move_cb(Move::none(), Move::none());
    return;
  }

  if (!initialized_) {
    if (best_move_cb)
      best_move_cb(Move::none(), Move::none());
    return;
  }

  stop();
  wait();

  // Verify all threads are truly dead before resetting shared state.
  // wait() guarantees joins, but this assertion catches logic errors.
  assert(!coordinator_thread_.joinable() &&
         "coordinator still joinable after wait()");
  assert(!mcts_thread_.joinable() && "mcts_thread still joinable after wait()");
  assert(!ab_thread_.joinable() && "ab_thread still joinable after wait()");

  stats_.reset();
  ab_state_.reset();
  mcts_state_.reset();
  stop_flag_.store(false, std::memory_order_release);
  searching_.store(true, std::memory_order_release);
  final_best_move_.store(0, std::memory_order_relaxed);
  final_ponder_move_.store(0, std::memory_order_relaxed);
  callback_invoked_.store(false, std::memory_order_relaxed);
  last_injected_ab_depth_.store(0, std::memory_order_relaxed);
  last_injected_ab_move_raw_.store(0, std::memory_order_relaxed);
  ab_policy_injections_.store(0, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = best_move_cb;
    info_callback_ = info_cb;
  }

  root_fen_ = pos.fen();
  limits_ = limits;
  search_start_ = std::chrono::steady_clock::now();
  time_budget_ms_ = calculate_time_budget();
  nn_policy_hints_.clear();

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

  mcts_thread_done_.store(false, std::memory_order_release);
  ab_thread_done_.store(false, std::memory_order_release);
  coordinator_thread_done_.store(false, std::memory_order_release);

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
  stop_flag_.store(true, std::memory_order_release);

  if (mcts_search_)
    mcts_search_->Stop();

  // engine_->stop() sets threads.stop = true, ensuring AB winds down in <1ms
  if (engine_)
    engine_->stop();
}

void ParallelHybridSearch::ponderhit() {
  if (!is_searching())
    return;

  limits_.ponderMode = false;
  search_start_ = std::chrono::steady_clock::now();
  time_budget_ms_ = calculate_time_budget();

  if (engine_)
    engine_->set_ponderhit(false);
}

void ParallelHybridSearch::wait() {
  // Ensure stop signals have been sent to sub-engines so they will terminate.
  stop_flag_.store(true, std::memory_order_release);
  if (mcts_search_)
    mcts_search_->Stop();
  if (engine_)
    engine_->stop();

  // Wait for all threads to signal completion. No timeout-break — threads
  // WILL finish because stop_flag + sub-engine stops guarantee that workers
  // exit after their current GPU dispatch (at most ~300ms on Apple Silicon).
  auto warn_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  bool warned = false;

  while (!all_threads_done()) {
    if (!warned && std::chrono::steady_clock::now() > warn_deadline) {
      std::cerr << "[HYB] wait() taking >10s - coord_done="
                << coordinator_thread_done_.load()
                << " mcts_done=" << mcts_thread_done_.load()
                << " ab_done=" << ab_thread_done_.load() << std::endl;
      warned = true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }

  {
    std::lock_guard<std::mutex> lock(thread_mutex_);

    if (coordinator_thread_.joinable())
      coordinator_thread_.join();
    if (mcts_thread_.joinable())
      mcts_thread_.join();
    if (ab_thread_.joinable())
      ab_thread_.join();

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

  stats_.reset();
  ab_state_.reset();
  mcts_state_.reset();
  callback_invoked_.store(false, std::memory_order_relaxed);
  last_injected_ab_depth_.store(0, std::memory_order_relaxed);
  last_injected_ab_move_raw_.store(0, std::memory_order_relaxed);
  ab_policy_injections_.store(0, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = nullptr;
    info_callback_ = nullptr;
  }

  if (mcts_search_) {
    mcts_search_ = std::make_unique<Search>(
        config_.mcts_config,
        std::make_unique<Backend>(config_.mcts_config.nn_weights_path,
                                  static_cast<size_t>(std::max(
                                      1, config_.mcts_config.nn_cache_size))));
    if (engine_) {
      shared_tt_reader_ = std::make_unique<SharedTTReader>(&engine_->get_tt());
      mcts_search_->SetSharedTT(shared_tt_reader_.get());
    }
  }
}

int ParallelHybridSearch::calculate_time_budget() const {
  if (limits_.ponderMode)
    return 0;
  if (limits_.movetime > 0)
    return limits_.movetime;
  if (limits_.infinite)
    return 0;
  if (limits_.nodes > 0)
    return 0;

  Position pos;
  StateInfo st;
  pos.set(root_fen_, false, &st);
  Color us = pos.side_to_move();

  int time_left = (us == WHITE) ? limits_.time[WHITE] : limits_.time[BLACK];
  int increment = (us == WHITE) ? limits_.inc[WHITE] : limits_.inc[BLACK];
  if (time_left <= 0) {
    return std::max(1000, increment);
  }

  if (time_left < 500)
    return std::max(50, std::min(time_left / 4, increment));

  const int moves_to_go = limits_.movestogo > 0 ? limits_.movestogo : 30;
  const int base = time_left / std::max(1, moves_to_go);
  const int inc_bonus = std::max(0, increment) * 3 / 4;
  const int budget = base + inc_bonus;
  const int hard_cap = std::max(500, time_left / 4);
  const int reserve_cap = std::max(1, time_left - 100);
  return std::max(250, std::min({budget, hard_cap, reserve_cap}));
}

bool HybridShouldContinueMCTSAfterAB(
    const ::MetalFish::Search::LimitsType &limits) {
  return limits.movetime > 0 || limits.nodes > 0 || limits.infinite;
}

bool ParallelHybridSearch::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire))
    return true;

  if (limits_.nodes > 0) {
    uint64_t mcts_n =
        mcts_search_
            ? mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed)
            : 0;
    uint64_t total =
        mcts_n + ab_state_.nodes_searched.load(std::memory_order_relaxed);
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

// MCTS thread - runs transformer-backed MCTS. NNUE remains CPU-only in AB.
void ParallelHybridSearch::mcts_thread_main() {
#ifdef __APPLE__
  set_thread_qos(QOS_CLASS_USER_INITIATED);
#endif

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

  // Give MCTS the real outer limits so its Lc0-style smart pruning, KLD, node
  // and time stoppers stay active. Ponder is the exception: before ponderhit
  // there is no move clock to spend, so the coordinator owns cancellation.
  ::MetalFish::Search::LimitsType mcts_limits = limits_;
  if (limits_.ponderMode) {
    mcts_limits = ::MetalFish::Search::LimitsType{};
    mcts_limits.infinite = 1;
  }
  mcts_limits.startTime = now();
  mcts_limits.searchmoves = limits_.searchmoves;

  // Check if stop was already requested before launching MCTS.
  // This prevents the race where stop() clears the flag before StartSearch
  // can see it (StartSearch resets its own stop_flag at the start).
  if (should_stop())
    return;

  Move best_move = Move::none();
  std::atomic<bool> mcts_done{false};

  auto mcts_callback = [&](Move move, Move ponder) {
    best_move = move;
    mcts_done = true;
  };

  // Start MCTS search with FEN string
  mcts_search_->StartSearch(root_fen_, mcts_limits, mcts_callback, nullptr);

  // Re-check after StartSearch: if stop was requested during launch, ensure
  // the MCTS search sees it (StartSearch resets its own internal stop_flag).
  if (should_stop()) {
    mcts_search_->Stop();
  }

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
      publish_mcts_state();

      uint64_t ab_counter =
          ab_state_.update_counter.load(std::memory_order_acquire);
      if (ab_counter > last_ab_counter) {
        update_mcts_policy_from_ab();
        last_ab_counter = ab_counter;
      }

      last_update = now_time;
    }
  }

  if (!mcts_done) {
    mcts_search_->Stop();
  }
  mcts_search_->Wait();

  publish_mcts_state();

  auto end = std::chrono::steady_clock::now();
  stats_.mcts_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  stats_.mcts_nodes = mcts_search_->Stats().total_nodes.load();
  stats_.transformer_evaluations = mcts_search_->Stats().nn_evaluations.load();
  stats_.transformer_batches = mcts_search_->Stats().total_batches.load();
}

void ParallelHybridSearch::publish_mcts_state() {
  if (!mcts_search_)
    return;

  const auto best_stats = mcts_search_->GetBestMoveStats();
  Move best = best_stats.move;
  if (best == Move::none())
    return;

  mcts_state_.best_move_raw.store(best.raw(), std::memory_order_relaxed);
  mcts_state_.best_q.store(best_stats.q, std::memory_order_relaxed);
  mcts_state_.best_visits.store(best_stats.visits, std::memory_order_relaxed);

  const auto root_moves = mcts_search_->GetRootMoveStats();
  uint64_t root_visits = 0;
  for (const auto &move : root_moves)
    root_visits += move.visits;
  mcts_state_.total_nodes.store(root_visits, std::memory_order_relaxed);

  const int top_count = std::min<int>(static_cast<int>(root_moves.size()),
                                      MCTSSharedState::MAX_TOP_MOVES);
  for (int i = 0; i < top_count; ++i) {
    mcts_state_.top_moves[i].move_raw.store(root_moves[i].move.raw(),
                                            std::memory_order_relaxed);
    mcts_state_.top_moves[i].policy.store(root_moves[i].policy,
                                          std::memory_order_relaxed);
    mcts_state_.top_moves[i].visits.store(root_moves[i].visits,
                                          std::memory_order_relaxed);
    mcts_state_.top_moves[i].q.store(root_moves[i].q,
                                     std::memory_order_relaxed);
  }
  for (int i = top_count; i < MCTSSharedState::MAX_TOP_MOVES; ++i) {
    mcts_state_.top_moves[i].move_raw.store(0, std::memory_order_relaxed);
    mcts_state_.top_moves[i].policy.store(0.0f, std::memory_order_relaxed);
    mcts_state_.top_moves[i].visits.store(0, std::memory_order_relaxed);
    mcts_state_.top_moves[i].q.store(0.0f, std::memory_order_relaxed);
  }
  mcts_state_.num_top_moves.store(top_count, std::memory_order_release);

  mcts_state_.has_result.store(true, std::memory_order_release);
  mcts_state_.update_counter.fetch_add(1, std::memory_order_release);
}

void ParallelHybridSearch::update_mcts_policy_from_ab() {
  if (config_.ab_policy_weight <= 0.0f)
    return;

  int depth = ab_state_.pv_depth.load(std::memory_order_acquire);
  if (depth < std::max(10, current_strategy_.ab_verify_depth))
    return;

  uint64_t mcts_nodes =
      mcts_search_
          ? mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed)
          : 0;
  if (mcts_nodes < 100)
    return;

  int pv_len = ab_state_.pv_length.load(std::memory_order_acquire);
  if (pv_len <= 0 || !mcts_search_)
    return;

  Move pv[ABSharedState::MAX_PV];
  for (int i = 0; i < pv_len; ++i) {
    pv[i] = Move(ab_state_.pv_moves[i].load(std::memory_order_relaxed));
  }
  if (pv[0] == Move::none())
    return;

  // Policy boosting is intentionally sparse. Re-applying the same AB line
  // every iterative-deepening update compounds the root prior and can turn the
  // GPU search into an AB echo. Only inject meaningfully new or deeper PVs.
  const uint32_t first_raw = pv[0].raw();
  const int last_depth =
      last_injected_ab_depth_.load(std::memory_order_acquire);
  const uint32_t last_move =
      last_injected_ab_move_raw_.load(std::memory_order_acquire);
  const int injections = ab_policy_injections_.load(std::memory_order_acquire);
  if (injections >= 4)
    return;
  if (first_raw == last_move && depth < last_depth + 4)
    return;

  mcts_search_->InjectPVBoost(pv, pv_len, depth, config_.ab_policy_weight);
  last_injected_ab_depth_.store(depth, std::memory_order_release);
  last_injected_ab_move_raw_.store(first_raw, std::memory_order_release);
  ab_policy_injections_.fetch_add(1, std::memory_order_acq_rel);
  stats_.policy_updates.fetch_add(1, std::memory_order_relaxed);
}

void ParallelHybridSearch::ab_thread_main() {
#ifdef __APPLE__
  set_thread_qos(QOS_CLASS_USER_INITIATED);
#endif

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
}

void ParallelHybridSearch::run_ab_search() {
  if (!engine_)
    return;
  if (should_stop())
    return;

  // Temporarily resize AB worker threads to avoid oversubscribing cores
  const int original_threads =
      static_cast<int>(engine_->get_options()["Threads"]);
  const int requested_ab_threads = std::max(1, config_.ab_threads);
  const bool needs_thread_resize = (requested_ab_threads != original_threads);

  auto set_engine_threads = [this](int n) {
    std::istringstream ss("name Threads value " + std::to_string(n));
    engine_->get_options().setoption(ss);
  };

  if (needs_thread_resize) {
    set_engine_threads(requested_ab_threads);
  }

  engine_->set_position(root_fen_, {});

  // Pass real time controls so Stockfish's time manager handles allocation
  ::MetalFish::Search::LimitsType ab_limits = limits_;
  ab_limits.startTime = now();

  auto saved_bestmove = engine_->get_on_bestmove();
  auto saved_update_full = engine_->get_on_update_full();
  Move ab_best_move = Move::none();
  int ab_score = 0;
  int ab_depth = 0;

  // Hook per-depth update to capture score and PV
  engine_->set_on_update_full([this, &ab_score,
                               &ab_depth](const Engine::InfoFull &info) {
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

    Position pos;
    StateInfo root_st;
    pos.set(root_fen_, false, &root_st);

    std::vector<Move> pv_moves;
    pv_moves.reserve(ABSharedState::MAX_PV);
    std::vector<StateInfo> pv_states;
    pv_states.reserve(ABSharedState::MAX_PV);

    std::istringstream pv_stream(std::string(info.pv));
    std::string move_token;
    while (pv_stream >> move_token && pv_moves.size() < ABSharedState::MAX_PV) {
      Move move = UCIEngine::to_move(pos, move_token);
      if (move == Move::none()) {
        break;
      }

      pv_moves.push_back(move);
      if (pv_moves.size() >= ABSharedState::MAX_PV) {
        break;
      }

      pv_states.emplace_back();
      pos.do_move(move, pv_states.back());
    }

    const Move current_best =
        pv_moves.empty() ? Move::none() : pv_moves.front();
    if (current_best != Move::none()) {
      ab_state_.publish_pv(pv_moves, ab_depth);
      publish_ab_state(current_best, ab_score, ab_depth,
                       engine_->threads_nodes_searched());
    }
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

  engine_->go(ab_limits);
  engine_->wait_for_search_finished();

  stats_.ab_nodes.store(engine_->threads_nodes_searched(),
                        std::memory_order_relaxed);
  stats_.ab_depth.store(ab_depth, std::memory_order_relaxed);

  engine_->set_on_update_full(std::move(saved_update_full));
  engine_->set_on_bestmove(std::move(saved_bestmove));

  if (needs_thread_resize) {
    set_engine_threads(original_threads);
  }
}

void ParallelHybridSearch::publish_ab_state(Move best, int score, int depth,
                                            uint64_t nodes) {
  ab_state_.set_best_move(best, score, depth, nodes);
}

// Coordinator thread - monitors both searches and makes final decision
void ParallelHybridSearch::coordinator_thread_main() {
#ifdef __APPLE__
  set_thread_qos(QOS_CLASS_UTILITY);
#endif

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

  // AB's time manager is the master clock. Coordinator stops when:
  // (1) AB finishes, (2) both agree on a move, or (3) safety hard cap fires.
  while (!should_stop()) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    bool ab_done = !ab_state_.ab_running.load(std::memory_order_acquire);
    bool mcts_done = !mcts_state_.mcts_running.load(std::memory_order_acquire);

    // AB finished -- for clock-managed games AB owns the move allocation,
    // but explicit analysis budgets should still give MCTS the requested time
    if (ab_done && ab_state_.has_result.load(std::memory_order_acquire)) {
      if (!HybridShouldContinueMCTSAfterAB(limits_) || mcts_done) {
        break;
      }
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Agreement-based early stopping (only safe in clock-managed play)
    uint32_t ab_move = ab_state_.best_move_raw.load(std::memory_order_relaxed);
    uint32_t mcts_move =
        mcts_state_.best_move_raw.load(std::memory_order_relaxed);

    if (ab_move != last_ab_move_raw || mcts_move != last_mcts_move_raw) {
      agreement_count = 0;
      last_ab_move_raw = ab_move;
      last_mcts_move_raw = mcts_move;
    }

    if (ab_move != 0 && mcts_move != 0) {
      if (ab_move == mcts_move) {
        agreement_count++;
        const bool fixed_movetime =
            limits_.movetime > 0 && limits_.time[WHITE] <= 0 &&
            limits_.time[BLACK] <= 0 && limits_.nodes <= 0;
        const int min_time = (time_budget_ms_ > 0) ? time_budget_ms_ / 4 : 500;
        const uint64_t mcts_nodes =
            mcts_search_ ? mcts_search_->Stats().total_nodes.load(
                               std::memory_order_relaxed)
                         : 0;
        const uint64_t min_mcts_nodes = static_cast<uint64_t>(
            std::max(100, current_strategy_.min_mcts_nodes));
        const int ab_depth =
            ab_state_.completed_depth.load(std::memory_order_relaxed);
        const int min_ab_depth =
            std::max(config_.ab_min_depth, current_strategy_.ab_verify_depth);

        if (!fixed_movetime && agreement_count >= 3 && ms > min_time &&
            mcts_nodes >= min_mcts_nodes && ab_depth >= min_ab_depth) {
          send_info_string("Hybrid: engines agree, stopping early at " +
                           std::to_string(ms) + "ms");
          break;
        }
      } else {
        agreement_count = 0;
      }
    }

    // Send combined info every ~500ms
    if (ms - last_info_ms >= 500) {
      last_info_ms = ms;
      uint64_t mcts_nodes = mcts_search_
                                ? mcts_search_->Stats().total_nodes.load(
                                      std::memory_order_relaxed)
                                : 0;
      uint64_t ab_nodes =
          ab_state_.nodes_searched.load(std::memory_order_relaxed);
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

  const bool external_stop = stop_flag_.load(std::memory_order_acquire);
  stop_flag_.store(true, std::memory_order_release);

  if (mcts_search_)
    mcts_search_->Stop();
  if (engine_)
    engine_->stop();

  // Wait for both AB and MCTS threads to finish before making decision.
  // MCTS must finish because it owns GPU resources that can't be accessed
  // after the coordinator fires the callback. AB is fast to stop (CPU-only).
  // No cap on MCTS wait — it WILL finish after Stop() since Metal dispatches
  // complete in bounded time (~500ms max on Apple Silicon).
  while (!mcts_thread_done_.load(std::memory_order_acquire) ||
         !ab_thread_done_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }

  // Capture freshest MCTS result before deciding
  if (mcts_search_) {
    publish_mcts_state();
  }

  Move final_move = make_final_decision();

  // Fallback: if no valid move, find any legal move
  if (final_move == Move::none()) {
    final_move = first_allowed_legal_move();
  }

  final_best_move_.store(final_move.raw(), std::memory_order_release);
  refresh_final_state(final_move);

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

  const uint64_t total_nodes =
      stats_.mcts_nodes.load(std::memory_order_relaxed) +
      stats_.ab_nodes.load(std::memory_order_relaxed);
  const int info_depth = std::max<int>(
      1, static_cast<int>(stats_.ab_depth.load(std::memory_order_relaxed)));
  const Move ab_best = ab_state_.get_best_move();
  const Move mcts_best = mcts_state_.get_best_move();
  const int info_score =
      final_move == mcts_best && final_move != ab_best
          ? QToNnueScore(mcts_state_.best_q.load(std::memory_order_relaxed))
          : ab_state_.best_score.load(std::memory_order_relaxed);
  send_info(info_depth, info_score, total_nodes,
            static_cast<int>(stats_.total_time_ms), final_pv_, "hybrid-final");

  send_info_string(
      "Final: MCTS=" + std::to_string(stats_.mcts_nodes.load()) + " MCTSBest=" +
      std::to_string(mcts_state_.best_visits.load(std::memory_order_relaxed)) +
      " AB=" + std::to_string(stats_.ab_nodes.load()) +
      " ABMove=" + UCIEngine::move(ab_best, false) +
      " MCTSMove=" + UCIEngine::move(mcts_best, false) +
      " agreements=" + std::to_string(stats_.move_agreements.load()) +
      " ab_overrides=" + std::to_string(stats_.ab_overrides.load()) +
      " mcts_overrides=" + std::to_string(stats_.mcts_overrides.load()) +
      " policy_updates=" + std::to_string(stats_.policy_updates.load()));

  invoke_best_move_callback(final_move, ponder_move);
}

void ParallelHybridSearch::refresh_final_state(Move final_move) {
  const uint64_t mcts_nodes =
      mcts_search_
          ? mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed)
          : 0;
  const uint64_t mcts_evals =
      mcts_search_
          ? mcts_search_->Stats().nn_evaluations.load(std::memory_order_relaxed)
          : 0;
  const uint64_t ab_nodes =
      ab_state_.nodes_searched.load(std::memory_order_relaxed);
  const uint64_t ab_depth =
      ab_state_.completed_depth.load(std::memory_order_relaxed);

  stats_.mcts_nodes.store(mcts_nodes, std::memory_order_relaxed);
  stats_.transformer_evaluations.store(mcts_evals, std::memory_order_relaxed);
  stats_.ab_nodes.store(ab_nodes, std::memory_order_relaxed);
  stats_.ab_depth.store(ab_depth, std::memory_order_relaxed);

  std::vector<Move> pv;
  const Move ab_best = ab_state_.get_best_move();
  const Move mcts_best = mcts_state_.get_best_move();

  if (final_move == ab_best) {
    const int pv_len =
        std::min<int>(ab_state_.pv_length.load(std::memory_order_acquire),
                      ABSharedState::MAX_PV);
    pv.reserve(static_cast<size_t>(pv_len));
    for (int i = 0; i < pv_len; ++i) {
      Move m(ab_state_.pv_moves[i].load(std::memory_order_relaxed));
      if (m == Move::none())
        break;
      pv.push_back(m);
    }
  } else if (final_move == mcts_best && mcts_search_ &&
             mcts_thread_done_.load(std::memory_order_acquire)) {
    pv = mcts_search_->GetPV();
  }

  if (final_move != Move::none() && (pv.empty() || pv.front() != final_move)) {
    pv.insert(pv.begin(), final_move);
  }

  {
    std::lock_guard<std::mutex> lock(pv_mutex_);
    final_pv_ = std::move(pv);
  }
}

void ParallelHybridSearch::invoke_best_move_callback(Move best, Move ponder) {
  bool expected = false;
  if (!callback_invoked_.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel)) {
    return;
  }

  BestMoveCallback callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = best_move_callback_;
    best_move_callback_ = nullptr;
  }

  if (callback)
    callback(best, ponder);
}

Move ParallelHybridSearch::make_final_decision() {
  Move ab_best = ab_state_.get_best_move();
  Move mcts_best = mcts_state_.get_best_move();
  int ab_score = ab_state_.best_score.load(std::memory_order_relaxed);
  float mcts_q = mcts_state_.best_q.load(std::memory_order_relaxed);
  const uint32_t mcts_visits =
      mcts_state_.best_visits.load(std::memory_order_relaxed);
  const uint64_t mcts_total_nodes =
      mcts_state_.total_nodes.load(std::memory_order_relaxed);

  const auto move_to_string = [](Move move) {
    return move == Move::none() ? std::string("none")
                                : UCIEngine::move(move, false);
  };
  const auto mode_to_string = [](ParallelHybridConfig::DecisionMode mode) {
    switch (mode) {
    case ParallelHybridConfig::DecisionMode::MCTS_PRIMARY:
      return "mcts_primary";
    case ParallelHybridConfig::DecisionMode::AB_PRIMARY:
      return "ab_primary";
    case ParallelHybridConfig::DecisionMode::VOTE_WEIGHTED:
      return "vote_weighted";
    case ParallelHybridConfig::DecisionMode::DYNAMIC:
      return "dynamic";
    }
    return "unknown";
  };
  const auto top_moves_to_string = [&]() {
    std::ostringstream ss;
    const int count = std::min<int>(
        mcts_state_.num_top_moves.load(std::memory_order_acquire), 5);
    ss << "[";
    for (int i = 0; i < count; ++i) {
      if (i > 0)
        ss << ",";
      Move move(
          mcts_state_.top_moves[i].move_raw.load(std::memory_order_relaxed));
      ss << move_to_string(move) << ":n="
         << mcts_state_.top_moves[i].visits.load(std::memory_order_relaxed)
         << ":q=" << std::fixed << std::setprecision(3)
         << mcts_state_.top_moves[i].q.load(std::memory_order_relaxed)
         << ":p=" << std::fixed << std::setprecision(3)
         << mcts_state_.top_moves[i].policy.load(std::memory_order_relaxed);
    }
    ss << "]";
    return ss.str();
  };
  const auto trace_simple = [&](const char *reason, Move selected) {
    if (!config_.trace_decisions)
      return;
    std::ostringstream ss;
    ss << "HybridTrace: reason=" << reason
       << " mode=" << mode_to_string(config_.decision_mode)
       << " selected=" << move_to_string(selected)
       << " ABMove=" << move_to_string(ab_best)
       << " MCTSMove=" << move_to_string(mcts_best) << " ABScore=" << ab_score
       << " MCTSQ=" << std::fixed << std::setprecision(3) << mcts_q
       << " MCTSBestVisits=" << mcts_visits
       << " MCTSRootVisits=" << mcts_total_nodes
       << " MCTSTop=" << top_moves_to_string();
    send_info_string(ss.str());
  };

  if (ab_best == Move::none() && mcts_best != Move::none()) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    trace_simple("mcts_only_result", mcts_best);
    return mcts_best;
  }
  if (mcts_best == Move::none() || ab_best == Move::none()) {
    if (ab_best != Move::none()) {
      stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
      trace_simple("ab_only_result", ab_best);
      return ab_best;
    }
    trace_simple("fallback_legal", Move::none());
    return first_allowed_legal_move();
  }

  if (ab_best == mcts_best) {
    stats_.move_agreements.fetch_add(1, std::memory_order_relaxed);
    trace_simple("engines_agree", ab_best);
    return ab_best;
  }

  const int mcts_cp = QToNnueScore(mcts_q);
  const int ab_depth =
      ab_state_.completed_depth.load(std::memory_order_relaxed);
  const float visit_share = mcts_total_nodes > 0
                                ? static_cast<float>(mcts_visits) /
                                      static_cast<float>(mcts_total_nodes)
                                : 0.0f;
  const bool fixed_budget =
      limits_.movetime > 0 || limits_.nodes > 0 || limits_.infinite;
  const uint64_t min_nodes =
      fixed_budget ? 180u
                   : static_cast<uint64_t>(
                         std::max(100, current_strategy_.min_mcts_nodes));
  const uint32_t min_visits =
      fixed_budget ? 72u
                   : static_cast<uint32_t>(
                         std::max(48, current_strategy_.min_mcts_nodes / 2));
  const bool mcts_reliable = mcts_total_nodes >= min_nodes &&
                             mcts_visits >= min_visits && visit_share >= 0.24f;
  const bool mcts_strong =
      mcts_reliable && (mcts_visits >= 512 || visit_share >= 0.55f);
  const bool mcts_overwhelming =
      mcts_total_nodes >= 5000 && mcts_visits >= 512 && visit_share >= 0.30f;
  const bool ab_verified =
      ab_depth >=
      std::max(config_.ab_min_depth, current_strategy_.ab_verify_depth);
  const int eval_delta = mcts_cp - ab_score;
  const bool ab_has_clear_preference = ab_verified && std::abs(ab_score) >= 15;
  bool mcts_root_rejects_ab = false;
  {
    const int top_count =
        std::min<int>(mcts_state_.num_top_moves.load(std::memory_order_acquire),
                      MCTSSharedState::MAX_TOP_MOVES);
    if (config_.mcts_root_reject && top_count > 0 && mcts_reliable &&
        mcts_visits >= 160 && visit_share >= 0.35f && ab_score >= -120 &&
        ab_score <= 40) {
      Move top_move(
          mcts_state_.top_moves[0].move_raw.load(std::memory_order_relaxed));
      if (top_move == mcts_best) {
        const float top_q =
            mcts_state_.top_moves[0].q.load(std::memory_order_relaxed);
        const uint32_t top_visits =
            mcts_state_.top_moves[0].visits.load(std::memory_order_relaxed);
        for (int i = 1; i < top_count; ++i) {
          Move move(mcts_state_.top_moves[i].move_raw.load(
              std::memory_order_relaxed));
          if (move != ab_best)
            continue;
          const uint32_t ab_visits =
              mcts_state_.top_moves[i].visits.load(std::memory_order_relaxed);
          const float ab_q =
              mcts_state_.top_moves[i].q.load(std::memory_order_relaxed);
          if (ab_visits >= 25 &&
              top_visits >= 3 * std::max<uint32_t>(1, ab_visits) &&
              top_q - ab_q >= 0.25f) {
            mcts_root_rejects_ab = true;
          }
          break;
        }
      }
    }
  }

  bool choose_mcts = false;
  const char *reason = "ab_default";
  switch (config_.decision_mode) {
  case ParallelHybridConfig::DecisionMode::MCTS_PRIMARY:
    choose_mcts =
        mcts_reliable && (!ab_has_clear_preference || eval_delta >= 180);
    if (choose_mcts)
      reason = "mcts_primary_reliable";
    break;
  case ParallelHybridConfig::DecisionMode::AB_PRIMARY:
    choose_mcts = mcts_overwhelming && eval_delta >= 250;
    if (choose_mcts)
      reason = "ab_primary_mcts_overwhelming";
    break;
  case ParallelHybridConfig::DecisionMode::VOTE_WEIGHTED:
  case ParallelHybridConfig::DecisionMode::DYNAMIC:
    if (mcts_overwhelming && eval_delta >= 180) {
      choose_mcts = true;
      reason = "mcts_overwhelming_delta";
    } else if (mcts_root_rejects_ab) {
      choose_mcts = true;
      reason = "mcts_root_rejects_ab";
    } else if (!ab_verified && mcts_reliable && eval_delta >= -80) {
      choose_mcts = true;
      reason = "mcts_reliable_ab_unverified";
    } else if (!ab_has_clear_preference && mcts_strong && eval_delta >= 80) {
      choose_mcts = true;
      reason = "mcts_strong_no_clear_ab_preference";
    }
    break;
  }

  if (config_.trace_decisions) {
    std::ostringstream ss;
    ss << "HybridTrace: reason=" << reason
       << " mode=" << mode_to_string(config_.decision_mode)
       << " selected=" << move_to_string(choose_mcts ? mcts_best : ab_best)
       << " ABMove=" << move_to_string(ab_best)
       << " MCTSMove=" << move_to_string(mcts_best) << " ABScore=" << ab_score
       << " ABDepth=" << ab_depth << " MCTSQ=" << std::fixed
       << std::setprecision(3) << mcts_q << " MCTSCP=" << mcts_cp
       << " EvalDelta=" << eval_delta << " MCTSBestVisits=" << mcts_visits
       << " MCTSRootVisits=" << mcts_total_nodes
       << " VisitShare=" << visit_share
       << " MCTSReliable=" << (mcts_reliable ? 1 : 0)
       << " MCTSStrong=" << (mcts_strong ? 1 : 0)
       << " MCTSOverwhelming=" << (mcts_overwhelming ? 1 : 0)
       << " ABVerified=" << (ab_verified ? 1 : 0)
       << " ABClearPreference=" << (ab_has_clear_preference ? 1 : 0)
       << " MCTSRootRejectsAB=" << (mcts_root_rejects_ab ? 1 : 0)
       << " MCTSTop=" << top_moves_to_string();
    send_info_string(ss.str());
  }

  if (choose_mcts) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    return mcts_best;
  }

  stats_.ab_overrides.fetch_add(1, std::memory_order_relaxed);
  return ab_best;
}

Move ParallelHybridSearch::first_allowed_legal_move() const {
  Position pos;
  StateInfo st;
  pos.set(root_fen_, false, &st);

  for (const auto &uci_move : limits_.searchmoves) {
    Move move = UCIEngine::to_move(pos, uci_move);
    if (move != Move::none())
      return move;
  }

  if (!limits_.searchmoves.empty())
    return Move::none();

  MoveList<LEGAL> moves(pos);
  return moves.size() > 0 ? *moves.begin() : Move::none();
}

void ParallelHybridSearch::send_info(int depth, int score, uint64_t nodes,
                                     int time_ms, const std::vector<Move> &pv,
                                     const std::string &source) {
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
      if (m != Move::none()) {
        has_valid = true;
        break;
      }
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
  InfoCallback callback;
  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback = info_callback_;
  }

  if (callback)
    callback("info string " + msg);
}

std::unique_ptr<ParallelHybridSearch>
create_parallel_hybrid_search(Engine *engine,
                              const ParallelHybridConfig &config) {
  auto search = std::make_unique<ParallelHybridSearch>();
  search->set_config(config);
  if (search->initialize(engine)) {
    return search;
  }
  return nullptr;
}

} // namespace MCTS
} // namespace MetalFish
