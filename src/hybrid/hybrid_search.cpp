/*
  MetalFish - Parallel Hybrid Search Implementation
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "hybrid_search.h"
#include "../core/misc.h"
#include "../core/movegen.h"
#include "../eval/evaluate.h"
#include "../mcts/core.h"
#include "../uci/engine.h"
#include "../uci/uci.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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

namespace {
int64_t SteadyNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
} // namespace

ParallelHybridSearch::ParallelHybridSearch() {
  config_.mcts_config.cpuct = 1.745f;
  config_.mcts_config.cpuct_at_root = 1.745f;
  config_.mcts_config.fpu_reduction = 0.330f;
  config_.mcts_config.cpuct_base = 38739.0f;
  config_.mcts_config.cpuct_factor = 3.894f;
  config_.mcts_config.high_policy_root_lever_selection = false;
  config_.mcts_config.low_policy_root_lever_selection = false;

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
  if (ane_root_hints_future_.valid())
    ane_root_hints_future_.wait();

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

  try {
    auto backend = std::make_unique<Backend>(
        config_.mcts_config.nn_weights_path,
        static_cast<size_t>(std::max(1, config_.mcts_config.nn_cache_size)),
        config_.mcts_config.GetBackendConfig());
    mcts_search_ =
        std::make_unique<Search>(config_.mcts_config, std::move(backend));
  } catch (const std::exception &e) {
    std::cerr << "[HYB] transformer backend creation failed: " << e.what()
              << std::endl;
    mcts_search_.reset();
    shared_tt_reader_.reset();
    initialized_ = false;
    return false;
  } catch (...) {
    std::cerr << "[HYB] transformer backend creation failed: unknown error"
              << std::endl;
    mcts_search_.reset();
    shared_tt_reader_.reset();
    initialized_ = false;
    return false;
  }

  ane_evaluator_.reset();
  if (config_.ane_root_probe) {
    if (config_.ane_weights_path.empty() || config_.ane_model_path.empty()) {
      std::cerr << "[HYB] ANE root probe disabled: weights/model path missing"
                << std::endl;
    } else {
#ifdef USE_COREML
      try {
        ane_evaluator_ = std::make_unique<NNMCTSEvaluator>(
            config_.ane_weights_path, "coreml", config_.ane_model_path,
            config_.ane_compute_units);
        StateInfo warmup_state;
        Position warmup_pos;
        warmup_pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                       "RNBQKBNR w KQkq - 0 1",
                       false, &warmup_state);
        (void)ane_evaluator_->Evaluate(warmup_pos);
      } catch (const std::exception &e) {
        std::cerr << "[HYB] ANE root probe disabled: " << e.what() << std::endl;
        ane_evaluator_.reset();
      } catch (...) {
        std::cerr << "[HYB] ANE root probe disabled: unknown error"
                  << std::endl;
        ane_evaluator_.reset();
      }
#else
      std::cerr
          << "[HYB] ANE root probe disabled: Core ML backend was not compiled "
             "into MetalFish"
          << std::endl;
#endif
    }
  }

  shared_tt_reader_ = std::make_unique<SharedTTReader>(&engine_->get_tt());
  mcts_search_->SetSharedTT(config_.use_shared_tt ? shared_tt_reader_.get()
                                                  : nullptr);

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
  mcts_search_started_.store(false, std::memory_order_release);
  ab_search_started_.store(false, std::memory_order_release);
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
  {
    std::lock_guard<std::mutex> lock(ab_root_mutex_);
    ab_root_moves_.clear();
    ab_root_order_hints_.clear();
    ab_verified_root_order_hints_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
    ane_root_hints_.clear();
    ane_root_hint_infos_.clear();
  }

  root_fen_ = pos.fen();
  limits_ = limits;
  ponderhit_received_.store(false, std::memory_order_release);
  search_start_ms_.store(SteadyNowMs(), std::memory_order_release);
  const int time_budget_ms = calculate_time_budget();
  time_budget_ms_.store(time_budget_ms, std::memory_order_release);
  nn_policy_hints_.clear();
  start_ane_root_probe();

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

  send_info_string("Time budget: " + std::to_string(time_budget_ms) + "ms");

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

  if (mcts_search_) {
    std::lock_guard<std::mutex> lock(mcts_start_mutex_);
    if (mcts_search_started_.load(std::memory_order_acquire))
      mcts_search_->Stop();
  }

  if (engine_) {
    std::lock_guard<std::mutex> lock(ab_start_mutex_);
    if (ab_search_started_.load(std::memory_order_acquire))
      engine_->stop();
  }
}

void ParallelHybridSearch::ponderhit() {
  if (!is_searching())
    return;

  if (ponderhit_received_.exchange(true, std::memory_order_acq_rel)) {
    if (engine_)
      engine_->set_ponderhit(false);
    return;
  }

  search_start_ms_.store(SteadyNowMs(), std::memory_order_release);
  time_budget_ms_.store(calculate_time_budget(), std::memory_order_release);
  start_ane_root_probe(true);

  if (mcts_search_) {
    std::lock_guard<std::mutex> lock(mcts_start_mutex_);
    if (mcts_search_started_.load(std::memory_order_acquire))
      mcts_search_->PonderHit();
  }

  if (engine_) {
    std::lock_guard<std::mutex> lock(ab_start_mutex_);
    if (ab_search_started_.load(std::memory_order_acquire))
      engine_->set_ponderhit(false);
  }
}

void ParallelHybridSearch::wait() {
  stop_flag_.store(true, std::memory_order_release);
  if (mcts_search_) {
    std::lock_guard<std::mutex> lock(mcts_start_mutex_);
    if (mcts_search_started_.load(std::memory_order_acquire))
      mcts_search_->Stop();
  }
  if (engine_) {
    std::lock_guard<std::mutex> lock(ab_start_mutex_);
    if (ab_search_started_.load(std::memory_order_acquire))
      engine_->stop();
  }

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

  if (ane_root_hints_future_.valid())
    ane_root_hints_future_.wait();

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
  ponderhit_received_.store(false, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    best_move_callback_ = nullptr;
    info_callback_ = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(ab_root_mutex_);
    ab_root_moves_.clear();
    ab_root_order_hints_.clear();
    ab_verified_root_order_hints_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
    ane_root_hints_.clear();
    ane_root_hint_infos_.clear();
  }

  if (mcts_search_) {
    mcts_search_->NewGame();
    if (engine_) {
      shared_tt_reader_ = std::make_unique<SharedTTReader>(&engine_->get_tt());
      mcts_search_->SetSharedTT(config_.use_shared_tt ? shared_tt_reader_.get()
                                                      : nullptr);
    } else {
      shared_tt_reader_.reset();
      mcts_search_->SetSharedTT(nullptr);
    }
  }
}

int ParallelHybridSearch::calculate_time_budget() const {
  if (limits_.ponderMode &&
      !ponderhit_received_.load(std::memory_order_acquire))
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
  if (limits.movetime > 0 || limits.nodes > 0 || limits.infinite)
    return true;
  if (limits.depth > 0 || limits.mate > 0)
    return false;
  return limits.time[WHITE] > 0 || limits.time[BLACK] > 0;
}

bool HybridCanStopEarlyOnAgreement(
    const ::MetalFish::Search::LimitsType &limits) {
  return limits.movetime <= 0 && limits.nodes <= 0 && !limits.infinite &&
         limits.depth <= 0 && limits.mate <= 0;
}

bool HybridHasMCTSDecisionBudget(const ::MetalFish::Search::LimitsType &limits,
                                 int time_budget_ms, bool ponderhit_received) {
  if (limits.movetime > 0 || limits.nodes > 0 || limits.infinite)
    return true;
  if (limits.depth > 0 || limits.mate > 0)
    return false;
  if (limits.ponderMode && !ponderhit_received)
    return false;
  if (time_budget_ms <= 0)
    return false;
  return limits.time[WHITE] > 0 || limits.time[BLACK] > 0;
}

bool HybridUseMCTSPrimaryForFixedNodeBudget(
    const ::MetalFish::Search::LimitsType &limits) {
  return limits.nodes > 0 && limits.nodes <= 1024 && limits.movetime <= 0 &&
         limits.depth <= 0 && limits.mate <= 0 && !limits.infinite &&
         !limits.ponderMode;
}

uint64_t HybridLowNodeABProbeNodes(uint64_t requested_nodes) {
  if (requested_nodes == 0)
    return 0;
  return std::clamp<uint64_t>(requested_nodes / 4, 1, 64);
}

bool HybridLowNodeMCTSPrimaryReady(bool enabled, uint64_t requested_nodes,
                                   uint64_t mcts_root_current_nodes,
                                   uint32_t mcts_best_current_visits,
                                   bool visit_evidence_sane) {
  if (!enabled || requested_nodes == 0 || !visit_evidence_sane)
    return false;

  const uint64_t root_min = std::max<uint64_t>(8, (requested_nodes * 3) / 4);
  const uint32_t best_min =
      static_cast<uint32_t>(std::max<uint64_t>(4, requested_nodes / 10));
  return mcts_root_current_nodes >= root_min &&
         mcts_best_current_visits >= best_min;
}

int HybridABCandidateVerifyBudgetMs(
    const ::MetalFish::Search::LimitsType &limits, int time_budget_ms,
    int requested_ms, bool waiting_for_ponderhit) {
  if (requested_ms <= 0)
    return 0;
  if (waiting_for_ponderhit)
    return 0;
  if (limits.nodes > 0 || limits.depth > 0 || limits.mate > 0 ||
      limits.infinite)
    return 0;

  int budget_cap = 0;
  if (limits.movetime > 0) {
    budget_cap = static_cast<int>(std::max<TimePoint>(0, limits.movetime)) / 6;
  } else if ((limits.time[WHITE] > 0 || limits.time[BLACK] > 0) &&
             time_budget_ms >= 1000) {
    budget_cap = time_budget_ms / 8;
  } else {
    return 0;
  }

  const int budget = std::min(requested_ms, budget_cap);
  return budget >= 10 ? budget : 0;
}

::MetalFish::Search::LimitsType
HybridBuildMCTSLimits(const ::MetalFish::Search::LimitsType &limits,
                      int time_budget_ms, bool waiting_for_ponderhit) {
  if (limits.ponderMode && waiting_for_ponderhit) {
    ::MetalFish::Search::LimitsType ponder_limits = limits;
    ponder_limits.ponderMode = true;
    return ponder_limits;
  }

  ::MetalFish::Search::LimitsType mcts_limits = limits;
  const bool fixed_or_analysis = limits.movetime > 0 || limits.nodes > 0 ||
                                 limits.infinite || limits.depth > 0 ||
                                 limits.mate > 0;
  const bool clock_search = limits.time[WHITE] > 0 || limits.time[BLACK] > 0;

  if (!fixed_or_analysis && clock_search && time_budget_ms > 0) {
    mcts_limits.movetime = time_budget_ms;
    mcts_limits.time[WHITE] = 0;
    mcts_limits.time[BLACK] = 0;
    mcts_limits.inc[WHITE] = 0;
    mcts_limits.inc[BLACK] = 0;
    mcts_limits.movestogo = 0;
    mcts_limits.ponderMode = false;
  }

  return mcts_limits;
}

bool HybridMCTSDecisiveFixedBudgetOverride(bool fixed_budget, bool mcts_strong,
                                           uint64_t mcts_total_nodes,
                                           uint32_t mcts_visits,
                                           float visit_share, int eval_delta) {
  return fixed_budget && mcts_strong && mcts_total_nodes >= 300 &&
         mcts_visits >= 210 && visit_share >= 0.60f && eval_delta >= 90;
}

bool HybridMCTSNoClearFixedBudgetOverride(bool fixed_budget, bool mcts_strong,
                                          uint32_t mcts_visits,
                                          float visit_share, int eval_delta) {
  return fixed_budget && mcts_strong && mcts_visits >= 225 &&
         visit_share >= 0.58f && eval_delta >= 30;
}

bool HybridMCTSRootDominantFixedBudgetOverride(
    bool fixed_budget, bool mcts_strong, uint64_t mcts_total_nodes,
    uint32_t mcts_visits, float visit_share, int mcts_cp, int eval_delta) {
  return fixed_budget && mcts_strong && mcts_total_nodes >= 250 &&
         mcts_visits >= 200 && visit_share >= 0.74f && mcts_cp >= 150 &&
         eval_delta >= 80;
}

bool HybridMCTSTacticalGapFixedBudgetOverride(
    bool fixed_budget, uint64_t mcts_total_nodes, uint32_t mcts_visits,
    float visit_share, float root_q_gap, int mcts_cp, int eval_delta) {
  return fixed_budget && mcts_total_nodes >= 250 && mcts_visits >= 55 &&
         visit_share >= 0.16f && root_q_gap >= 0.12f && mcts_cp >= 300 &&
         eval_delta >= 250;
}

bool HybridMCTSRootConfidenceFixedBudgetOverride(
    bool fixed_budget, bool mcts_strong, uint64_t mcts_total_nodes,
    uint32_t mcts_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta) {
  if (!fixed_budget || !mcts_strong || mcts_total_nodes < 230 ||
      mcts_visits < 180 || mcts_cp < 170) {
    return false;
  }

  const bool value_gap_confident =
      visit_share >= 0.65f && root_q_gap >= 0.12f && eval_delta >= 110;
  const bool compact_root_confident =
      visit_share >= 0.74f && root_q_gap >= 0.06f && eval_delta >= 70;
  const bool clear_root_q_confident =
      mcts_total_nodes >= 300 && mcts_visits >= 240 && visit_share >= 0.64f &&
      root_q_gap >= 0.12f && mcts_cp >= 200 && eval_delta >= 60;
  return value_gap_confident || compact_root_confident ||
         clear_root_q_confident;
}

bool HybridMCTSLowNodeRootConfidenceOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_has_clear_preference,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int ab_average_score,
    int mcts_average_score, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, bool mcts_in_ab_upperbound,
    uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || ab_has_clear_preference)
    return false;

  if (mcts_root_visits < 50 || mcts_root_visits > 180 ||
      mcts_best_visits < 40 || visit_share < 0.70f || root_q_gap < 0.30f ||
      mcts_cp < 180 || eval_delta < 150) {
    return false;
  }

  if (ab_average_score - mcts_average_score > 35)
    return false;

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_rank > 4 ||
      ab_in_mcts_current_visits > 12) {
    return false;
  }

  if (mcts_q - ab_in_mcts_q < 0.40f)
    return false;

  const bool ab_barely_touched_mcts =
      mcts_in_ab_rank >= 2 && mcts_in_ab_rank <= 4 &&
      mcts_in_ab_score == -VALUE_INFINITE && !mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_effort <= 1000;
  const bool ab_shallow_lower_bound =
      mcts_in_ab_rank >= 2 && mcts_in_ab_rank <= 3 && mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_score >= -50 &&
      mcts_in_ab_effort >= 50000 && mcts_in_ab_effort <= 250000;

  return ab_barely_touched_mcts || ab_shallow_lower_bound;
}

bool HybridMCTSShortRootTacticalOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_root_rejects_mcts,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int ab_average_score,
    int mcts_average_score, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane)
    return false;

  if (mcts_root_visits < 150 || mcts_root_visits >= 230 ||
      mcts_best_visits < 95 || visit_share < 0.58f || root_q_gap < 0.16f ||
      mcts_cp < 200 || eval_delta < 45) {
    return false;
  }

  const int max_average_gap = ab_root_rejects_mcts ? 70 : 25;
  if (ab_average_score - mcts_average_score > max_average_gap)
    return false;

  if (mcts_in_ab_rank <= 0 || mcts_in_ab_rank > 3 ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_in_ab_lowerbound ||
      mcts_in_ab_effort < 200000 || mcts_in_ab_effort > 1500000) {
    return false;
  }

  if (ab_in_mcts_rank < 4 || ab_in_mcts_current_visits > 16)
    return false;

  return mcts_q - ab_in_mcts_q >= 0.24f;
}

bool HybridMCTSABLowerBoundConfirmedOverride(
    bool fixed_budget, bool visit_evidence_sane, uint64_t mcts_root_visits,
    uint32_t mcts_best_visits, float visit_share, float root_q_gap,
    int ab_score, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits) {
  if (!fixed_budget)
    return false;

  if (mcts_best_visits > mcts_root_visits)
    return false;

  if (visit_evidence_sane) {
    if (mcts_root_visits < 50 || mcts_root_visits > 140 ||
        mcts_best_visits < 45 || visit_share < 0.75f || root_q_gap < 0.30f) {
      return false;
    }

    if (ab_score > 500 || mcts_in_ab_rank <= 1 || mcts_in_ab_rank > 3 ||
        !mcts_in_ab_lowerbound || mcts_in_ab_score < 250 ||
        mcts_in_ab_score - ab_score < 250 || mcts_in_ab_effort < 100000 ||
        mcts_in_ab_effort > 1500000) {
      return false;
    }

    return ab_in_mcts_rank > 1 && ab_in_mcts_current_visits <= 8;
  }

  if (mcts_root_visits < 1500 || mcts_root_visits > 6000 ||
      mcts_best_visits < 1500 || visit_share < 0.95f || root_q_gap < 0.50f) {
    return false;
  }

  if (ab_score > 100 || mcts_in_ab_rank <= 1 || mcts_in_ab_rank > 3 ||
      !mcts_in_ab_lowerbound || mcts_in_ab_score < 500 ||
      mcts_in_ab_score - ab_score < 450 || mcts_in_ab_effort < 100000 ||
      mcts_in_ab_effort > 1500000) {
    return false;
  }

  return ab_in_mcts_rank > 1 && ab_in_mcts_current_visits <= 32;
}

bool HybridMCTSCompactFixedBudgetOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_has_clear_preference,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int ab_average_score,
    int mcts_average_score) {
  if (!fixed_budget || !visit_evidence_sane || ab_has_clear_preference ||
      mcts_root_visits < 50 || mcts_root_visits >= 180 ||
      mcts_best_visits < 48 || visit_share < 0.82f || root_q_gap < 0.55f ||
      mcts_cp < 140 || eval_delta < 140) {
    return false;
  }

  return ab_average_score - mcts_average_score <= 90;
}

bool HybridMCTSCompactPawnEndgameOverride(
    bool fixed_budget, bool visit_evidence_sane, bool pawn_only_endgame,
    bool ab_has_clear_preference, uint64_t mcts_root_visits,
    uint32_t mcts_best_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta, int ab_average_score, int mcts_average_score,
    int mcts_in_ab_rank, int mcts_in_ab_score, bool mcts_in_ab_lowerbound,
    uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || !pawn_only_endgame ||
      ab_has_clear_preference) {
    return false;
  }

  if (mcts_root_visits < 35 || mcts_root_visits >= 80 ||
      mcts_best_visits < 32 || visit_share < 0.88f || root_q_gap < 0.90f ||
      mcts_cp < 220 || eval_delta < 220) {
    return false;
  }

  if (ab_average_score - mcts_average_score > 90)
    return false;

  if (mcts_in_ab_rank != 2 || mcts_in_ab_score != -VALUE_INFINITE ||
      mcts_in_ab_lowerbound || mcts_in_ab_effort < 100000 ||
      mcts_in_ab_effort > 600000) {
    return false;
  }

  if (ab_in_mcts_rank != 2 || ab_in_mcts_current_visits > 2)
    return false;

  return mcts_q - ab_in_mcts_q >= 0.90f;
}

bool HybridMCTSCompactClearPreferenceOverride(
    bool fixed_budget, bool visit_evidence_sane, uint64_t mcts_root_visits,
    uint32_t mcts_best_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta, int ab_score, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, bool mcts_in_ab_upperbound,
    uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane)
    return false;

  if (mcts_root_visits < 50 || mcts_root_visits >= 180 ||
      mcts_best_visits < 48 || visit_share < 0.90f || root_q_gap < 0.75f ||
      mcts_cp < 280 || eval_delta < 300 || std::abs(ab_score) > 40) {
    return false;
  }

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_current_visits > 2)
    return false;

  if (mcts_q - ab_in_mcts_q < 0.75f)
    return false;

  return mcts_in_ab_rank == 2 && mcts_in_ab_score == -VALUE_INFINITE &&
         !mcts_in_ab_lowerbound && mcts_in_ab_upperbound &&
         mcts_in_ab_effort >= 20000 && mcts_in_ab_effort <= 200000;
}

bool HybridMCTSCrossRootConfidenceOverride(
    bool fixed_budget, bool mcts_strong, uint64_t mcts_total_nodes,
    uint32_t mcts_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta, int ab_average_score, int mcts_in_ab_rank,
    int mcts_in_ab_score, int mcts_average_score, uint64_t mcts_effort,
    int ab_in_mcts_rank, uint32_t ab_in_mcts_visits, float ab_in_mcts_q,
    float mcts_q) {
  if (!fixed_budget || !mcts_strong || mcts_total_nodes < 250 ||
      mcts_visits < 170 || visit_share < 0.62f || root_q_gap < 0.12f ||
      mcts_cp < 170 || eval_delta < 40) {
    return false;
  }

  if (mcts_in_ab_rank <= 0 || mcts_in_ab_rank > 4 ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_effort < 100000) {
    return false;
  }

  if (ab_average_score - mcts_average_score > 80)
    return false;

  if (ab_in_mcts_rank < 4 || ab_in_mcts_visits == 0)
    return false;

  return mcts_visits >= 3 * std::max<uint32_t>(1, ab_in_mcts_visits) &&
         mcts_q - ab_in_mcts_q >= 0.20f;
}

bool HybridMCTSRootConfidenceRejectOverride(
    bool root_confidence_fixed_budget, bool ab_root_rejects_mcts,
    int mcts_in_ab_rank, int mcts_in_ab_score, bool mcts_in_ab_lowerbound,
    bool mcts_in_ab_upperbound, uint64_t mcts_in_ab_effort,
    int ab_average_score, int mcts_average_score, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_visits, float ab_in_mcts_q, float mcts_q) {
  if (!root_confidence_fixed_budget || !ab_root_rejects_mcts)
    return false;

  if (mcts_in_ab_rank <= 0 || mcts_in_ab_score != -VALUE_INFINITE ||
      mcts_in_ab_lowerbound || mcts_in_ab_upperbound ||
      mcts_in_ab_effort > 10000) {
    return false;
  }

  const int average_gap = ab_average_score - mcts_average_score;
  const float q_gap_to_ab = mcts_q - ab_in_mcts_q;

  if (mcts_in_ab_rank > 3) {
    if (mcts_in_ab_rank > 10 || average_gap > 90 || ab_in_mcts_rank < 2 ||
        ab_in_mcts_rank > 4 || ab_in_mcts_visits == 0 ||
        ab_in_mcts_visits > 16) {
      return false;
    }

    return q_gap_to_ab >= 0.75f;
  }

  if (average_gap > 120)
    return false;

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_visits < 10)
    return false;

  return q_gap_to_ab >= 0.60f;
}

bool HybridMCTSRootRejectLowMaterialPushOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_root_rejects_mcts,
    bool low_material_root, bool mcts_kingside_pawn_push,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int ab_average_score,
    int mcts_average_score, int mcts_in_ab_rank, int mcts_in_ab_score,
    uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || !ab_root_rejects_mcts ||
      !low_material_root || !mcts_kingside_pawn_push) {
    return false;
  }

  if (mcts_root_visits < 700 || mcts_root_visits > 1600 ||
      mcts_best_visits < 500 || visit_share < 0.68f || root_q_gap < 0.45f ||
      mcts_cp < 140 || eval_delta < 130) {
    return false;
  }

  if (ab_average_score - mcts_average_score > 30)
    return false;

  if (mcts_in_ab_rank <= 1 || mcts_in_ab_rank > 3 ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_in_ab_effort < 20000 ||
      mcts_in_ab_effort > 80000) {
    return false;
  }

  if (ab_in_mcts_rank != 2 || ab_in_mcts_current_visits < 120)
    return false;

  return mcts_q - ab_in_mcts_q >= 0.45f;
}

bool HybridMCTSRootRejectRookEndgamePawnPushOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_root_rejects_mcts,
    bool rook_endgame_root, bool mcts_quiet_central_pawn_push,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int mcts_in_ab_rank,
    int mcts_in_ab_score, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !ab_root_rejects_mcts || !rook_endgame_root ||
      !mcts_quiet_central_pawn_push || mcts_best_visits > mcts_root_visits) {
    return false;
  }

  if (visit_evidence_sane) {
    if (mcts_root_visits < 45 || mcts_root_visits > 1600 ||
        mcts_best_visits < 40 || visit_share < 0.88f || root_q_gap < 0.60f ||
        mcts_cp < 430 || eval_delta < 300) {
      return false;
    }
  } else {
    if (mcts_root_visits < 1500 || mcts_root_visits > 6000 ||
        mcts_best_visits < 1500 || visit_share < 0.95f || root_q_gap < 0.50f ||
        mcts_cp < 450 || eval_delta < 380) {
      return false;
    }
  }

  if (mcts_in_ab_rank <= 1 || mcts_in_ab_rank > 6 ||
      mcts_in_ab_score != -VALUE_INFINITE ||
      mcts_in_ab_effort > (visit_evidence_sane ? 20000 : 2000)) {
    return false;
  }

  if (ab_in_mcts_rank < 2 || ab_in_mcts_rank > 3 ||
      ab_in_mcts_current_visits == 0 || ab_in_mcts_current_visits > 32) {
    return false;
  }

  const uint32_t dominance_multiplier = visit_evidence_sane ? 6 : 50;
  if (mcts_best_visits <
      dominance_multiplier * std::max<uint32_t>(1, ab_in_mcts_current_visits)) {
    return false;
  }

  return mcts_q - ab_in_mcts_q >= 0.60f;
}

bool HybridMCTSRootRejectQuietQueenMoveOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_root_rejects_mcts,
    bool mcts_quiet_central_queen_move, uint64_t mcts_root_visits,
    uint32_t mcts_best_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, bool mcts_in_ab_upperbound,
    uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || !ab_root_rejects_mcts ||
      !mcts_quiet_central_queen_move) {
    return false;
  }

  if (mcts_root_visits < 50 || mcts_root_visits > 1200 ||
      mcts_best_visits < 47 || visit_share < 0.85f || root_q_gap < 1.05f ||
      mcts_cp < 330 || eval_delta < 330) {
    return false;
  }

  if (mcts_in_ab_rank <= 1 || mcts_in_ab_rank > 6 ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_in_ab_lowerbound ||
      mcts_in_ab_upperbound || mcts_in_ab_effort > 512) {
    return false;
  }

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_rank > 4 ||
      ab_in_mcts_current_visits == 0 || ab_in_mcts_current_visits > 4) {
    return false;
  }

  if (mcts_best_visits <
      12 * std::max<uint32_t>(1, ab_in_mcts_current_visits)) {
    return false;
  }

  return mcts_q - ab_in_mcts_q >= 1.05f;
}

bool HybridMCTSRootRejectQuietMinorMajorAttackOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_has_clear_preference,
    bool ab_root_rejects_mcts, bool mcts_quiet_minor_major_attack,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int mcts_in_ab_rank,
    int mcts_in_ab_score, bool mcts_in_ab_lowerbound,
    bool mcts_in_ab_upperbound, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || !ab_has_clear_preference ||
      !ab_root_rejects_mcts || !mcts_quiet_minor_major_attack) {
    return false;
  }

  if (mcts_root_visits < 55 || mcts_root_visits > 90 || mcts_best_visits < 30 ||
      visit_share < 0.50f || root_q_gap < 0.19f || mcts_cp < 180 ||
      eval_delta < 75) {
    return false;
  }

  if (mcts_in_ab_rank < 5 || mcts_in_ab_rank > 12 ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_in_ab_lowerbound ||
      mcts_in_ab_effort > 10000) {
    return false;
  }

  const float q_gap_to_ab = mcts_q - ab_in_mcts_q;
  const bool outranks_ab = ab_in_mcts_rank >= 2 && ab_in_mcts_rank <= 3 &&
                           ab_in_mcts_current_visits <= 24 &&
                           q_gap_to_ab >= 0.19f;
  const bool lightly_sampled_ab =
      ab_in_mcts_rank >= 4 && ab_in_mcts_current_visits <= 4 &&
      mcts_best_visits >=
          12 * std::max<uint32_t>(1, ab_in_mcts_current_visits) &&
      q_gap_to_ab >= 0.45f;
  return outranks_ab || lightly_sampled_ab;
}

bool HybridMCTSReusedRootConfidenceOverride(
    bool fixed_budget, uint64_t mcts_root_visits, uint32_t mcts_best_visits,
    uint64_t mcts_root_current_visits, uint32_t mcts_best_current_visits,
    float absolute_visit_share, float current_visit_share, int mcts_cp,
    int eval_delta, int ab_average_score, int mcts_average_score,
    int mcts_in_ab_rank, int mcts_in_ab_score, uint64_t mcts_in_ab_effort,
    int ab_in_mcts_rank, uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q,
    float mcts_q) {
  if (!fixed_budget)
    return false;

  const float q_gap_to_ab = mcts_q - ab_in_mcts_q;
  const int average_gap = ab_average_score - mcts_average_score;
  const bool high_cache_near_current =
      mcts_root_visits >= 500 && mcts_best_visits >= 500 &&
      mcts_root_current_visits >= 18 && mcts_best_current_visits >= 18 &&
      mcts_in_ab_effort <= 100000;
  const bool moderate_cache_current =
      mcts_root_visits >= 400 && mcts_best_visits >= 400 &&
      mcts_root_current_visits >= 20 && mcts_best_current_visits >= 20 &&
      mcts_in_ab_effort <= 150000;
  const bool cache_heavy_near_current =
      (high_cache_near_current || moderate_cache_current) &&
      mcts_root_visits >= 3 * std::max<uint64_t>(1, mcts_root_current_visits) &&
      mcts_best_visits >= 3 * static_cast<uint32_t>(std::max<uint32_t>(
                                  1, mcts_best_current_visits)) &&
      absolute_visit_share >= 0.96f && current_visit_share >= 0.96f &&
      mcts_cp >= 400 && eval_delta >= 400 && average_gap <= 75 &&
      mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 2 &&
      mcts_in_ab_score == -VALUE_INFINITE && ab_in_mcts_rank == 2 &&
      ab_in_mcts_current_visits <= 1 && q_gap_to_ab >= 0.85f;
  if (cache_heavy_near_current)
    return true;

  if (mcts_root_visits < 200 || mcts_best_visits < 180 ||
      mcts_root_current_visits < 24 || mcts_best_current_visits < 24 ||
      absolute_visit_share < 0.90f || current_visit_share < 0.90f) {
    return false;
  }

  if (mcts_root_visits < 3 * std::max<uint64_t>(1, mcts_root_current_visits) ||
      mcts_best_visits < 3 * static_cast<uint32_t>(std::max<uint32_t>(
                                 1, mcts_best_current_visits))) {
    return false;
  }

  if (mcts_cp < 300 || eval_delta < 300)
    return false;

  if (mcts_in_ab_rank <= 0 || mcts_in_ab_rank > 3 ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_in_ab_effort > 250000) {
    return false;
  }

  if (average_gap > 75)
    return false;

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_current_visits > 2)
    return false;

  return q_gap_to_ab >= 0.75f;
}

bool HybridMCTSReusedRootCurrentOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_has_clear_preference,
    bool ab_root_rejects_mcts, uint64_t mcts_root_visits,
    uint32_t mcts_best_visits, uint64_t mcts_root_current_visits,
    uint32_t mcts_best_current_visits, float absolute_visit_share,
    float current_visit_share, float root_q_gap, int mcts_cp, int eval_delta,
    int mcts_in_ab_rank, int mcts_in_ab_score, bool mcts_in_ab_lowerbound,
    bool mcts_in_ab_upperbound, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || ab_has_clear_preference)
    return false;

  if (mcts_root_visits < 350 || mcts_best_visits < 300 ||
      mcts_root_current_visits < 24 || mcts_best_current_visits < 24 ||
      absolute_visit_share < 0.84f || current_visit_share < 0.92f ||
      mcts_cp < 300 || eval_delta < 300) {
    return false;
  }

  if (mcts_root_visits < 3 * std::max<uint64_t>(1, mcts_root_current_visits) ||
      mcts_best_visits < 3 * static_cast<uint32_t>(std::max<uint32_t>(
                                 1, mcts_best_current_visits))) {
    return false;
  }

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_current_visits > 1)
    return false;

  const float q_gap_to_ab = mcts_q - ab_in_mcts_q;
  if (q_gap_to_ab < 0.47f)
    return false;

  const bool ab_searched_lower_bound =
      mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 2 && mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_score >= 300 &&
      mcts_in_ab_effort >= 250000;
  const bool late_lower_bound =
      mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 2 && mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_effort >= 250000 &&
      ((mcts_in_ab_score >= 0 && absolute_visit_share >= 0.93f &&
        mcts_cp >= 400 && q_gap_to_ab >= 0.58f) ||
       (mcts_in_ab_score >= -50 && absolute_visit_share >= 0.95f &&
        current_visit_share >= 0.95f && mcts_cp >= 400 && eval_delta >= 400 &&
        q_gap_to_ab >= 0.85f));
  const bool ab_barely_touched_mcts =
      ab_root_rejects_mcts && mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 4 &&
      mcts_in_ab_score == -VALUE_INFINITE && mcts_in_ab_effort <= 10000 &&
      root_q_gap >= 0.75f && q_gap_to_ab >= 0.50f;

  return ab_searched_lower_bound || late_lower_bound || ab_barely_touched_mcts;
}

bool HybridMCTSBishopEndgameRetreatOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_has_clear_preference,
    bool bishop_endgame_root, bool mcts_bishop_back_rank_retreat,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits,
    uint64_t mcts_root_current_visits, uint32_t mcts_best_current_visits,
    float absolute_visit_share, float current_visit_share, int mcts_cp,
    int eval_delta, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, bool mcts_in_ab_upperbound,
    uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !visit_evidence_sane || !bishop_endgame_root ||
      !mcts_bishop_back_rank_retreat) {
    return false;
  }

  const float q_gap_to_ab = mcts_q - ab_in_mcts_q;
  if (mcts_best_visits > mcts_root_visits ||
      mcts_best_current_visits > mcts_root_current_visits) {
    return false;
  }

  const bool barely_touched_score =
      mcts_in_ab_score == -VALUE_INFINITE && !mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_effort <= 10000;
  const bool low_node_barely_touched =
      !ab_has_clear_preference && barely_touched_score && mcts_in_ab_rank > 0 &&
      mcts_in_ab_rank <= 4 && mcts_root_visits >= 80 &&
      mcts_root_visits <= 190 && mcts_best_visits >= 30 &&
      mcts_root_current_visits >= 12 && mcts_best_current_visits >= 12 &&
      absolute_visit_share >= 0.35f && current_visit_share >= 0.70f &&
      mcts_cp >= 190 && eval_delta >= 180 && ab_in_mcts_rank >= 3 &&
      ab_in_mcts_rank <= 4 && ab_in_mcts_current_visits <= 12 &&
      q_gap_to_ab >= 0.30f;
  if (low_node_barely_touched)
    return true;

  const bool low_node_q_gap_barely_touched =
      !ab_has_clear_preference && barely_touched_score && mcts_in_ab_rank > 0 &&
      mcts_in_ab_rank <= 4 && mcts_root_visits >= 100 &&
      mcts_root_visits <= 130 && mcts_best_visits >= 45 &&
      mcts_root_current_visits >= 55 && mcts_best_current_visits >= 36 &&
      absolute_visit_share >= 0.43f && current_visit_share >= 0.58f &&
      mcts_cp >= 210 && eval_delta >= 200 && ab_in_mcts_rank >= 2 &&
      ab_in_mcts_rank <= 3 && ab_in_mcts_current_visits <= 12 &&
      q_gap_to_ab >= 0.29f;
  if (low_node_q_gap_barely_touched)
    return true;

  const bool low_node_lowerbound_confirms =
      !ab_has_clear_preference && mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 2 &&
      mcts_in_ab_lowerbound && !mcts_in_ab_upperbound &&
      mcts_in_ab_score >= 140 && mcts_in_ab_effort >= 200000 &&
      mcts_root_visits >= 90 && mcts_root_visits <= 130 &&
      mcts_best_visits >= 40 && mcts_root_current_visits >= 50 &&
      mcts_best_current_visits >= 30 && absolute_visit_share >= 0.40f &&
      current_visit_share >= 0.58f && mcts_cp >= 220 && eval_delta >= 210 &&
      ab_in_mcts_rank >= 3 && ab_in_mcts_current_visits <= 12 &&
      q_gap_to_ab >= 0.38f;
  if (low_node_lowerbound_confirms)
    return true;

  if (mcts_root_visits < 200 || mcts_best_visits < 180 ||
      mcts_root_current_visits < 24 || absolute_visit_share < 0.75f ||
      current_visit_share < 0.94f || mcts_cp < 280 || eval_delta < 210) {
    return false;
  }

  if (ab_in_mcts_rank <= 1 || ab_in_mcts_current_visits > 1)
    return false;

  const bool high_confidence_one_visit_short =
      mcts_best_current_visits >= 23 && mcts_root_current_visits >= 24 &&
      mcts_cp >= 360 && eval_delta >= 360 && q_gap_to_ab >= 0.58f;
  if (mcts_best_current_visits < 24 && !high_confidence_one_visit_short)
    return false;

  const bool low_gap_barely_touched =
      barely_touched_score && mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 4 &&
      ab_in_mcts_current_visits == 0 && absolute_visit_share >= 0.80f &&
      current_visit_share >= 0.99f && mcts_cp >= 300 && eval_delta >= 290;
  if (q_gap_to_ab < (low_gap_barely_touched ? 0.46f : 0.50f))
    return false;

  const bool barely_touched_rank_ok =
      mcts_in_ab_rank > 0 &&
      (mcts_in_ab_rank <= 4 ||
       (mcts_in_ab_rank == 5 && mcts_cp >= 360 && eval_delta >= 350));
  const bool ab_barely_touched_mcts =
      barely_touched_score && barely_touched_rank_ok;
  const bool ab_lowerbound_confirms_mcts =
      mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 2 && mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_score >= 0 &&
      mcts_in_ab_effort >= 250000;
  const bool ab_shallow_lowerbound_confirms_mcts =
      mcts_in_ab_rank > 0 && mcts_in_ab_rank <= 2 && mcts_in_ab_lowerbound &&
      !mcts_in_ab_upperbound && mcts_in_ab_score >= 150 &&
      mcts_in_ab_effort >= 150000 && absolute_visit_share >= 0.90f &&
      mcts_cp >= 380 && eval_delta >= 360 && q_gap_to_ab >= 0.60f;

  if (ab_has_clear_preference)
    return ab_lowerbound_confirms_mcts && mcts_in_ab_score >= 300 &&
           eval_delta >= 220;

  return ab_barely_touched_mcts || ab_lowerbound_confirms_mcts ||
         ab_shallow_lowerbound_confirms_mcts;
}

bool HybridMCTSVisitEvidenceSane(uint64_t mcts_playouts, uint64_t mcts_evals,
                                 uint64_t root_visits, uint32_t best_visits) {
  if (mcts_playouts == 0) {
    return root_visits == 0 && best_visits == 0;
  }

  if (best_visits > root_visits)
    return false;

  const uint64_t visit_slack = std::max<uint64_t>(64, mcts_playouts / 32);
  if (root_visits > mcts_playouts + visit_slack)
    return false;

  if (mcts_evals == 0)
    return true;

  const uint64_t best_eval_limit = std::max<uint64_t>(512, mcts_evals * 64);
  const uint64_t root_eval_limit = std::max<uint64_t>(1024, mcts_evals * 96);
  return best_visits <= best_eval_limit && root_visits <= root_eval_limit;
}

bool HybridANEConfirmedMCTSOverride(bool enabled, bool ane_agrees_mcts,
                                    bool fixed_budget, bool visit_evidence_sane,
                                    uint64_t mcts_root_visits,
                                    uint32_t mcts_best_visits,
                                    float visit_share, float root_q_gap,
                                    int mcts_cp, int eval_delta,
                                    float ane_score_margin) {
  const bool large_ane_margin_low_cp =
      mcts_cp >= 80 && eval_delta >= 80 && ane_score_margin >= 0.45f;
  if (!enabled || !ane_agrees_mcts || !fixed_budget || root_q_gap < 0.20f ||
      (!large_ane_margin_low_cp && mcts_cp < 120) || eval_delta < 60) {
    return false;
  }
  if (mcts_best_visits > mcts_root_visits)
    return false;

  if (!visit_evidence_sane)
    return false;

  if (mcts_root_visits >= 80 && mcts_best_visits >= 64 &&
      visit_share >= 0.70f && ane_score_margin >= 0.10f) {
    return true;
  }

  if (mcts_root_visits < 80 && mcts_root_visits >= 50 &&
      mcts_best_visits >= 45 && mcts_best_visits <= 49 &&
      visit_share >= 0.85f && root_q_gap >= 0.55f && mcts_cp >= 130 &&
      eval_delta >= 120 && ane_score_margin >= 0.20f) {
    return true;
  }

  if (mcts_root_visits < 80 && mcts_root_visits >= 56 &&
      mcts_best_visits >= 50 && visit_share >= 0.88f && root_q_gap >= 0.60f &&
      mcts_cp >= 300 && eval_delta >= 200 && ane_score_margin >= 0.12f) {
    return true;
  }

  if (mcts_root_visits < 80 && mcts_root_visits >= 60 &&
      mcts_best_visits >= 55 && visit_share >= 0.86f && root_q_gap >= 0.40f &&
      mcts_cp >= 80 && eval_delta >= 80 && ane_score_margin >= 0.45f) {
    return true;
  }

  if (mcts_root_visits < 80 && mcts_root_visits >= 30 &&
      mcts_best_visits >= 24 && visit_share >= 0.44f &&
      root_q_gap >= 1.0f && mcts_cp >= 280 && eval_delta >= 300 &&
      ane_score_margin >= 0.15f) {
    return true;
  }

  return mcts_root_visits < 80 && mcts_root_visits >= 50 &&
         mcts_best_visits >= 50 && visit_share >= 0.90f &&
         root_q_gap >= 0.50f && mcts_cp >= 200 && eval_delta >= 150 &&
         ane_score_margin >= 0.10f;
}

bool HybridANEQSupportedRootOverride(
    bool enabled, bool fixed_budget, bool visit_evidence_sane,
    bool ab_root_rejects_mcts, int ane_rank, float ane_score_margin,
    uint64_t mcts_root_visits, int ane_in_mcts_rank,
    uint32_t ane_in_mcts_current_visits, float ane_in_mcts_q,
    float ane_in_mcts_policy, uint32_t selected_mcts_current_visits,
    float selected_mcts_q, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, int ane_in_ab_rank,
    int ane_in_ab_score, int ane_in_ab_average_score,
    bool ane_in_ab_upperbound, uint64_t ane_in_ab_effort,
    int selected_ab_average_score) {
  if (!enabled || !fixed_budget || !visit_evidence_sane ||
      !ab_root_rejects_mcts || ane_rank != 1 || ane_score_margin < 0.15f) {
    return false;
  }

  if (mcts_root_visits < 30 || mcts_root_visits > 120 ||
      ane_in_mcts_rank <= 0 || ane_in_mcts_rank > 2 ||
      ane_in_mcts_current_visits < 20 || selected_mcts_current_visits < 20 ||
      ane_in_mcts_policy < 0.08f || ane_in_mcts_q < 0.55f) {
    return false;
  }

  if (ane_in_mcts_current_visits > mcts_root_visits ||
      selected_mcts_current_visits > mcts_root_visits) {
    return false;
  }

  if (static_cast<uint64_t>(ane_in_mcts_current_visits) * 100 <
      mcts_root_visits * 35) {
    return false;
  }

  if (ane_in_mcts_q - selected_mcts_q < 0.75f)
    return false;

  if (ab_in_mcts_rank > 0) {
    if (ab_in_mcts_current_visits > mcts_root_visits)
      return false;
    if (ane_in_mcts_q - ab_in_mcts_q < 0.75f)
      return false;
  }

  const bool ab_barely_evaluated_ane =
      ane_in_ab_score <= -30000 && ane_in_ab_effort <= 1000;
  if (ane_in_ab_rank <= 0 || (!ab_barely_evaluated_ane && ane_in_ab_rank > 4))
    return false;

  const bool ab_did_not_refute_ane =
      ane_in_ab_upperbound &&
      selected_ab_average_score - ane_in_ab_average_score <= 140 &&
      ane_in_ab_effort <= 10000;
  return ab_barely_evaluated_ane || ab_did_not_refute_ane;
}

bool HybridABRootRejectsMCTS(bool ab_verified, int ab_rank, int mcts_rank,
                             int ab_average_score, int mcts_average_score,
                             uint64_t ab_effort, uint64_t mcts_effort,
                             int mcts_score) {
  if (!ab_verified || ab_rank != 1 || mcts_rank <= 1)
    return false;

  const int average_gap = ab_average_score - mcts_average_score;
  if (average_gap >= 30 && ab_effort >= 10000)
    return true;

  return mcts_score <= -30000 && ab_effort >= 10000 &&
         ab_effort >= 10 * std::max<uint64_t>(1, mcts_effort);
}

bool HybridRootPolicyTieBreak(bool fixed_budget, uint64_t root_visits,
                              uint32_t top_visits, float top_q,
                              float top_policy, uint32_t candidate_visits,
                              float candidate_q, float candidate_policy) {
  return fixed_budget && root_visits >= 600 && top_visits >= 256 &&
         candidate_visits >= 256 &&
         static_cast<uint64_t>(candidate_visits) * 100 >=
             static_cast<uint64_t>(top_visits) * 82 &&
         top_q - candidate_q <= 0.05f && candidate_policy >= 0.30f &&
         candidate_policy >= top_policy * 2.0f;
}

bool HybridRootQConflictTieBreak(bool fixed_budget, bool visit_evidence_sane,
                                 uint64_t root_visits, uint32_t selected_visits,
                                 float selected_q, float selected_policy,
                                 uint32_t candidate_visits, float candidate_q,
                                 float candidate_policy,
                                 int selected_average_score,
                                 int candidate_average_score) {
  if (!fixed_budget || !visit_evidence_sane || root_visits < 40 ||
      root_visits > 90 || selected_visits < 20 || candidate_visits < 20) {
    return false;
  }

  if (static_cast<uint64_t>(candidate_visits) * 100 <
      static_cast<uint64_t>(selected_visits) * 75) {
    return false;
  }

  if (selected_q > 0.05f || candidate_q < 0.10f ||
      candidate_q - selected_q < 0.45f) {
    return false;
  }

  if (selected_policy < 0.20f || candidate_policy > selected_policy * 0.60f) {
    return false;
  }

  return selected_average_score - candidate_average_score <= 60;
}

bool HybridMCTSRootRejectsAB(bool fixed_budget, bool visit_evidence_sane,
                             bool mcts_strong, bool ab_has_clear_preference,
                             uint32_t top_visits, uint32_t ab_visits,
                             float top_q, float ab_q, float visit_share,
                             int eval_delta) {
  if (!visit_evidence_sane || ab_visits == 0)
    return false;

  const float root_q_gap = top_q - ab_q;
  const bool hard_reject = ab_visits >= 25 &&
                           top_visits >= 3 * std::max<uint32_t>(1, ab_visits) &&
                           root_q_gap >= 0.25f;
  const bool no_clear_ab_reject =
      fixed_budget && mcts_strong && !ab_has_clear_preference &&
      top_visits >= 180 && top_visits >= 4 * std::max<uint32_t>(1, ab_visits) &&
      ab_visits <= 64 && visit_share >= 0.65f && root_q_gap >= 0.12f &&
      eval_delta >= 25;
  return hard_reject || no_clear_ab_reject;
}

bool HybridMCTSRootRejectQGapOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_root_rejects_mcts,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int mcts_in_ab_rank,
    int mcts_in_ab_score, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  if (!fixed_budget || !ab_root_rejects_mcts ||
      mcts_best_visits > mcts_root_visits) {
    return false;
  }

  if (!visit_evidence_sane || mcts_root_visits < 40 || mcts_root_visits > 90 ||
      mcts_best_visits < 25 || mcts_best_visits > 40 || visit_share < 0.40f ||
      visit_share > 0.60f || root_q_gap < 0.60f || mcts_cp < 80 ||
      eval_delta < 150) {
    const bool medium_root_discovery =
        visit_evidence_sane && mcts_root_visits >= 90 &&
        mcts_root_visits <= 300 && mcts_best_visits >= 60 &&
        visit_share >= 0.60f && root_q_gap >= 0.75f && mcts_cp >= 140 &&
        eval_delta >= 220 && mcts_in_ab_rank == 2 &&
        mcts_in_ab_score == -VALUE_INFINITE && mcts_in_ab_effort <= 1500 &&
        ab_in_mcts_rank == 2 && ab_in_mcts_current_visits >= 20 &&
        mcts_q - ab_in_mcts_q >= 0.75f;
    const bool cache_heavy_discovery =
        !visit_evidence_sane && mcts_root_visits >= 1400 &&
        mcts_root_visits <= 2600 && mcts_best_visits >= 1500 &&
        visit_share >= 0.94f && root_q_gap >= 1.0f && mcts_cp >= 300 &&
        eval_delta >= 350 && mcts_in_ab_rank >= 2 && mcts_in_ab_rank <= 3 &&
        mcts_in_ab_score == -VALUE_INFINITE && mcts_in_ab_effort <= 2600 &&
        ab_in_mcts_rank == 2 && ab_in_mcts_current_visits >= 20 &&
        ab_in_mcts_current_visits <= 64 && mcts_q - ab_in_mcts_q >= 1.0f;
    return medium_root_discovery || cache_heavy_discovery;
  }

  if ((mcts_in_ab_rank != 2 && mcts_in_ab_rank != 3) ||
      mcts_in_ab_score != -VALUE_INFINITE || mcts_in_ab_effort > 2500) {
    return false;
  }

  if (ab_in_mcts_rank != 2 || ab_in_mcts_current_visits < 20 ||
      static_cast<uint64_t>(ab_in_mcts_current_visits) * 100 <
          static_cast<uint64_t>(mcts_best_visits) * 75) {
    return false;
  }

  return mcts_q - ab_in_mcts_q >= 0.55f;
}

bool HybridMCTSClockRootRejectQGapOverride(
    bool clock_managed, bool visit_evidence_sane, bool ab_root_rejects_mcts,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int mcts_in_ab_rank,
    int mcts_in_ab_score, uint64_t mcts_in_ab_effort, int ab_in_mcts_rank,
    uint32_t ab_in_mcts_current_visits, float ab_in_mcts_q, float mcts_q) {
  return clock_managed && visit_evidence_sane && ab_root_rejects_mcts &&
         mcts_root_visits >= 100 && mcts_root_visits <= 300 &&
         mcts_best_visits >= 72 && visit_share >= 0.65f &&
         root_q_gap >= 0.90f && mcts_cp >= 220 && eval_delta >= 300 &&
         mcts_in_ab_rank >= 2 && mcts_in_ab_rank <= 6 &&
         mcts_in_ab_score == -VALUE_INFINITE && mcts_in_ab_effort <= 1000 &&
         ab_in_mcts_rank >= 2 && ab_in_mcts_rank <= 6 &&
         ab_in_mcts_current_visits <= 4 && mcts_q - ab_in_mcts_q >= 0.90f;
}

bool HybridPreserveLeadingHintAfterABVerify(
    bool fixed_budget, bool visit_evidence_sane, bool candidate_shape,
    uint64_t mcts_current_root_visits, uint32_t mcts_current_best_visits,
    float current_visit_share, float mcts_q, float displaced_q,
    float mcts_policy, float displaced_policy, int displaced_average_score,
    int mcts_average_score, int displaced_score, bool displaced_lowerbound,
    int mcts_score, bool mcts_lowerbound, bool mcts_upperbound,
    uint64_t mcts_effort) {
  if (!fixed_budget || !visit_evidence_sane || !candidate_shape)
    return false;

  if (mcts_current_root_visits < 18 || mcts_current_root_visits > 80 ||
      mcts_current_best_visits < 17 ||
      mcts_current_best_visits > mcts_current_root_visits ||
      current_visit_share < 0.88f) {
    return false;
  }

  if (mcts_q < 0.72f || mcts_q - displaced_q < 0.40f)
    return false;

  if (mcts_policy + 0.03f < displaced_policy)
    return false;

  if (displaced_average_score - mcts_average_score > 35)
    return false;

  if (displaced_lowerbound || displaced_score > 500)
    return false;

  const bool unresolved_or_shallow = mcts_score == -VALUE_INFINITE ||
                                     mcts_lowerbound || mcts_upperbound ||
                                     mcts_effort < 500000;
  return unresolved_or_shallow;
}

bool HybridMCTSDiscoveredPawnPushOverride(
    bool fixed_budget, bool visit_evidence_sane, bool candidate_shape,
    uint64_t mcts_current_root_visits, uint32_t mcts_current_best_visits,
    uint32_t ab_current_visits, float current_visit_share, float q_gap_to_ab,
    int mcts_cp, int eval_delta, int ab_score, int ab_average_score,
    int mcts_average_score, int mcts_in_ab_rank, int mcts_in_ab_score,
    bool mcts_in_ab_lowerbound, bool mcts_in_ab_upperbound,
    uint64_t mcts_in_ab_effort) {
  if (!fixed_budget || !visit_evidence_sane || !candidate_shape)
    return false;

  if (mcts_current_root_visits < 18 || mcts_current_root_visits > 64 ||
      mcts_current_best_visits < 17 ||
      mcts_current_best_visits > mcts_current_root_visits ||
      ab_current_visits > 4 || current_visit_share < 0.90f) {
    return false;
  }

  if (q_gap_to_ab < 0.40f || mcts_cp < 280 || eval_delta < 240 ||
      std::abs(ab_score) > 120) {
    return false;
  }

  if (ab_average_score - mcts_average_score > 50)
    return false;

  if (mcts_in_ab_rank <= 0 || mcts_in_ab_rank > 3 ||
      mcts_in_ab_effort > 500000) {
    return false;
  }

  if (mcts_in_ab_score == -VALUE_INFINITE)
    return true;

  return mcts_in_ab_lowerbound && !mcts_in_ab_upperbound &&
         mcts_in_ab_score >= 30;
}

bool HybridRootPawnLeverAgreementTieBreak(bool fixed_budget,
                                          bool visit_evidence_sane,
                                          float agreement_visit_share,
                                          float root_q_gap) {
  return fixed_budget && visit_evidence_sane && agreement_visit_share < 0.55f &&
         root_q_gap < 0.08f;
}

bool HybridRootPawnLeverCandidate(
    int selected_average_score, int candidate_average_score,
    uint64_t candidate_effort, int mcts_rank, uint32_t mcts_current_visits,
    int selected_mcts_rank, float selected_mcts_q, float selected_mcts_policy,
    float best_mcts_q, float candidate_mcts_q, float candidate_mcts_policy) {
  const bool high_policy_lever =
      selected_mcts_rank > 0 && candidate_mcts_policy >= 0.25f &&
      candidate_mcts_policy >= selected_mcts_policy * 1.15f;
  const int max_average_gap = high_policy_lever ? 80 : 60;
  const uint32_t min_current_visits = mcts_rank >= 5 ? 7 : 8;
  const bool low_visit_agreement_lever =
      !high_policy_lever && selected_mcts_rank > 0 && selected_mcts_rank <= 3 &&
      mcts_rank >= 5 && mcts_rank <= 6 && mcts_current_visits >= 3 &&
      selected_average_score - candidate_average_score <= 40 &&
      candidate_effort >= 300 && candidate_mcts_policy >= 0.035f &&
      selected_mcts_q - candidate_mcts_q <= 0.07f &&
      best_mcts_q - candidate_mcts_q <= 0.07f;
  const bool defensive_low_visit_lever =
      !high_policy_lever && selected_average_score <= -250 &&
      selected_mcts_rank == 1 && mcts_rank >= 5 && mcts_rank <= 6 &&
      mcts_current_visits >= 2 &&
      selected_average_score - candidate_average_score <= 40 &&
      candidate_effort >= 900 && candidate_mcts_policy >= 0.035f &&
      selected_mcts_q - candidate_mcts_q <= 0.09f &&
      best_mcts_q - candidate_mcts_q <= 0.09f;
  const bool low_visit_lever =
      low_visit_agreement_lever || defensive_low_visit_lever;
  if (mcts_rank <= 0 || mcts_rank > 8 ||
      (mcts_current_visits < min_current_visits && !low_visit_lever) ||
      selected_average_score - candidate_average_score > max_average_gap ||
      (!high_policy_lever && candidate_effort < 150)) {
    return false;
  }

  if (selected_mcts_rank > 0) {
    const float q_gap = selected_mcts_q - candidate_mcts_q;
    if (mcts_rank == 4 && candidate_average_score < selected_average_score &&
        q_gap > 0.15f)
      return false;
    if (mcts_rank < selected_mcts_rank &&
        candidate_mcts_policy < selected_mcts_policy && q_gap > -0.07f)
      return false;
    if (mcts_rank > 4 && q_gap > 0.060f && !low_visit_lever)
      return false;
    if (candidate_average_score < selected_average_score && mcts_rank <= 4 &&
        q_gap > 0.05f && candidate_mcts_policy <= selected_mcts_policy)
      return false;
  }

  if (!high_policy_lever) {
    const float top_q_gap = best_mcts_q - candidate_mcts_q;
    if (top_q_gap > (defensive_low_visit_lever ? 0.09f : 0.07f))
      return false;
    if (candidate_mcts_policy >= 0.20f && top_q_gap > 0.03f)
      return false;
  }

  return true;
}

bool HybridANERootPawnLeverCandidate(
    bool ane_root_probe, int selected_ane_rank, float selected_ane_score,
    int candidate_ane_rank, float candidate_ane_score,
    int selected_average_score, int candidate_average_score,
    uint64_t candidate_effort, int selected_mcts_rank, float selected_mcts_q,
    int candidate_mcts_rank, uint32_t candidate_mcts_current_visits,
    float candidate_mcts_q, float candidate_mcts_policy) {
  if (!ane_root_probe || selected_ane_rank <= 0 || candidate_ane_rank <= 0)
    return false;
  if (selected_mcts_rank != 1)
    return false;
  if (candidate_ane_rank > 2 || candidate_ane_rank >= selected_ane_rank)
    return false;
  if (candidate_ane_score - selected_ane_score < 0.015f)
    return false;
  if (candidate_mcts_rank <= 0 || candidate_mcts_rank > 6 ||
      candidate_mcts_current_visits < 2 || candidate_mcts_policy < 0.035f) {
    return false;
  }
  const bool defensive_position = selected_average_score <= -100;
  const uint64_t min_effort = defensive_position ? 75 : 150;
  const int max_average_gap = defensive_position ? 50 : 40;
  if (candidate_effort < min_effort)
    return false;
  if (selected_average_score - candidate_average_score > max_average_gap)
    return false;
  return selected_mcts_q - candidate_mcts_q <= 0.09f;
}

bool HybridIsPawnOnlyEndgame(const Position &pos) {
  return pos.non_pawn_material() == VALUE_ZERO;
}

bool HybridIsRookEndgame(const Position &pos) {
  return pos.count<ROOK>() > 0 && pos.count<QUEEN>() == 0 &&
         pos.count<BISHOP>() == 0 && pos.count<KNIGHT>() == 0;
}

bool HybridIsQuietCentralPawnPush(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != PAWN || !pos.empty(to))
    return false;

  const File file = file_of(from);
  if (file != FILE_D && file != FILE_E)
    return false;

  if (file != file_of(to))
    return false;

  const Color us = color_of(piece);
  const Direction push = pawn_push(us);
  if (to != from + push && to != from + push + push)
    return false;

  return relative_rank(us, to) >= RANK_4;
}

bool HybridIsQuietCentralQueenMove(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != QUEEN || !pos.empty(to))
    return false;

  if (rank_of(from) != rank_of(to))
    return false;

  const int file_delta = std::abs(int(file_of(to)) - int(file_of(from)));
  if (file_delta != 1)
    return false;

  const File to_file = file_of(to);
  if (to_file != FILE_D && to_file != FILE_E)
    return false;

  return relative_rank(color_of(piece), to) == RANK_3;
}

bool HybridIsQuietMinorMajorAttack(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || !pos.empty(to))
    return false;

  const PieceType piece_type = type_of(piece);
  if (piece_type != BISHOP && piece_type != KNIGHT)
    return false;

  const Color us = color_of(piece);
  const Bitboard occupied_after = (pos.pieces() ^ from) | to;
  return bool(attacks_bb(piece_type, to, occupied_after) &
              pos.pieces(~us, QUEEN, ROOK));
}

bool HybridIsBishopOnlyEndgame(const Position &pos) {
  return pos.count<BISHOP>() > 0 && pos.count<QUEEN>() == 0 &&
         pos.count<ROOK>() == 0 && pos.count<KNIGHT>() == 0;
}

bool HybridIsQuietBishopBackRankRetreat(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != BISHOP || !pos.empty(to))
    return false;

  const Color us = color_of(piece);
  return relative_rank(us, from) >= RANK_2 && relative_rank(us, to) == RANK_1;
}

bool HybridANEProbeAllowedForPosition(const Position &pos,
                                      bool only_pawn_endgames) {
  return !only_pawn_endgames || HybridIsPawnOnlyEndgame(pos);
}

bool HybridIsPawnLever(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != PAWN || !pos.empty(to))
    return false;

  if (file_of(from) != file_of(to))
    return false;

  const Color us = color_of(piece);
  const Direction push = pawn_push(us);
  if (to != from + push && to != from + push + push)
    return false;

  if (relative_rank(us, to) < RANK_4)
    return false;

  return bool(attacks_bb<PAWN>(to, us) & pos.pieces(~us, PAWN));
}

bool HybridIsPawnOnlyMCTSANECandidate(const Position &pos, Move selected,
                                      Move candidate) {
  if (!HybridIsPawnOnlyEndgame(pos) || selected == Move::none() ||
      candidate == Move::none() || selected == candidate ||
      selected.type_of() != NORMAL || candidate.type_of() != NORMAL) {
    return false;
  }

  const Piece candidate_piece = pos.piece_on(candidate.from_sq());
  if (candidate_piece == NO_PIECE)
    return false;

  if (HybridIsPawnLever(pos, candidate))
    return true;

  if (HybridIsPawnOnlyPawnCaptureCandidate(pos, selected, candidate))
    return true;

  return HybridIsPawnOnlyKingRecaptureCandidate(pos, selected, candidate);
}

bool HybridIsPawnOnlyPawnCaptureCandidate(const Position &pos, Move selected,
                                          Move candidate) {
  if (!HybridIsPawnOnlyEndgame(pos) || selected == Move::none() ||
      candidate == Move::none() || selected == candidate ||
      candidate.type_of() != NORMAL) {
    return false;
  }

  const Piece candidate_piece = pos.piece_on(candidate.from_sq());
  const Piece captured_piece = pos.piece_on(candidate.to_sq());
  if (candidate_piece == NO_PIECE || captured_piece == NO_PIECE ||
      type_of(candidate_piece) != PAWN || type_of(captured_piece) != PAWN ||
      color_of(candidate_piece) == color_of(captured_piece)) {
    return false;
  }

  return bool(attacks_bb<PAWN>(candidate.from_sq(), color_of(candidate_piece)) &
              candidate.to_sq());
}

bool HybridIsPawnOnlyKingRecaptureCandidate(const Position &pos, Move selected,
                                            Move candidate) {
  if (!HybridIsPawnOnlyEndgame(pos) || selected == Move::none() ||
      candidate == Move::none() || selected == candidate ||
      selected.type_of() != NORMAL || candidate.type_of() != NORMAL) {
    return false;
  }

  const Piece candidate_piece = pos.piece_on(candidate.from_sq());
  if (candidate_piece == NO_PIECE)
    return false;

  if (!pos.capture(selected) || !pos.capture(candidate) ||
      selected.to_sq() != candidate.to_sq()) {
    return false;
  }

  const Piece selected_piece = pos.piece_on(selected.from_sq());
  const Piece captured_piece = pos.piece_on(candidate.to_sq());
  return selected_piece != NO_PIECE && captured_piece != NO_PIECE &&
         type_of(selected_piece) == PAWN && type_of(candidate_piece) == KING &&
         type_of(captured_piece) == PAWN;
}

bool HybridPawnOnlyANEMCTSOverride(
    bool enabled, bool ane_agrees_mcts, bool fixed_budget,
    bool visit_evidence_sane, bool candidate_shape, bool king_recapture_shape,
    bool pawn_lever_shape, uint64_t mcts_root_visits, uint32_t mcts_best_visits,
    uint64_t mcts_current_root_visits, uint32_t mcts_current_best_visits,
    uint32_t ab_mcts_visits, float visit_share, float root_q_gap,
    float q_gap_to_ab, int mcts_cp, int eval_delta, int ab_average_score,
    int mcts_average_score, float ane_score_margin) {
  if (!enabled || !ane_agrees_mcts || !fixed_budget || !candidate_shape ||
      mcts_best_visits > mcts_root_visits ||
      mcts_current_best_visits > mcts_current_root_visits) {
    return false;
  }

  const bool cache_heavy_current_evidence =
      !visit_evidence_sane && !king_recapture_shape &&
      mcts_current_root_visits >= 1500 && mcts_current_best_visits >= 1500 &&
      visit_share >= 0.95f && root_q_gap >= 0.90f && q_gap_to_ab >= 0.90f &&
      mcts_cp >= 400 && eval_delta >= 400 && ane_score_margin >= 0.75f &&
      ab_mcts_visits <= 32;
  const bool cache_heavy_pawn_lever =
      !visit_evidence_sane && pawn_lever_shape && !king_recapture_shape &&
      mcts_current_root_visits >= 1500 && mcts_current_best_visits >= 1500 &&
      visit_share >= 0.95f && root_q_gap >= 0.25f && q_gap_to_ab >= 0.25f &&
      mcts_cp >= 500 && eval_delta >= 450 && ane_score_margin >= 0.30f &&
      ab_mcts_visits <= 80 && std::abs(ab_average_score) <= 200 &&
      std::abs(mcts_average_score) <= 200 &&
      std::abs(ab_average_score - mcts_average_score) <= 180;
  if (cache_heavy_pawn_lever)
    return true;
  if (!visit_evidence_sane && !cache_heavy_current_evidence)
    return false;

  const bool root_gap_ok = root_q_gap >= 0.25f || q_gap_to_ab >= 0.50f;
  if (mcts_root_visits < 40 || mcts_best_visits < 32 ||
      mcts_current_root_visits < 20 || mcts_current_best_visits < 20 ||
      visit_share < 0.75f || !root_gap_ok || q_gap_to_ab < 0.25f ||
      mcts_cp < 80 || eval_delta < 80 || ane_score_margin < 0.15f) {
    return false;
  }

  if (king_recapture_shape &&
      (visit_share < 0.90f || q_gap_to_ab < 0.60f || mcts_cp < 220 ||
       eval_delta < 220 || ane_score_margin < 0.30f)) {
    return false;
  }

  const int max_average_score = king_recapture_shape ? 170 : 100;
  const int max_average_gap = king_recapture_shape ? 90 : 40;
  if (std::abs(ab_average_score) > max_average_score ||
      std::abs(mcts_average_score) > max_average_score ||
      std::abs(ab_average_score - mcts_average_score) > max_average_gap) {
    return false;
  }

  return ab_mcts_visits <= std::max<uint32_t>(8, mcts_best_visits / 4);
}

bool HybridIsKingsidePawnLever(const Position &pos, Move move) {
  return HybridIsPawnLever(pos, move) && file_of(move.from_sq()) >= FILE_F;
}

bool HybridIsKingsidePawnPush(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != PAWN || !pos.empty(to))
    return false;

  const Color us = color_of(piece);
  const Direction push = pawn_push(us);
  return file_of(from) >= FILE_F && file_of(from) == file_of(to) &&
         (to == from + push || to == from + push + push);
}

bool HybridIsPawnDiscoveredKingAttack(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != PAWN || !pos.empty(to))
    return false;

  const Color us = color_of(piece);
  const Direction push = pawn_push(us);
  if (file_of(from) != file_of(to) ||
      (to != from + push && to != from + push + push)) {
    return false;
  }

  const Square enemy_king = pos.square<KING>(~us);
  const Bitboard before_occupied = pos.pieces();
  const Bitboard after_occupied = (before_occupied ^ from) | to;
  const Bitboard sliders = pos.pieces(us, ROOK, QUEEN);
  const Bitboard diagonals = pos.pieces(us, BISHOP, QUEEN);
  const bool attacked_before =
      bool(attacks_bb<ROOK>(enemy_king, before_occupied) & sliders) ||
      bool(attacks_bb<BISHOP>(enemy_king, before_occupied) & diagonals);
  const bool attacked_after =
      bool(attacks_bb<ROOK>(enemy_king, after_occupied) & sliders) ||
      bool(attacks_bb<BISHOP>(enemy_king, after_occupied) & diagonals);
  return !attacked_before && attacked_after;
}

bool HybridRootPawnLeverCanChallengeSelected(const Position &pos, Move selected,
                                             bool allow_non_pawn_selected) {
  if (selected == Move::none() || selected.type_of() == CASTLING)
    return false;

  const Piece selected_piece = pos.piece_on(selected.from_sq());
  if (!allow_non_pawn_selected &&
      (selected_piece == NO_PIECE || type_of(selected_piece) != PAWN)) {
    return false;
  }

  return !HybridIsPawnLever(pos, selected) &&
         !HybridIsKingsidePawnPush(pos, selected);
}

bool HybridHighPolicyRootLeverHint(const Position &pos, Move move, float policy,
                                   float leader_policy) {
  return policy >= 0.25f && policy >= leader_policy * 1.15f &&
         HybridIsKingsidePawnLever(pos, move);
}

float HybridVisitedRootQGap(float best_q, const uint32_t *candidate_visits,
                            const float *candidate_qs, int candidate_count) {
  if (!candidate_visits || !candidate_qs || candidate_count <= 0) {
    return 0.0f;
  }

  float next_best_q = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < candidate_count; ++i) {
    if (candidate_visits[i] == 0) {
      continue;
    }
    next_best_q = std::max(next_best_q, candidate_qs[i]);
  }

  return std::isfinite(next_best_q) ? best_q - next_best_q : 0.0f;
}

int HybridSubsearchJoinGraceMs(bool external_stop, int time_budget_ms) {
  if (external_stop)
    return 2500;
  if (time_budget_ms <= 0)
    return 1000;
  return std::clamp(time_budget_ms / 4, 1000, 5000);
}

bool ParallelHybridSearch::should_stop() const {
  if (stop_flag_.load(std::memory_order_acquire))
    return true;

  if (limits_.ponderMode &&
      !ponderhit_received_.load(std::memory_order_acquire))
    return false;

  if (limits_.nodes > 0) {
    uint64_t mcts_n =
        mcts_search_
            ? mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed)
            : 0;
    if (HybridUseMCTSPrimaryForFixedNodeBudget(limits_) && mcts_search_) {
      if (mcts_n >= limits_.nodes)
        return true;
      return false;
    }
    uint64_t total =
        mcts_n + ab_state_.nodes_searched.load(std::memory_order_relaxed);
    if (total >= limits_.nodes)
      return true;
  }

  const int time_budget_ms = time_budget_ms_.load(std::memory_order_acquire);
  if (time_budget_ms > 0) {
    const int64_t search_start_ms =
        search_start_ms_.load(std::memory_order_acquire);
    const int64_t elapsed_ms = SteadyNowMs() - search_start_ms;
    if (elapsed_ms >= time_budget_ms)
      return true;
  }

  return false;
}

void ParallelHybridSearch::mcts_thread_main() {
#ifdef __APPLE__
  set_thread_qos(QOS_CLASS_UTILITY);
#endif

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

  const int mcts_time_budget_ms =
      time_budget_ms_.load(std::memory_order_acquire);
  ::MetalFish::Search::LimitsType mcts_limits = HybridBuildMCTSLimits(
      limits_, mcts_time_budget_ms,
      limits_.ponderMode &&
          !ponderhit_received_.load(std::memory_order_acquire));
  mcts_limits.startTime = now();
  mcts_limits.searchmoves = limits_.searchmoves;

  // Prevent race: stop() may fire before StartSearch resets its own stop_flag.
  if (should_stop())
    return;

  Move best_move = Move::none();
  std::atomic<bool> mcts_done{false};

  auto mcts_callback = [&](Move move, Move ponder) {
    best_move = move;
    mcts_done = true;
  };

  if (shared_tt_reader_) {
    mcts_search_->SetSharedTT(config_.use_shared_tt ? shared_tt_reader_.get()
                                                    : nullptr);
  }

  {
    std::lock_guard<std::mutex> lock(mcts_start_mutex_);
    if (should_stop())
      return;
    mcts_search_->StartSearch(root_fen_, mcts_limits, mcts_callback, nullptr);
    mcts_search_started_.store(true, std::memory_order_release);
    if (limits_.ponderMode &&
        ponderhit_received_.load(std::memory_order_acquire)) {
      mcts_search_->PonderHit();
    }
  }

  if (should_stop()) {
    mcts_search_->Stop();
  }

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
  mcts_state_.best_current_visits.store(best_stats.current_visits,
                                        std::memory_order_relaxed);

  const auto root_moves = mcts_search_->GetRootMoveStats();
  uint64_t root_visits = 0;
  uint64_t root_current_visits = 0;
  for (const auto &move : root_moves) {
    root_visits += move.visits;
    root_current_visits += move.current_visits;
  }
  mcts_state_.total_nodes.store(root_visits, std::memory_order_relaxed);
  mcts_state_.total_current_nodes.store(root_current_visits,
                                        std::memory_order_relaxed);

  const int top_count = std::min<int>(static_cast<int>(root_moves.size()),
                                      MCTSSharedState::MAX_TOP_MOVES);
  for (int i = 0; i < top_count; ++i) {
    mcts_state_.top_moves[i].move_raw.store(root_moves[i].move.raw(),
                                            std::memory_order_relaxed);
    mcts_state_.top_moves[i].policy.store(root_moves[i].policy,
                                          std::memory_order_relaxed);
    mcts_state_.top_moves[i].visits.store(root_moves[i].visits,
                                          std::memory_order_relaxed);
    mcts_state_.top_moves[i].current_visits.store(root_moves[i].current_visits,
                                                  std::memory_order_relaxed);
    mcts_state_.top_moves[i].q.store(root_moves[i].q,
                                     std::memory_order_relaxed);
  }
  for (int i = top_count; i < MCTSSharedState::MAX_TOP_MOVES; ++i) {
    mcts_state_.top_moves[i].move_raw.store(0, std::memory_order_relaxed);
    mcts_state_.top_moves[i].policy.store(0.0f, std::memory_order_relaxed);
    mcts_state_.top_moves[i].visits.store(0, std::memory_order_relaxed);
    mcts_state_.top_moves[i].current_visits.store(0, std::memory_order_relaxed);
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

std::vector<Move> ParallelHybridSearch::collect_mcts_root_order_hints() {
  std::vector<Move> hints;
  if (!config_.mcts_ab_root_hints || !mcts_search_)
    return hints;
  if (limits_.ponderMode &&
      !ponderhit_received_.load(std::memory_order_acquire))
    return hints;

  int hint_count = std::clamp(config_.mcts_ab_root_hint_count, 1, 16);
  if (config_.ab_candidate_verify_ms > 0) {
    hint_count =
        std::max(hint_count, std::clamp(config_.ab_candidate_verify_count, 1,
                                        MCTSSharedState::MAX_TOP_MOVES));
  }
  const int delay_ms = std::max(0, config_.mcts_ab_root_hint_delay_ms);
  const int64_t deadline_ms = SteadyNowMs() + delay_ms;
  StateInfo root_state;
  Position root_pos;
  root_pos.set(root_fen_, false, &root_state);

  auto collect_latest_hints = [&]() {
    std::vector<Move> latest_hints;
    auto add_latest_hint = [&latest_hints, hint_count](Move move) {
      if (move == Move::none())
        return;
      if (static_cast<int>(latest_hints.size()) >= hint_count)
        return;
      if (std::find(latest_hints.begin(), latest_hints.end(), move) ==
          latest_hints.end()) {
        latest_hints.push_back(move);
      }
    };

    auto root_moves = mcts_search_->GetRootMoveStats();
    const float leader_policy =
        root_moves.empty() ? 0.0f : root_moves.front().policy;
    for (const auto &root_move : root_moves) {
      if (root_move.policy >= 0.08f &&
          root_move.policy >= leader_policy * 0.35f &&
          HybridIsQuietMinorMajorAttack(root_pos, root_move.move)) {
        add_latest_hint(root_move.move);
      }
    }
    for (const auto &root_move : root_moves) {
      if (HybridHighPolicyRootLeverHint(root_pos, root_move.move,
                                        root_move.policy, leader_policy)) {
        add_latest_hint(root_move.move);
      }
    }
    const int visit_hints = std::max(1, hint_count / 2);
    for (int i = 0; i < static_cast<int>(root_moves.size()) &&
                    static_cast<int>(latest_hints.size()) < visit_hints;
         ++i) {
      add_latest_hint(root_moves[i].move);
    }

    std::stable_sort(
        root_moves.begin(), root_moves.end(),
        [](const Search::RootMoveStats &a, const Search::RootMoveStats &b) {
          if (std::abs(a.policy - b.policy) > 0.000001f)
            return a.policy > b.policy;
          if (a.visits != b.visits)
            return a.visits > b.visits;
          return a.q > b.q;
        });
    for (const auto &root_move : root_moves)
      add_latest_hint(root_move.move);

    return latest_hints;
  };

  while (!should_stop()) {
    auto latest_hints = collect_latest_hints();
    if (!latest_hints.empty())
      hints = std::move(latest_hints);
    if (delay_ms == 0 || SteadyNowMs() >= deadline_ms)
      break;
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }

  if (hints.empty()) {
    hints = collect_latest_hints();
  }

  if (config_.trace_decisions && !hints.empty()) {
    std::ostringstream ss;
    ss << "Hybrid: AB root hints from MCTS";
    for (Move hint : hints)
      ss << " " << UCIEngine::move(hint, false);
    send_info_string(ss.str());
  }
  return hints;
}

void ParallelHybridSearch::start_ane_root_probe(bool reuse_existing) {
  auto trace_skip = [&](const std::string &reason) {
    if (config_.trace_decisions)
      send_info_string("Hybrid: ANE root probe skipped: " + reason);
  };
  auto consume_existing_future = [&]() -> std::vector<Move> {
    if (!ane_root_hints_future_.valid())
      return {};
    try {
      return ane_root_hints_future_.get();
    } catch (const std::exception &e) {
      if (config_.trace_decisions) {
        send_info_string(std::string("Hybrid: ANE root probe failed: ") +
                         e.what());
      }
    } catch (...) {
      if (config_.trace_decisions)
        send_info_string("Hybrid: ANE root probe failed: unknown error");
    }
    return {};
  };

  if (reuse_existing) {
    std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
    if (!ane_root_hints_.empty()) {
      trace_skip("already ready");
      return;
    }
  }

  if (ane_root_hints_future_.valid()) {
    if (reuse_existing) {
      const auto status =
          ane_root_hints_future_.wait_for(std::chrono::milliseconds(0));
      if (status != std::future_status::ready) {
        trace_skip("already running");
        return;
      }
      {
        std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
        if (!ane_root_hints_.empty()) {
          trace_skip("already ready");
          return;
        }
      }
    }
    std::vector<Move> existing_hints = consume_existing_future();
    if (reuse_existing && !existing_hints.empty()) {
      std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
      ane_root_hints_ = std::move(existing_hints);
      trace_skip("already ready");
      return;
    }
    if (!reuse_existing) {
      std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
      ane_root_hints_.clear();
      ane_root_hint_infos_.clear();
    }
  }

  if (!config_.ane_root_probe)
    return;
  if (!ane_evaluator_) {
    trace_skip("no evaluator");
    return;
  }
  const int min_budget = std::max(0, config_.ane_min_budget_ms);
  const int time_budget = time_budget_ms_.load(std::memory_order_acquire);
  if (min_budget > 0 && time_budget > 0 && time_budget < min_budget) {
    trace_skip("budget " + std::to_string(time_budget) + "ms < minimum " +
               std::to_string(min_budget) + "ms");
    return;
  }
  if (min_budget > 0 && time_budget == 0 &&
      (limits_.nodes > 0 || limits_.infinite || limits_.depth > 0 ||
       limits_.mate > 0)) {
    trace_skip("fixed search budget");
    return;
  }
  if (config_.ane_only_pawn_endgames) {
    StateInfo root_state;
    Position root_pos;
    root_pos.set(root_fen_, false, &root_state);
    if (!HybridANEProbeAllowedForPosition(root_pos, true)) {
      trace_skip("not a pawn-only endgame");
      return;
    }
  }

  if (config_.trace_decisions)
    send_info_string("Hybrid: ANE root probe started");
  ane_root_hints_future_ =
      std::async(std::launch::async,
                 &ParallelHybridSearch::compute_ane_root_order_hints, this);
}

std::vector<Move> ParallelHybridSearch::compute_ane_root_order_hints() {
  std::vector<Move> hints;
  const int64_t start_ms = SteadyNowMs();
  if (!ane_evaluator_)
    return hints;
  if (should_stop()) {
    if (config_.trace_decisions)
      send_info_string("Hybrid: ANE root probe aborted before evaluation");
    return hints;
  }

  try {
    StateInfo root_state;
    Position root_pos;
    root_pos.set(root_fen_, false, &root_state);

    MoveList<LEGAL> legal_moves(root_pos);
    std::vector<Move> candidate_moves;
    candidate_moves.reserve(legal_moves.size());
    if (!limits_.searchmoves.empty()) {
      for (const auto &uci_move : limits_.searchmoves) {
        Move move = UCIEngine::to_move(root_pos, uci_move);
        if (move != Move::none() && legal_moves.contains(move) &&
            std::find(candidate_moves.begin(), candidate_moves.end(), move) ==
                candidate_moves.end()) {
          candidate_moves.push_back(move);
        }
      }
    } else {
      for (Move move : legal_moves)
        candidate_moves.push_back(move);
    }
    if (candidate_moves.empty()) {
      if (config_.trace_decisions)
        send_info_string("Hybrid: ANE root probe found no legal candidates");
      return hints;
    }

    struct ChildInput {
      Move move = Move::none();
      StateInfo base_state;
      StateInfo child_state;
      Position pos;
      float score = 0.0f;
      bool scored = false;
    };

    std::vector<std::unique_ptr<ChildInput>> children;
    children.reserve(candidate_moves.size());
    std::vector<const Position *> pending_positions;
    pending_positions.reserve(candidate_moves.size());
    std::vector<size_t> pending_indices;
    pending_indices.reserve(candidate_moves.size());

    for (Move move : candidate_moves) {
      auto child = std::make_unique<ChildInput>();
      child->move = move;
      child->pos.copy_from(root_pos, &child->base_state);
      child->pos.do_move(move, child->child_state, nullptr);

      MoveList<LEGAL> child_moves(child->pos);
      if (child_moves.size() == 0) {
        child->score = child->pos.checkers() ? 1.0f : 0.0f;
        child->scored = true;
      } else {
        pending_positions.push_back(&child->pos);
        pending_indices.push_back(children.size());
      }
      children.push_back(std::move(child));
    }

    size_t result_count = 0;
    if (!pending_positions.empty()) {
      constexpr size_t kANERootProbeChunkSize = 8;
      for (size_t offset = 0; offset < pending_positions.size();
           offset += kANERootProbeChunkSize) {
        if (should_stop()) {
          if (config_.trace_decisions)
            send_info_string(
                "Hybrid: ANE root probe aborted before Core ML batch");
          break;
        }
        const size_t chunk_size =
            std::min(kANERootProbeChunkSize, pending_positions.size() - offset);
        auto results = ane_evaluator_->EvaluateBatch(
            pending_positions.data() + offset, chunk_size);
        const size_t chunk_count = std::min(results.size(), chunk_size);
        for (size_t i = 0; i < chunk_count; ++i) {
          auto &child = *children[pending_indices[offset + i]];
          const float root_score = -results[i].value;
          child.score = std::isfinite(root_score) ? root_score : 0.0f;
          child.scored = true;
        }
        result_count += chunk_count;
        if (chunk_count < chunk_size)
          break;
      }
    }

    std::vector<size_t> order;
    order.reserve(children.size());
    for (size_t i = 0; i < children.size(); ++i) {
      if (children[i]->scored)
        order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
      return children[lhs]->score > children[rhs]->score;
    });

    const int hint_count = std::clamp(config_.ane_root_hint_count, 1, 32);
    hints.reserve(std::min<int>(hint_count, static_cast<int>(order.size())));
    std::vector<ANERootHintInfo> hint_infos;
    hint_infos.reserve(
        std::min<int>(hint_count, static_cast<int>(order.size())));
    for (size_t idx : order) {
      if (should_stop())
        break;
      hints.push_back(children[idx]->move);
      hint_infos.push_back({children[idx]->move, children[idx]->score});
      if (static_cast<int>(hints.size()) >= hint_count)
        break;
    }
    {
      std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
      ane_root_hints_ = hints;
      ane_root_hint_infos_ = hint_infos;
    }
    if (config_.trace_decisions) {
      std::ostringstream ss;
      ss << "Hybrid: ANE root probe completed " << hints.size() << "/"
         << candidate_moves.size() << " moves";
      if (!hints.empty()) {
        ss << " top " << UCIEngine::move(hints.front(), false) << " score "
           << std::fixed << std::setprecision(3) << hint_infos.front().score;
      }
      ss << " evals " << result_count << " time "
         << std::max<int64_t>(0, SteadyNowMs() - start_ms) << "ms";
      send_info_string(ss.str());
    }
  } catch (const std::exception &e) {
    {
      std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
      ane_root_hints_.clear();
      ane_root_hint_infos_.clear();
    }
    if (config_.trace_decisions) {
      send_info_string(std::string("Hybrid: ANE root probe failed: ") +
                       e.what());
    }
    return {};
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
      ane_root_hints_.clear();
      ane_root_hint_infos_.clear();
    }
    if (config_.trace_decisions)
      send_info_string("Hybrid: ANE root probe failed: unknown error");
    return {};
  }

  return hints;
}

std::vector<Move> ParallelHybridSearch::collect_ane_root_order_hints() {
  std::vector<Move> hints;
  if (!config_.ane_root_probe || !ane_root_hints_future_.valid())
    return hints;

  const int wait_ms = std::max(0, config_.ane_root_hint_wait_ms);
  const auto status =
      wait_ms == 0
          ? ane_root_hints_future_.wait_for(std::chrono::milliseconds(0))
          : ane_root_hints_future_.wait_for(std::chrono::milliseconds(wait_ms));
  if (status != std::future_status::ready)
    return hints;

  hints = ane_root_hints_future_.get();
  {
    std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
    ane_root_hints_ = hints;
    if (ane_root_hint_infos_.size() != ane_root_hints_.size()) {
      ane_root_hint_infos_.clear();
      ane_root_hint_infos_.reserve(ane_root_hints_.size());
      for (Move move : ane_root_hints_)
        ane_root_hint_infos_.push_back({move, 0.0f});
    }
  }
  if (config_.trace_decisions && !hints.empty()) {
    std::ostringstream ss;
    ss << (config_.ane_root_hints ? "Hybrid: AB root hints from ANE"
                                  : "Hybrid: ANE root probe ready");
    for (Move hint : hints)
      ss << " " << UCIEngine::move(hint, false);
    send_info_string(ss.str());
  }
  return hints;
}

std::vector<Move> ParallelHybridSearch::collect_root_order_hints() {
  std::vector<Move> hints;
  auto add_hint = [&hints](Move move) {
    if (move == Move::none())
      return;
    if (std::find(hints.begin(), hints.end(), move) == hints.end())
      hints.push_back(move);
  };

  for (Move move : collect_mcts_root_order_hints())
    add_hint(move);
  if (config_.ane_root_hints) {
    for (Move move : collect_ane_root_order_hints())
      add_hint(move);
  }

  const int max_hints =
      std::clamp(config_.mcts_ab_root_hint_count, 1, 16) +
      (config_.ane_root_hints ? std::clamp(config_.ane_root_hint_count, 1, 32)
                              : 0);
  if (static_cast<int>(hints.size()) > max_hints)
    hints.resize(max_hints);
  return hints;
}

std::vector<Move> ParallelHybridSearch::verify_ab_root_candidates(
    const std::vector<Move> &candidates, int verify_ms) {
  std::vector<Move> verified_order;
  if (!engine_ || candidates.empty() || verify_ms <= 0)
    return verified_order;

  const int candidate_count = std::clamp(config_.ab_candidate_verify_count, 1,
                                         MCTSSharedState::MAX_TOP_MOVES);
  std::vector<Move> limited_candidates;
  limited_candidates.reserve(
      std::min<int>(candidate_count, static_cast<int>(candidates.size())));
  for (Move candidate : candidates) {
    if (candidate == Move::none())
      continue;
    if (std::find(limited_candidates.begin(), limited_candidates.end(),
                  candidate) != limited_candidates.end())
      continue;
    limited_candidates.push_back(candidate);
    if (static_cast<int>(limited_candidates.size()) >= candidate_count)
      break;
  }
  if (limited_candidates.empty())
    return verified_order;

  ::MetalFish::Search::LimitsType verify_limits;
  verify_limits.movetime = verify_ms;
  verify_limits.startTime = now();
  verify_limits.ponderMode = false;
  verify_limits.root_order_hints = limited_candidates;
  verify_limits.searchmoves.reserve(limited_candidates.size());
  for (Move candidate : limited_candidates) {
    verify_limits.searchmoves.push_back(UCIEngine::move(candidate, false));
  }

  auto saved_bestmove = engine_->get_on_bestmove();
  auto saved_update_full = engine_->get_on_update_full();
  engine_->set_on_update_full([](const Engine::InfoFull &) {});
  engine_->set_on_bestmove([](std::string_view, std::string_view) {});

  bool verify_started = false;
  {
    std::lock_guard<std::mutex> lock(ab_start_mutex_);
    if (!should_stop()) {
      engine_->go(verify_limits);
      ab_search_started_.store(true, std::memory_order_release);
      verify_started = true;
    }
  }

  std::vector<Engine::RootMoveSnapshot> verify_snapshot;
  if (verify_started) {
    if (should_stop()) {
      engine_->stop();
    }
    engine_->wait_for_search_finished();

    verify_snapshot = engine_->root_move_snapshot(candidate_count);
    for (const auto &root_move : verify_snapshot) {
      if (root_move.move == Move::none())
        continue;
      if (std::find(limited_candidates.begin(), limited_candidates.end(),
                    root_move.move) == limited_candidates.end())
        continue;
      if (std::find(verified_order.begin(), verified_order.end(),
                    root_move.move) == verified_order.end()) {
        verified_order.push_back(root_move.move);
      }
    }
  }

  engine_->set_on_update_full(std::move(saved_update_full));
  engine_->set_on_bestmove(std::move(saved_bestmove));

  if (verify_started && !limited_candidates.empty() &&
      !verified_order.empty() &&
      verified_order.front() != limited_candidates.front() && mcts_search_) {
    const Move leading_hint = limited_candidates.front();
    const Move verified_top = verified_order.front();
    StateInfo root_state;
    Position root_pos;
    root_pos.set(root_fen_, false, &root_state);

    auto root_moves = mcts_search_->GetRootMoveStats();
    const auto find_mcts_root = [&root_moves](Move target) {
      Search::RootMoveStats result;
      for (const auto &root_move : root_moves) {
        if (root_move.move == target) {
          result = root_move;
          break;
        }
      }
      return result;
    };
    const auto find_verify_root = [&verify_snapshot](Move target) {
      Engine::RootMoveSnapshot result;
      for (const auto &root_move : verify_snapshot) {
        if (root_move.move == target) {
          result = root_move;
          break;
        }
      }
      return result;
    };

    const auto leading_mcts = find_mcts_root(leading_hint);
    const auto displaced_mcts = find_mcts_root(verified_top);
    uint64_t current_root_visits = 0;
    for (const auto &root_move : root_moves)
      current_root_visits += root_move.current_visits;
    const float current_visit_share =
        current_root_visits > 0
            ? static_cast<float>(leading_mcts.current_visits) /
                  static_cast<float>(current_root_visits)
            : 0.0f;

    const auto leading_verify = find_verify_root(leading_hint);
    const auto displaced_verify = find_verify_root(verified_top);
    const bool fixed_budget = HybridHasMCTSDecisionBudget(
        limits_, time_budget_ms_.load(std::memory_order_acquire),
        ponderhit_received_.load(std::memory_order_acquire));
    const bool visit_evidence_sane = HybridMCTSVisitEvidenceSane(
        mcts_search_->Stats().total_nodes.load(std::memory_order_relaxed),
        mcts_search_->Stats().nn_evaluations.load(std::memory_order_relaxed),
        current_root_visits, leading_mcts.current_visits);

    if (leading_mcts.move == leading_hint &&
        displaced_mcts.move == verified_top &&
        HybridPreserveLeadingHintAfterABVerify(
            fixed_budget, visit_evidence_sane,
            HybridIsKingsidePawnPush(root_pos, leading_hint) &&
                HybridIsPawnDiscoveredKingAttack(root_pos, leading_hint),
            current_root_visits, leading_mcts.current_visits,
            current_visit_share, leading_mcts.q, displaced_mcts.q,
            leading_mcts.policy, displaced_mcts.policy,
            displaced_verify.average_score, leading_verify.average_score,
            displaced_verify.score, displaced_verify.score_lowerbound,
            leading_verify.score, leading_verify.score_lowerbound,
            leading_verify.score_upperbound, leading_verify.effort)) {
      verified_order.erase(std::remove(verified_order.begin(),
                                       verified_order.end(), leading_hint),
                           verified_order.end());
      verified_order.insert(verified_order.begin(), leading_hint);
      if (config_.trace_decisions) {
        send_info_string("Hybrid: preserving leading MCTS root hint after "
                         "shallow AB verify " +
                         UCIEngine::move(leading_hint, false) + " over " +
                         UCIEngine::move(verified_top, false));
      }
    }
  }

  for (Move candidate : limited_candidates) {
    if (std::find(verified_order.begin(), verified_order.end(), candidate) ==
        verified_order.end()) {
      verified_order.push_back(candidate);
    }
  }

  if (config_.trace_decisions && verify_started && !verified_order.empty()) {
    std::ostringstream ss;
    ss << "Hybrid: AB candidate verify " << verify_ms << "ms";
    for (Move move : verified_order)
      ss << " " << UCIEngine::move(move, false);
    send_info_string(ss.str());
  }

  return verified_order;
}

void ParallelHybridSearch::ab_thread_main() {
#ifdef __APPLE__
  set_thread_qos(QOS_CLASS_USER_INITIATED);
#endif

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

  const int original_threads =
      static_cast<int>(engine_->get_options()["Threads"]);
  const int requested_ab_threads = std::max(1, config_.ab_threads);
  const bool needs_thread_resize = (requested_ab_threads != original_threads);
  const int original_multipv =
      static_cast<int>(engine_->get_options()["MultiPV"]);
  const bool needs_multipv_reset = original_multipv != 1;

  auto set_engine_option = [this](const std::string &name, int value) {
    std::istringstream ss("name " + name + " value " + std::to_string(value));
    engine_->get_options().setoption(ss);
  };

  if (needs_thread_resize) {
    set_engine_option("Threads", requested_ab_threads);
  }
  if (needs_multipv_reset) {
    set_engine_option("MultiPV", 1);
  }

  engine_->set_position(root_fen_, {});

  ::MetalFish::Search::LimitsType ab_limits = limits_;
  ab_limits.startTime = now();
  if (HybridUseMCTSPrimaryForFixedNodeBudget(limits_)) {
    ab_limits.nodes = HybridLowNodeABProbeNodes(limits_.nodes);
  }
  if (limits_.ponderMode &&
      ponderhit_received_.load(std::memory_order_acquire)) {
    ab_limits.ponderMode = false;
  }
  ab_limits.root_order_hints = collect_root_order_hints();
  {
    std::lock_guard<std::mutex> lock(ab_root_mutex_);
    ab_root_order_hints_ = ab_limits.root_order_hints;
    ab_verified_root_order_hints_.clear();
  }
  const int candidate_verify_ms = HybridABCandidateVerifyBudgetMs(
      limits_, time_budget_ms_.load(std::memory_order_acquire),
      config_.ab_candidate_verify_ms,
      limits_.ponderMode &&
          !ponderhit_received_.load(std::memory_order_acquire));
  if (candidate_verify_ms > 0 && !ab_limits.root_order_hints.empty()) {
    std::vector<Move> verified_order = verify_ab_root_candidates(
        ab_limits.root_order_hints, candidate_verify_ms);
    if (!verified_order.empty()) {
      for (Move hint : ab_limits.root_order_hints) {
        if (std::find(verified_order.begin(), verified_order.end(), hint) ==
            verified_order.end()) {
          verified_order.push_back(hint);
        }
      }
      ab_limits.root_order_hints = std::move(verified_order);
    }
    {
      std::lock_guard<std::mutex> lock(ab_root_mutex_);
      ab_verified_root_order_hints_ = ab_limits.root_order_hints;
    }
    ab_limits.startTime = now();
  }

  auto saved_bestmove = engine_->get_on_bestmove();
  auto saved_update_full = engine_->get_on_update_full();
  Move ab_best_move = Move::none();
  int ab_score = 0;
  int ab_depth = 0;

  engine_->set_on_update_full([this, &ab_score,
                               &ab_depth](const Engine::InfoFull &info) {
    if (info.multiPV != 1)
      return;

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

  bool ab_started = false;
  {
    std::lock_guard<std::mutex> lock(ab_start_mutex_);
    if (!should_stop()) {
      if (limits_.ponderMode &&
          ponderhit_received_.load(std::memory_order_acquire)) {
        ab_limits.ponderMode = false;
      }
      engine_->go(ab_limits);
      ab_search_started_.store(true, std::memory_order_release);
      ab_started = true;
    }
  }
  if (ab_started) {
    if (should_stop()) {
      engine_->stop();
    }
    if (limits_.ponderMode &&
        ponderhit_received_.load(std::memory_order_acquire)) {
      engine_->set_ponderhit(false);
    }
    engine_->wait_for_search_finished();

    stats_.ab_nodes.store(engine_->threads_nodes_searched(),
                          std::memory_order_relaxed);
    stats_.ab_depth.store(ab_depth, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(ab_root_mutex_);
      ab_root_moves_.clear();
      for (const auto &rm :
           engine_->root_move_snapshot(ABSharedState::MAX_PV)) {
        ABRootMoveInfo item;
        item.move = rm.move;
        item.score = rm.score;
        item.previous_score = rm.previous_score;
        item.average_score = rm.average_score;
        item.score_lowerbound = rm.score_lowerbound;
        item.score_upperbound = rm.score_upperbound;
        item.effort = rm.effort;
        ab_root_moves_.push_back(item);
      }
    }
  }

  engine_->set_on_update_full(std::move(saved_update_full));
  engine_->set_on_bestmove(std::move(saved_bestmove));

  if (needs_multipv_reset) {
    set_engine_option("MultiPV", original_multipv);
  }
  if (needs_thread_resize) {
    set_engine_option("Threads", original_threads);
  }
}

void ParallelHybridSearch::publish_ab_state(Move best, int score, int depth,
                                            uint64_t nodes) {
  ab_state_.set_best_move(best, score, depth, nodes);
}

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

  while (!should_stop()) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    if (limits_.ponderMode &&
        !ponderhit_received_.load(std::memory_order_acquire)) {
      continue;
    }

    bool ab_done = !ab_state_.ab_running.load(std::memory_order_acquire);
    bool mcts_done = !mcts_state_.mcts_running.load(std::memory_order_acquire);

    if (ab_done && ab_state_.has_result.load(std::memory_order_acquire)) {
      if (!HybridShouldContinueMCTSAfterAB(limits_) || mcts_done) {
        break;
      }
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

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
        const bool allow_agreement_stop =
            HybridCanStopEarlyOnAgreement(limits_);
        const int time_budget_ms =
            time_budget_ms_.load(std::memory_order_acquire);
        const int min_time = (time_budget_ms > 0) ? time_budget_ms / 4 : 500;
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

        if (allow_agreement_stop && agreement_count >= 3 && ms > min_time &&
            mcts_nodes >= min_mcts_nodes && ab_depth >= min_ab_depth) {
          send_info_string("Hybrid: engines agree, stopping early at " +
                           std::to_string(ms) + "ms");
          break;
        }
      } else {
        agreement_count = 0;
      }
    }

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
  if (!external_stop && config_.ane_root_probe)
    collect_ane_root_order_hints();
  stop_flag_.store(true, std::memory_order_release);

  if (mcts_search_)
    mcts_search_->Stop();
  if (engine_)
    engine_->stop();

  const int join_grace_ms = HybridSubsearchJoinGraceMs(
      external_stop, time_budget_ms_.load(std::memory_order_acquire));
  const auto join_deadline = std::chrono::steady_clock::now() +
                             std::chrono::milliseconds(join_grace_ms);
  bool join_timed_out = false;
  while (!mcts_thread_done_.load(std::memory_order_acquire) ||
         !ab_thread_done_.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= join_deadline) {
      join_timed_out = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }

  if (join_timed_out) {
    send_info_string("Hybrid: subsearch stop grace expired; returning best "
                     "available move");
  }

  if (mcts_search_) {
    publish_mcts_state();
  }

  Move final_move = make_final_decision();

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
      "Final: MCTSPlayouts=" + std::to_string(stats_.mcts_nodes.load()) +
      " MCTSBest=" +
      std::to_string(mcts_state_.best_visits.load(std::memory_order_relaxed)) +
      " MCTSBestCurrent=" +
      std::to_string(
          mcts_state_.best_current_visits.load(std::memory_order_relaxed)) +
      " MCTSRootVisits=" +
      std::to_string(mcts_state_.total_nodes.load(std::memory_order_relaxed)) +
      " MCTSRootCurrentVisits=" +
      std::to_string(
          mcts_state_.total_current_nodes.load(std::memory_order_relaxed)) +
      " MCTSEvals=" +
      std::to_string(
          stats_.transformer_evaluations.load(std::memory_order_relaxed)) +
      " MCTSBatches=" +
      std::to_string(
          stats_.transformer_batches.load(std::memory_order_relaxed)) +
      " MCTSTimeMs=" + std::to_string(static_cast<int>(stats_.mcts_time_ms)) +
      " AB=" + std::to_string(stats_.ab_nodes.load()) +
      " ABDepth=" + std::to_string(stats_.ab_depth.load()) +
      " ABTimeMs=" + std::to_string(static_cast<int>(stats_.ab_time_ms)) +
      " TotalTimeMs=" + std::to_string(static_cast<int>(stats_.total_time_ms)) +
      " ABMove=" + UCIEngine::move(ab_best, false) +
      " MCTSMove=" + UCIEngine::move(mcts_best, false) +
      " agreements=" + std::to_string(stats_.move_agreements.load()) +
      " ab_overrides=" + std::to_string(stats_.ab_overrides.load()) +
      " mcts_overrides=" + std::to_string(stats_.mcts_overrides.load()) +
      " policy_updates=" + std::to_string(stats_.policy_updates.load()));

  invoke_best_move_callback(final_move, ponder_move);

  if (join_timed_out) {
    while (!mcts_thread_done_.load(std::memory_order_acquire) ||
           !ab_thread_done_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }
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

  if (!pv.empty()) {
    Position pv_pos;
    StateInfo root_state;
    pv_pos.set(root_fen_, false, &root_state);

    std::vector<StateInfo> states(pv.size());
    std::vector<Move> legal_pv;
    legal_pv.reserve(pv.size());

    for (size_t i = 0; i < pv.size(); ++i) {
      Move move = pv[i];
      if (move == Move::none())
        break;

      MoveList<LEGAL> legal_moves(pv_pos);
      if (!legal_moves.contains(move)) {
        if (config_.trace_decisions) {
          send_info_string("Hybrid: final PV truncated before illegal move " +
                           UCIEngine::move(move, false));
        }
        break;
      }

      legal_pv.push_back(move);
      pv_pos.do_move(move, states[i], nullptr);
    }

    pv = std::move(legal_pv);
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
  const uint32_t mcts_current_visits =
      mcts_state_.best_current_visits.load(std::memory_order_relaxed);
  const uint64_t mcts_total_nodes =
      mcts_state_.total_nodes.load(std::memory_order_relaxed);
  const uint64_t mcts_total_current_nodes =
      mcts_state_.total_current_nodes.load(std::memory_order_relaxed);
  const uint64_t mcts_playouts =
      stats_.mcts_nodes.load(std::memory_order_relaxed);
  const uint64_t mcts_evals =
      stats_.transformer_evaluations.load(std::memory_order_relaxed);

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
    const int count =
        std::min<int>(mcts_state_.num_top_moves.load(std::memory_order_acquire),
                      MCTSSharedState::MAX_TOP_MOVES);
    ss << "[";
    for (int i = 0; i < count; ++i) {
      if (i > 0)
        ss << ",";
      Move move(
          mcts_state_.top_moves[i].move_raw.load(std::memory_order_relaxed));
      ss << move_to_string(move) << ":n="
         << mcts_state_.top_moves[i].visits.load(std::memory_order_relaxed)
         << ":cn="
         << mcts_state_.top_moves[i].current_visits.load(
                std::memory_order_relaxed)
         << ":q=" << std::fixed << std::setprecision(3)
         << mcts_state_.top_moves[i].q.load(std::memory_order_relaxed)
         << ":p=" << std::fixed << std::setprecision(3)
         << mcts_state_.top_moves[i].policy.load(std::memory_order_relaxed);
    }
    ss << "]";
    return ss.str();
  };
  const auto ab_root_moves_to_string = [&]() {
    std::vector<ABRootMoveInfo> root_moves;
    {
      std::lock_guard<std::mutex> lock(ab_root_mutex_);
      root_moves = ab_root_moves_;
    }

    std::ostringstream ss;
    ss << "[";
    const int count = std::min<int>(static_cast<int>(root_moves.size()),
                                    ABSharedState::MAX_PV);
    for (int i = 0; i < count; ++i) {
      if (i > 0)
        ss << ",";
      ss << move_to_string(root_moves[i].move) << ":s=" << root_moves[i].score
         << ":ps=" << root_moves[i].previous_score
         << ":avg=" << root_moves[i].average_score
         << ":lb=" << (root_moves[i].score_lowerbound ? 1 : 0)
         << ":ub=" << (root_moves[i].score_upperbound ? 1 : 0)
         << ":eff=" << root_moves[i].effort;
    }
    ss << "]";
    return ss.str();
  };
  const auto move_list_to_string = [&](const std::vector<Move> &moves) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < moves.size(); ++i) {
      if (i > 0)
        ss << ",";
      ss << move_to_string(moves[i]);
    }
    ss << "]";
    return ss.str();
  };
  std::vector<Move> ab_root_order_hints_snapshot;
  std::vector<Move> ab_verified_root_order_hints_snapshot;
  {
    std::lock_guard<std::mutex> lock(ab_root_mutex_);
    ab_root_order_hints_snapshot = ab_root_order_hints_;
    ab_verified_root_order_hints_snapshot = ab_verified_root_order_hints_;
  }
  struct MCTSRootLookup {
    int rank = -1;
    uint32_t visits = 0;
    uint32_t current_visits = 0;
    float q = 0.0f;
    float policy = 0.0f;
  };
  const auto find_mcts_root_move = [&](Move target) {
    MCTSRootLookup result;
    if (target == Move::none())
      return result;

    const int count =
        std::min<int>(mcts_state_.num_top_moves.load(std::memory_order_acquire),
                      MCTSSharedState::MAX_TOP_MOVES);
    for (int i = 0; i < count; ++i) {
      Move move(
          mcts_state_.top_moves[i].move_raw.load(std::memory_order_relaxed));
      if (move != target)
        continue;
      result.rank = i + 1;
      result.visits =
          mcts_state_.top_moves[i].visits.load(std::memory_order_relaxed);
      result.current_visits = mcts_state_.top_moves[i].current_visits.load(
          std::memory_order_relaxed);
      result.q = mcts_state_.top_moves[i].q.load(std::memory_order_relaxed);
      result.policy =
          mcts_state_.top_moves[i].policy.load(std::memory_order_relaxed);
      break;
    }
    return result;
  };
  struct ABRootLookup {
    int rank = -1;
    int score = 0;
    int average_score = 0;
    uint64_t effort = 0;
    bool score_lowerbound = false;
    bool score_upperbound = false;
  };
  const auto find_ab_root_move = [&](Move target) {
    ABRootLookup result;
    if (target == Move::none())
      return result;

    std::lock_guard<std::mutex> lock(ab_root_mutex_);
    const int count = std::min<int>(static_cast<int>(ab_root_moves_.size()),
                                    ABSharedState::MAX_PV);
    for (int i = 0; i < count; ++i) {
      if (ab_root_moves_[i].move != target)
        continue;
      result.rank = i + 1;
      result.score = ab_root_moves_[i].score;
      result.average_score = ab_root_moves_[i].average_score;
      result.effort = ab_root_moves_[i].effort;
      result.score_lowerbound = ab_root_moves_[i].score_lowerbound;
      result.score_upperbound = ab_root_moves_[i].score_upperbound;
      break;
    }
    return result;
  };
  std::vector<Move> ane_root_hints_snapshot;
  std::vector<ANERootHintInfo> ane_root_hint_infos_snapshot;
  {
    std::lock_guard<std::mutex> lock(ane_root_hints_mutex_);
    ane_root_hints_snapshot = ane_root_hints_;
    ane_root_hint_infos_snapshot = ane_root_hint_infos_;
  }
  const Move ane_top = ane_root_hints_snapshot.empty()
                           ? Move::none()
                           : ane_root_hints_snapshot.front();
  const bool ane_agrees_mcts = config_.ane_root_probe && ane_top == mcts_best &&
                               mcts_best != Move::none();
  const float ane_top_score = ane_root_hint_infos_snapshot.empty()
                                  ? 0.0f
                                  : ane_root_hint_infos_snapshot[0].score;
  const float ane_second_score = ane_root_hint_infos_snapshot.size() < 2
                                     ? ane_top_score
                                     : ane_root_hint_infos_snapshot[1].score;
  const float ane_score_margin = ane_root_hint_infos_snapshot.empty()
                                     ? 0.0f
                                     : ane_top_score - ane_second_score;
  bool ane_confirmed_mcts_override = false;
  const auto ane_root_hints_to_string = [&]() {
    std::ostringstream ss;
    ss << "[";
    const bool has_scores =
        ane_root_hint_infos_snapshot.size() == ane_root_hints_snapshot.size();
    for (size_t i = 0; i < ane_root_hints_snapshot.size(); ++i) {
      if (i > 0)
        ss << ",";
      ss << move_to_string(ane_root_hints_snapshot[i]);
      if (has_scores) {
        ss << ":v=" << std::fixed << std::setprecision(3)
           << ane_root_hint_infos_snapshot[i].score;
      }
    }
    ss << "]";
    return ss.str();
  };
  struct ANERootLookup {
    int rank = -1;
    float score = 0.0f;
  };
  const auto find_ane_root_move = [&](Move target) {
    ANERootLookup result;
    if (!config_.ane_root_probe || target == Move::none())
      return result;

    const bool has_scores =
        ane_root_hint_infos_snapshot.size() == ane_root_hints_snapshot.size();
    for (size_t i = 0; i < ane_root_hints_snapshot.size(); ++i) {
      if (ane_root_hints_snapshot[i] != target)
        continue;
      result.rank = static_cast<int>(i) + 1;
      result.score = has_scores ? ane_root_hint_infos_snapshot[i].score : 0.0f;
      break;
    }
    return result;
  };
  StateInfo decision_root_state;
  Position decision_root_pos;
  bool decision_root_pos_ready = false;
  const auto get_decision_root_pos = [&]() -> const Position & {
    if (!decision_root_pos_ready) {
      decision_root_pos.set(root_fen_, false, &decision_root_state);
      decision_root_pos_ready = true;
    }
    return decision_root_pos;
  };
  const auto append_cross_root_trace = [&](std::ostringstream &ss) {
    const MCTSRootLookup ab_in_mcts = find_mcts_root_move(ab_best);
    const ABRootLookup mcts_in_ab = find_ab_root_move(mcts_best);
    ss << " ABInMCTSRank=" << ab_in_mcts.rank
       << " ABInMCTSVisits=" << ab_in_mcts.visits
       << " ABInMCTSCurrentVisits=" << ab_in_mcts.current_visits
       << " ABInMCTSQ=" << std::fixed << std::setprecision(3) << ab_in_mcts.q
       << " ABInMCTSPolicy=" << std::fixed << std::setprecision(3)
       << ab_in_mcts.policy << " MCTSInABRank=" << mcts_in_ab.rank
       << " MCTSInABScore=" << mcts_in_ab.score
       << " MCTSInABAvg=" << mcts_in_ab.average_score
       << " MCTSInABEffort=" << mcts_in_ab.effort
       << " MCTSInABLB=" << (mcts_in_ab.score_lowerbound ? 1 : 0)
       << " MCTSInABUB=" << (mcts_in_ab.score_upperbound ? 1 : 0);
  };
  const auto root_q_gap_for_best = [&]() {
    const int count =
        std::min<int>(mcts_state_.num_top_moves.load(std::memory_order_acquire),
                      MCTSSharedState::MAX_TOP_MOVES);
    if (count <= 1) {
      return 0.0f;
    }
    Move top_move(
        mcts_state_.top_moves[0].move_raw.load(std::memory_order_relaxed));
    if (top_move != mcts_best) {
      return 0.0f;
    }
    const float best_q =
        mcts_state_.top_moves[0].q.load(std::memory_order_relaxed);
    uint32_t candidate_visits[MCTSSharedState::MAX_TOP_MOVES]{};
    float candidate_qs[MCTSSharedState::MAX_TOP_MOVES]{};
    int candidate_count = 0;
    for (int i = 1; i < count; ++i) {
      candidate_visits[candidate_count] =
          mcts_state_.top_moves[i].current_visits.load(
              std::memory_order_relaxed);
      candidate_qs[candidate_count] =
          mcts_state_.top_moves[i].q.load(std::memory_order_relaxed);
      ++candidate_count;
    }
    return HybridVisitedRootQGap(best_q, candidate_visits, candidate_qs,
                                 candidate_count);
  };
  const bool mcts_decision_budget = HybridHasMCTSDecisionBudget(
      limits_, time_budget_ms_.load(std::memory_order_acquire),
      ponderhit_received_.load(std::memory_order_acquire));
  const bool clock_managed_search =
      limits_.movetime <= 0 && limits_.nodes <= 0 && limits_.depth <= 0 &&
      limits_.mate <= 0 && !limits_.infinite &&
      (limits_.time[WHITE] > 0 || limits_.time[BLACK] > 0);
  const uint32_t mcts_confidence_visits = mcts_current_visits;
  const uint64_t mcts_confidence_total_nodes = mcts_total_current_nodes;
  const bool mcts_visit_evidence_sane = HybridMCTSVisitEvidenceSane(
      mcts_playouts, mcts_evals, mcts_confidence_total_nodes,
      mcts_confidence_visits);
  const bool low_node_mcts_primary =
      HybridUseMCTSPrimaryForFixedNodeBudget(limits_);
  const bool low_node_mcts_primary_ready = HybridLowNodeMCTSPrimaryReady(
      low_node_mcts_primary, limits_.nodes, mcts_confidence_total_nodes,
      mcts_confidence_visits, mcts_visit_evidence_sane);
  const auto find_root_pawn_lever_tiebreak = [&](Move selected,
                                                 bool allow_non_pawn_selected) {
    if (!config_.root_pawn_lever_tiebreak || selected == Move::none() ||
        low_node_mcts_primary || !mcts_decision_budget ||
        !mcts_visit_evidence_sane) {
      return Move::none();
    }

    const Position &root_pos = get_decision_root_pos();
    if (!HybridRootPawnLeverCanChallengeSelected(root_pos, selected,
                                                 allow_non_pawn_selected)) {
      return Move::none();
    }

    const ABRootLookup selected_ab = find_ab_root_move(selected);
    if (selected_ab.rank <= 0)
      return Move::none();
    if (std::abs(selected_ab.average_score) > 1000)
      return Move::none();
    const MCTSRootLookup selected_mcts = find_mcts_root_move(selected);
    const MCTSRootLookup best_mcts = find_mcts_root_move(mcts_best);
    const ANERootLookup selected_ane = find_ane_root_move(selected);
    const float best_mcts_q =
        best_mcts.rank > 0 ? best_mcts.q : selected_mcts.q;

    std::vector<ABRootMoveInfo> root_moves;
    {
      std::lock_guard<std::mutex> lock(ab_root_mutex_);
      root_moves = ab_root_moves_;
    }

    Move best_lever = Move::none();
    int best_lever_average = std::numeric_limits<int>::min();
    int best_lever_rank = MCTSSharedState::MAX_TOP_MOVES + 1;
    const int count = std::min<int>(static_cast<int>(root_moves.size()),
                                    ABSharedState::MAX_PV);
    for (int i = 0; i < count; ++i) {
      const ABRootMoveInfo &candidate = root_moves[i];
      if (candidate.move == Move::none() || candidate.move == selected)
        continue;
      if (!HybridIsKingsidePawnLever(root_pos, candidate.move))
        continue;
      const MCTSRootLookup mcts_lookup = find_mcts_root_move(candidate.move);
      const bool mcts_confirms_lever = HybridRootPawnLeverCandidate(
          selected_ab.average_score, candidate.average_score, candidate.effort,
          mcts_lookup.rank, mcts_lookup.current_visits, selected_mcts.rank,
          selected_mcts.q, selected_mcts.policy, best_mcts_q, mcts_lookup.q,
          mcts_lookup.policy);
      const ANERootLookup candidate_ane = find_ane_root_move(candidate.move);
      const bool ane_confirms_lever = HybridANERootPawnLeverCandidate(
          config_.ane_root_probe, selected_ane.rank, selected_ane.score,
          candidate_ane.rank, candidate_ane.score, selected_ab.average_score,
          candidate.average_score, candidate.effort, selected_mcts.rank,
          selected_mcts.q, mcts_lookup.rank, mcts_lookup.current_visits,
          mcts_lookup.q, mcts_lookup.policy);
      if (!mcts_confirms_lever && !ane_confirms_lever)
        continue;

      if (candidate.average_score > best_lever_average ||
          (candidate.average_score == best_lever_average &&
           mcts_lookup.rank < best_lever_rank)) {
        best_lever = candidate.move;
        best_lever_average = candidate.average_score;
        best_lever_rank = mcts_lookup.rank;
      }
    }
    return best_lever;
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
       << " MCTSPlayouts=" << mcts_playouts << " MCTSEvals=" << mcts_evals
       << " MCTSBestVisits=" << mcts_visits
       << " MCTSBestCurrentVisits=" << mcts_current_visits
       << " MCTSRootVisits=" << mcts_total_nodes
       << " MCTSRootCurrentVisits=" << mcts_total_current_nodes
       << " MCTSConfidenceVisits=" << mcts_confidence_visits
       << " MCTSConfidenceRootVisits=" << mcts_confidence_total_nodes
       << " MCTSDecisionBudget=" << (mcts_decision_budget ? 1 : 0)
       << " MCTSVisitEvidenceSane=" << (mcts_visit_evidence_sane ? 1 : 0)
       << " LowNodeMCTSPrimary=" << (low_node_mcts_primary ? 1 : 0)
       << " LowNodeMCTSReady=" << (low_node_mcts_primary_ready ? 1 : 0)
       << " ANETop=" << move_to_string(ane_top)
       << " ANEAgreesMCTS=" << (ane_agrees_mcts ? 1 : 0)
       << " ANEConfirmedMCTS=" << (ane_confirmed_mcts_override ? 1 : 0)
       << " ANETopScore=" << std::fixed << std::setprecision(3) << ane_top_score
       << " ANEScoreMargin=" << std::fixed << std::setprecision(3)
       << ane_score_margin << " ANERoot=" << ane_root_hints_to_string()
       << " ABHints=" << move_list_to_string(ab_root_order_hints_snapshot)
       << " ABVerifiedHints="
       << move_list_to_string(ab_verified_root_order_hints_snapshot)
       << " MCTSTop=" << top_moves_to_string();
    append_cross_root_trace(ss);
    ss << " ABRoot=" << ab_root_moves_to_string();
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
    Move policy_tiebreak = Move::none();
    Move q_conflict_tiebreak = Move::none();
    const int top_count =
        std::min<int>(mcts_state_.num_top_moves.load(std::memory_order_acquire),
                      MCTSSharedState::MAX_TOP_MOVES);
    if (mcts_decision_budget && mcts_visit_evidence_sane && top_count > 1) {
      Move top_move(
          mcts_state_.top_moves[0].move_raw.load(std::memory_order_relaxed));
      if (top_move == ab_best) {
        const uint32_t top_visits =
            mcts_state_.top_moves[0].current_visits.load(
                std::memory_order_relaxed);
        const float top_q =
            mcts_state_.top_moves[0].q.load(std::memory_order_relaxed);
        const float top_policy =
            mcts_state_.top_moves[0].policy.load(std::memory_order_relaxed);
        float best_policy = 0.0f;
        float best_q_conflict_gap = 0.0f;
        const ABRootLookup selected_ab = find_ab_root_move(top_move);
        for (int i = 1; i < top_count; ++i) {
          Move move(mcts_state_.top_moves[i].move_raw.load(
              std::memory_order_relaxed));
          if (move == Move::none())
            continue;
          const uint32_t candidate_visits =
              mcts_state_.top_moves[i].current_visits.load(
                  std::memory_order_relaxed);
          const float candidate_q =
              mcts_state_.top_moves[i].q.load(std::memory_order_relaxed);
          const float candidate_policy =
              mcts_state_.top_moves[i].policy.load(std::memory_order_relaxed);
          if (candidate_policy > best_policy &&
              HybridRootPolicyTieBreak(mcts_decision_budget,
                                       mcts_confidence_total_nodes, top_visits,
                                       top_q, top_policy, candidate_visits,
                                       candidate_q, candidate_policy)) {
            best_policy = candidate_policy;
            policy_tiebreak = move;
          }
          const ABRootLookup candidate_ab = find_ab_root_move(move);
          const float q_gap = candidate_q - top_q;
          if (q_gap > best_q_conflict_gap &&
              HybridRootQConflictTieBreak(
                  mcts_decision_budget, mcts_visit_evidence_sane,
                  mcts_confidence_total_nodes, top_visits, top_q, top_policy,
                  candidate_visits, candidate_q, candidate_policy,
                  selected_ab.average_score, candidate_ab.average_score)) {
            best_q_conflict_gap = q_gap;
            q_conflict_tiebreak = move;
          }
        }
      }
    }
    if (policy_tiebreak != Move::none()) {
      stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      trace_simple("root_policy_tiebreak", policy_tiebreak);
      return policy_tiebreak;
    }
    if (q_conflict_tiebreak != Move::none()) {
      stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      trace_simple("root_q_conflict_tiebreak", q_conflict_tiebreak);
      return q_conflict_tiebreak;
    }

    const float agreement_visit_share =
        mcts_confidence_total_nodes > 0
            ? static_cast<float>(mcts_confidence_visits) /
                  static_cast<float>(mcts_confidence_total_nodes)
            : 0.0f;
    const bool allow_non_pawn_agreement_lever =
        HybridRootPawnLeverAgreementTieBreak(
            mcts_decision_budget, mcts_visit_evidence_sane,
            agreement_visit_share, root_q_gap_for_best());
    Move pawn_lever_tiebreak =
        find_root_pawn_lever_tiebreak(ab_best, allow_non_pawn_agreement_lever);
    if (pawn_lever_tiebreak != Move::none()) {
      stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
      trace_simple("root_pawn_lever_tiebreak", pawn_lever_tiebreak);
      return pawn_lever_tiebreak;
    }

    stats_.move_agreements.fetch_add(1, std::memory_order_relaxed);
    trace_simple("engines_agree", ab_best);
    return ab_best;
  }

  const int mcts_cp = QToNnueScore(mcts_q);
  const int ab_depth =
      ab_state_.completed_depth.load(std::memory_order_relaxed);
  const float absolute_visit_share =
      mcts_total_nodes > 0 ? static_cast<float>(mcts_visits) /
                                 static_cast<float>(mcts_total_nodes)
                           : 0.0f;
  const float visit_share =
      mcts_confidence_total_nodes > 0
          ? static_cast<float>(mcts_confidence_visits) /
                static_cast<float>(mcts_confidence_total_nodes)
          : 0.0f;
  const uint64_t min_nodes = mcts_decision_budget
                                 ? 180u
                                 : static_cast<uint64_t>(std::max(
                                       100, current_strategy_.min_mcts_nodes));
  const uint32_t min_visits =
      mcts_decision_budget ? 72u
                           : static_cast<uint32_t>(std::max(
                                 48, current_strategy_.min_mcts_nodes / 2));
  const bool mcts_reliable = mcts_confidence_total_nodes >= min_nodes &&
                             mcts_confidence_visits >= min_visits &&
                             visit_share >= 0.24f;
  const bool mcts_strong =
      mcts_reliable && (mcts_confidence_visits >= 512 || visit_share >= 0.55f);
  const bool mcts_overwhelming = mcts_confidence_total_nodes >= 5000 &&
                                 mcts_confidence_visits >= 512 &&
                                 visit_share >= 0.30f;
  const bool ab_verified =
      ab_depth >=
      std::max(config_.ab_min_depth, current_strategy_.ab_verify_depth);
  const int eval_delta = mcts_cp - ab_score;
  const bool ab_has_clear_preference = ab_verified && std::abs(ab_score) >= 15;
  const ABRootLookup ab_in_ab = find_ab_root_move(ab_best);
  const ABRootLookup mcts_in_ab = find_ab_root_move(mcts_best);
  const MCTSRootLookup ab_in_mcts = find_mcts_root_move(ab_best);
  const ABRootLookup ane_in_ab = find_ab_root_move(ane_top);
  const MCTSRootLookup ane_in_mcts = find_mcts_root_move(ane_top);
  const ANERootLookup ane_top_lookup = find_ane_root_move(ane_top);
  const bool ab_root_rejects_mcts =
      config_.ab_root_reject_mcts &&
      HybridABRootRejectsMCTS(ab_verified, ab_in_ab.rank, mcts_in_ab.rank,
                              ab_in_ab.average_score, mcts_in_ab.average_score,
                              ab_in_ab.effort, mcts_in_ab.effort,
                              mcts_in_ab.score);
  const float root_q_gap = root_q_gap_for_best();
  const bool mcts_decisive_fixed_budget =
      mcts_visit_evidence_sane &&
      HybridMCTSDecisiveFixedBudgetOverride(
          mcts_decision_budget, mcts_strong, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, eval_delta);
  const bool mcts_no_clear_fixed_budget =
      mcts_visit_evidence_sane && !ab_has_clear_preference &&
      HybridMCTSNoClearFixedBudgetOverride(mcts_decision_budget, mcts_strong,
                                           mcts_confidence_visits, visit_share,
                                           eval_delta);
  const bool mcts_root_dominant_fixed_budget =
      mcts_visit_evidence_sane &&
      HybridMCTSRootDominantFixedBudgetOverride(
          mcts_decision_budget, mcts_strong, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, mcts_cp, eval_delta);
  const bool mcts_tactical_gap_fixed_budget =
      mcts_visit_evidence_sane &&
      HybridMCTSTacticalGapFixedBudgetOverride(
          mcts_decision_budget, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta);
  const bool mcts_root_confidence_fixed_budget =
      mcts_visit_evidence_sane &&
      HybridMCTSRootConfidenceFixedBudgetOverride(
          mcts_decision_budget, mcts_strong, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta);
  const bool mcts_low_node_root_confidence =
      HybridMCTSLowNodeRootConfidenceOverride(
          mcts_decision_budget, mcts_visit_evidence_sane,
          ab_has_clear_preference, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta,
          ab_in_ab.average_score, mcts_in_ab.average_score, mcts_in_ab.rank,
          mcts_in_ab.score, mcts_in_ab.score_lowerbound,
          mcts_in_ab.score_upperbound, mcts_in_ab.effort, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_short_root_tactical = HybridMCTSShortRootTacticalOverride(
      mcts_decision_budget, mcts_visit_evidence_sane, ab_root_rejects_mcts,
      mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
      root_q_gap, mcts_cp, eval_delta, ab_in_ab.average_score,
      mcts_in_ab.average_score, mcts_in_ab.rank, mcts_in_ab.score,
      mcts_in_ab.score_lowerbound, mcts_in_ab.effort, ab_in_mcts.rank,
      ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_ab_lowerbound_confirmed =
      HybridMCTSABLowerBoundConfirmedOverride(
          mcts_decision_budget, mcts_visit_evidence_sane,
          mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
          root_q_gap, ab_score, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.score_lowerbound, mcts_in_ab.effort, ab_in_mcts.rank,
          ab_in_mcts.current_visits);
  const Position &decision_root = get_decision_root_pos();
  const bool pawn_only_root = HybridIsPawnOnlyEndgame(decision_root);
  const bool mcts_compact_fixed_budget = HybridMCTSCompactFixedBudgetOverride(
      mcts_decision_budget, mcts_visit_evidence_sane, ab_has_clear_preference,
      mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
      root_q_gap, mcts_cp, eval_delta, ab_in_ab.average_score,
      mcts_in_ab.average_score);
  const bool mcts_compact_pawn_endgame = HybridMCTSCompactPawnEndgameOverride(
      mcts_decision_budget, mcts_visit_evidence_sane, pawn_only_root,
      ab_has_clear_preference, mcts_confidence_total_nodes,
      mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta,
      ab_in_ab.average_score, mcts_in_ab.average_score, mcts_in_ab.rank,
      mcts_in_ab.score, mcts_in_ab.score_lowerbound, mcts_in_ab.effort,
      ab_in_mcts.rank, ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_compact_clear_preference =
      HybridMCTSCompactClearPreferenceOverride(
          mcts_decision_budget, mcts_visit_evidence_sane,
          mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
          root_q_gap, mcts_cp, eval_delta, ab_score, mcts_in_ab.rank,
          mcts_in_ab.score, mcts_in_ab.score_lowerbound,
          mcts_in_ab.score_upperbound, mcts_in_ab.effort, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_cross_root_confidence_fixed_budget =
      mcts_visit_evidence_sane &&
      HybridMCTSCrossRootConfidenceOverride(
          mcts_decision_budget, mcts_strong, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta,
          ab_in_ab.average_score, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.average_score, mcts_in_ab.effort, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_root_confidence_reject_override =
      HybridMCTSRootConfidenceRejectOverride(
          mcts_root_confidence_fixed_budget, ab_root_rejects_mcts,
          mcts_in_ab.rank, mcts_in_ab.score, mcts_in_ab.score_lowerbound,
          mcts_in_ab.score_upperbound, mcts_in_ab.effort,
          ab_in_ab.average_score, mcts_in_ab.average_score, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool low_material_root = decision_root.non_pawn_material() <= RookValue;
  const bool mcts_kingside_pawn_push =
      HybridIsKingsidePawnPush(decision_root, mcts_best);
  const bool mcts_discovered_pawn_push =
      mcts_kingside_pawn_push &&
      HybridIsPawnDiscoveredKingAttack(decision_root, mcts_best);
  const bool mcts_discovered_pawn_push_override =
      HybridMCTSDiscoveredPawnPushOverride(
          mcts_decision_budget, mcts_visit_evidence_sane,
          mcts_discovered_pawn_push, mcts_confidence_total_nodes,
          mcts_confidence_visits, ab_in_mcts.current_visits, visit_share,
          ab_in_mcts.rank > 0 ? mcts_q - ab_in_mcts.q : 0.0f, mcts_cp,
          eval_delta, ab_score, ab_in_ab.average_score,
          mcts_in_ab.average_score, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.score_lowerbound, mcts_in_ab.score_upperbound,
          mcts_in_ab.effort);
  const bool mcts_root_reject_low_material_push =
      HybridMCTSRootRejectLowMaterialPushOverride(
          mcts_decision_budget, mcts_visit_evidence_sane, ab_root_rejects_mcts,
          low_material_root, mcts_kingside_pawn_push,
          mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
          root_q_gap, mcts_cp, eval_delta, ab_in_ab.average_score,
          mcts_in_ab.average_score, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.effort, ab_in_mcts.rank, ab_in_mcts.current_visits,
          ab_in_mcts.q, mcts_q);
  const bool rook_endgame_root = HybridIsRookEndgame(decision_root);
  const bool mcts_quiet_central_pawn_push =
      HybridIsQuietCentralPawnPush(decision_root, mcts_best);
  const bool mcts_root_reject_rook_pawn_push =
      HybridMCTSRootRejectRookEndgamePawnPushOverride(
          mcts_decision_budget, mcts_visit_evidence_sane, ab_root_rejects_mcts,
          rook_endgame_root, mcts_quiet_central_pawn_push,
          mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
          root_q_gap, mcts_cp, eval_delta, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.effort, ab_in_mcts.rank, ab_in_mcts.current_visits,
          ab_in_mcts.q, mcts_q);
  const bool mcts_quiet_central_queen_move =
      HybridIsQuietCentralQueenMove(decision_root, mcts_best);
  const bool mcts_root_reject_quiet_queen_move =
      HybridMCTSRootRejectQuietQueenMoveOverride(
          mcts_decision_budget, mcts_visit_evidence_sane, ab_root_rejects_mcts,
          mcts_quiet_central_queen_move, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta,
          mcts_in_ab.rank, mcts_in_ab.score, mcts_in_ab.score_lowerbound,
          mcts_in_ab.score_upperbound, mcts_in_ab.effort, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_quiet_minor_major_attack =
      HybridIsQuietMinorMajorAttack(decision_root, mcts_best);
  const bool mcts_root_reject_quiet_minor_major_attack =
      HybridMCTSRootRejectQuietMinorMajorAttackOverride(
          mcts_decision_budget, mcts_visit_evidence_sane,
          ab_has_clear_preference, ab_root_rejects_mcts,
          mcts_quiet_minor_major_attack, mcts_confidence_total_nodes,
          mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta,
          mcts_in_ab.rank, mcts_in_ab.score, mcts_in_ab.score_lowerbound,
          mcts_in_ab.score_upperbound, mcts_in_ab.effort, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool mcts_reused_root_confidence =
      HybridMCTSReusedRootConfidenceOverride(
          mcts_decision_budget, mcts_total_nodes, mcts_visits,
          mcts_total_current_nodes, mcts_current_visits, absolute_visit_share,
          visit_share, mcts_cp, eval_delta, ab_in_ab.average_score,
          mcts_in_ab.average_score, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.effort, ab_in_mcts.rank, ab_in_mcts.current_visits,
          ab_in_mcts.q, mcts_q);
  const bool mcts_reused_root_current = HybridMCTSReusedRootCurrentOverride(
      mcts_decision_budget, mcts_visit_evidence_sane, ab_has_clear_preference,
      ab_root_rejects_mcts, mcts_total_nodes, mcts_visits,
      mcts_total_current_nodes, mcts_current_visits, absolute_visit_share,
      visit_share, root_q_gap, mcts_cp, eval_delta, mcts_in_ab.rank,
      mcts_in_ab.score, mcts_in_ab.score_lowerbound,
      mcts_in_ab.score_upperbound, mcts_in_ab.effort, ab_in_mcts.rank,
      ab_in_mcts.current_visits, ab_in_mcts.q, mcts_q);
  const bool bishop_endgame_root = HybridIsBishopOnlyEndgame(decision_root);
  const bool mcts_bishop_back_rank_retreat =
      HybridIsQuietBishopBackRankRetreat(decision_root, mcts_best);
  const bool mcts_bishop_endgame_retreat =
      HybridMCTSBishopEndgameRetreatOverride(
          mcts_decision_budget, mcts_visit_evidence_sane,
          ab_has_clear_preference, bishop_endgame_root,
          mcts_bishop_back_rank_retreat, mcts_total_nodes, mcts_visits,
          mcts_total_current_nodes, mcts_current_visits, absolute_visit_share,
          visit_share, mcts_cp, eval_delta, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.score_lowerbound, mcts_in_ab.score_upperbound,
          mcts_in_ab.effort, ab_in_mcts.rank, ab_in_mcts.current_visits,
          ab_in_mcts.q, mcts_q);
  const bool mcts_root_reject_q_gap = HybridMCTSRootRejectQGapOverride(
      mcts_decision_budget, mcts_visit_evidence_sane, ab_root_rejects_mcts,
      mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
      root_q_gap, mcts_cp, eval_delta, mcts_in_ab.rank, mcts_in_ab.score,
      mcts_in_ab.effort, ab_in_mcts.rank, ab_in_mcts.current_visits,
      ab_in_mcts.q, mcts_q);
  const bool mcts_clock_root_reject_q_gap =
      HybridMCTSClockRootRejectQGapOverride(
          clock_managed_search, mcts_visit_evidence_sane, ab_root_rejects_mcts,
          mcts_confidence_total_nodes, mcts_confidence_visits, visit_share,
          root_q_gap, mcts_cp, eval_delta, mcts_in_ab.rank, mcts_in_ab.score,
          mcts_in_ab.effort, ab_in_mcts.rank, ab_in_mcts.current_visits,
          ab_in_mcts.q, mcts_q);
  ane_confirmed_mcts_override = HybridANEConfirmedMCTSOverride(
      config_.ane_confirm_mcts_override, ane_agrees_mcts, mcts_decision_budget,
      mcts_visit_evidence_sane, mcts_confidence_total_nodes,
      mcts_confidence_visits, visit_share, root_q_gap, mcts_cp, eval_delta,
      ane_score_margin);
  const bool pawn_only_ane_mcts_shape =
      config_.ane_root_probe && ane_agrees_mcts &&
      HybridIsPawnOnlyMCTSANECandidate(decision_root, ab_best, mcts_best);
  const bool pawn_only_ane_mcts_king_recapture =
      pawn_only_ane_mcts_shape &&
      HybridIsPawnOnlyKingRecaptureCandidate(decision_root, ab_best, mcts_best);
  const bool pawn_only_ane_mcts_pawn_lever =
      pawn_only_ane_mcts_shape && HybridIsPawnLever(decision_root, mcts_best);
  const bool pawn_only_ane_mcts_override = HybridPawnOnlyANEMCTSOverride(
      config_.ane_confirm_mcts_override, ane_agrees_mcts, mcts_decision_budget,
      mcts_visit_evidence_sane, pawn_only_ane_mcts_shape,
      pawn_only_ane_mcts_king_recapture, pawn_only_ane_mcts_pawn_lever,
      mcts_total_nodes, mcts_visits, mcts_confidence_total_nodes,
      mcts_confidence_visits, ab_in_mcts.current_visits, visit_share,
      root_q_gap, ab_in_mcts.rank > 0 ? mcts_q - ab_in_mcts.q : 0.0f, mcts_cp,
      eval_delta, ab_in_ab.average_score, mcts_in_ab.average_score,
      ane_score_margin);
  const bool ane_q_supported_root_override =
      ane_top != Move::none() && ane_top != ab_best && ane_top != mcts_best &&
      HybridANEQSupportedRootOverride(
          config_.ane_confirm_mcts_override, mcts_decision_budget,
          mcts_visit_evidence_sane, ab_root_rejects_mcts, ane_top_lookup.rank,
          ane_score_margin, mcts_confidence_total_nodes, ane_in_mcts.rank,
          ane_in_mcts.current_visits, ane_in_mcts.q, ane_in_mcts.policy,
          mcts_current_visits, mcts_q, ab_in_mcts.rank,
          ab_in_mcts.current_visits, ab_in_mcts.q, ane_in_ab.rank,
          ane_in_ab.score, ane_in_ab.average_score,
          ane_in_ab.score_upperbound, ane_in_ab.effort,
          ab_in_ab.average_score);
  bool mcts_root_rejects_ab = false;
  {
    const int top_count =
        std::min<int>(mcts_state_.num_top_moves.load(std::memory_order_acquire),
                      MCTSSharedState::MAX_TOP_MOVES);
    if (config_.mcts_root_reject && top_count > 0 && mcts_reliable &&
        mcts_confidence_visits >= 160 && visit_share >= 0.35f &&
        ab_score >= -120 && ab_score <= 40) {
      Move top_move(
          mcts_state_.top_moves[0].move_raw.load(std::memory_order_relaxed));
      if (top_move == mcts_best) {
        const float top_q =
            mcts_state_.top_moves[0].q.load(std::memory_order_relaxed);
        const uint32_t top_visits =
            mcts_state_.top_moves[0].current_visits.load(
                std::memory_order_relaxed);
        for (int i = 1; i < top_count; ++i) {
          Move move(mcts_state_.top_moves[i].move_raw.load(
              std::memory_order_relaxed));
          if (move != ab_best)
            continue;
          const uint32_t ab_visits =
              mcts_state_.top_moves[i].current_visits.load(
                  std::memory_order_relaxed);
          const float ab_q =
              mcts_state_.top_moves[i].q.load(std::memory_order_relaxed);
          if (HybridMCTSRootRejectsAB(
                  mcts_decision_budget, mcts_visit_evidence_sane, mcts_strong,
                  ab_has_clear_preference, top_visits, ab_visits, top_q, ab_q,
                  visit_share, eval_delta)) {
            mcts_root_rejects_ab = true;
          }
          break;
        }
      }
    }
  }
  const bool mcts_override_allowed =
      low_node_mcts_primary_ready || !ab_root_rejects_mcts ||
      ane_confirmed_mcts_override || pawn_only_ane_mcts_override ||
      ane_q_supported_root_override ||
      mcts_short_root_tactical || mcts_compact_fixed_budget ||
      mcts_compact_pawn_endgame || mcts_ab_lowerbound_confirmed ||
      mcts_low_node_root_confidence || mcts_compact_clear_preference ||
      mcts_cross_root_confidence_fixed_budget ||
      mcts_root_confidence_reject_override || mcts_reused_root_confidence ||
      mcts_root_reject_low_material_push || mcts_root_reject_rook_pawn_push ||
      mcts_root_reject_quiet_queen_move ||
      mcts_root_reject_quiet_minor_major_attack ||
      mcts_bishop_endgame_retreat || mcts_root_reject_q_gap ||
      mcts_clock_root_reject_q_gap || mcts_discovered_pawn_push_override ||
      mcts_reused_root_current || mcts_root_rejects_ab ||
      (mcts_overwhelming && eval_delta >= 250);

  bool choose_mcts = false;
  const char *reason = "ab_default";
  switch (config_.decision_mode) {
  case ParallelHybridConfig::DecisionMode::MCTS_PRIMARY:
    choose_mcts =
        mcts_override_allowed &&
        (low_node_mcts_primary_ready || ane_confirmed_mcts_override ||
         pawn_only_ane_mcts_override || mcts_ab_lowerbound_confirmed ||
         mcts_low_node_root_confidence || mcts_discovered_pawn_push_override ||
         mcts_root_reject_low_material_push ||
         mcts_root_reject_rook_pawn_push || mcts_root_reject_quiet_queen_move ||
         mcts_root_reject_quiet_minor_major_attack ||
         mcts_bishop_endgame_retreat || mcts_root_reject_q_gap ||
         mcts_clock_root_reject_q_gap ||
         (mcts_reliable && (!ab_has_clear_preference || eval_delta >= 180)));
    if (choose_mcts)
      reason =
          low_node_mcts_primary_ready          ? "low_node_mcts_primary"
          : ane_confirmed_mcts_override        ? "ane_confirmed_mcts"
          : pawn_only_ane_mcts_override        ? "pawn_only_ane_mcts"
          : mcts_compact_pawn_endgame          ? "mcts_compact_pawn_endgame"
          : mcts_ab_lowerbound_confirmed       ? "mcts_ab_lowerbound_confirmed"
          : mcts_low_node_root_confidence      ? "mcts_low_node_root_confidence"
          : mcts_discovered_pawn_push_override ? "mcts_discovered_pawn_push"
          : mcts_root_reject_low_material_push
              ? "mcts_root_reject_low_material_push"
          : mcts_root_reject_rook_pawn_push ? "mcts_root_reject_rook_pawn_push"
          : mcts_root_reject_quiet_queen_move
              ? "mcts_root_reject_quiet_queen_move"
          : mcts_root_reject_quiet_minor_major_attack
              ? "mcts_root_reject_quiet_minor_major_attack"
          : mcts_bishop_endgame_retreat  ? "mcts_bishop_endgame_retreat"
          : mcts_root_reject_q_gap       ? "mcts_root_reject_q_gap"
          : mcts_clock_root_reject_q_gap ? "mcts_clock_root_reject_q_gap"
                                         : "mcts_primary_reliable";
    break;
  case ParallelHybridConfig::DecisionMode::AB_PRIMARY:
    choose_mcts =
        mcts_override_allowed && mcts_overwhelming && eval_delta >= 250;
    if (choose_mcts)
      reason = "ab_primary_mcts_overwhelming";
    break;
  case ParallelHybridConfig::DecisionMode::VOTE_WEIGHTED:
  case ParallelHybridConfig::DecisionMode::DYNAMIC:
    if (!mcts_override_allowed) {
      reason = "ab_root_rejects_mcts";
    } else if (low_node_mcts_primary_ready) {
      choose_mcts = true;
      reason = "low_node_mcts_primary";
    } else if (ane_confirmed_mcts_override) {
      choose_mcts = true;
      reason = "ane_confirmed_mcts";
    } else if (pawn_only_ane_mcts_override) {
      choose_mcts = true;
      reason = "pawn_only_ane_mcts";
    } else if (ane_q_supported_root_override) {
      reason = "ane_q_supported_root";
    } else if (mcts_visit_evidence_sane && mcts_overwhelming &&
               eval_delta >= 180) {
      choose_mcts = true;
      reason = "mcts_overwhelming_delta";
    } else if (mcts_decisive_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_decisive_fixed_budget";
    } else if (mcts_tactical_gap_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_tactical_gap_fixed_budget";
    } else if (mcts_root_dominant_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_root_dominant_fixed_budget";
    } else if (mcts_root_confidence_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_root_confidence_fixed_budget";
    } else if (mcts_low_node_root_confidence) {
      choose_mcts = true;
      reason = "mcts_low_node_root_confidence";
    } else if (mcts_short_root_tactical) {
      choose_mcts = true;
      reason = "mcts_short_root_tactical";
    } else if (mcts_ab_lowerbound_confirmed) {
      choose_mcts = true;
      reason = "mcts_ab_lowerbound_confirmed";
    } else if (mcts_discovered_pawn_push_override) {
      choose_mcts = true;
      reason = "mcts_discovered_pawn_push";
    } else if (mcts_root_reject_low_material_push) {
      choose_mcts = true;
      reason = "mcts_root_reject_low_material_push";
    } else if (mcts_root_reject_rook_pawn_push) {
      choose_mcts = true;
      reason = "mcts_root_reject_rook_pawn_push";
    } else if (mcts_root_reject_quiet_queen_move) {
      choose_mcts = true;
      reason = "mcts_root_reject_quiet_queen_move";
    } else if (mcts_root_reject_quiet_minor_major_attack) {
      choose_mcts = true;
      reason = "mcts_root_reject_quiet_minor_major_attack";
    } else if (mcts_bishop_endgame_retreat) {
      choose_mcts = true;
      reason = "mcts_bishop_endgame_retreat";
    } else if (mcts_root_reject_q_gap) {
      choose_mcts = true;
      reason = "mcts_root_reject_q_gap";
    } else if (mcts_clock_root_reject_q_gap) {
      choose_mcts = true;
      reason = "mcts_clock_root_reject_q_gap";
    } else if (mcts_reused_root_confidence) {
      choose_mcts = true;
      reason = "mcts_reused_root_confidence";
    } else if (mcts_reused_root_current) {
      choose_mcts = true;
      reason = "mcts_reused_root_current";
    } else if (mcts_compact_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_compact_fixed_budget";
    } else if (mcts_compact_pawn_endgame) {
      choose_mcts = true;
      reason = "mcts_compact_pawn_endgame";
    } else if (mcts_compact_clear_preference) {
      choose_mcts = true;
      reason = "mcts_compact_clear_preference";
    } else if (mcts_cross_root_confidence_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_cross_root_confidence_fixed_budget";
    } else if (mcts_root_rejects_ab) {
      choose_mcts = true;
      reason = "mcts_root_rejects_ab";
    } else if (mcts_no_clear_fixed_budget) {
      choose_mcts = true;
      reason = "mcts_no_clear_fixed_budget";
    } else if (mcts_visit_evidence_sane && !ab_verified && mcts_reliable &&
               eval_delta >= -80) {
      choose_mcts = true;
      reason = "mcts_reliable_ab_unverified";
    }
    break;
  }

  const auto trace_final = [&](const char *final_reason, Move final_selected) {
    if (!config_.trace_decisions)
      return;
    std::ostringstream ss;
    ss << "HybridTrace: reason=" << final_reason
       << " mode=" << mode_to_string(config_.decision_mode)
       << " selected=" << move_to_string(final_selected)
       << " ABMove=" << move_to_string(ab_best)
       << " MCTSMove=" << move_to_string(mcts_best) << " ABScore=" << ab_score
       << " ABDepth=" << ab_depth << " MCTSQ=" << std::fixed
       << std::setprecision(3) << mcts_q << " MCTSCP=" << mcts_cp
       << " EvalDelta=" << eval_delta << " MCTSBestVisits=" << mcts_visits
       << " MCTSBestCurrentVisits=" << mcts_current_visits
       << " MCTSPlayouts=" << mcts_playouts << " MCTSEvals=" << mcts_evals
       << " MCTSRootVisits=" << mcts_total_nodes
       << " MCTSRootCurrentVisits=" << mcts_total_current_nodes
       << " MCTSConfidenceVisits=" << mcts_confidence_visits
       << " MCTSConfidenceRootVisits=" << mcts_confidence_total_nodes
       << " AbsoluteVisitShare=" << absolute_visit_share
       << " VisitShare=" << visit_share
       << " MCTSDecisionBudget=" << (mcts_decision_budget ? 1 : 0)
       << " MCTSVisitEvidenceSane=" << (mcts_visit_evidence_sane ? 1 : 0)
       << " LowNodeMCTSPrimary=" << (low_node_mcts_primary ? 1 : 0)
       << " LowNodeMCTSReady=" << (low_node_mcts_primary_ready ? 1 : 0)
       << " RootQGap=" << root_q_gap
       << " MCTSReliable=" << (mcts_reliable ? 1 : 0)
       << " MCTSStrong=" << (mcts_strong ? 1 : 0)
       << " MCTSNoClearFixed=" << (mcts_no_clear_fixed_budget ? 1 : 0)
       << " MCTSRootDominant=" << (mcts_root_dominant_fixed_budget ? 1 : 0)
       << " MCTSTacticalGap=" << (mcts_tactical_gap_fixed_budget ? 1 : 0)
       << " MCTSRootConfidence=" << (mcts_root_confidence_fixed_budget ? 1 : 0)
       << " MCTSLowNodeRootConfidence="
       << (mcts_low_node_root_confidence ? 1 : 0)
       << " MCTSShortRootTactical=" << (mcts_short_root_tactical ? 1 : 0)
       << " MCTSABLowerBoundConfirmed="
       << (mcts_ab_lowerbound_confirmed ? 1 : 0)
       << " MCTSCompact=" << (mcts_compact_fixed_budget ? 1 : 0)
       << " MCTSCompactPawnEndgame=" << (mcts_compact_pawn_endgame ? 1 : 0)
       << " MCTSCompactClearPreference="
       << (mcts_compact_clear_preference ? 1 : 0) << " MCTSCrossRootConfidence="
       << (mcts_cross_root_confidence_fixed_budget ? 1 : 0)
       << " MCTSRootConfidenceRejectOverride="
       << (mcts_root_confidence_reject_override ? 1 : 0)
       << " MCTSDiscoveredPawnPush="
       << (mcts_discovered_pawn_push_override ? 1 : 0)
       << " MCTSRootRejectLowMaterialPush="
       << (mcts_root_reject_low_material_push ? 1 : 0)
       << " MCTSRootRejectRookPawnPush="
       << (mcts_root_reject_rook_pawn_push ? 1 : 0)
       << " MCTSRootRejectQuietQueenMove="
       << (mcts_root_reject_quiet_queen_move ? 1 : 0)
       << " MCTSRootRejectQuietMinorMajorAttack="
       << (mcts_root_reject_quiet_minor_major_attack ? 1 : 0)
       << " MCTSBishopEndgameRetreat=" << (mcts_bishop_endgame_retreat ? 1 : 0)
       << " MCTSRootRejectQGap=" << (mcts_root_reject_q_gap ? 1 : 0)
       << " MCTSClockRootRejectQGap=" << (mcts_clock_root_reject_q_gap ? 1 : 0)
       << " MCTSReusedRootConfidence=" << (mcts_reused_root_confidence ? 1 : 0)
       << " MCTSReusedRootCurrent=" << (mcts_reused_root_current ? 1 : 0)
       << " MCTSOverwhelming=" << (mcts_overwhelming ? 1 : 0)
       << " ABVerified=" << (ab_verified ? 1 : 0)
       << " ABClearPreference=" << (ab_has_clear_preference ? 1 : 0)
       << " ABRootRejectsMCTS=" << (ab_root_rejects_mcts ? 1 : 0)
       << " MCTSOverrideAllowed=" << (mcts_override_allowed ? 1 : 0)
       << " MCTSRootRejectsAB=" << (mcts_root_rejects_ab ? 1 : 0)
       << " ANETop=" << move_to_string(ane_top)
       << " ANEAgreesMCTS=" << (ane_agrees_mcts ? 1 : 0)
       << " ANEConfirmedMCTS=" << (ane_confirmed_mcts_override ? 1 : 0)
       << " PawnOnlyANEMCTS=" << (pawn_only_ane_mcts_override ? 1 : 0)
       << " ANEQSupportedRoot=" << (ane_q_supported_root_override ? 1 : 0)
       << " ANETopScore=" << std::fixed << std::setprecision(3) << ane_top_score
       << " ANEScoreMargin=" << std::fixed << std::setprecision(3)
       << ane_score_margin << " ANETopInMCTSRank=" << ane_in_mcts.rank
       << " ANETopInMCTSCurrentVisits=" << ane_in_mcts.current_visits
       << " ANETopInMCTSQ=" << std::fixed << std::setprecision(3)
       << ane_in_mcts.q << " ANETopInMCTSPolicy=" << std::fixed
       << std::setprecision(3) << ane_in_mcts.policy
       << " ANETopInABRank=" << ane_in_ab.rank
       << " ANETopInABScore=" << ane_in_ab.score
       << " ANETopInABAvg=" << ane_in_ab.average_score
       << " ANETopInABEffort=" << ane_in_ab.effort
       << " ANETopInABUB=" << (ane_in_ab.score_upperbound ? 1 : 0)
       << " ANERoot=" << ane_root_hints_to_string()
       << " ABHints=" << move_list_to_string(ab_root_order_hints_snapshot)
       << " ABVerifiedHints="
       << move_list_to_string(ab_verified_root_order_hints_snapshot)
       << " MCTSTop=" << top_moves_to_string();
    append_cross_root_trace(ss);
    ss << " ABRoot=" << ab_root_moves_to_string();
    send_info_string(ss.str());
  };

  Move selected = ane_q_supported_root_override
                      ? ane_top
                      : (choose_mcts ? mcts_best : ab_best);
  trace_final(reason, selected);

  Move pawn_lever_tiebreak = find_root_pawn_lever_tiebreak(selected, true);
  if (pawn_lever_tiebreak != Move::none()) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    trace_final("root_pawn_lever_tiebreak", pawn_lever_tiebreak);
    return pawn_lever_tiebreak;
  }

  if (ane_q_supported_root_override) {
    stats_.mcts_overrides.fetch_add(1, std::memory_order_relaxed);
    return ane_top;
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
