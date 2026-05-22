/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Parallel Hybrid Search - MCTS and Alpha-Beta running simultaneously
  on Apple Silicon with unified memory for zero-copy data sharing.

  Licensed under GPL-3.0
*/

#pragma once

#include "../mcts/search.h"
#include "../search/search.h"
#include "../search/tt.h"
#include "classifier.h"
#include "shared_tt.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace MetalFish {

class Engine;

namespace MCTS {

constexpr size_t APPLE_CACHE_LINE_SIZE = 128;

// Lock-free structure for AB to communicate best moves to MCTS
struct alignas(APPLE_CACHE_LINE_SIZE) ABSharedState {
  std::atomic<uint32_t> best_move_raw{0};
  std::atomic<int32_t> best_score{0};
  std::atomic<int32_t> completed_depth{0};
  std::atomic<uint64_t> nodes_searched{0};

  std::atomic<uint64_t> update_counter{0};
  std::atomic<bool> ab_running{false};
  std::atomic<bool> has_result{false};

  void reset() {
    best_move_raw.store(0, std::memory_order_relaxed);
    best_score.store(0, std::memory_order_relaxed);
    completed_depth.store(0, std::memory_order_relaxed);
    nodes_searched.store(0, std::memory_order_relaxed);
    update_counter.store(0, std::memory_order_relaxed);
    ab_running.store(false, std::memory_order_relaxed);
    has_result.store(false, std::memory_order_relaxed);
    pv_length.store(0, std::memory_order_relaxed);
    pv_depth.store(0, std::memory_order_relaxed);
    for (int i = 0; i < MAX_PV; ++i) {
      pv_moves[i].store(0, std::memory_order_relaxed);
    }
  }

  Move get_best_move() const {
    uint32_t raw = best_move_raw.load(std::memory_order_acquire);
    return Move(raw);
  }

  void set_best_move(Move m, int score, int depth, uint64_t nodes) {
    best_move_raw.store(m.raw(), std::memory_order_relaxed);
    best_score.store(score, std::memory_order_relaxed);
    completed_depth.store(depth, std::memory_order_relaxed);
    nodes_searched.store(nodes, std::memory_order_relaxed);
    has_result.store(true, std::memory_order_release);
    update_counter.fetch_add(1, std::memory_order_release);
  }

  // PV from AB iterative deepening, injected into the MCTS tree in real-time
  static constexpr int MAX_PV = 16;
  std::atomic<uint16_t> pv_moves[MAX_PV]{};
  std::atomic<int> pv_length{0};
  std::atomic<int> pv_depth{0};

  void publish_pv(const std::vector<Move> &pv, int depth) {
    int len = std::min(static_cast<int>(pv.size()), MAX_PV);
    for (int i = 0; i < len; ++i) {
      pv_moves[i].store(pv[i].raw(), std::memory_order_relaxed);
    }
    pv_depth.store(depth, std::memory_order_relaxed);
    pv_length.store(len, std::memory_order_release);
  }
};

struct alignas(APPLE_CACHE_LINE_SIZE) MCTSSharedState {
  std::atomic<uint32_t> best_move_raw{0};
  std::atomic<float> best_q{0.0f};
  std::atomic<uint32_t> best_visits{0};
  std::atomic<uint32_t> best_current_visits{0};
  // Sum of root-edge visits in the current MCTS root. This can differ from
  // per-search node stats when a tree is reused, and is the right denominator
  // for root confidence / visit-share decisions.
  std::atomic<uint64_t> total_nodes{0};
  std::atomic<uint64_t> total_current_nodes{0};

  static constexpr int MAX_TOP_MOVES = 10;
  struct TopMove {
    std::atomic<uint32_t> move_raw{0};
    std::atomic<float> policy{0.0f};
    std::atomic<uint32_t> visits{0};
    std::atomic<uint32_t> current_visits{0};
    std::atomic<float> q{0.0f};
  };
  TopMove top_moves[MAX_TOP_MOVES];
  std::atomic<int> num_top_moves{0};

  std::atomic<uint64_t> update_counter{0};
  std::atomic<bool> mcts_running{false};
  std::atomic<bool> has_result{false};

  void reset() {
    best_move_raw.store(0, std::memory_order_relaxed);
    best_q.store(0.0f, std::memory_order_relaxed);
    best_visits.store(0, std::memory_order_relaxed);
    best_current_visits.store(0, std::memory_order_relaxed);
    total_nodes.store(0, std::memory_order_relaxed);
    total_current_nodes.store(0, std::memory_order_relaxed);
    num_top_moves.store(0, std::memory_order_relaxed);
    update_counter.store(0, std::memory_order_relaxed);
    mcts_running.store(false, std::memory_order_relaxed);
    has_result.store(false, std::memory_order_relaxed);
    for (int i = 0; i < MAX_TOP_MOVES; ++i) {
      top_moves[i].move_raw.store(0, std::memory_order_relaxed);
      top_moves[i].policy.store(0.0f, std::memory_order_relaxed);
      top_moves[i].visits.store(0, std::memory_order_relaxed);
      top_moves[i].current_visits.store(0, std::memory_order_relaxed);
      top_moves[i].q.store(0.0f, std::memory_order_relaxed);
    }
  }

  Move get_best_move() const {
    uint32_t raw = best_move_raw.load(std::memory_order_acquire);
    return Move(raw);
  }
};

struct ParallelSearchStats {
  std::atomic<uint64_t> mcts_nodes{0};
  std::atomic<uint64_t> mcts_iterations{0};
  std::atomic<uint64_t> transformer_evaluations{0};
  std::atomic<uint64_t> transformer_batches{0};

  std::atomic<uint64_t> ab_nodes{0};
  std::atomic<uint64_t> ab_depth{0};
  std::atomic<uint64_t> ab_iterations{0};

  std::atomic<uint64_t> policy_updates{0};
  std::atomic<uint64_t> move_agreements{0};
  std::atomic<uint64_t> ab_overrides{0};
  std::atomic<uint64_t> mcts_overrides{0};

  double mcts_time_ms = 0;
  double ab_time_ms = 0;
  double total_time_ms = 0;

  void reset() {
    mcts_nodes = 0;
    mcts_iterations = 0;
    transformer_evaluations = 0;
    transformer_batches = 0;
    ab_nodes = 0;
    ab_depth = 0;
    ab_iterations = 0;
    policy_updates = 0;
    move_agreements = 0;
    ab_overrides = 0;
    mcts_overrides = 0;
    mcts_time_ms = 0;
    ab_time_ms = 0;
    total_time_ms = 0;
  }
};

struct ParallelHybridConfig {
  SearchParams mcts_config;
  int mcts_threads = 4;
  int ab_threads = 4;

  int ab_min_depth = 8;
  int ab_max_depth = 64;
  bool ab_use_time = true;

  // Experimental AB PV -> MCTS prior mutation. Keep opt-in until it is
  // Elo-proven; unsafe prior mutation can overpower the network root policy.
  float ab_policy_weight = 0.0f;
  float agreement_threshold = 0.3f;
  float override_threshold = 1.0f;
  int policy_update_interval_ms = 50;

  float time_fraction = 0.05f;
  float max_time_fraction = 0.20f;
  float increment_usage = 0.75f;

  bool use_position_classifier = true;

  // Transformer batching. NNUE stays on CPU; these values tune MCTS network
  // inference only.
  int transformer_batch_size = 128;
  int transformer_batch_timeout_us = 200;
  bool use_transformer_prefetch = true;
  bool trace_decisions = false;
  bool ab_root_reject_mcts = true;
  bool mcts_root_reject = true;
  bool use_shared_tt = false;
  bool mcts_ab_root_hints = true;
  int mcts_ab_root_hint_delay_ms = 25;
  int mcts_ab_root_hint_count = 8;
  int ab_candidate_verify_ms = 120;
  int ab_candidate_verify_count = 4;
  bool root_pawn_lever_tiebreak = true;
  bool ane_root_probe = false;
  bool ane_root_hints = false;
  bool ane_confirm_mcts_override = true;
  std::string ane_weights_path;
  std::string ane_model_path;
  std::string ane_compute_units = "cpu-ne";
  int ane_root_hint_count = 10;
  int ane_root_hint_wait_ms = 0;
  int ane_min_budget_ms = 1000;

  enum class DecisionMode {
    MCTS_PRIMARY,  // Trust MCTS unless AB strongly disagrees
    AB_PRIMARY,    // Trust AB unless MCTS strongly disagrees
    VOTE_WEIGHTED, // Weighted combination based on confidence
    DYNAMIC        // Choose based on position type
  };
  DecisionMode decision_mode = DecisionMode::DYNAMIC;
};

class ParallelHybridSearch {
public:
  using BestMoveCallback = std::function<void(Move, Move)>;
  using InfoCallback = std::function<void(const std::string &)>;

  ParallelHybridSearch();
  ~ParallelHybridSearch();

  ParallelHybridSearch(const ParallelHybridSearch &) = delete;
  ParallelHybridSearch &operator=(const ParallelHybridSearch &) = delete;
  ParallelHybridSearch(ParallelHybridSearch &&) = delete;
  ParallelHybridSearch &operator=(ParallelHybridSearch &&) = delete;

  bool initialize(Engine *engine);
  bool is_ready() const { return initialized_; }

  void set_config(const ParallelHybridConfig &config) { config_ = config; }
  const ParallelHybridConfig &config() const { return config_; }

  void start_search(const Position &pos,
                    const ::MetalFish::Search::LimitsType &limits,
                    BestMoveCallback best_move_cb,
                    InfoCallback info_cb = nullptr);
  void ponderhit();
  void stop();
  void wait();
  bool is_searching() const {
    return searching_.load(std::memory_order_acquire);
  }

  const ParallelSearchStats &stats() const { return stats_; }
  Move get_best_move() const;
  Move get_ponder_move() const;

  void new_game();

private:
  struct ABRootMoveInfo {
    Move move = Move::none();
    int score = 0;
    int previous_score = 0;
    int average_score = 0;
    bool score_lowerbound = false;
    bool score_upperbound = false;
    uint64_t effort = 0;
  };

  struct ANERootHintInfo {
    Move move = Move::none();
    float score = 0.0f;
  };

  bool initialized_ = false;
  ParallelHybridConfig config_;
  ParallelSearchStats stats_;

  std::unique_ptr<Search> mcts_search_;
  std::unique_ptr<NNMCTSEvaluator> ane_evaluator_;
  std::unique_ptr<SharedTTReader> shared_tt_reader_;
  Engine *engine_ = nullptr;

  PositionClassifier classifier_;
  StrategySelector strategy_selector_;
  SearchStrategy current_strategy_;

  std::unordered_map<uint16_t, float> nn_policy_hints_;

  ABSharedState ab_state_;
  MCTSSharedState mcts_state_;

  enum class ThreadState { IDLE, RUNNING, STOPPING };

  std::mutex thread_mutex_;
  std::condition_variable thread_cv_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> searching_{false};
  std::atomic<bool> shutdown_requested_{false};
  std::atomic<int> last_injected_ab_depth_{0};
  std::atomic<uint32_t> last_injected_ab_move_raw_{0};
  std::atomic<int> ab_policy_injections_{0};

  std::thread mcts_thread_;
  std::thread ab_thread_;
  std::thread coordinator_thread_;

  std::atomic<ThreadState> mcts_thread_state_{ThreadState::IDLE};
  std::atomic<ThreadState> ab_thread_state_{ThreadState::IDLE};
  std::atomic<ThreadState> coordinator_thread_state_{ThreadState::IDLE};

  std::atomic<bool> mcts_thread_done_{true};
  std::atomic<bool> ab_thread_done_{true};
  std::atomic<bool> coordinator_thread_done_{true};
  std::mutex mcts_start_mutex_;
  std::mutex ab_start_mutex_;
  std::atomic<bool> mcts_search_started_{false};
  std::atomic<bool> ab_search_started_{false};
  std::future<std::vector<Move>> ane_root_hints_future_;
  std::mutex ane_root_hints_mutex_;
  std::vector<Move> ane_root_hints_;
  std::vector<ANERootHintInfo> ane_root_hint_infos_;

  std::string root_fen_;
  ::MetalFish::Search::LimitsType limits_;
  std::atomic<int64_t> search_start_ms_{0};
  std::atomic<int> time_budget_ms_{0};
  std::atomic<bool> ponderhit_received_{false};

  std::atomic<uint32_t> final_best_move_{0};
  std::atomic<uint32_t> final_ponder_move_{0};
  std::vector<Move> final_pv_;
  mutable std::mutex pv_mutex_;

  std::mutex callback_mutex_;
  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;
  std::mutex ab_root_mutex_;
  std::vector<ABRootMoveInfo> ab_root_moves_;
  std::atomic<bool> callback_invoked_{false};

  void mcts_thread_main();
  void ab_thread_main();
  void coordinator_thread_main();

  void update_mcts_policy_from_ab();
  void publish_mcts_state();
  std::vector<Move> collect_mcts_root_order_hints();
  void start_ane_root_probe();
  std::vector<Move> compute_ane_root_order_hints();
  std::vector<Move> collect_ane_root_order_hints();
  std::vector<Move> collect_root_order_hints();
  std::vector<Move>
  verify_ab_root_candidates(const std::vector<Move> &candidates, int verify_ms);

  void run_ab_search();
  void publish_ab_state(Move best, int score, int depth, uint64_t nodes);

  Move make_final_decision();
  void refresh_final_state(Move final_move);
  Move first_allowed_legal_move() const;
  int calculate_time_budget() const;
  bool should_stop() const;

  void join_all_threads();
  void signal_thread_done(std::atomic<bool> &done_flag);
  bool all_threads_done() const;

  void send_info(int depth, int score, uint64_t nodes, int time_ms,
                 const std::vector<Move> &pv, const std::string &source);
  void send_info_string(const std::string &msg);

  void invoke_best_move_callback(Move best, Move ponder);
};

std::unique_ptr<ParallelHybridSearch> create_parallel_hybrid_search(
    Engine *engine,
    const ParallelHybridConfig &config = ParallelHybridConfig());

// Returns true when the coordinator, not AB's time manager, owns the outer
// search budget and should keep MCTS running after AB has produced a result.
bool HybridShouldContinueMCTSAfterAB(
    const ::MetalFish::Search::LimitsType &limits);

bool HybridCanStopEarlyOnAgreement(
    const ::MetalFish::Search::LimitsType &limits);

bool HybridHasMCTSDecisionBudget(const ::MetalFish::Search::LimitsType &limits,
                                 int time_budget_ms, bool ponderhit_received);

::MetalFish::Search::LimitsType
HybridBuildMCTSLimits(const ::MetalFish::Search::LimitsType &limits,
                      int time_budget_ms, bool waiting_for_ponderhit);

int HybridABCandidateVerifyBudgetMs(
    const ::MetalFish::Search::LimitsType &limits, int time_budget_ms,
    int requested_ms, bool waiting_for_ponderhit);

bool HybridMCTSDecisiveFixedBudgetOverride(bool fixed_budget, bool mcts_strong,
                                           uint64_t mcts_total_nodes,
                                           uint32_t mcts_visits,
                                           float visit_share, int eval_delta);

bool HybridMCTSNoClearFixedBudgetOverride(bool fixed_budget, bool mcts_strong,
                                          uint32_t mcts_visits,
                                          float visit_share, int eval_delta);

bool HybridMCTSRootDominantFixedBudgetOverride(
    bool fixed_budget, bool mcts_strong, uint64_t mcts_total_nodes,
    uint32_t mcts_visits, float visit_share, int mcts_cp, int eval_delta);

bool HybridMCTSTacticalGapFixedBudgetOverride(
    bool fixed_budget, uint64_t mcts_total_nodes, uint32_t mcts_visits,
    float visit_share, float root_q_gap, int mcts_cp, int eval_delta);

bool HybridMCTSRootConfidenceFixedBudgetOverride(
    bool fixed_budget, bool mcts_strong, uint64_t mcts_total_nodes,
    uint32_t mcts_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta);

bool HybridMCTSCompactFixedBudgetOverride(
    bool fixed_budget, bool visit_evidence_sane, bool ab_has_clear_preference,
    uint64_t mcts_root_visits, uint32_t mcts_best_visits, float visit_share,
    float root_q_gap, int mcts_cp, int eval_delta, int ab_average_score,
    int mcts_average_score);

bool HybridMCTSCrossRootConfidenceOverride(
    bool fixed_budget, bool mcts_strong, uint64_t mcts_total_nodes,
    uint32_t mcts_visits, float visit_share, float root_q_gap, int mcts_cp,
    int eval_delta, int ab_average_score, int mcts_in_ab_rank,
    int mcts_in_ab_score, int mcts_average_score, uint64_t mcts_effort,
    int ab_in_mcts_rank, uint32_t ab_in_mcts_visits, float ab_in_mcts_q,
    float mcts_q);

bool HybridMCTSVisitEvidenceSane(uint64_t mcts_playouts, uint64_t mcts_evals,
                                 uint64_t root_visits, uint32_t best_visits);

bool HybridANEConfirmedMCTSOverride(bool enabled, bool ane_agrees_mcts,
                                    bool fixed_budget, bool visit_evidence_sane,
                                    uint64_t mcts_root_visits,
                                    uint32_t mcts_best_visits,
                                    float visit_share, float root_q_gap,
                                    int mcts_cp, int eval_delta,
                                    float ane_score_margin);

bool HybridABRootRejectsMCTS(bool ab_verified, int ab_rank, int mcts_rank,
                             int ab_average_score, int mcts_average_score,
                             uint64_t ab_effort, uint64_t mcts_effort,
                             int mcts_score);

bool HybridRootPolicyTieBreak(bool fixed_budget, uint64_t root_visits,
                              uint32_t top_visits, float top_q,
                              float top_policy, uint32_t candidate_visits,
                              float candidate_q, float candidate_policy);

bool HybridMCTSRootRejectsAB(bool fixed_budget, bool visit_evidence_sane,
                             bool mcts_strong, bool ab_has_clear_preference,
                             uint32_t top_visits, uint32_t ab_visits,
                             float top_q, float ab_q, float visit_share,
                             int eval_delta);

bool HybridRootPawnLeverAgreementTieBreak(bool fixed_budget,
                                          bool visit_evidence_sane,
                                          float agreement_visit_share,
                                          float root_q_gap);

bool HybridRootPawnLeverCandidate(
    int selected_average_score, int candidate_average_score,
    uint64_t candidate_effort, int mcts_rank, uint32_t mcts_current_visits,
    int selected_mcts_rank, float selected_mcts_q, float selected_mcts_policy,
    float best_mcts_q, float candidate_mcts_q, float candidate_mcts_policy);

bool HybridIsPawnLever(const Position &pos, Move move);

bool HybridIsKingsidePawnLever(const Position &pos, Move move);

bool HybridIsKingsidePawnPush(const Position &pos, Move move);

bool HybridRootPawnLeverCanChallengeSelected(const Position &pos, Move selected,
                                             bool allow_non_pawn_selected);

bool HybridHighPolicyRootLeverHint(const Position &pos, Move move, float policy,
                                   float leader_policy);

float HybridVisitedRootQGap(float best_q, const uint32_t *candidate_visits,
                            const float *candidate_qs, int candidate_count);

int HybridSubsearchJoinGraceMs(bool external_stop, int time_budget_ms);

} // namespace MCTS
} // namespace MetalFish
