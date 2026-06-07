/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  Licensed under GPL-3.0
*/

#pragma once

#include "backend_adapter.h"
#include "core.h"
#include "node.h"
#include "search_params.h"
#include "stoppers.h"

#include "../search/search.h"
#include "../uci/uci.h"

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <utility>

#ifdef __aarch64__
#define CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define PREFETCH(addr)                                                         \
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#else
#define PREFETCH(addr) (void)(addr)
#endif

namespace MetalFish {
namespace MCTS {

class SharedTTReader;

struct PipelineStats {
  std::atomic<uint64_t> total_nodes{0};
  std::atomic<uint64_t> nn_evaluations{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> cache_misses{0};
  std::atomic<uint64_t> total_batches{0};

  void reset() {
    total_nodes.store(0, std::memory_order_relaxed);
    nn_evaluations.store(0, std::memory_order_relaxed);
    cache_hits.store(0, std::memory_order_relaxed);
    cache_misses.store(0, std::memory_order_relaxed);
    total_batches.store(0, std::memory_order_relaxed);
  }
};

struct SearchWorkerCtx {
  int worker_id = 0;
  Position root_pos;
  StateInfo root_pos_st;
  Position pos;
  StateInfo root_st;
  std::vector<StateInfo> state_stack;
  std::vector<Move> move_stack;
  std::vector<uint64_t> hash_stack;
  std::vector<int> rule50_stack;
  std::vector<int> repetition_stack;
  std::mt19937 rng;

  uint64_t local_cache_hits = 0;
  uint64_t local_cache_misses = 0;

  std::string cached_root_fen;

  SearchWorkerCtx() : rng(std::random_device{}()) {
    state_stack.reserve(256);
    move_stack.reserve(256);
    hash_stack.reserve(256);
    rule50_stack.reserve(256);
    repetition_stack.reserve(256);
  }

  void SetRootFen(const std::string &fen) {
    if (cached_root_fen == fen && !hash_stack.empty()) {
      ResetToRoot();
      return;
    }

    cached_root_fen = fen;
    LoadRootPosition();
  }

  void ResetToRoot() {
    while (!move_stack.empty())
      UndoMove();

    if (!hash_stack.empty())
      return;

    LoadRootPosition();
  }

  void LoadRootPosition() {
    state_stack.clear();
    move_stack.clear();
    hash_stack.clear();
    rule50_stack.clear();
    repetition_stack.clear();
    root_pos.set(cached_root_fen, false, &root_pos_st);
    pos.copy_from(root_pos, &root_st);
    hash_stack.push_back(pos.raw_key());
    rule50_stack.push_back(pos.rule50_count());
    repetition_stack.push_back(pos.repetition_distance());
  }

  void DoMove(Move m) {
    state_stack.emplace_back();
    pos.do_move(m, state_stack.back());
    move_stack.push_back(m);
    hash_stack.push_back(pos.raw_key());
    rule50_stack.push_back(pos.rule50_count());
    repetition_stack.push_back(pos.repetition_distance());
  }

  void UndoMove() {
    if (move_stack.empty())
      return;
    pos.undo_move(move_stack.back());
    move_stack.pop_back();
    state_stack.pop_back();
    hash_stack.pop_back();
    rule50_stack.pop_back();
    repetition_stack.pop_back();
  }

  // Position history (last 8 positions) for NN encoding.
  // HistoryBuffer must not be moved after population (raw StateInfo pointers).
  struct HistoryBuffer {
    static constexpr int kMaxHistory = 8;
    static constexpr int kInlineStateCapacity = 128;
    std::array<Position, kMaxHistory> positions;
    StateInfo root_states[kMaxHistory];
    std::array<StateInfo, kInlineStateCapacity> replay_inline_states;
    std::vector<StateInfo> replay_overflow_states;
    const Position *ptrs[kMaxHistory];
    int depth = 0;

    HistoryBuffer() = default;
    HistoryBuffer(const HistoryBuffer &) = delete;
    HistoryBuffer &operator=(const HistoryBuffer &) = delete;
    HistoryBuffer(HistoryBuffer &&) = delete;
    HistoryBuffer &operator=(HistoryBuffer &&) = delete;

    void PrepareReplayStateStack(int total_plies) {
      if (total_plies > kInlineStateCapacity) {
        replay_overflow_states.resize(total_plies);
      } else {
        replay_overflow_states.clear();
      }
    }

    StateInfo &ReplayStateAt(int ply, int total_plies) {
      if (total_plies > kInlineStateCapacity)
        return replay_overflow_states[ply];
      return replay_inline_states[ply];
    }
  };

  std::deque<HistoryBuffer> batch_histories;
  std::deque<HistoryBuffer> prefetch_histories;

  void BuildHistory(HistoryBuffer &buf) {
    int total_plies = static_cast<int>(move_stack.size());
    buf.depth = std::min(total_plies + 1, HistoryBuffer::kMaxHistory);
    int start_ply = total_plies - (buf.depth - 1);

    if (total_plies >= HistoryBuffer::kMaxHistory) {
      // For deep paths, only the NN history tail is needed. Walk the
      // live position back to the first retained ply, snapshot while
      // replaying forward, and leave ctx.pos restored at the leaf.
      for (int ply = total_plies; ply > start_ply; --ply) {
        pos.undo_move(move_stack[ply - 1]);
      }

      for (int h = 0; h < buf.depth; ++h) {
        buf.positions[h].copy_from(pos, &buf.root_states[h]);
        buf.ptrs[h] = &buf.positions[h];

        if (h + 1 < buf.depth) {
          const int move_index = start_ply + h;
          pos.do_move(move_stack[move_index], state_stack[move_index]);
        }
      }
      return;
    }

    Position replay;
    StateInfo replay_root;
    replay.copy_from(root_pos, &replay_root);
    buf.PrepareReplayStateStack(total_plies);

    int next_history = 0;
    for (int ply = 0; ply <= total_plies && next_history < buf.depth; ++ply) {
      if (ply == start_ply + next_history) {
        buf.positions[next_history].copy_from(replay,
                                              &buf.root_states[next_history]);
        buf.ptrs[next_history] = &buf.positions[next_history];
        ++next_history;
      }

      if (ply < total_plies) {
        replay.do_move(move_stack[ply], buf.ReplayStateAt(ply, total_plies));
      }
    }
  }

  std::unique_ptr<HistoryBuffer> BuildHistory() {
    auto buf = std::make_unique<HistoryBuffer>();
    BuildHistory(*buf);
    return buf;
  }

  uint64_t CurrentNNCacheKey(
      int cache_history_length = HistoryBuffer::kMaxHistory - 1) const {
    const int total_positions = static_cast<int>(hash_stack.size());
    if (total_positions <= 0)
      return ComputeNNCacheKeyFromState(nullptr, nullptr, nullptr, 0);

    const int requested_positions =
        std::clamp(cache_history_length + 1, 1, HistoryBuffer::kMaxHistory);
    const int depth = std::min(total_positions, requested_positions);
    const int start = total_positions - depth;
    return ComputeNNCacheKeyFromState(hash_stack.data() + start,
                                      rule50_stack.data() + start,
                                      repetition_stack.data() + start, depth);
  }
};

class Search {
public:
  using BestMoveCallback = std::function<void(Move best, Move ponder)>;
  using InfoCallback = std::function<void(const std::string &)>;

  Search(const SearchParams &params, std::unique_ptr<Backend> backend);
  ~Search();

  Search(const Search &) = delete;
  Search &operator=(const Search &) = delete;

  void StartSearch(const std::string &fen,
                   const ::MetalFish::Search::LimitsType &limits,
                   BestMoveCallback best_cb = nullptr,
                   InfoCallback info_cb = nullptr);
  void Stop();
  void PonderHit();
  void Wait();
  void ClearCallbacks();
  void NewGame();

  struct RootMoveStats {
    Move move = Move::none();
    float q = 0.0f;
    uint32_t visits = 0;
    float policy = 0.0f;
    uint32_t current_visits = 0;
  };

  Move GetBestMove() const;
  Move GetBestMoveWithTemperature(float temperature) const;
  float GetBestQ() const;
  RootMoveStats GetBestMoveStats() const;
  std::vector<RootMoveStats> GetRootMoveStats(int max_moves = 0) const;
  std::vector<Move> GetPV() const;
  const PipelineStats &Stats() const { return stats_; }

  void InjectPVBoost(const Move *pv, int pv_len, int ab_depth, float weight);

  void SetSharedTT(SharedTTReader *tt) { shared_tt_ = tt; }

private:
  void WorkerThreadMain(int thread_id);
  bool IsSearchActive() const;
  bool ShouldStop() const;
  SearchStats CollectSearchStats() const;
  void ConfigureStopper();
  void SendInfo();
  int64_t CalculateTimeBudget();
  RootMoveStats GetBestMoveStatsLocked() const;
  bool TryGetRootTablebaseMoveStatsLocked(RootMoveStats *out) const;

  void RunIteration(SearchWorkerCtx &ctx);
  void RunIterationSemaphore(SearchWorkerCtx &ctx);
  struct SelectedLeaf {
    Node *node = nullptr;
    int multivisit = 1;
  };
  SelectedLeaf SelectLeaf(SearchWorkerCtx &ctx);
  struct PuctResult {
    int best_idx;
    int visits_to_assign;
  };
  PuctResult SelectChildPuct(Node *node, bool is_root, SearchWorkerCtx &ctx);
  void Backpropagate(Node *node, float value, float draw, float moves_left,
                     int multivisit = 1);
  void CancelPathScoreUpdate(Node *leaf, int multivisit);
  void AddDirichletNoise(Node *root);
  void UpdateBackendLatencyMargin(int64_t elapsed_ms);
  void BuildRootSearchMoves(const Position &root_pos);
  void CreateLeafEdges(Node *leaf, const MoveList<LEGAL> &moves);
  Move FirstRootMoveOrLegal() const;
  void CaptureRootVisitBaselineLocked();
  uint32_t RootVisitBaselineLocked(Move move) const;

  void MaybePrefetchIntoCache(
      SearchWorkerCtx &ctx, BackendComputation *computation,
      std::deque<SearchWorkerCtx::HistoryBuffer> &prefetch_histories);
  int PrefetchIntoCache(
      Node *node, int budget, SearchWorkerCtx &ctx,
      BackendComputation *computation,
      std::deque<SearchWorkerCtx::HistoryBuffer> &prefetch_histories);

  static void ApplyNNPolicy(Node *node, const EvaluationResult &result,
                            float softmax_temp);
  float PolicySoftmaxTempForNode(const Node *node) const;

  SearchParams params_;
  std::unique_ptr<Backend> backend_;
  NodeTree tree_;

  std::vector<std::unique_ptr<BackendComputation>> computation_pool_;
  std::mutex pool_mutex_;

  std::unique_ptr<BackendComputation> AcquireComputation() {
    {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      if (!computation_pool_.empty()) {
        auto c = std::move(computation_pool_.back());
        computation_pool_.pop_back();
        c->Reset();
        return c;
      }
    }
    return backend_->CreateComputation();
  }

  void ReleaseComputation(std::unique_ptr<BackendComputation> c) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    computation_pool_.push_back(std::move(c));
  }

  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};
  std::atomic<bool> ponder_mode_active_{false};
  ::MetalFish::Search::LimitsType limits_;
  std::atomic<int64_t> search_start_ms_{0};
  std::atomic<int64_t> time_budget_ms_{0};
  std::atomic<uint64_t> nodes_at_movestart_{0};
  std::atomic<uint64_t> batches_at_movestart_{0};
  Color root_color_ = WHITE;
  bool root_search_filter_active_ = false;
  bool active_root_search_filter_active_ = false;
  std::atomic<bool> smart_pruning_enabled_{false};
  std::vector<Move> root_search_moves_;
  std::vector<Move> active_root_search_moves_;
  std::vector<std::pair<uint32_t, uint32_t>> root_visit_baseline_;

  BestMoveCallback best_move_cb_;
  InfoCallback info_cb_;

  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<SearchWorkerCtx>> worker_ctxs_;
  PipelineStats stats_;

  std::atomic<int> gathering_permit_{1};
  std::atomic<int> backend_waiting_{0};
  std::atomic<int64_t> backend_latency_margin_ms_{0};

  std::unique_ptr<ChainedStopper> stopper_;
  mutable StoppersHints latest_hints_;
  mutable std::mutex stopper_mutex_;
  mutable std::shared_mutex tree_structure_mutex_;

  mutable std::atomic<int64_t> first_eval_time_ms_{-1};

  // Lc0-style smooth time management state (persistent across searches)
  struct TimeManagerState {
    float nps = 20000.0f;
    bool nps_reliable = false;
    float tree_reuse = 0.52f;
    float timeuse = 0.70f;
    float avg_ms_per_move = 0.0f;
    float move_allocated_time_ms = 0.0f;
    int64_t last_move_final_nodes = 0;
    int64_t piggybank_ms = 0;
    bool first_move = true;
  };
  TimeManagerState tmgr_;

  SharedTTReader *shared_tt_ = nullptr;
};

bool MCTSIsKingsidePawnLever(const Position &pos, Move move);
bool MCTSIsMinorCentralPawnCapture(const Position &pos, Move move);
bool MCTSIsMinorKingPawnCheckCapture(const Position &pos, Move move);
bool MCTSIsMinorHighValueCapture(const Position &pos, Move move);
bool MCTSIsMinorCentralQuietMove(const Position &pos, Move move);
bool MCTSIsMinorQuietAttacksMajor(const Position &pos, Move move);
bool MCTSIsMinorQuietAttacksQueen(const Position &pos, Move move);
bool MCTSIsMinorFifthRankQuietMove(const Position &pos, Move move);
bool MCTSHasHeavyPieceOnSeventh(const Position &pos, Color us);
bool MCTSIsAdvancedPromotionSupportQueenMove(const Position &pos, Move move);
bool MCTSIsQuietQueenCheck(const Position &pos, Move move);
bool MCTSIsQuietQueenKingNetMove(const Position &pos, Move move);
bool MCTSRootHighPolicyLeverCandidate(
    uint32_t root_visits, uint32_t best_visits, uint32_t candidate_visits,
    float best_policy, float best_q, float candidate_policy, float candidate_q);
bool MCTSRootLowPolicyLeverCandidate(uint32_t root_visits, uint32_t best_visits,
                                     uint32_t candidate_visits,
                                     int candidate_rank, float best_policy,
                                     float best_q, float candidate_policy,
                                     float candidate_q);
bool MCTSRootLowPolicyLeverProbeCandidate(uint32_t root_visits,
                                          int candidate_policy_rank,
                                          float candidate_policy);
bool MCTSRootTinyLowVisitQOverrideCandidate(
    uint32_t root_visits, uint32_t best_visits, uint32_t candidate_visits,
    float best_policy, float best_q, float candidate_policy, float candidate_q);
bool MCTSRootTacticalCaptureProbeCandidate(uint32_t root_visits,
                                           int candidate_policy_rank,
                                           float candidate_policy);
bool MCTSRootHighValueCaptureProbeCandidate(uint32_t root_visits,
                                            int candidate_policy_rank,
                                            float candidate_policy);
bool MCTSRootTacticalQuietProbeCandidate(uint32_t root_visits,
                                         int candidate_policy_rank,
                                         float candidate_policy);
bool MCTSRootQuietMajorAttackProbeCandidate(uint32_t root_visits,
                                            int candidate_policy_rank,
                                            float candidate_policy);
bool MCTSRootDeepTacticalQuietProbeCandidate(uint32_t root_visits,
                                             int candidate_policy_rank,
                                             float candidate_policy);
bool MCTSRootFifthRankQuietProbeCandidate(uint32_t root_visits,
                                          int candidate_policy_rank,
                                          float candidate_policy);
bool MCTSRootFifthRankCurrentOverrideCandidate(
    uint32_t root_visits, uint32_t best_current_visits,
    uint32_t candidate_current_visits, float best_q, float candidate_q,
    float candidate_policy);
bool MCTSRootQuietQueenCheckProbeCandidate(uint32_t root_visits,
                                           int candidate_policy_rank,
                                           float candidate_policy);
bool MCTSRootQuietQueenKingNetProbeCandidate(uint32_t root_visits,
                                             int candidate_policy_rank,
                                             float candidate_policy);
bool MCTSRootAdvancedPromotionSupportCandidate(
    uint32_t root_visits, uint32_t best_visits, uint32_t candidate_visits,
    float best_policy, float best_q, float candidate_policy, float candidate_q);
bool MCTSRootPawnEndgameEnPassantCandidate(uint32_t root_visits,
                                           uint32_t best_visits,
                                           uint32_t candidate_visits,
                                           bool best_is_capture,
                                           bool candidate_is_en_passant,
                                           float best_q, float candidate_q);
bool MCTSRootMinorPawnEndgameCaptureProtected(
    const Position &pos, Move best_move, Move candidate_move, float best_policy,
    float best_q, float candidate_policy, float candidate_q);
bool MCTSRootLowVisitQOverrideCandidate(
    uint32_t best_visits, uint32_t candidate_visits, float best_q,
    float candidate_q, float near_equal_required_gap = 0.05f,
    float candidate_policy = 1.0f, bool allow_strong_gap_candidate = false);
bool MCTSRootHighPolicyVisitLeaderProtected(uint32_t best_visits,
                                            uint32_t candidate_visits,
                                            float best_policy, float best_q,
                                            float candidate_policy,
                                            float candidate_q);
bool MCTSRootClockLowVisitQOverrideCandidate(uint32_t root_current_visits,
                                             uint32_t best_current_visits,
                                             uint32_t candidate_current_visits,
                                             float best_q, float candidate_q,
                                             float candidate_policy);

std::unique_ptr<Search> CreateSearch(const SearchParams &config);

} // namespace MCTS
} // namespace MetalFish
