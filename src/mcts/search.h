/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Search Pipeline - Optimized for Apple Silicon
  Licensed under GPL-3.0
*/

#pragma once

#include "node.h"
#include "search_params.h"
#include "backend_adapter.h"
#include "stoppers.h"
#include "core.h"

#include "../search/search.h"
#include "../uci/uci.h"

#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <functional>
#include <sstream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <pthread/qos.h>
#endif

#ifdef __aarch64__
#define CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define PREFETCH(addr) \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#else
#define PREFETCH(addr) (void)(addr)
#endif

namespace MetalFish {
namespace MCTS {

// ============================================================================
// Search statistics exposed to UCI layer
// ============================================================================

struct PipelineStats {
    std::atomic<uint64_t> total_nodes{0};
    std::atomic<uint64_t> nn_evaluations{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};

    void reset() {
        total_nodes.store(0, std::memory_order_relaxed);
        nn_evaluations.store(0, std::memory_order_relaxed);
        cache_hits.store(0, std::memory_order_relaxed);
        cache_misses.store(0, std::memory_order_relaxed);
    }
};

// ============================================================================
// Worker context - thread-local state for each search thread
// ============================================================================

struct SearchWorkerCtx {
    int worker_id = 0;
    Position pos;
    StateInfo root_st;
    std::vector<StateInfo> state_stack;
    std::vector<Move> move_stack;
    std::vector<uint64_t> hash_stack;
    std::mt19937 rng;

    uint64_t local_cache_hits = 0;
    uint64_t local_cache_misses = 0;

    std::string cached_root_fen;

    SearchWorkerCtx() : rng(std::random_device{}()) {
        state_stack.reserve(256);
        move_stack.reserve(256);
        hash_stack.reserve(256);
    }

    void SetRootFen(const std::string& fen) {
        cached_root_fen = fen;
        ResetToRoot();
    }

    void ResetToRoot() {
        state_stack.clear();
        move_stack.clear();
        hash_stack.clear();
        pos.set(cached_root_fen, false, &root_st);
        hash_stack.push_back(pos.raw_key());
    }

    void DoMove(Move m) {
        state_stack.emplace_back();
        pos.do_move(m, state_stack.back());
        move_stack.push_back(m);
        hash_stack.push_back(pos.raw_key());
    }

    // Build position history (last 8 positions) for NN encoding.
    // Reconstructs intermediate positions from the move path.
    // Caller owns the returned buffer and must keep it alive
    // while the history pointers are in use.
    struct HistoryBuffer {
        static constexpr int kMaxHistory = 8;
        std::unique_ptr<Position> positions[kMaxHistory];
        StateInfo root_states[kMaxHistory];
        std::vector<StateInfo> state_stacks[kMaxHistory];
        const Position* ptrs[kMaxHistory];
        int depth = 0;

        HistoryBuffer() {
            for (int i = 0; i < kMaxHistory; ++i)
                positions[i] = std::make_unique<Position>();
        }

        HistoryBuffer(const HistoryBuffer&) = delete;
        HistoryBuffer& operator=(const HistoryBuffer&) = delete;
        HistoryBuffer(HistoryBuffer&&) = default;
        HistoryBuffer& operator=(HistoryBuffer&&) = default;
    };

    HistoryBuffer BuildHistory() {
        HistoryBuffer buf;
        int total_plies = static_cast<int>(move_stack.size());
        buf.depth = std::min(total_plies + 1, HistoryBuffer::kMaxHistory);
        int start_ply = total_plies - (buf.depth - 1);

        for (int h = 0; h < buf.depth; ++h) {
            int target_ply = start_ply + h;
            buf.positions[h]->set(cached_root_fen, false, &buf.root_states[h]);
            buf.state_stacks[h].clear();
            buf.state_stacks[h].reserve(target_ply);
            for (int p = 0; p < target_ply; ++p) {
                buf.state_stacks[h].emplace_back();
                buf.positions[h]->do_move(move_stack[p],
                                           buf.state_stacks[h].back());
            }
            buf.ptrs[h] = buf.positions[h].get();
        }
        return buf;
    }
};

// ============================================================================
// MCTS Search Engine
// ============================================================================

class Search {
public:
    using BestMoveCallback = std::function<void(Move best, Move ponder)>;
    using InfoCallback = std::function<void(const std::string&)>;

    Search(const SearchParams& params, std::unique_ptr<Backend> backend);
    ~Search();

    Search(const Search&) = delete;
    Search& operator=(const Search&) = delete;

    void StartSearch(const std::string& fen,
                     const ::MetalFish::Search::LimitsType& limits,
                     BestMoveCallback best_cb = nullptr,
                     InfoCallback info_cb = nullptr);
    void Stop();
    void Wait();
    void ClearCallbacks();

    Move GetBestMove() const;
    float GetBestQ() const;
    std::vector<Move> GetPV() const;
    const PipelineStats& Stats() const { return stats_; }

    void InjectPVBoost(const Move* pv, int pv_len, int ab_depth);

private:
    void WorkerThreadMain(int thread_id);
    bool IsSearchActive() const;
    bool ShouldStop() const;
    void SendInfo();
    int64_t CalculateTimeBudget();

    // Core MCTS algorithms
    void RunIteration(SearchWorkerCtx& ctx);
    void RunIterationSemaphore(SearchWorkerCtx& ctx, int num_workers);
    struct SelectedLeaf {
        Node* node = nullptr;
        int multivisit = 1;
    };
    SelectedLeaf SelectLeaf(SearchWorkerCtx& ctx);
    struct PuctResult { int best_idx; int visits_to_assign; };
    PuctResult SelectChildPuct(Node* node, bool is_root, SearchWorkerCtx& ctx);
    void Backpropagate(Node* node, float value, float draw, float moves_left,
                       int multivisit = 1);
    void AddDirichletNoise(Node* root);

    // NN policy application
    static void ApplyNNPolicy(Node* node, const EvaluationResult& result,
                              float softmax_temp);

    SearchParams params_;
    std::unique_ptr<Backend> backend_;
    NodeTree tree_;

    // BackendComputation pool to avoid per-iteration heap allocation
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
    ::MetalFish::Search::LimitsType limits_;
    std::chrono::steady_clock::time_point search_start_;
    int64_t time_budget_ms_ = 0;
    Color root_color_ = WHITE;

    BestMoveCallback best_move_cb_;
    InfoCallback info_cb_;

    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<SearchWorkerCtx>> worker_ctxs_;
    PipelineStats stats_;

    std::atomic<int> gathering_permit_{1};
    std::atomic<int> backend_waiting_{0};

    std::unique_ptr<KLDGainStopper> kld_stopper_;

    // Lc0-style smooth time management (persistent across searches)
    struct TimeManagerState {
        float nps = 200.0f;
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
};

// Factory function matching the old create_thread_safe_mcts pattern
std::unique_ptr<Search> CreateSearch(const SearchParams& config);

} // namespace MCTS
} // namespace MetalFish
