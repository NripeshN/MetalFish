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
        std::vector<std::unique_ptr<Position>> positions;
        std::vector<std::unique_ptr<StateInfo>> root_states;
        std::vector<std::vector<StateInfo>> state_stacks;
        std::vector<const Position*> ptrs;
    };

    HistoryBuffer BuildHistory() const {
        HistoryBuffer buf;
        int total_plies = static_cast<int>(move_stack.size());
        int history_depth = std::min(total_plies + 1, 8);
        int start_ply = total_plies - (history_depth - 1);

        buf.positions.reserve(history_depth);
        buf.root_states.reserve(history_depth);
        buf.state_stacks.resize(history_depth);
        buf.ptrs.reserve(history_depth);

        for (int h = 0; h < history_depth; ++h) {
            int target_ply = start_ply + h;
            buf.positions.push_back(std::make_unique<Position>());
            buf.root_states.push_back(std::make_unique<StateInfo>());
            buf.positions.back()->set(cached_root_fen, false,
                                       buf.root_states.back().get());
            buf.state_stacks[h].reserve(target_ply);
            for (int p = 0; p < target_ply; ++p) {
                buf.state_stacks[h].emplace_back();
                buf.positions.back()->do_move(move_stack[p],
                                               buf.state_stacks[h].back());
            }
            buf.ptrs.push_back(buf.positions.back().get());
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
    int64_t CalculateTimeBudget() const;

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
};

// Factory function matching the old create_thread_safe_mcts pattern
std::unique_ptr<Search> CreateSearch(const SearchParams& config);

} // namespace MCTS
} // namespace MetalFish
