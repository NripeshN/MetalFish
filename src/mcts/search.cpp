/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Search Pipeline Implementation - Optimized for Apple Silicon
  Licensed under GPL-3.0
*/

#include "search.h"
#include "core.h"

#include "../core/movegen.h"
#include "../uci/uci.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

namespace MetalFish {
namespace MCTS {

static_assert(sizeof(Node) >= sizeof(double), "Node must contain WL");
static_assert(sizeof(Node) <= 128, "Node exceeds two cache lines");
static_assert(alignof(Node) == 64, "Node alignment mismatch");

namespace {

constexpr int kMaxEdges = 256;
constexpr uint64_t kFNVOffset = 1469598103934665603ULL;
constexpr uint64_t kFNVPrime = 1099511628211ULL;

inline uint64_t Mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Hash only current position (not full history) to enable transposition hits across move orders.
uint64_t ComputePositionCacheKey(const Position* const* history, int count) {
    if (count <= 0 || !history[count - 1]) return kFNVOffset;
    const Position* current = history[count - 1];
    uint64_t key = kFNVOffset;
    key ^= Mix64(current->raw_key());
    key *= kFNVPrime;
    key ^= Mix64(static_cast<uint64_t>(
        static_cast<int64_t>(current->repetition_distance())));
    key *= kFNVPrime;
    return key;
}

bool IsInsufficientMaterial(const Position& pos) {
    const int pawns = popcount(pos.pieces(PAWN));
    const int rooks = popcount(pos.pieces(ROOK));
    const int queens = popcount(pos.pieces(QUEEN));
    if (pawns || rooks || queens) return false;

    const int wn = popcount(pos.pieces(WHITE, KNIGHT));
    const int bn = popcount(pos.pieces(BLACK, KNIGHT));
    const int wb = popcount(pos.pieces(WHITE, BISHOP));
    const int bb = popcount(pos.pieces(BLACK, BISHOP));
    const int minors = wn + bn + wb + bb;

    if (minors == 0) return true;   // K vs K
    if (minors == 1) return true;   // K+N/B vs K
    if (wb == 0 && bb == 0 && wn <= 2 && bn == 0) return true; // K+NN vs K
    if (wb == 0 && bb == 0 && bn <= 2 && wn == 0) return true; // K vs K+NN
    if (wn == 0 && bn == 0 && wb == 1 && bb == 1) return true; // KB vs KB

    return false;
}

bool ShouldAdjudicateRepetitionDraw(const Position& pos, int plies_from_root,
                                    bool two_fold_draws, float* moves_left_out,
                                    Node::Terminal* terminal_out) {
    const int rep = pos.repetition_distance();
    if (rep == 0) return false;

    if (rep < 0) {
        if (moves_left_out) *moves_left_out = 0.0f;
        if (terminal_out) *terminal_out = Node::Terminal::EndOfGame;
        return true;
    }

    if (!two_fold_draws) return false;

    if (plies_from_root >= 4 && plies_from_root >= rep) {
        if (moves_left_out) *moves_left_out = static_cast<float>(rep);
        if (terminal_out) *terminal_out = Node::Terminal::TwoFold;
        return true;
    }

    return false;
}

void ApplyNNPolicyToNode(Node* node, const EvaluationResult& result,
                         float softmax_temp) {
    const int num_edges = node->NumEdges();
    if (num_edges == 0) return;

    const float inv_temp = 1.0f / softmax_temp;
    const int n = std::min(num_edges, kMaxEdges);

    float logits_buf[kMaxEdges];
    float priors_buf[kMaxEdges];

    Edge* edges = node->Edges();

#ifdef __APPLE__
    for (int i = 0; i < n; ++i) {
        logits_buf[i] = result.get_policy(edges[i].move);
    }

    float max_logit;
    vDSP_maxv(logits_buf, 1, &max_logit, static_cast<vDSP_Length>(n));
    float neg_max = -max_logit;
    vDSP_vsadd(logits_buf, 1, &neg_max, logits_buf, 1, static_cast<vDSP_Length>(n));
    vDSP_vsmul(logits_buf, 1, &inv_temp, logits_buf, 1, static_cast<vDSP_Length>(n));
    int vn = n;
    vvexpf(priors_buf, logits_buf, &vn);
    float sum;
    vDSP_sve(priors_buf, 1, &sum, static_cast<vDSP_Length>(n));
#else
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        logits_buf[i] = result.get_policy(edges[i].move);
        if (logits_buf[i] > max_logit) max_logit = logits_buf[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        priors_buf[i] = FastMath::FastExp((logits_buf[i] - max_logit) * inv_temp);
        sum += priors_buf[i];
    }
#endif

    if (sum <= 0.0f) {
        const float uniform = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) edges[i].SetP(uniform);
        return;
    }

    const float inv_sum = 1.0f / sum;
#ifdef __APPLE__
    vDSP_vsmul(priors_buf, 1, &inv_sum, priors_buf, 1, static_cast<vDSP_Length>(n));
#else
    for (int i = 0; i < n; ++i) priors_buf[i] *= inv_sum;
#endif

    for (int i = 0; i < n; ++i) edges[i].SetP(priors_buf[i]);
}

} // anonymous namespace

// ============================================================================
// Search Lifecycle
// ============================================================================

Search::Search(const SearchParams& params, std::unique_ptr<Backend> backend)
    : params_(params), backend_(std::move(backend)) {
    if (!backend_) {
        std::string path = params_.nn_weights_path;
        if (path.empty()) {
            const char* env = std::getenv("METALFISH_NN_WEIGHTS");
            if (env) path = env;
        }
        if (!path.empty()) {
            try {
                backend_ = std::make_unique<Backend>(path);
                std::cerr << "[MCTS] Loaded transformer weights: "
                          << path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[MCTS] Failed to load weights ("
                          << path << "): " << e.what() << std::endl;
            }
        } else {
            std::cerr << "[MCTS] WARNING: No transformer weights path set. "
                      << "Set via UCI option NNWeights or env "
                      << "METALFISH_NN_WEIGHTS." << std::endl;
        }
    }

    // Warm up MPS graph with batch sizes 1 and 8 to pre-compile Metal shaders.
    if (backend_) {
        Position warmup_pos;
        StateInfo warmup_st;
        warmup_pos.set(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false, &warmup_st);
        {
            auto comp = backend_->CreateComputation();
            comp->AddInput(warmup_pos, warmup_pos.key());
            comp->ComputeBlocking();
        }
        {
            auto comp = backend_->CreateComputation();
            for (int i = 0; i < 8; ++i)
                comp->AddInput(warmup_pos, warmup_pos.key());
            comp->ComputeBlocking();
        }
    }
}

Search::~Search() {
    Stop();
    Wait();
}

void Search::StartSearch(const std::string& fen,
                         const ::MetalFish::Search::LimitsType& limits,
                         BestMoveCallback best_cb,
                         InfoCallback info_cb) {
    Stop();
    Wait();

    stats_.reset();
    stop_flag_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    limits_ = limits;
    best_move_cb_ = best_cb;
    info_cb_ = info_cb;
    search_start_ = std::chrono::steady_clock::now();

    if (!tree_.TryReuse(fen)) {
        tree_.Reset(fen);
    }

    {
        Position root_pos;
        StateInfo root_st;
        root_pos.set(fen, false, &root_st);
        root_color_ = root_pos.side_to_move();
    }

    time_budget_ms_ = CalculateTimeBudget();

    gathering_permit_.store(1, std::memory_order_relaxed);
    backend_waiting_.store(0, std::memory_order_relaxed);

    if (params_.use_kld_gain_stopper) {
        kld_stopper_ = std::make_unique<KLDGainStopper>(
            params_.kld_gain_min, params_.kld_gain_average_interval);
    } else {
        kld_stopper_.reset();
    }

    int num_threads = params_.GetNumThreads();
    worker_ctxs_.clear();
    workers_.clear();

    for (int i = 0; i < num_threads; ++i) {
        auto ctx = std::make_unique<SearchWorkerCtx>();
        ctx->worker_id = i;
        worker_ctxs_.push_back(std::move(ctx));
    }

    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&Search::WorkerThreadMain, this, i);
    }
}

void Search::Stop() {
    stop_flag_.store(true, std::memory_order_release);
}

namespace {
float ExponentialDecay(float from, float to, float halflife_steps, float steps) {
    return to - (to - from) * std::pow(0.5f, steps / halflife_steps);
}
} // namespace

void Search::Wait() {
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
    running_.store(false, std::memory_order_release);

    // Update time management state for next search
    auto elapsed = std::chrono::steady_clock::now() - search_start_;
    int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    uint64_t total_nodes = stats_.total_nodes.load(std::memory_order_relaxed);

    if (elapsed_ms > 0 && total_nodes > 0) {
        float actual_nps = 1000.0f * total_nodes / elapsed_ms;
        if (tmgr_.nps_reliable) {
            tmgr_.nps = tmgr_.nps * 0.5f + actual_nps * 0.5f;
        } else {
            tmgr_.nps = actual_nps;
            tmgr_.nps_reliable = true;
        }
    }

    if (tmgr_.move_allocated_time_ms > 0 && elapsed_ms > 0) {
        float actual_timeuse = static_cast<float>(elapsed_ms) / tmgr_.move_allocated_time_ms;
        float update_rate = tmgr_.avg_ms_per_move > 0 ?
            static_cast<float>(elapsed_ms) / tmgr_.avg_ms_per_move : 1.0f;
        tmgr_.timeuse = ExponentialDecay(tmgr_.timeuse, actual_timeuse,
                                         5.51f, update_rate);
        tmgr_.timeuse = std::max(tmgr_.timeuse, 0.34f);
    }

    tmgr_.last_move_final_nodes = static_cast<int64_t>(total_nodes);

    auto cb_copy = best_move_cb_;
    best_move_cb_ = nullptr;

    if (cb_copy) {
        Move best = GetBestMove();
        std::vector<Move> pv = GetPV();
        Move ponder = pv.size() > 1 ? pv[1] : Move::none();
        cb_copy(best, ponder);
    }
}

void Search::ClearCallbacks() {
    best_move_cb_ = nullptr;
    info_cb_ = nullptr;
}

bool Search::IsSearchActive() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// Time Management
// ============================================================================

namespace {

// Log-logistic distribution for estimating remaining moves in a game.
// From Lc0: midpoint=45.2, steepness=5.93 (fitted to game length data).
float EstimateMovesToGo(int ply, float midpoint = 45.2f,
                        float steepness = 5.93f) {
    float move = ply / 2.0f;
    return midpoint * std::pow(1.0f + 2.0f * std::pow(move / midpoint, steepness),
                               1.0f / steepness) - move;
}

} // namespace

int64_t Search::CalculateTimeBudget() {
    if (limits_.nodes > 0 && limits_.movetime <= 0 && !limits_.infinite &&
        limits_.time[WHITE] <= 0 && limits_.time[BLACK] <= 0) {
        return 0;
    }
    if (limits_.movetime > 0) return limits_.movetime;
    if (limits_.infinite) return 0;

    Color us = root_color_;
    int64_t time_left = limits_.time[us];
    int64_t inc = limits_.inc[us];
    if (time_left <= 0) return std::max(int64_t(50), inc);

    const int64_t overhead = static_cast<int64_t>(params_.move_overhead_ms);

    // Emergency: very low time, just use increment or tiny fraction
    if (time_left < 500) {
        return std::max(int64_t(50), std::min(time_left / 4, inc));
    }

    // Initialize piggybank on first move (9% of initial time, from Lc0)
    if (tmgr_.first_move) {
        tmgr_.piggybank_ms = static_cast<int64_t>(time_left * 0.09f);
        tmgr_.first_move = false;
    }

    // Update tree reuse estimate from previous search
    uint32_t current_nodes = tree_.Root() ? tree_.Root()->GetN() : 0;
    if (tmgr_.last_move_final_nodes > 0 && current_nodes > 0) {
        float this_reuse = static_cast<float>(current_nodes) /
                           static_cast<float>(tmgr_.last_move_final_nodes);
        float update_rate = tmgr_.avg_ms_per_move > 0.0f ?
            tmgr_.move_allocated_time_ms / tmgr_.avg_ms_per_move : 1.0f;
        tmgr_.tree_reuse = ExponentialDecay(tmgr_.tree_reuse, this_reuse,
                                            3.39f, update_rate);
        tmgr_.tree_reuse = std::min(tmgr_.tree_reuse, 0.73f);
    }

    // Estimate remaining moves using log-logistic distribution
    // Approximate game ply from the position (use tree root's move count)
    int ply = 2 * (limits_.movestogo > 0 ? 40 : 30);
    {
        Position pos;
        StateInfo st;
        std::string fen = tree_.RootFen();
        if (!fen.empty()) {
            pos.set(fen, false, &st);
            ply = pos.game_ply();
        }
    }

    float remaining_moves = EstimateMovesToGo(ply);
    if (limits_.movestogo > 0 && limits_.movestogo < remaining_moves) {
        remaining_moves = static_cast<float>(limits_.movestogo);
    }
    remaining_moves = std::max(remaining_moves, 1.0f);

    // Max piggybank: proportional to average time per move
    float max_piggybank = 36.5f *
        std::max(0.0f, static_cast<float>(time_left) +
                 static_cast<float>(inc) * (remaining_moves - 1.0f) - overhead) /
        remaining_moves;
    tmgr_.piggybank_ms = std::min(tmgr_.piggybank_ms,
                                  static_cast<int64_t>(max_piggybank));

    // Total remaining time (excluding piggybank reserve)
    float total_remaining = std::max(0.0f,
        static_cast<float>(time_left) - static_cast<float>(tmgr_.piggybank_ms) +
        static_cast<float>(inc) * (remaining_moves - 1.0f) -
        static_cast<float>(overhead));

    // Average nodes per move (using NPS estimate)
    float remaining_game_nodes = total_remaining * tmgr_.nps / 1000.0f;
    float avg_nodes_per_move = remaining_game_nodes / remaining_moves;
    tmgr_.avg_ms_per_move = total_remaining / remaining_moves;

    // Account for tree reuse: we get some nodes for free
    float nodes_with_reuse = avg_nodes_per_move / (1.0f - tmgr_.tree_reuse);
    float new_nodes_needed = std::max(0.0f, nodes_with_reuse - current_nodes);

    // Expected thinking time for this move
    float expected_ms = new_nodes_needed / tmgr_.nps * 1000.0f;

    // Save 12% of move time to piggybank (from Lc0)
    float to_piggybank = std::min(max_piggybank - tmgr_.piggybank_ms,
                                  expected_ms * 0.12f);
    expected_ms -= to_piggybank;

    // Adjust for smart pruning time use (we typically don't use all allocated time)
    tmgr_.move_allocated_time_ms = expected_ms / tmgr_.timeuse;

    // Cap at 42% of remaining time (from Lc0: max-move-budget=0.42)
    float max_move_time = static_cast<float>(time_left) * 0.42f;
    if (tmgr_.move_allocated_time_ms > max_move_time) {
        tmgr_.move_allocated_time_ms = max_move_time;
    }

    // Hard cap: never exceed remaining time minus overhead
    float hard_cap = static_cast<float>(time_left - overhead);
    if (tmgr_.move_allocated_time_ms > hard_cap) {
        tmgr_.move_allocated_time_ms = std::max(0.0f, hard_cap);
    }

    // Add piggybank to budget
    tmgr_.piggybank_ms += static_cast<int64_t>(to_piggybank);

    // Minimum: at least 50ms or 1/3 of remaining
    float min_time = std::min(50.0f, hard_cap / 3.0f);
    tmgr_.move_allocated_time_ms = std::max(tmgr_.move_allocated_time_ms, min_time);

    return static_cast<int64_t>(tmgr_.move_allocated_time_ms);
}

// ============================================================================
// Stop Conditions
// ============================================================================

bool Search::ShouldStop() const {
    if (stop_flag_.load(std::memory_order_acquire)) return true;

    if (time_budget_ms_ > 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - search_start_).count();
        if (elapsed >= time_budget_ms_) return true;

        if (params_.smart_pruning_factor > 0.0f &&
            stats_.total_nodes.load(std::memory_order_relaxed) % 32 == 0) {
            const Node* root = tree_.Root();
            if (root && root->NumEdges() > 0) {
                int num_edges = root->NumEdges();
                const Edge* edges = root->Edges();
                uint32_t best_n = 0, second_n = 0;
                for (int i = 0; i < num_edges; ++i) {
                    Node* child = edges[i].child.load(std::memory_order_relaxed);
                    if (child) {
                        uint32_t cn = child->GetN();
                        if (cn > best_n) { second_n = best_n; best_n = cn; }
                        else if (cn > second_n) { second_n = cn; }
                    }
                }
                float elapsed_frac = static_cast<float>(elapsed) /
                                     static_cast<float>(time_budget_ms_);
                uint64_t total = stats_.total_nodes.load(std::memory_order_relaxed);
                float est_remaining = static_cast<float>(total) *
                    (1.0f - elapsed_frac) / std::max(elapsed_frac, 0.01f);
                if (best_n > second_n + static_cast<uint32_t>(
                        est_remaining * params_.smart_pruning_factor))
                    return true;
            }
        }
    }

    if (kld_stopper_) {
        SearchStats kld_stats;
        kld_stats.total_nodes = stats_.total_nodes.load(std::memory_order_relaxed);
        const Node* root = tree_.Root();
        if (root && root->NumEdges() > 0) {
            const Edge* edges = root->Edges();
            for (int i = 0; i < root->NumEdges(); ++i) {
                Node* child = edges[i].child.load(std::memory_order_relaxed);
                kld_stats.edge_n.push_back(child ? child->GetN() : 0);
            }
            if (kld_stopper_->ShouldStop(kld_stats)) return true;
        }
    }

    if (limits_.nodes > 0 && stats_.total_nodes >= limits_.nodes) return true;

    return false;
}

// ============================================================================
// Worker Thread Entry Point
// ============================================================================

void Search::WorkerThreadMain(int thread_id) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    SearchWorkerCtx& ctx = *worker_ctxs_[thread_id];
    ctx.SetRootFen(tree_.RootFen());

    auto last_info = std::chrono::steady_clock::now();
    int num_threads = params_.GetNumThreads();
    bool use_semaphore = (num_threads > 1) && backend_;

    while (!ShouldStop()) {
        if (use_semaphore)
            RunIterationSemaphore(ctx, num_threads);
        else
            RunIteration(ctx);

        if (thread_id == 0) {
            auto now = std::chrono::steady_clock::now();
            auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - last_info).count();
            if (since >= 1000) {
                SendInfo();
                last_info = now;
            }
        }
    }

    stats_.cache_hits.fetch_add(ctx.local_cache_hits, std::memory_order_relaxed);
    stats_.cache_misses.fetch_add(ctx.local_cache_misses, std::memory_order_relaxed);
}

// ============================================================================
// Single-threaded iteration (no semaphore)
// ============================================================================

void Search::RunIteration(SearchWorkerCtx& ctx) {
    ctx.ResetToRoot();

    auto selected = SelectLeaf(ctx);
    Node* leaf = selected.node;
    int multivisit = std::max(1, selected.multivisit);
    if (!leaf) return;

    MoveList<LEGAL> moves(ctx.pos);
    if (moves.size() == 0) {
        bool in_check = ctx.pos.checkers() != 0;
        float value = in_check ? 1.0f : 0.0f;
        float draw = in_check ? 0.0f : 1.0f;
        leaf->MakeTerminal(Node::Terminal::EndOfGame, value, draw, 0.0f);
        leaf->FinalizeScoreUpdate(value, draw, 0.0f, multivisit);
        if (leaf->Parent())
            Backpropagate(leaf->Parent(), -value, draw, 1.0f, multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        return;
    }

    const bool is_root_leaf = (leaf == tree_.Root());
    if (!is_root_leaf) {
        if (IsInsufficientMaterial(ctx.pos) || ctx.pos.rule50_count() > 99) {
            leaf->MakeTerminal(Node::Terminal::EndOfGame, 0.0f, 1.0f, 0.0f);
            leaf->FinalizeScoreUpdate(0.0f, 1.0f, 0.0f, multivisit);
            if (leaf->Parent())
                Backpropagate(leaf->Parent(), 0.0f, 1.0f, 1.0f, multivisit);
            stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
            return;
        }

        float rep_moves_left = 0.0f;
        Node::Terminal rep_terminal = Node::Terminal::EndOfGame;
        const int plies_from_root = static_cast<int>(ctx.move_stack.size());
        if (ShouldAdjudicateRepetitionDraw(ctx.pos, plies_from_root,
                                           params_.two_fold_draws,
                                           &rep_moves_left, &rep_terminal)) {
            leaf->MakeTerminal(rep_terminal, 0.0f, 1.0f, rep_moves_left);
            leaf->FinalizeScoreUpdate(0.0f, 1.0f, rep_moves_left, multivisit);
            if (leaf->Parent())
                Backpropagate(leaf->Parent(), 0.0f, 1.0f, rep_moves_left + 1.0f,
                              multivisit);
            stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
            return;
        }
    }

    float value = 0.0f, draw = 0.0f, moves_left_val = 30.0f;

    if (leaf->NumEdges() == 0 && backend_) {
        auto history = ctx.BuildHistory();
        uint64_t cache_key = ComputePositionCacheKey(history.ptrs, history.depth);
        auto computation = AcquireComputation();
        auto add_result = computation->AddInputWithHistory(
            std::vector<const Position*>(history.ptrs, history.ptrs + history.depth), cache_key);

        if (add_result == BackendComputation::CACHE_HIT) {
            ctx.local_cache_hits++;
            const auto& result = computation->GetResult(0);
            if (leaf->NumEdges() == 0) {
                leaf->CreateEdges(moves);
                ApplyNNPolicyToNode(leaf, result, params_.policy_softmax_temp);
                leaf->SortEdges();
            }
            if (params_.add_dirichlet_noise && leaf == tree_.Root())
                AddDirichletNoise(leaf);
            value = -result.value;
            draw = result.has_wdl ? result.wdl[1] : 0.0f;
            moves_left_val = result.has_moves_left ? result.moves_left : 30.0f;
        } else {
            ctx.local_cache_misses++;
            computation->ComputeBlocking();
            const auto& result = computation->GetResult(0);
            if (leaf->NumEdges() == 0) {
                leaf->CreateEdges(moves);
                ApplyNNPolicyToNode(leaf, result, params_.policy_softmax_temp);
                leaf->SortEdges();
            }
            if (params_.add_dirichlet_noise && leaf == tree_.Root())
                AddDirichletNoise(leaf);
            value = -result.value;
            draw = result.has_wdl ? result.wdl[1] : 0.0f;
            moves_left_val = result.has_moves_left ? result.moves_left : 30.0f;
            stats_.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (leaf->NumEdges() > 0 && leaf->GetN() > 0) {
        // Collision: another thread already expanded this node.
        // Use its existing Q instead of backpropagating a fake 0.0 value.
        value = leaf->GetWL();
        draw = leaf->GetD();
        moves_left_val = leaf->GetM();
    } else if (leaf->NumEdges() > 0) {
        // Node expanded but not yet backed up -- cancel virtual loss, skip
        leaf->CancelScoreUpdate(multivisit);
        return;
    }

    Backpropagate(leaf, value, draw, moves_left_val, multivisit);
    stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
}

// ============================================================================
// Semaphore-based iteration (multi-threaded batched GPU eval)
// ============================================================================

void Search::RunIterationSemaphore(SearchWorkerCtx& ctx, int num_workers) {
    while (true) {
        if (ShouldStop()) return;
        int expected = 1;
        if (gathering_permit_.compare_exchange_weak(
                expected, 0, std::memory_order_acquire))
            break;
        CPU_PAUSE();
    }

    struct BatchEntry {
        Node* leaf;
        SearchWorkerCtx::HistoryBuffer history;
        int multivisit;
        int computation_idx;
    };

    std::vector<BatchEntry> local_batch;
    local_batch.reserve(params_.minibatch_size);

    std::unique_ptr<BackendComputation> computation;
    if (backend_) computation = AcquireComputation();

    int collision_events = 0;
    int collision_visits = 0;
    int out_of_order_count = 0;
    int max_out_of_order = params_.out_of_order_eval
        ? static_cast<int>(
              params_.minibatch_size * params_.max_out_of_order_evals_factor)
        : 0;
    uint32_t tree_size = tree_.Root() ? tree_.Root()->GetN() : 0;

    CollisionStats coll_stats;
    coll_stats.max_collision_events = params_.max_collision_events;
    coll_stats.max_collision_visits = params_.max_collision_visits;
    coll_stats.scaling_start = params_.max_collision_visits_scaling_start;
    coll_stats.scaling_end = params_.max_collision_visits_scaling_end;
    coll_stats.scaling_power = params_.max_collision_visits_scaling_power;
    int max_coll_visits = coll_stats.GetMaxCollisionVisits(tree_size);

    while (static_cast<int>(local_batch.size()) < params_.minibatch_size &&
           !ShouldStop()) {
        if (collision_events >= params_.max_collision_events ||
            collision_visits >= max_coll_visits)
            break;

        ctx.ResetToRoot();

        auto selected = SelectLeaf(ctx);
        Node* leaf = selected.node;
        int multivisit = std::max(1, selected.multivisit);
        if (!leaf) {
            collision_events++;
            collision_visits++;
            continue;
        }

        MoveList<LEGAL> moves(ctx.pos);
        if (moves.size() == 0) {
            bool in_check = ctx.pos.checkers() != 0;
            float v = in_check ? 1.0f : 0.0f;
            float d = in_check ? 0.0f : 1.0f;
            leaf->MakeTerminal(Node::Terminal::EndOfGame, v, d, 0.0f);
            leaf->FinalizeScoreUpdate(v, d, 0.0f, multivisit);
            if (leaf->Parent())
                Backpropagate(leaf->Parent(), -v, d, 1.0f, multivisit);
            stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
            out_of_order_count++;
            if (out_of_order_count >= max_out_of_order) break;
            continue;
        }

        const bool is_root_leaf = (leaf == tree_.Root());
        if (!is_root_leaf) {
            if (IsInsufficientMaterial(ctx.pos) || ctx.pos.rule50_count() > 99) {
                leaf->MakeTerminal(Node::Terminal::EndOfGame, 0.0f, 1.0f, 0.0f);
                leaf->FinalizeScoreUpdate(0.0f, 1.0f, 0.0f, multivisit);
                if (leaf->Parent())
                    Backpropagate(leaf->Parent(), 0.0f, 1.0f, 1.0f, multivisit);
                stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
                out_of_order_count++;
                if (out_of_order_count >= max_out_of_order) break;
                continue;
            }

            float rep_moves_left = 0.0f;
            Node::Terminal rep_terminal = Node::Terminal::EndOfGame;
            const int plies_from_root = static_cast<int>(ctx.move_stack.size());
            if (ShouldAdjudicateRepetitionDraw(ctx.pos, plies_from_root,
                                               params_.two_fold_draws,
                                               &rep_moves_left, &rep_terminal)) {
                leaf->MakeTerminal(rep_terminal, 0.0f, 1.0f, rep_moves_left);
                leaf->FinalizeScoreUpdate(0.0f, 1.0f, rep_moves_left, multivisit);
                if (leaf->Parent())
                    Backpropagate(leaf->Parent(), 0.0f, 1.0f, rep_moves_left + 1.0f,
                                  multivisit);
                stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
                out_of_order_count++;
                if (out_of_order_count >= max_out_of_order) break;
                continue;
            }
        }

        if (leaf->NumEdges() > 0 || !backend_) {
            collision_events++;
            collision_visits++;
            leaf->CancelScoreUpdate(multivisit);
            continue;
        }

        auto hist_buf_sem = ctx.BuildHistory();
        uint64_t cache_key = ComputePositionCacheKey(hist_buf_sem.ptrs, hist_buf_sem.depth);
        auto add_result = computation->AddInputWithHistory(
            std::vector<const Position*>(hist_buf_sem.ptrs, hist_buf_sem.ptrs + hist_buf_sem.depth), cache_key);

        if (add_result == BackendComputation::CACHE_HIT) {
            ctx.local_cache_hits++;
            const auto& result = computation->GetResult(computation->TotalInputs() - 1);
            if (leaf->NumEdges() == 0) {
                leaf->CreateEdges(moves);
                ApplyNNPolicyToNode(leaf, result, params_.policy_softmax_temp);
                leaf->SortEdges();
            }
            if (params_.add_dirichlet_noise && leaf == tree_.Root())
                AddDirichletNoise(leaf);
            float v = -result.value;
            float d = result.has_wdl ? result.wdl[1] : 0.0f;
            float ml = result.has_moves_left ? result.moves_left : 30.0f;
            Backpropagate(leaf, v, d, ml, multivisit);
            stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
            out_of_order_count++;
            if (out_of_order_count >= max_out_of_order) break;
            continue;
        }

        ctx.local_cache_misses++;
        BatchEntry entry;
        entry.leaf = leaf;
        entry.history = std::move(hist_buf_sem);
        entry.multivisit = multivisit;
        entry.computation_idx = computation->TotalInputs() - 1;
        local_batch.push_back(std::move(entry));
    }

    if (local_batch.empty()) {
        gathering_permit_.store(1, std::memory_order_release);
        return;
    }

    backend_waiting_.fetch_add(1, std::memory_order_release);
    gathering_permit_.store(1, std::memory_order_release);

    MaybePrefetchIntoCache(ctx, computation.get());

    try {
        computation->ComputeBlocking();
    } catch (...) {
        uint64_t failed_visits = 0;
        for (auto& entry : local_batch) {
            Backpropagate(entry.leaf, 0.0f, 1.0f, 30.0f, entry.multivisit);
            failed_visits += static_cast<uint64_t>(entry.multivisit);
        }
        backend_waiting_.fetch_sub(1, std::memory_order_relaxed);
        stats_.total_nodes.fetch_add(failed_visits, std::memory_order_relaxed);
        return;
    }

    backend_waiting_.fetch_sub(1, std::memory_order_relaxed);

    uint64_t total_visits = 0;
    for (auto& entry : local_batch) {
        const auto& result = computation->GetResult(entry.computation_idx);

        if (entry.leaf->NumEdges() == 0) {
            const Position& leaf_pos = *entry.history.ptrs[entry.history.depth - 1];
            MoveList<LEGAL> leaf_moves(leaf_pos);
            if (leaf_moves.size() > 0) {
                entry.leaf->CreateEdges(leaf_moves);
                ApplyNNPolicyToNode(entry.leaf, result, params_.policy_softmax_temp);
                entry.leaf->SortEdges();
                if (params_.add_dirichlet_noise && entry.leaf == tree_.Root())
                    AddDirichletNoise(entry.leaf);
            }
        }

        float v = -result.value;
        float d = result.has_wdl ? result.wdl[1] : 0.0f;
        float ml = result.has_moves_left ? result.moves_left : 30.0f;

        Backpropagate(entry.leaf, v, d, ml, entry.multivisit);
        total_visits += static_cast<uint64_t>(entry.multivisit);
    }

    stats_.total_nodes.fetch_add(total_visits, std::memory_order_relaxed);
    stats_.nn_evaluations.fetch_add(local_batch.size(), std::memory_order_relaxed);

    if (computation) ReleaseComputation(std::move(computation));
}

// ============================================================================
// PUCT Leaf Selection
// ============================================================================

Search::SelectedLeaf Search::SelectLeaf(SearchWorkerCtx& ctx) {
    Node* node = tree_.Root();
    int path_multivisit = std::max(1, params_.virtual_loss);

    while (node->NumEdges() > 0 && !node->IsTerminal()) {
        bool is_root = (node == tree_.Root());
        auto [best_idx, visits_to_assign] = SelectChildPuct(node, is_root, ctx);
        if (best_idx < 0) break;

        // Use multivisit from PUCT at root to reduce tree traversals
        if (is_root && visits_to_assign > 1) {
            path_multivisit = std::min(visits_to_assign, 128);
        }

        Edge& edge = node->Edges()[best_idx];

        Node* child = edge.child.load(std::memory_order_acquire);
        if (!child) {
            Node* new_child = tree_.AllocateNode(node, best_idx);
            Node* expected = nullptr;
            if (edge.child.compare_exchange_strong(
                    expected, new_child,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                child = new_child;
            } else {
                child = expected;
            }
        }

        if (!child->TryStartScoreUpdate(path_multivisit)) {
            break;
        }

        ctx.DoMove(edge.move);
        node = child;
    }

    return {node, path_multivisit};
}

Search::PuctResult Search::SelectChildPuct(Node* node, bool is_root, SearchWorkerCtx& ctx) {
    int num_edges = node->NumEdges();
    if (num_edges == 0) return {-1, 1};

    uint32_t parent_n = node->GetN() + node->GetNInFlight();
    float parent_q = parent_n > 0 ? -node->GetQ(-params_.draw_score) : 0.0f;

    float cpuct = params_.GetCpuct(is_root);
    float cpuct_base = params_.GetCpuctBase(is_root);
    float cpuct_factor = params_.GetCpuctFactor(is_root);
    float effective_cpuct = cpuct + cpuct_factor *
        std::log(
            (static_cast<float>(parent_n) + cpuct_base) / cpuct_base);

    uint32_t children_visits = node->GetChildrenVisits();
    float cpuct_sqrt_n = effective_cpuct *
        std::sqrt(static_cast<float>(std::max(children_visits, 1u)));

    float visited_policy = node->GetVisitedPolicy();
    float fpu;
    if (params_.GetFpuAbsolute(is_root)) {
        fpu = params_.GetFpuValue(is_root);
    } else {
        float reduction = is_root ? params_.fpu_reduction_at_root
                                  : params_.fpu_reduction;
        fpu = parent_q - reduction * std::sqrt(visited_policy);
    }

    MovesLeftEvaluator m_eval(params_, node->GetM(), parent_q);

    const Edge* edges = node->Edges();
    int best_idx = 0;
    float best_score = -1e9f;
    float second_best_score = -1e9f;
    int best_n_started = 0;
    float best_policy = 0.0f;

    PREFETCH(&edges[0]);
    if (num_edges > 1) PREFETCH(&edges[1]);

    for (int i = 0; i < num_edges; ++i) {
        if (i + 2 < num_edges) PREFETCH(&edges[i + 2]);

        const Edge& edge = edges[i];
        Node* child = edge.child.load(std::memory_order_acquire);

        // Early cutoff: edges sorted by descending policy, so once we see
        // two consecutive unvisited edges (no child pointer), all remaining
        // also unvisited and have lower policy — can't beat current best.
        if (!child && i > 0) {
            Node* prev = edges[i - 1].child.load(std::memory_order_relaxed);
            if (!prev) break;
        }

        float q, m_utility = 0.0f;
        float policy = edge.GetP();
        int n_started = 0;

        if (child) {
            uint32_t cn = child->GetN();
            uint32_t cn_flight = child->GetNInFlight();
            n_started = static_cast<int>(cn + cn_flight);

            q = (cn > 0) ? child->GetQ(params_.draw_score) : fpu;

            if (cn > 0 && m_eval.IsEnabled()) {
                m_utility = m_eval.GetMUtility(child->GetM(), q);
            }
        } else {
            q = fpu;
            m_utility = m_eval.GetDefaultMUtility();
        }

        float u = cpuct_sqrt_n * policy / (1.0f + static_cast<float>(n_started));
        float score = q + u + m_utility;

        if (score > best_score) {
            second_best_score = best_score;
            best_score = score;
            best_idx = i;
            best_n_started = n_started;
            best_policy = policy;
        } else if (score > second_best_score) {
            second_best_score = score;
        }
    }

    // Compute multivisit: how many visits can best child get before
    // second-best overtakes it? (Lc0's visits_to_perform logic)
    int visits_to_assign = 1;
    if (num_edges > 1 && second_best_score > -1e8f) {
        // After adding V visits to best, its U becomes:
        // cpuct_sqrt_n * best_policy / (1 + best_n_started + V)
        // We want: best_q + U_new >= second_best_score
        // U_new = cpuct_sqrt_n * best_policy / (1 + N + V)
        // V = cpuct_sqrt_n * best_policy / (second_best_score - best_q) - N - 1
        float best_q_component = best_score -
            cpuct_sqrt_n * best_policy / (1.0f + static_cast<float>(best_n_started));
        float margin = second_best_score - best_q_component;
        if (margin > 0.0f) {
            float v = cpuct_sqrt_n * best_policy / margin -
                      static_cast<float>(best_n_started) - 1.0f;
            visits_to_assign = std::max(1, static_cast<int>(v));
            // Cap at reasonable maximum
            visits_to_assign = std::min(visits_to_assign, 128);
        }
    }

    return {best_idx, visits_to_assign};
}

// ============================================================================
// Backpropagation
// ============================================================================

void Search::Backpropagate(Node* node, float value, float draw,
                           float moves_left, int multivisit) {
    const int visits = std::max(1, multivisit);
    while (node) {
        node->FinalizeScoreUpdate(value, draw, moves_left, visits);
        if (params_.sticky_endgames) node->MaybeSetBounds();
        if (!node->IsSolid() &&
            node->GetN() >= static_cast<uint32_t>(params_.solid_tree_threshold)) {
            node->MakeSolid();
        }
        value = -value;
        moves_left += 1.0f;
        node = node->Parent();
    }
}

// ============================================================================
// Prefetch into NN Cache
// ============================================================================

void Search::MaybePrefetchIntoCache(SearchWorkerCtx& ctx, BackendComputation* computation) {
    if (stop_flag_.load(std::memory_order_acquire)) return;
    if (!computation || computation->UsedBatchSize() <= 0) return;
    if (computation->UsedBatchSize() >= params_.max_prefetch) return;

    int budget = params_.max_prefetch - computation->UsedBatchSize();
    ctx.ResetToRoot();
    PrefetchIntoCache(tree_.Root(), budget, ctx, computation);
}

int Search::PrefetchIntoCache(Node* node, int budget, SearchWorkerCtx& ctx,
                               BackendComputation* computation) {
    if (budget <= 0 || !node || stop_flag_.load(std::memory_order_acquire)) return 0;

    if (node->GetNStarted() == 0) {
        if (node->NumEdges() == 0) return 0;
        auto history = ctx.BuildHistory();
        uint64_t key = ComputePositionCacheKey(history.ptrs, history.depth);
        computation->AddInputWithHistory(
            std::vector<const Position*>(history.ptrs, history.ptrs + history.depth), key);
        return 1;
    }

    if (node->GetN() == 0 || node->IsTerminal() || node->NumEdges() == 0) return 0;

    int num_edges = node->NumEdges();
    const Edge* edges = node->Edges();
    bool is_root = (node == tree_.Root());

    float cpuct = params_.GetCpuct(is_root);
    float cpuct_base = params_.GetCpuctBase(is_root);
    float cpuct_factor = params_.GetCpuctFactor(is_root);
    float effective_cpuct = cpuct + cpuct_factor *
        std::log((static_cast<float>(node->GetN()) + cpuct_base) / cpuct_base);
    float puct_mult = effective_cpuct *
        std::sqrt(static_cast<float>(std::max(node->GetChildrenVisits(), 1u)));

    float fpu;
    if (params_.GetFpuAbsolute(is_root)) {
        fpu = params_.GetFpuValue(is_root);
    } else {
        float reduction = is_root ? params_.fpu_reduction_at_root : params_.fpu_reduction;
        fpu = -node->GetQ(-params_.draw_score) - reduction * std::sqrt(node->GetVisitedPolicy());
    }

    int best_idx = -1;
    float best_score = -1e9f;
    for (int i = 0; i < num_edges && i < 8; ++i) {
        Node* child = edges[i].child.load(std::memory_order_acquire);
        float policy = edges[i].GetP();
        if (policy <= 0.0f) continue;
        float q = (child && child->GetN() > 0) ? child->GetQ(params_.draw_score) : fpu;
        int n_started = child ? child->GetNStarted() : 0;
        float u = puct_mult * policy / (1.0f + static_cast<float>(n_started));
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) return 0;

    Node* child = edges[best_idx].child.load(std::memory_order_acquire);
    ctx.DoMove(edges[best_idx].move);
    int spent = PrefetchIntoCache(child, budget, ctx, computation);
    ctx.UndoMove();

    return spent;
}

// ============================================================================
// Dirichlet Noise
// ============================================================================

void Search::AddDirichletNoise(Node* root) {
    int num_edges = root->NumEdges();
    if (num_edges == 0 || params_.noise_epsilon <= 0.0f) return;

    Edge* edges = root->Edges();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(params_.noise_alpha, 1.0f);

    std::vector<float> noise(num_edges);
    float noise_sum = 0.0f;
    for (int i = 0; i < num_edges; ++i) {
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }

    if (noise_sum < std::numeric_limits<float>::min()) return;

    for (int i = 0; i < num_edges; ++i) {
        float current = edges[i].GetP();
        float noisy = (1.0f - params_.noise_epsilon) * current +
                      params_.noise_epsilon * (noise[i] / noise_sum);
        edges[i].SetP(noisy);
    }
}

// ============================================================================
// Static NN Policy Application
// ============================================================================

void Search::ApplyNNPolicy(Node* node, const EvaluationResult& result,
                           float softmax_temp) {
    ApplyNNPolicyToNode(node, result, softmax_temp);
}

// ============================================================================
// Best Move / PV / Q Extraction
// ============================================================================

Move Search::GetBestMove() const {
    const Node* root = tree_.Root();
    if (!root || root->NumEdges() == 0) return Move::none();

    int num_edges = root->NumEdges();
    const Edge* edges = root->Edges();

    int best_idx = -1;
    uint32_t best_n = 0;
    float best_q = -2.0f;
    float best_m = 999.0f;
    bool best_is_terminal_win = false;

    for (int i = 0; i < num_edges; ++i) {
        Node* child = edges[i].child.load(std::memory_order_acquire);
        if (!child) continue;

        uint32_t cn = child->GetN();
        if (cn == 0) continue;

        float cq = child->GetWL();
        bool is_terminal = child->IsTerminal();
        bool is_win = is_terminal && cq > 0.5f;
        bool is_loss = is_terminal && cq < -0.5f;

        bool prefer = false;
        if (best_idx < 0) {
            prefer = true;
        } else if (is_win && !best_is_terminal_win) {
            prefer = true;
        } else if (is_win && best_is_terminal_win) {
            prefer = child->GetM() < best_m;
        } else if (!is_win && best_is_terminal_win) {
            prefer = false;
        } else if (is_loss) {
            prefer = false;
        } else {
            if (cn != best_n) prefer = cn > best_n;
            else if (std::abs(cq - best_q) > 0.001f) prefer = cq > best_q;
            else prefer = edges[i].GetP() > edges[best_idx].GetP();
        }

        if (prefer) {
            best_idx = i;
            best_n = cn;
            best_q = cq;
            best_m = child->GetM();
            best_is_terminal_win = is_win;
        }
    }

    if (best_idx < 0) {
        int best_policy_idx = 0;
        float best_policy = edges[0].GetP();
        for (int i = 1; i < num_edges; ++i) {
            float p = edges[i].GetP();
            if (p > best_policy) {
                best_policy = p;
                best_policy_idx = i;
            }
        }
        return edges[best_policy_idx].move;
    }
    return edges[best_idx].move;
}

std::vector<Move> Search::GetPV() const {
    std::vector<Move> pv;
    const Node* node = tree_.Root();

    while (node && node->NumEdges() > 0) {
        int num_edges = node->NumEdges();
        const Edge* edges = node->Edges();

        int best_idx = -1;
        uint32_t best_n = 0;
        for (int i = 0; i < num_edges; ++i) {
            Node* child = edges[i].child.load(std::memory_order_acquire);
            if (child && child->GetN() > best_n) {
                best_n = child->GetN();
                best_idx = i;
            }
        }

        if (best_idx < 0) break;
        pv.push_back(edges[best_idx].move);
        node = edges[best_idx].child.load(std::memory_order_acquire);
    }

    return pv;
}

float Search::GetBestQ() const {
    const Node* root = tree_.Root();
    if (!root || root->NumEdges() == 0) return 0.0f;

    int num_edges = root->NumEdges();
    const Edge* edges = root->Edges();

    int best_idx = -1;
    uint32_t best_n = 0;
    for (int i = 0; i < num_edges; ++i) {
        Node* child = edges[i].child.load(std::memory_order_acquire);
        if (child && child->GetN() > best_n) {
            best_n = child->GetN();
            best_idx = i;
        }
    }

    if (best_idx < 0) return root->GetWL();
    Node* best = edges[best_idx].child.load(std::memory_order_acquire);
    return best ? best->GetWL() : 0.0f;
}

// ============================================================================
// UCI Info Output
// ============================================================================

void Search::SendInfo() {
    if (!info_cb_) return;

    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - search_start_).count();

    uint64_t nn_evals = stats_.nn_evaluations.load(std::memory_order_relaxed);
    uint64_t nodes = stats_.total_nodes.load(std::memory_order_relaxed);
    uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

    float q = GetBestQ();
    int cp;
    if (std::abs(q) > 0.99f) {
        cp = q > 0 ? 10000 : -10000;
    } else {
        cp = static_cast<int>(111.714 * std::tan(1.5621 * q));
    }

    std::vector<Move> pv = GetPV();
    int depth = static_cast<int>(pv.size());
    if (depth < 1) depth = 1;

    std::ostringstream ss;
    ss << "info depth " << depth
       << " seldepth " << depth
       << " nodes " << nodes
       << " nps " << nps
       << " time " << elapsed_ms
       << " score cp " << cp
       << " string nn_evals " << nn_evals;

    if (!pv.empty()) {
        ss << " pv";
        for (const Move& m : pv) {
            ss << " " << UCIEngine::move(m, false);
        }
    }

    info_cb_(ss.str());
}

// ============================================================================
// PV Boost Injection (Hybrid AB/MCTS)
// ============================================================================

void Search::InjectPVBoost(const Move* pv, int pv_len, int ab_depth) {
    if (pv_len <= 0) return;

    float boost = std::min(1.0f, static_cast<float>(ab_depth) / 20.0f);
    Node* node = tree_.Root();

    for (int i = 0; i < pv_len && node && node->NumEdges() > 0; ++i) {
        Edge* edges = node->Edges();
        int num = node->NumEdges();
        bool found = false;

        float total = 0.0f;
        for (int e = 0; e < num; ++e) total += edges[e].GetP();
        if (total <= 0.0f) break;

        for (int e = 0; e < num; ++e) {
            if (edges[e].move == pv[i]) {
                float depth_boost = boost * (1.0f / (1.0f + 0.5f * i));
                float current = edges[e].GetP();
                float boosted = current * (1.0f + depth_boost);

                float new_total = total - current + boosted;
                if (new_total > 0.0f) {
                    float scale = total / new_total;
                    for (int j = 0; j < num; ++j) {
                        if (j != e) {
                            edges[j].SetP(edges[j].GetP() * scale);
                        } else {
                            edges[j].SetP(boosted * scale);
                        }
                    }
                }

                node = edges[e].child.load(std::memory_order_relaxed);
                found = true;
                break;
            }
        }
        if (!found) break;
    }
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Search> CreateSearch(const SearchParams& config) {
    std::unique_ptr<Backend> backend;
    if (!config.nn_weights_path.empty()) {
        try {
            backend = std::make_unique<Backend>(config.nn_weights_path);
        } catch (const std::exception& e) {
            std::cerr << "[MCTS] CreateSearch: backend creation failed: "
                      << e.what() << std::endl;
        }
    }
    return std::make_unique<Search>(config, std::move(backend));
}

} // namespace MCTS
} // namespace MetalFish
