/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Node and Tree - Optimized for Apple Silicon

  Licensed under GPL-3.0
*/

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "../core/movegen.h"
#include "../core/position.h"
#include "../core/types.h"

#ifdef __APPLE__
#include <os/lock.h>
#else
#include <mutex>

struct os_unfair_lock_s {
    std::mutex mtx;
};
using os_unfair_lock_t = os_unfair_lock_s*;
using os_unfair_lock = os_unfair_lock_s;

#define OS_UNFAIR_LOCK_INIT os_unfair_lock_s{}
inline void os_unfair_lock_lock(os_unfair_lock_t lock) { lock->mtx.lock(); }
inline void os_unfair_lock_unlock(os_unfair_lock_t lock) { lock->mtx.unlock(); }
#endif

namespace MetalFish {
namespace MCTS {

#ifdef __APPLE__
constexpr size_t CACHE_LINE_SIZE = 128;
#else
constexpr size_t CACHE_LINE_SIZE = 64;
#endif

// Forward declarations
class Node;

// MCTS Edge with compressed policy and embedded child pointer (TSEdge pattern).
// Edges are packed contiguously for cache-friendly PUCT scans.
struct Edge {
    Move move = Move::none();
    uint16_t p_ = 0;
    std::atomic<Node*> child{nullptr};

    Edge() = default;
    Edge(Move m, float p) : move(m), child(nullptr) { SetP(p); }

    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;

    Edge(Edge&& other) noexcept
        : move(other.move),
          p_(other.p_),
          child(other.child.load(std::memory_order_relaxed)) {}

    Edge& operator=(Edge&& other) noexcept {
        move = other.move;
        p_ = other.p_;
        child.store(other.child.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
        return *this;
    }

    // 16-bit compressed policy: store bits 27..12 of IEEE 754 float
    void SetP(float p) {
        constexpr int32_t roundings = (1 << 11) - (3 << 28);
        int32_t tmp;
        std::memcpy(&tmp, &p, sizeof(float));
        tmp += roundings;
        p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
    }

    float GetP() const {
        uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
        float ret;
        std::memcpy(&ret, &tmp, sizeof(uint32_t));
        return ret;
    }
};

// MCTS Node - cache-line aligned, ~64 bytes core data on ARM64.
// Children are stored as atomic<Node*> inside each Edge, matching the old
// TSEdge pattern for lock-free child installation via CAS.
class alignas(64) Node {
public:
    enum class Terminal : uint8_t {
        NonTerminal = 0,
        EndOfGame   = 1,
        Tablebase   = 2,
        TwoFold     = 3
    };

    explicit Node(Node* parent = nullptr, int edge_idx = -1)
        : parent_(parent),
          index_(edge_idx >= 0 ? static_cast<uint16_t>(edge_idx) : 0) {}

    ~Node() {
        if (solid_children_ && solid_base_) {
            for (int i = 0; i < num_edges_; ++i) {
                solid_base_[i].~Node();
            }
            std::allocator<Node> alloc;
            alloc.deallocate(solid_base_, num_edges_);
            solid_base_ = nullptr;
        }
    }

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    // --- Visit counts ---

    uint32_t GetN() const { return n_.load(std::memory_order_acquire); }

    uint32_t GetNInFlight() const {
        return n_in_flight_.load(std::memory_order_acquire);
    }

    int GetNStarted() const {
        return static_cast<int>(GetN()) + static_cast<int>(GetNInFlight());
    }

    uint32_t GetChildrenVisits() const {
        uint32_t n = GetN();
        return n > 0 ? n - 1 : 0;
    }

    // --- Value accessors ---

    float GetQ(float draw_score = 0.0f) const {
        return static_cast<float>(wl_.load(std::memory_order_acquire)) +
               draw_score * d_.load(std::memory_order_acquire);
    }

    float GetWL() const {
        return static_cast<float>(wl_.load(std::memory_order_acquire));
    }
    float GetD() const { return d_.load(std::memory_order_acquire); }
    float GetM() const { return m_.load(std::memory_order_acquire); }

    // Sum of policy priors for children that have been visited (N > 0)
    float GetVisitedPolicy() const {
        float dominated = 0.0f;
        for (int i = 0; i < num_edges_; ++i) {
            Node* ch = edges_[i].child.load(std::memory_order_acquire);
            if (ch && ch->GetN() > 0) {
                dominated += edges_[i].GetP();
            }
        }
        return dominated;
    }

    // --- Score update ---

    // Collision-aware virtual loss. Returns false when another thread is
    // already expanding this node (N==0 && N_in_flight>0), signalling
    // a collision the caller should back off from.
    bool TryStartScoreUpdate(int multivisit = 1) {
        const uint32_t visits = static_cast<uint32_t>(std::max(1, multivisit));
        uint32_t n = n_.load(std::memory_order_acquire);
        if (n == 0 && n_in_flight_.load(std::memory_order_acquire) > 0) {
            return false;
        }
        n_in_flight_.fetch_add(visits, std::memory_order_acq_rel);
        return true;
    }

    void CancelScoreUpdate(int count = 1) {
        uint32_t to_sub = static_cast<uint32_t>(std::max(1, count));
        uint32_t cur = n_in_flight_.load(std::memory_order_acquire);
        while (true) {
            const uint32_t next = (cur > to_sub) ? (cur - to_sub) : 0;
            if (n_in_flight_.compare_exchange_weak(
                    cur, next, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                break;
            }
        }
    }

    // Running-average backpropagation with double-precision WL accumulation.
    void FinalizeScoreUpdate(float v, float d, float m, int multivisit = 1) {
#ifdef __APPLE__
        os_unfair_lock_lock(&score_lock_);
#else
        std::lock_guard<std::mutex> _guard(score_lock_);
#endif
        const uint32_t visits = static_cast<uint32_t>(std::max(1, multivisit));
        const uint32_t old_n = n_.load(std::memory_order_relaxed);
        const double total = static_cast<double>(old_n + visits);

        const double old_wl = wl_.load(std::memory_order_relaxed);
        const float old_d = d_.load(std::memory_order_relaxed);
        const float old_m = m_.load(std::memory_order_relaxed);

        wl_.store((old_wl * old_n + static_cast<double>(v) * visits) / total,
                  std::memory_order_release);
        d_.store(static_cast<float>((static_cast<double>(old_d) * old_n +
                                     static_cast<double>(d) * visits) / total),
                 std::memory_order_release);
        m_.store(static_cast<float>((static_cast<double>(old_m) * old_n +
                                     static_cast<double>(m) * visits) / total),
                 std::memory_order_release);

        n_.store(old_n + visits, std::memory_order_release);

        uint32_t cur = n_in_flight_.load(std::memory_order_acquire);
        while (true) {
            const uint32_t next = (cur > visits) ? (cur - visits) : 0;
            if (n_in_flight_.compare_exchange_weak(
                    cur, next, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                break;
            }
        }
#ifdef __APPLE__
        os_unfair_lock_unlock(&score_lock_);
#endif
    }

    // Propagate proven terminal bounds up the tree ("sticky endgames").
    // If every child is terminal, this node becomes terminal too.
    void MaybeSetBounds() {
        if (num_edges_ == 0) return;

        bool all_terminal = true;
        bool any_loss = false;
        bool any_draw = false;
        float best_wl = -2.0f;
        float best_d  = 0.0f;
        float best_m  = 999.0f;

        for (int i = 0; i < num_edges_; ++i) {
            Node* ch = edges_[i].child.load(std::memory_order_acquire);
            if (!ch || !ch->IsTerminal()) {
                all_terminal = false;
                break;
            }
            float child_wl = ch->GetWL();
            float child_d  = ch->GetD();
            float child_m  = ch->GetM() + 1.0f;

            if (child_wl > best_wl) {
                best_wl = child_wl;
                best_d  = child_d;
                best_m  = child_m;
            }
            if (child_wl < 0.0f) any_loss = true;
            if (child_d > 0.5f) any_draw = true;
        }

        if (!all_terminal) return;

        if (best_wl > 0.0f) {
            MakeTerminal(Terminal::EndOfGame, -best_wl, best_d, best_m);
        } else if (any_draw) {
            MakeTerminal(Terminal::EndOfGame, 0.0f, 1.0f, best_m);
        } else if (any_loss) {
            MakeTerminal(Terminal::EndOfGame, -best_wl, best_d, best_m);
        }
    }

    // --- Edge / children management ---

    void CreateEdges(const MoveList<LEGAL>& moves) {
        int count = static_cast<int>(moves.size());
        if (count == 0) return;
        edges_ = std::make_unique<Edge[]>(count);
        float uniform = 1.0f / static_cast<float>(count);
        int idx = 0;
        for (const auto& m : moves) {
            edges_[idx].move = m;
            edges_[idx].SetP(uniform);
            edges_[idx].child.store(nullptr, std::memory_order_relaxed);
            ++idx;
        }
        num_edges_ = static_cast<uint8_t>(count);
    }

    void SortEdges() {
        if (!edges_ || num_edges_ <= 1) return;
        std::sort(edges_.get(), edges_.get() + num_edges_,
                  [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
    }

    int NumEdges() const { return num_edges_; }
    Edge* Edges() { return edges_.get(); }
    const Edge* Edges() const { return edges_.get(); }

    // --- Tree navigation ---

    Node* Parent() const { return parent_; }
    int EdgeIndex() const { return static_cast<int>(index_); }
    void DetachFromParentForReuse() {
        parent_ = nullptr;
        index_ = 0;
    }

    // --- Terminal state ---

    bool IsTerminal() const {
        return terminal_type_.load(std::memory_order_acquire) != 0;
    }

    Terminal GetTerminalType() const {
        return static_cast<Terminal>(
            terminal_type_.load(std::memory_order_acquire));
    }

    void MakeTerminal(Terminal type, float wl, float d, float m) {
#ifdef __APPLE__
        os_unfair_lock_lock(&score_lock_);
#else
        std::lock_guard<std::mutex> _guard(score_lock_);
#endif
        wl_.store(static_cast<double>(wl), std::memory_order_release);
        d_.store(d, std::memory_order_release);
        m_.store(m, std::memory_order_release);
        terminal_type_.store(static_cast<uint8_t>(type), std::memory_order_release);
#ifdef __APPLE__
        os_unfair_lock_unlock(&score_lock_);
#endif
    }

    // Legacy helper: sets WL from a single value, D=0, M=0
    void SetTerminal(Terminal type, float value) {
        MakeTerminal(type, value, 0.0f, 0.0f);
    }

    void RevertTerminal() {
        terminal_type_.store(static_cast<uint8_t>(Terminal::NonTerminal),
                             std::memory_order_release);
    }

    void MaybeRevertTwoFold(int depth_from_root) {
        if (GetTerminalType() != Terminal::TwoFold) return;
        if (depth_from_root < 4) {
            RevertTerminal();
        }
    }

    // --- Node solidification ---

    bool IsSolid() const { return solid_children_; }

    void TakeEdgesFrom(Node& other) {
        edges_ = std::move(other.edges_);
        num_edges_ = other.num_edges_;
        other.num_edges_ = 0;
    }

    bool MakeSolid() {
        if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;

        for (int i = 0; i < num_edges_; ++i) {
            Node* ch = edges_[i].child.load(std::memory_order_acquire);
            if (ch && ch->GetN() == 0 && ch->GetNInFlight() > 0) return false;
        }

        std::allocator<Node> alloc;
        Node* arr = alloc.allocate(num_edges_);
        for (int i = 0; i < num_edges_; ++i) {
            new (&arr[i]) Node(this, i);
            Node* old = edges_[i].child.load(std::memory_order_acquire);
            if (old) {
                if (old->GetN() > 0) {
                    arr[i].FinalizeScoreUpdate(old->GetWL(), old->GetD(),
                                               old->GetM(),
                                               static_cast<int>(old->GetN()));
                }
                if (old->NumEdges() > 0) {
                    arr[i].TakeEdgesFrom(*old);
                }
                if (old->IsTerminal()) {
                    arr[i].MakeTerminal(old->GetTerminalType(), old->GetWL(),
                                        old->GetD(), old->GetM());
                }
            }
            edges_[i].child.store(&arr[i], std::memory_order_release);
        }

        solid_base_ = arr;
        solid_children_ = true;
        return true;
    }

    // --- Legacy accessors (compatibility with old ThreadSafeNode API) ---

    uint32_t n() const { return GetN(); }
    float q() const { return GetWL(); }
    float d() const { return GetD(); }
    float m() const { return GetM(); }
    bool is_terminal() const { return IsTerminal(); }

private:
    std::atomic<double>       wl_{0.0};           // 8  bytes - Win-Loss (double precision)
    std::unique_ptr<Edge[]>   edges_;              // 8  bytes
    Node*                     parent_ = nullptr;   // 8  bytes
    std::atomic<float>        d_{0.0f};            // 4  bytes - Draw probability
    std::atomic<float>        m_{0.0f};            // 4  bytes - Moves left estimate
    std::atomic<uint32_t>     n_{0};               // 4  bytes - Visit count
    std::atomic<uint32_t>     n_in_flight_{0};     // 4  bytes - In-flight virtual visits
    uint16_t                  index_ = 0;          // 2  bytes - Edge index in parent
    uint8_t                   num_edges_ = 0;      // 1  byte  - Number of edges
    std::atomic<uint8_t>      terminal_type_{0};   // 1  byte  - Terminal enum
#ifdef __APPLE__
    mutable os_unfair_lock    score_lock_ = OS_UNFAIR_LOCK_INIT;  // 4 bytes
#else
    mutable std::mutex        score_lock_;
#endif
    Node*                     solid_base_ = nullptr;     // 8 bytes — contiguous child array
    bool                      solid_children_ = false;   // 1 byte  — children are solidified
    // Total: 48 + 8 + 1 + 7 padding = 64 bytes on Apple
};

// ============================================================================
// NodeGarbageCollector - Background deallocation of pruned subtrees
// ============================================================================

class NodeGarbageCollector {
public:
    NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

    ~NodeGarbageCollector() {
        stop_.store(true, std::memory_order_release);
        if (gc_thread_.joinable()) gc_thread_.join();
    }

    NodeGarbageCollector(const NodeGarbageCollector&) = delete;
    NodeGarbageCollector& operator=(const NodeGarbageCollector&) = delete;

    void AddToQueue(std::unique_ptr<Node> node) {
        if (!node) return;
        std::lock_guard<std::mutex> lock(gc_mutex_);
        gc_queue_.push_back(std::move(node));
    }

private:
    void Worker() {
        while (!stop_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ProcessQueue();
        }
        ProcessQueue();
    }

    void ProcessQueue() {
        while (true) {
            std::unique_ptr<Node> item;
            {
                std::lock_guard<std::mutex> lock(gc_mutex_);
                if (gc_queue_.empty()) return;
                item = std::move(gc_queue_.back());
                gc_queue_.pop_back();
            }
            item.reset();
        }
    }

    std::thread gc_thread_;
    std::atomic<bool> stop_{false};
    std::mutex gc_mutex_;
    std::vector<std::unique_ptr<Node>> gc_queue_;
};

inline NodeGarbageCollector& GetNodeGC() {
    static NodeGarbageCollector gc;
    return gc;
}

// ============================================================================
// NodeTree - Arena-allocated tree with reuse support
// ============================================================================

class NodeTree {
public:
    NodeTree() { AllocateNewArena(); }
    ~NodeTree() = default;

    NodeTree(const NodeTree&) = delete;
    NodeTree& operator=(const NodeTree&) = delete;

    // Full reset: discard the entire tree and start from the given FEN.
    void Reset(const std::string& fen) {
        arenas_.clear();
        current_arena_.store(0, std::memory_order_relaxed);
        AllocateNewArena();
        if (root_heap_) GetNodeGC().AddToQueue(std::move(root_heap_));
        root_arena_.reset();
        root_is_arena_ = false;

        root_heap_ = std::make_unique<Node>();
        {
            std::unique_lock<std::shared_mutex> lock(fen_mutex_);
            root_fen_ = fen;
        }
    }

    // Attempt 1–2 ply lookahead reuse: if the new FEN is reachable from the
    // current root via one of its expanded children (or grandchildren),
    // re-root the tree instead of discarding it. Returns true on success.
    bool TryReuse(const std::string& new_fen) {
        Node* r = Root();
        if (!r || r->NumEdges() == 0) return false;

        std::string old_fen = RootFen();
        if (old_fen.empty()) return false;

        Position new_pos;
        StateInfo ns;
        new_pos.set(new_fen, false, &ns);
        uint64_t target_key = new_pos.key();

        Edge* edges = r->Edges();
        int num = r->NumEdges();

        for (int i = 0; i < num; ++i) {
            Node* child = edges[i].child.load(std::memory_order_acquire);
            if (!child) continue;

            Position test;
            StateInfo ts, ts2;
            test.set(old_fen, false, &ts);
            test.do_move(edges[i].move, ts2);

            if (test.key() == target_key) {
                edges[i].child.store(nullptr, std::memory_order_relaxed);
                child->DetachFromParentForReuse();
                if (root_heap_) GetNodeGC().AddToQueue(std::move(root_heap_));
                root_arena_.reset(child);
                root_is_arena_ = true;
                {
                    std::unique_lock<std::shared_mutex> lock(fen_mutex_);
                    root_fen_ = new_fen;
                }
                return true;
            }

            if (child->NumEdges() > 0) {
                Edge* ce = child->Edges();
                for (int j = 0; j < child->NumEdges(); ++j) {
                    Node* gc = ce[j].child.load(std::memory_order_acquire);
                    if (!gc) continue;
                    StateInfo ts3;
                    test.do_move(ce[j].move, ts3);
                    if (test.key() == target_key) {
                        ce[j].child.store(nullptr, std::memory_order_relaxed);
                        gc->DetachFromParentForReuse();
                        if (root_heap_) GetNodeGC().AddToQueue(std::move(root_heap_));
                        root_arena_.reset(gc);
                        root_is_arena_ = true;
                        {
                            std::unique_lock<std::shared_mutex> lock(fen_mutex_);
                            root_fen_ = new_fen;
                        }
                        return true;
                    }
                    test.undo_move(ce[j].move);
                }
            }
        }
        return false;
    }

    Node* Root() {
        return root_is_arena_ ? root_arena_.get() : root_heap_.get();
    }

    const Node* Root() const {
        return root_is_arena_ ? root_arena_.get() : root_heap_.get();
    }

    std::string RootFen() const {
        std::shared_lock<std::shared_mutex> lock(fen_mutex_);
        return root_fen_;
    }

    // Arena allocation: O(1) amortised, thread-safe.
    Node* AllocateNode(Node* parent, int edge_idx) {
        while (true) {
            size_t arena_idx = current_arena_.load(std::memory_order_acquire);
            if (arena_idx < arenas_.size()) {
                NodeArena& arena = *arenas_[arena_idx];
                size_t slot = arena.next.fetch_add(1, std::memory_order_acq_rel);
                if (slot < ARENA_SIZE) {
                    Node* n = &arena.nodes[slot];
                    new (n) Node(parent, edge_idx);
                    return n;
                }
            }
            // Current arena is full – allocate a new one under lock.
#ifdef __APPLE__
            os_unfair_lock_lock(&arena_lock_);
#else
            os_unfair_lock_lock(&arena_lock_);
#endif
            if (current_arena_.load(std::memory_order_acquire) == arena_idx) {
                AllocateNewArena();
                current_arena_.store(arenas_.size() - 1,
                                     std::memory_order_release);
            }
#ifdef __APPLE__
            os_unfair_lock_unlock(&arena_lock_);
#else
            os_unfair_lock_unlock(&arena_lock_);
#endif
        }
    }

private:
    // Root can live on the heap (initial) or in an arena (after reuse).
    struct NoDelete { void operator()(Node*) const {} };
    std::unique_ptr<Node>            root_heap_;
    std::unique_ptr<Node, NoDelete>  root_arena_;
    bool                             root_is_arena_ = false;

    std::string                      root_fen_;
    mutable std::shared_mutex        fen_mutex_;

    // Arena pool
    static constexpr size_t ARENA_SIZE = 8192;

    struct NodeArena {
        std::unique_ptr<Node[]> nodes;
        std::atomic<size_t>     next{0};

        NodeArena() : nodes(std::make_unique<Node[]>(ARENA_SIZE)) {}
    };

    std::vector<std::unique_ptr<NodeArena>> arenas_;
    std::atomic<size_t>                     current_arena_{0};

#ifdef __APPLE__
    os_unfair_lock arena_lock_ = OS_UNFAIR_LOCK_INIT;
#else
    os_unfair_lock arena_lock_ = OS_UNFAIR_LOCK_INIT;
#endif

    void AllocateNewArena() {
        arenas_.push_back(std::make_unique<NodeArena>());
    }
};

} // namespace MCTS
} // namespace MetalFish
