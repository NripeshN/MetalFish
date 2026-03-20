# Hybrid Engine Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite hybrid engine with deep AB+MCTS integration via Shared TT. Target: hybrid ≥ AB on BK suite (≥20/24).

**Architecture:** AB-Primary with 4 integration points in Apple Silicon unified memory.

**Tech Stack:** C++20, Apple Silicon, Metal/MPS, CMake

**Build & Test:**
```bash
cd /Users/nripeshn/Documents/PythonPrograms/metalfish
cmake --build build -j$(sysctl -n hw.ncpu)
./build/metalfish_tests
python3 -u tests/bk_parity.py --engine metalfish --movetime 10000 --threads 2
```

---

## Task 1: Shared TT Reader for MCTS Leaf Evaluation

**The biggest win.** When MCTS expands a leaf node, check if AB's transposition table has a deep entry for this position. If yes, use AB's score (converted to WDL) instead of queuing an expensive GPU NN evaluation.

**Files:**
- Create: `src/hybrid/shared_tt.h` — SharedTT interface wrapping Stockfish's TT
- Modify: `src/mcts/search.h` — add SharedTT pointer to Search class
- Modify: `src/mcts/search.cpp` — check SharedTT before NN eval in RunIteration/RunIterationSemaphore
- Modify: `src/hybrid/hybrid_search.cpp` — pass TT reference to MCTS Search

**Step 1:** Create `src/hybrid/shared_tt.h`:
```cpp
#pragma once
#include "../search/tt.h"
#include "../core/position.h"

namespace MetalFish {
namespace MCTS {

class SharedTTReader {
public:
    explicit SharedTTReader(TranspositionTable* tt) : tt_(tt) {}

    // Try to get an AB evaluation for this position.
    // Returns true if a usable entry was found.
    // depth_threshold: minimum AB search depth to trust
    struct TTResult {
        float value;   // WDL value in [-1, 1]
        float draw;    // Draw probability
        int depth;     // AB search depth
        bool found;
    };

    TTResult Probe(const Position& pos, int depth_threshold = 6) const {
        TTResult result{0.0f, 0.0f, 0, false};
        if (!tt_) return result;

        auto [found, tte] = tt_->probe(pos.key());
        if (!found) return result;

        int entry_depth = tte->depth();
        if (entry_depth < depth_threshold) return result;

        // Convert centipawn score to WDL value
        int cp = tte->value();
        // Clamp to reasonable range
        cp = std::clamp(cp, -10000, 10000);

        // Logistic conversion: cp → win probability
        // Using standard Elo formula: P(win) = 1 / (1 + 10^(-cp/400))
        float win_prob = 1.0f / (1.0f + std::pow(10.0f, -cp / 400.0f));
        float loss_prob = 1.0f - win_prob;

        // Map to WDL: value = win - loss, draw = estimated from score magnitude
        float draw_est = std::max(0.0f, 1.0f - 2.0f * std::abs(win_prob - 0.5f));
        float w = win_prob * (1.0f - draw_est);
        float l = loss_prob * (1.0f - draw_est);

        result.value = w - l;  // In [-1, 1] range
        result.draw = draw_est;
        result.depth = entry_depth;
        result.found = true;
        return result;
    }

private:
    TranspositionTable* tt_ = nullptr;
};

}} // namespace
```

**Step 2:** Add `SharedTTReader* shared_tt_ = nullptr;` to MCTS `Search` class and a setter method.

**Step 3:** In `RunIteration` and `RunIterationSemaphore`, BEFORE the NN evaluation (before `BuildHistory` / `ComputePositionCacheKey`), add:
```cpp
// Check if AB has already evaluated this position deeply
if (shared_tt_) {
    auto tt_result = shared_tt_->Probe(ctx.pos, 8);
    if (tt_result.found) {
        // Use AB's evaluation instead of expensive NN eval
        if (leaf->NumEdges() == 0) {
            leaf->CreateEdges(moves);
            // Use uniform policy (no NN policy available from TT)
        }
        float v = -tt_result.value; // Negate for parent perspective
        float d = tt_result.draw;
        float ml = 30.0f; // Estimate
        Backpropagate(leaf, v, d, ml, multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        return; // Skip NN eval entirely
    }
}
```

**Step 4:** In `hybrid_search.cpp`, pass the TT to MCTS:
```cpp
mcts_search_->SetSharedTT(new SharedTTReader(&engine_->get_tt()));
```
(Need to expose TT from Engine — check if `engine_->tt` is accessible or add a getter.)

**Test:** `bk_parity.py --engine metalfish --movetime 10000 --threads 2` on hybrid → verify ≥19/24
**Commit:** `feat(hybrid): MCTS reads AB's transposition table at leaf nodes`

---

## Task 2: Single NN Inference for AB Root Move Ordering

At search start, run ONE transformer inference on the root position. The policy head produces a probability distribution over all legal moves. Use this to reorder AB's root moves — AB explores NN-preferred moves first, potentially finding the best move earlier and pruning more effectively.

**Files:**
- Modify: `src/hybrid/hybrid_search.cpp` — add policy inference at search start
- Modify: `src/hybrid/hybrid_search.h` — store policy hints

**Step 1:** In `start_search`, after position classification, run a single NN inference:
```cpp
// Single NN inference for AB root move ordering
if (mcts_search_ && mcts_search_->HasBackend()) {
    auto computation = mcts_search_->GetBackend()->CreateComputation();
    Position root_pos;
    StateInfo st;
    root_pos.set(root_fen_, false, &st);
    computation->AddInput(root_pos, root_pos.key());
    computation->ComputeBlocking();
    const auto& result = computation->GetResult(0);
    // Store policy hints for AB move ordering
    nn_policy_hints_.clear();
    for (const auto& [move, prob] : result.policy_priors) {
        nn_policy_hints_[move.raw()] = prob;
    }
}
```

**Step 2:** Expose `nn_policy_hints_` to the AB thread for root move ordering.
The AB thread calls `engine_->go()` which starts Stockfish's search. We need to inject the policy hints into the root move ordering. This is trickier — Stockfish's `MovePicker` determines move ordering. The simplest approach: sort the root moves in `Search::iterative_deepening` by NN policy before the first iteration.

Actually, a simpler approach: publish the top 5 NN-suggested moves as "root move hints" and have the hybrid coordinator set them as `searchmoves` for the AB engine if the NN strongly prefers certain moves. But this limits AB's search.

**Simplest effective approach:** The AB thread in `run_ab_search()` already calls `engine_->set_position()` then `engine_->go()`. We can't easily inject move ordering into Stockfish's internals without modifying `search.cpp`. Instead, log the policy hints for the final decision — if the NN's top policy move matches AB's best move, we have higher confidence in that move.

For now, skip the AB move ordering injection and just store the policy for the final decision. This is a future optimization.

**Step 3:** In `make_final_decision()`, if the NN policy strongly favors one move, use it as a tiebreaker.

**Test:** `paper_benchmarks.py` — hybrid should maintain ≥19/24
**Commit:** `feat(hybrid): single NN inference at root for policy hints`

---

## Task 3: MCTS Visit Counts → AB Root Awareness

The `MCTSSharedState.top_moves` array already publishes MCTS's top moves with visit counts and Q-values. The coordinator can use this data to inform the final decision — if MCTS explored a move heavily and AB explored the same move heavily, that's strong agreement.

**Files:**
- Modify: `src/hybrid/hybrid_search.cpp` — enhance `make_final_decision` with MCTS data

**Step 1:** In `make_final_decision`, read MCTS top moves and check if AB's best move appears in MCTS's top-3. If yes, boost confidence in that move. If AB's best move is NOT in MCTS's top-3, it may be a purely tactical move the NN doesn't like — still trust AB but flag lower confidence.

```cpp
// Check MCTS top moves for agreement
int mcts_top_count = mcts_state_.num_top_moves.load(std::memory_order_acquire);
bool ab_in_mcts_top3 = false;
float ab_mcts_q = 0.0f;
for (int i = 0; i < std::min(mcts_top_count, 3); ++i) {
    Move m = Move(mcts_state_.top_moves[i].move_raw.load(std::memory_order_relaxed));
    if (m == ab_best) {
        ab_in_mcts_top3 = true;
        ab_mcts_q = mcts_state_.top_moves[i].q.load(std::memory_order_relaxed);
        break;
    }
}
```

**Step 2:** Use this in the decision logic:
- If AB and MCTS agree (same best move) → AB move, maximum confidence
- If AB's best is in MCTS top-3 → AB move, high confidence  
- If AB's best is NOT in MCTS top-3 AND MCTS has >5000 visits → consider MCTS's top move
- Otherwise → AB move (safe default)

**Test:** `paper_benchmarks.py` — hybrid ≥19/24
**Commit:** `feat(hybrid): use MCTS visit data in final decision`

---

## Task 4: Rewrite Final Decision Logic

Replace the entire `make_final_decision()` with a clean, principled implementation:

**Files:**
- Modify: `src/hybrid/hybrid_search.cpp`

**The new decision logic:**

```cpp
Move ParallelHybridSearch::make_final_decision() {
    Move ab_best = ab_state_.get_best_move();
    Move mcts_best = mcts_state_.get_best_move();
    int ab_score = ab_state_.best_score.load(std::memory_order_relaxed);
    int ab_depth = ab_state_.completed_depth.load(std::memory_order_relaxed);
    float mcts_q = mcts_state_.best_q.load(std::memory_order_relaxed);
    uint32_t mcts_visits = mcts_state_.best_visits.load(std::memory_order_relaxed);

    // Rule 1: If only one engine has a result, use it
    if (ab_best == Move::none()) return mcts_best != Move::none() ? mcts_best : fallback();
    if (mcts_best == Move::none()) return ab_best;

    // Rule 2: If both agree, instant return (highest confidence)
    if (ab_best == mcts_best) return ab_best;

    // Rule 3: AB is primary. MCTS can only override with:
    //   a) Very high visit count (>10000 visits = reliable evaluation)
    //   b) Strong disagreement (>200cp difference)
    //   c) Position is NOT highly tactical (AB excels at tactics)
    bool mcts_reliable = mcts_visits > 10000;
    int mcts_cp = QToNnueScore(mcts_q);
    bool strong_disagreement = std::abs(mcts_cp - ab_score) > 200;
    bool is_tactical = (current_strategy_.position_type == PositionType::HIGHLY_TACTICAL ||
                        current_strategy_.position_type == PositionType::TACTICAL);

    if (mcts_reliable && strong_disagreement && !is_tactical && mcts_cp > ab_score + 200) {
        return mcts_best;
    }

    // Default: trust AB
    return ab_best;
}
```

This is drastically simpler and more conservative than the current 4-mode switch statement. The key insight: AB is almost always right, and MCTS should only override in extreme cases with overwhelming evidence.

**Test:** `paper_benchmarks.py` — hybrid ≥20/24 (should match or exceed pure AB)
**Commit:** `feat(hybrid): rewrite final decision — AB-primary with conservative MCTS override`

---

## Task 5: Full Validation + Benchmark Comparison

**No code changes — validation only.**

Run the complete paper_benchmarks.py and compare:

| Engine | Target | Old Baseline | Phase 1 Result |
|--------|--------|-------------|----------------|
| MetalFish-AB | ≥20/24 | 20/24 | 21/24 |
| MetalFish-MCTS | ≥19/24 | 18/24 | 20/24 |
| MetalFish-Hybrid | **≥20/24** | 20/24 | ? |

Also verify:
- Unit tests pass
- Hybrid doesn't crash
- All modes produce bestmove reliably
- NPS is reasonable (hybrid NPS ≥ AB NPS since AB is primary)

**Test:** Full `python3 -u tests/paper_benchmarks.py`
**Compare:** With `results/paper_summary.md`

---

## Summary

| Task | What | Key Innovation |
|------|------|---------------|
| 1 | Shared TT reader | MCTS gets AB's deep evals for free |
| 2 | Single NN inference | AB gets NN policy hints at root |
| 3 | MCTS visits → decision | Agreement detection for confidence |
| 4 | Rewrite final decision | AB-primary, conservative MCTS override |
| 5 | Full validation | Verify hybrid ≥ AB on all benchmarks |

