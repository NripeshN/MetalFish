# MCTS Phase 2: Hybrid Fix + Remaining Lc0 Gaps

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix hybrid regression (15/24 → 20/24+) and close remaining lc0 implementation gaps.

**Architecture:** Fix hybrid first (biggest regression), then systematic lc0 gap closure.

**Tech Stack:** C++20, Apple Silicon, CMake

**Build & Test:**
```bash
cd /Users/nripeshn/Documents/PythonPrograms/metalfish
cmake --build build -j$(sysctl -n hw.ncpu)
./build/metalfish_tests
python3 -u tests/bk_parity.py --engine metalfish --movetime 10000 --threads 2
python3 -u tests/paper_benchmarks.py  # Full validation
```

**Current state:** MCTS 20/24, AB 21/24, Hybrid 15/24 (regression), Lc0 20/24

---

## Task 1: Fix Hybrid Regression — Isolate MCTS Policy Injection

**Priority: CRITICAL — 15/24 → 20/24+**

The hybrid lost 5 positions (BK.08, BK.11, BK.15, BK.21, BK.23) because our MCTS changes (moves_left_slope=0.0027, new PUCT params, solidification) altered the policy priors that get injected into the AB search via `InjectPVBoost`. The MCTS component now gives different PV suggestions that lead the AB astray.

**Root cause analysis needed:** Read `src/hybrid/hybrid_search.cpp` function `update_mcts_policy_from_ab()` and `make_final_decision()`. The hybrid injects MCTS PV into AB policy every 50ms. With our MCTS changes, the PV is different (and potentially better for MCTS-style play but worse when mixed into AB).

**Fix options (test each):**
1. **Reduce PV boost weight** — lower `ab_policy_weight` from 0.3 to 0.1
2. **Only inject when MCTS agrees with AB** — skip injection if MCTS bestmove differs from AB
3. **Disable injection entirely** — let both engines run independently, only use MCTS for final decision tiebreaking
4. **Use MCTS Q for decision weighting, not PV injection** — the hybrid decision should weight based on MCTS evaluation strength, not inject MCTS moves into AB policy

**Files:** `src/hybrid/hybrid_search.cpp`, `src/hybrid/hybrid_search.h`

**Step 1:** Read `update_mcts_policy_from_ab()` and `make_final_decision()`
**Step 2:** Try option 3 first (disable PV injection, keep independent) — this is the safest and most principled approach
**Step 3:** Run `paper_benchmarks.py` on hybrid only to verify
**Step 4:** If hybrid is back to 20/24+, commit. If not, try option 1.

**Test:** `python3 -u tests/paper_benchmarks.py` — hybrid should be ≥20/24
**Commit:** `fix(hybrid): disable MCTS PV injection that regressed AB quality`

---

## Task 2: Smart Pruning Tolerances (Match Lc0)

**Priority: IMPORTANT — prevents premature stopping**

Lc0's smart pruning has safeguards our inline ShouldStop lacks:
- `first_eval_time_` — don't compute NPS until first NN eval returns
- `kSmartPruningToleranceMs = 200` — wait 200ms after first eval
- `kSmartPruningToleranceNodes = 300` — add 300 nodes to denominator
- `minimum_batches` — require minimum batch count

**Files:** `src/mcts/search.cpp` (ShouldStop function, lines 486-564)

**Step 1:** Add a `first_eval_time_` member to Search (set on first `total_nodes > 0`)
**Step 2:** In smart pruning section, add tolerance checks:
```cpp
// Skip smart pruning until 200ms after first eval
if (elapsed < first_eval_time_ms_ + 200) continue;
// Add tolerance nodes to NPS calculation
uint64_t adj_nodes = total + 300;
```
**Step 3:** Build, test with `bk_parity.py --movetime 10000 --threads 2`
**Commit:** `fix(mcts): add lc0-style smart pruning tolerances`

---

## Task 3: NNCache Thread Safety

**Priority: IMPORTANT — prevents torn reads with 2+ threads**

The NNCache `Lookup` and `Insert` have no synchronization. With 2 threads, concurrent read/write to the same slot can read a partially-written entry.

**Files:** `src/mcts/backend_adapter.h`, `src/mcts/backend_adapter.cpp`

**Step 1:** Add an `std::atomic<uint64_t> generation` field to each Entry
**Step 2:** On Insert: increment generation before write, increment again after
**Step 3:** On Lookup: read generation before and after — if they differ or odd, retry
**Step 4:** This is a seqlock pattern — fast reads, no mutex needed

**Test:** `./build/metalfish_tests` + `bk_parity.py --threads 2`
**Commit:** `fix(mcts): add seqlock to NNCache for thread-safe concurrent access`

---

## Task 4: Validate All Modes — No Regressions

**Priority: CRITICAL — final verification**

Run the complete `paper_benchmarks.py` and compare all scores:

| Engine | Target | Baseline |
|--------|--------|----------|
| MetalFish-AB | ≥20/24 | 20/24 |
| MetalFish-MCTS | ≥20/24 | 18/24 |
| MetalFish-Hybrid | ≥20/24 | 20/24 |

Also verify:
- All unit tests pass
- Hybrid doesn't crash with piped input
- All three modes produce bestmove in <15s

**Test:** Full `paper_benchmarks.py` run
**Commit:** No code change — validation only

---

## Summary

| Task | What | Target |
|------|------|--------|
| 1 | Fix hybrid regression | 15/24 → 20/24+ |
| 2 | Smart pruning tolerances | Match lc0 safeguards |
| 3 | NNCache thread safety | Prevent torn reads |
| 4 | Full validation | All modes ≥20/24 |

