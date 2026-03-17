# Codebase Cleanup & Production Readiness Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all dead code, redundant files, unused modules, and consolidate duplicated patterns to make the MetalFish codebase production-ready.

**Architecture:** The codebase has three search modes: Alpha-Beta (Stockfish-derived), MCTS (Lc0-style transformer), and Hybrid (parallel AB+MCTS). After the recent MCTS rewrite achieving Lc0 parity, several legacy modules, duplicate parameter structs, dead functions, and stale test infrastructure remain. This plan systematically removes them layer by layer, verifying the build and tests after each task.

**Tech Stack:** C++17, Objective-C++ (.mm), Metal shaders, CMake, Python 3 (test scripts)

---

## Task 1: Delete Dead MCTS Legacy Modules

These files are compiled by CMakeLists.txt but provide no symbols used by any active code path. `apple_silicon.h` is only included by its own `.cpp`. `accumulator_cache.h` is only included by its own `.cpp`.

**Files:**
- Delete: `src/mcts/apple_silicon.h`
- Delete: `src/mcts/apple_silicon.cpp`
- Delete: `src/eval/accumulator_cache.h`
- Delete: `src/eval/accumulator_cache.cpp`
- Modify: `CMakeLists.txt` (remove from MCTS_SOURCES and EVAL_SOURCES)

**Step 1: Remove from CMakeLists.txt**

In CMakeLists.txt, remove `src/mcts/apple_silicon.cpp` from `MCTS_SOURCES` and `src/eval/accumulator_cache.cpp` from `EVAL_SOURCES`.

**Step 2: Delete the files**

```bash
rm src/mcts/apple_silicon.h src/mcts/apple_silicon.cpp
rm src/eval/accumulator_cache.h src/eval/accumulator_cache.cpp
```

**Step 3: Build to verify**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -5
```

Expected: Clean build with 0 errors.

**Step 4: Run unit tests**

```bash
./build/metalfish_tests
```

Expected: All 5 sections pass.

**Step 5: Commit**

```bash
git add -A && git commit -m "chore: remove dead apple_silicon and accumulator_cache modules"
```

---

## Task 2: Delete Dead ab_bridge Module

The entire `ab_bridge.h/.cpp` module is dead in production. `ParallelHybridSearch` uses the real `Engine` directly for AB search, never touching `ABSearcher`, `ABPolicyGenerator`, `TacticalAnalyzer`, or `HybridSearchBridge`. The `initialize_hybrid_bridge()` function is never called. Only `shutdown_hybrid_bridge()` is called from `main.cpp` (as a no-op since the bridge is never initialized).

**Files:**
- Delete: `src/hybrid/ab_bridge.h`
- Delete: `src/hybrid/ab_bridge.cpp`
- Modify: `CMakeLists.txt` (remove `src/hybrid/ab_bridge.cpp` from HYBRID_SOURCES)
- Modify: `src/main.cpp` (remove `shutdown_hybrid_bridge()` call and its include)
- Modify: `src/hybrid/hybrid_search.h` (remove `#include "ab_bridge.h"`)

**Step 1: Remove include from hybrid_search.h**

In `src/hybrid/hybrid_search.h`, remove the line `#include "ab_bridge.h"`.

**Step 2: Remove shutdown call from main.cpp**

In `src/main.cpp`, remove the `MCTS::shutdown_hybrid_bridge()` call and its associated include/forward declaration.

**Step 3: Remove from CMakeLists.txt**

Remove `src/hybrid/ab_bridge.cpp` from `HYBRID_SOURCES`.

**Step 4: Delete the files**

```bash
rm src/hybrid/ab_bridge.h src/hybrid/ab_bridge.cpp
```

**Step 5: Build to verify**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -5
```

Expected: Clean build. If `test_hybrid.cpp` references `ABSearcher` or `HybridSearchBridge`, those test sections must also be removed (see Task 7).

**Step 6: Run unit tests**

```bash
./build/metalfish_tests
```

Expected: All sections pass (hybrid tests may need adjustment — handle in Task 7).

**Step 7: Commit**

```bash
git add -A && git commit -m "chore: remove dead ab_bridge module (superseded by ParallelHybridSearch)"
```

---

## Task 3: Remove Empty node.cpp Translation Unit

`src/mcts/node.cpp` is essentially empty — just boilerplate comments and two `static_assert`s. All methods live inline in `node.h`. Move the static_asserts to `search.cpp` (which already includes `node.h`) and delete the file.

**Files:**
- Delete: `src/mcts/node.cpp`
- Modify: `src/mcts/search.cpp` (add the two static_asserts at the top)
- Modify: `CMakeLists.txt` (remove `src/mcts/node.cpp` from MCTS_SOURCES)

**Step 1: Read current static_asserts from node.cpp**

Check what's there. Expected: two `static_assert`s about `Node` size and cache alignment.

**Step 2: Add static_asserts to search.cpp**

Add them inside the `namespace MetalFish { namespace MCTS {` block at the top of `search.cpp`, after the includes.

**Step 3: Remove from CMakeLists.txt and delete**

Remove `src/mcts/node.cpp` from MCTS_SOURCES. Delete the file.

**Step 4: Build and test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

Expected: Clean build, all tests pass.

**Step 5: Commit**

```bash
git add -A && git commit -m "chore: remove empty node.cpp, move static_asserts to search.cpp"
```

---

## Task 4: Consolidate Duplicate MCTS Parameter Structs

`src/mcts/core.h` defines `MCTSSearchParams` and `src/mcts/search_params.h` defines `SearchParams` — same fields, slightly different defaults (e.g., cpuct 1.75 vs 1.745). `MCTSSearchParams` is only used to construct `MovesLeftEvaluator`. Consolidate by making `MovesLeftEvaluator` accept `SearchParams` directly, then remove the duplicate struct.

**Files:**
- Modify: `src/mcts/core.h` (remove `MCTSSearchParams`, update `MovesLeftEvaluator` to accept `SearchParams`)
- Modify: `src/mcts/search.cpp` (update `MovesLeftEvaluator` construction to pass `SearchParams` directly)

**Step 1: Update MovesLeftEvaluator in core.h**

Change `MovesLeftEvaluator`'s constructor to accept `const SearchParams&` (add `#include "search_params.h"` to `core.h`). Map fields: `params.moves_left_weight` → `weight_`, `params.moves_left_max_effect` → `max_effect_`, `params.moves_left_slope` → `slope_`, `params.moves_left_mid_point` → `mid_`, `params.moves_left_quadratic_factor` → `quadratic_factor_`.

**Step 2: Remove MCTSSearchParams struct**

Delete the entire `MCTSSearchParams` struct from `core.h`.

**Step 3: Update search.cpp**

Where `MCTSSearchParams temp;` is constructed and passed to `MovesLeftEvaluator`, replace with passing `params_` directly.

**Step 4: Build and test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

**Step 5: Commit**

```bash
git add -A && git commit -m "refactor: consolidate MCTSSearchParams into SearchParams"
```

---

## Task 5: Prune Dead Code from core.h

After Task 4, `core.h` still contains many dead functions that have been superseded by inline implementations in `search.cpp` and `node.h`. Remove all functions that have zero callers outside `core.h` itself.

**Dead functions to remove from `src/mcts/core.h`:**
- `ComputeCpuct()` (superseded by `Search::SelectChildPuct`)
- `ComputeFpu()` (superseded by `Search::SelectChildPuct`)
- `ComputePuctScore()` (superseded by `Search::SelectChildPuct`)
- `SelectBestChildPuct()` (superseded by `Search::SelectChildPuct`)
- `ApplyDirichletNoise()` (superseded by `Search::AddDirichletNoise`)
- `CompareEdgesForBestMove()`, `GetEdgeRank()`, `EdgeRank` enum (superseded by `Search::GetBestMove`)
- `FinalizeScoreUpdate()`, `FinalizeScoreUpdateAtomic()` (superseded by `Node::FinalizeScoreUpdate`)
- `CalculateTimeForMove()`, `TimeManagerParams` (superseded by `Search::CalculateTimeBudget`)
- `CanTerminateEarly()`, `EarlyTerminationParams` (superseded by `Search::ShouldStop`)
- `QToCentipawns()`, `ComputeWDL()`, `WDLRescale()`, `WDLRescaleParams`
- `NnueScoreToQ()`, `QToNnueScore()` (only used by deleted apple_silicon.cpp)
- `ShouldSolidify()`, `SOLID_TREE_THRESHOLD`, `NodeUpdateParams`

**Keep alive in core.h:**
- `FastMath` namespace (used by `MovesLeftEvaluator`)
- `MovesLeftEvaluator` class
- `CollisionStats` struct

**Step 1: Delete all dead functions/structs listed above**

**Step 2: Build and test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

If any build errors occur, a function was actually referenced — add it back.

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: remove dead algorithm functions from core.h"
```

---

## Task 6: Remove Legacy Aliases from gpu_constants.h

Lines 46-52 of `src/eval/gpu_constants.h` define 6 legacy aliases (`NNUE_FEATURE_DIM_BIG`, `NNUE_FEATURE_DIM_SMALL`, `MAX_BATCH_SIZE`, `MAX_FEATURES_PER_POSITION`, `HALFKA_DIMS`, `PSQT_DIMS`) that are never used in actual code — only `GPU_*` prefixed versions are used.

**Files:**
- Modify: `src/eval/gpu_constants.h` (remove lines 44-52, the legacy alias block and its comment)

**Step 1: Remove the legacy aliases block**

Delete the `// Legacy aliases for backward compatibility` comment and all 6 `constexpr` lines below it.

**Step 2: Build and test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: remove unused legacy NNUE constant aliases"
```

---

## Task 7: Clean Up Test Files

### 7a: Consolidate C++ test framework

All test files should use the shared `test_common.h` framework instead of duplicating their own `TestCase` class. Currently, `test_core.cpp`, `test_search_module.cpp`, `test_hybrid.cpp` each define their own local `TestCase` + `EXPECT`. `test_mcts_module.cpp` defines `TestCounter` + `expect()`.

**Files:**
- Modify: `tests/test_core.cpp` — remove local `TestCase`/`EXPECT`, `#include "test_common.h"`
- Modify: `tests/test_search_module.cpp` — same
- Modify: `tests/test_hybrid.cpp` — same; remove dead ab_bridge tests (`test_ab_integration` tests for `ABSearchResult`, `ABSearchConfig`, `HybridBridgeStats` that reference deleted ab_bridge types)
- Modify: `tests/test_mcts_module.cpp` — replace `TestCounter`/`expect()` with `test_common.h`

**Step 1: Update test_core.cpp**

Remove the anonymous namespace containing `struct TestCase` and `#define EXPECT`. Add `#include "test_common.h"` and use `Test::TestCase` / `EXPECT` from there. Adjust `run_core_tests()` return type if needed.

**Step 2: Update test_search_module.cpp**

Same refactoring. Also remove the trivial `test_stack()` and `test_info()` tests that just assign struct fields and read them back.

**Step 3: Update test_hybrid.cpp**

Same refactoring. Remove tests that reference `ABSearchResult`, `ABSearchConfig`, `HybridBridgeStats`, and `HybridSearchBridge` (all from deleted ab_bridge). Remove trivial struct-field-assignment tests (e.g., "Decision modes" test that just sets an enum and checks it).

**Step 4: Update test_mcts_module.cpp**

Replace `TestCounter`/`expect()` with `Test::TestCase`/`EXPECT`. Fix the skip-counts-as-pass issue in `test_deterministic_repro()` — skipped tests should not increment `passed`.

**Step 5: Build and run**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

Expected: All tests pass with fewer total test count (removed trivial tests).

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: consolidate test framework, remove trivial and dead tests"
```

### 7b: Remove redundant Python test wrappers

**Files:**
- Delete: `tests/bk_test.py`
- Delete: `tests/bk_test_lc0.py`
- Delete: `tests/suites/bk.epd` (positions are hardcoded in `bk_parity.py`, EPD file is unreferenced)
- Delete: `tests/suites/` directory (empty after removing bk.epd)

**Step 1: Delete files**

```bash
rm tests/bk_test.py tests/bk_test_lc0.py
rm -rf tests/suites
```

**Step 2: Fix requirements.txt**

Change `python-chess>=1.999` to `python-chess>=1.10` (realistic version).

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: remove redundant BK test wrappers and unused EPD file"
```

---

## Task 8: Clean Up Dead Code in Hybrid Module

Remove dead functions and plumbing from the hybrid module while keeping the active `ParallelHybridSearch` functional.

**Files:**
- Modify: `src/hybrid/hybrid_search.h`
- Modify: `src/hybrid/hybrid_search.cpp`
- Modify: `src/hybrid/classifier.h`
- Modify: `src/hybrid/classifier.cpp`

**Step 1: Clean hybrid_search.h/.cpp**

Remove:
- `#include "ab_bridge.h"` (already done in Task 2 — verify)
- `#include "position_adapter.h"` (not needed by ParallelHybridSearch)
- `GPUResidentBatch` struct and `initialize_gpu_batches()` declarations
- `gpu_batch_[2]`, `current_batch_` members
- `async_mutex_`, `async_cv_`, `pending_evaluations_`, `wait_gpu_evaluations()`
- `apply_move(Move)` method (no-op, never called)

**Step 2: Clean classifier.h/.cpp**

Remove dead functions never called from production code:
- `PositionClassifier::quick_classify()`
- `PositionClassifier::tactical_score()`
- `PositionClassifier::strategic_score()`
- `PositionClassifier::is_tactical()`
- `PositionClassifier::is_strategic()`
- `PositionClassifier::has_forcing_moves()`
- `PositionClassifier::is_quiet()`
- `StrategySelector::get_strategy(const Position&)` (Position overload — only the PositionFeatures overload is used)

**Step 3: Build and test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

**Step 4: Commit**

```bash
git add -A && git commit -m "chore: remove dead code from hybrid module"
```

---

## Task 9: Clean Up Stale Includes and Dead Code in position_adapter

Since `gpu_backend.cpp/h` still uses `position_adapter` through the hybrid path, we can't delete `position_adapter` entirely. But we can remove its dead items.

**Files:**
- Modify: `src/hybrid/position_adapter.h` — remove `MCTSNode`/`MCTSTree` forward declarations, `GameResult` enum
- Modify: `src/hybrid/position_adapter.cpp` — remove `MCTSEncoder::decode_policy()`, `MCTSEncoder::move_to_policy_index()`, `MCTSPosition::undo_move()`, `MCTSPositionHistory` class, `GameResult` usage
- Modify: `src/uci/uci.cpp` — remove unused `#include "hybrid/position_adapter.h"`

**Step 1: Remove dead items from position_adapter.h**

Remove `MCTSNode`/`MCTSTree` forward declarations and `GameResult` enum.

**Step 2: Remove dead functions from position_adapter.cpp**

Remove `decode_policy()`, `move_to_policy_index()`, `undo_move()`, `MCTSPositionHistory` class.

**Step 3: Remove stale include from uci.cpp**

Remove `#include "hybrid/position_adapter.h"` from `src/uci/uci.cpp`.

**Step 4: Build and test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu) && ./metalfish_tests
```

**Step 5: Commit**

```bash
git add -A && git commit -m "chore: remove dead code from position_adapter"
```

---

## Task 10: Remove Stale Python Cache and Verify .gitignore

**Files:**
- Delete: `tests/__pycache__/` (if tracked by git)
- Modify: `.gitignore` (ensure `__pycache__/` is covered)

**Step 1: Check if __pycache__ is tracked**

```bash
git ls-files tests/__pycache__
```

If any files show up, remove them:

```bash
git rm -r --cached tests/__pycache__
```

**Step 2: Verify .gitignore covers __pycache__**

Check if `__pycache__/` or `*.pyc` is in `.gitignore`. If not, add `__pycache__/` to the file.

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: remove tracked __pycache__, update .gitignore"
```

---

## Task 11: Final Verification — Full Build + All Tests

Run every test and verify nothing was broken.

**Step 1: Clean rebuild**

```bash
cd build && rm -rf * && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(sysctl -n hw.ncpu)
```

Expected: 3 targets built: `metalfish`, `metalfish_tests`, `test_nn_comparison` (plus `metal_shaders`).

**Step 2: Run C++ unit tests**

```bash
./metalfish_tests
```

Expected: All sections pass.

**Step 3: Run NN comparison tests**

```bash
METALFISH_NN_WEIGHTS=../networks/BT4-1024x15x32h-swa-6147500.pb ./test_nn_comparison
```

Expected: All pass.

**Step 4: UCI smoke test**

```bash
printf "uci\nsetoption name UseMCTS value true\nsetoption name NNWeights value $(pwd)/../networks/BT4-1024x15x32h-swa-6147500.pb\nisready\nposition startpos\ngo movetime 5000\n" | timeout 15 ./metalfish 2>&1 | grep "bestmove"
```

Expected: `bestmove d2d4 ponder d7d5` (or similar reasonable move).

**Step 5: Python integration tests**

```bash
cd ../tests && python3 testing.py
```

Expected: All UCI and perft tests pass.

**Step 6: BK parity test**

```bash
python3 bk_parity.py --engine metalfish --movetime 10000
```

Expected: Score >= 17/24 (confirming no regression from cleanup).

**Step 7: Final commit if any fixups needed**

```bash
git add -A && git commit -m "chore: final verification after codebase cleanup"
```

---

## Summary: What Gets Removed

| Category | Files/Items | Lines Removed (approx) |
|----------|------------|----------------------|
| Dead MCTS modules | `apple_silicon.h/.cpp` | ~1,079 |
| Dead eval module | `accumulator_cache.h/.cpp` | ~557 |
| Dead hybrid module | `ab_bridge.h/.cpp` | ~1,455 |
| Empty translation unit | `node.cpp` | ~25 |
| Dead functions in `core.h` | ~20 template functions | ~400 |
| Dead functions in hybrid | `classifier`, `position_adapter`, `hybrid_search` | ~200 |
| Legacy constants | `gpu_constants.h` aliases | ~8 |
| Redundant test wrappers | `bk_test.py`, `bk_test_lc0.py`, `suites/bk.epd` | ~79 |
| Trivial tests | struct-field-assignment tests | ~60 |
| Duplicate test frameworks | 3 local TestCase copies | ~90 |
| **Total** | | **~3,953 lines** |
