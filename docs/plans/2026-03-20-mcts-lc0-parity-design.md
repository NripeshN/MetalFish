# MCTS Lc0 Parity Design

## Goal

Close the 411 Elo gap between MetalFish-MCTS (~3289) and Lc0 (~3700) by implementing all missing lc0 search features systematically, with step-by-step testing to validate each improvement.

## Background

MetalFish-MCTS uses the same BT4 transformer network as Lc0 and achieves 1.26x higher throughput at 2 threads, yet is ~411 Elo weaker. The paper identifies three root causes:

1. **0% NN cache hit rate** — full-history cache keys prevent transposition hits
2. **Missing search heuristics** — lc0 has ~15 features MetalFish lacks
3. **Weaker batching** — multivisit not propagated, no prefetching

## Approach

Bottom-up: fix foundational issues (cache, batching, node layout) before adding heuristics (stoppers, tablebases, temperature). Each section is independently testable.

## Sections

### 1. NN Cache Key Fix (+100-150 Elo)

Replace `ComputePositionCacheKey()` which hashes all 8 history positions (including repetition_distance) with a position-only key using the current board's Zobrist hash + repetition count. This matches lc0's `ComputeEvalPositionHash` which calls `pos.pos.back().Hash()`.

Files: `src/mcts/search.cpp`, `src/mcts/backend_adapter.cpp`
Test: Cache hit rate from 0% to >20%. BK suite regression. Self-play Elo.

### 2. Activate Multivisit Propagation (+10-20 Elo)

`SelectChildPuct` already computes `visits_to_assign` but `SelectLeaf` discards it. Propagate it through selection, use as virtual loss count and backpropagation multiplier.

Files: `src/mcts/search.cpp`
Test: NPS improvement. BK suite correctness.

### 3. Edge Sorting + Early Cutoff (+10-20 Elo)

Sort edges by descending policy on creation. Short-circuit PUCT scan when hitting two consecutive unvisited edges with N_started=0.

Files: `src/mcts/node.h`, `src/mcts/search.cpp`
Test: Selection time profiling. BK correctness.

### 4. Node Solidification (+15-30 Elo)

Convert hot subtrees (>100 visits) from sparse edge-atomic-pointer children to contiguous Node arrays. Improves cache locality for PUCT scans.

Files: `src/mcts/node.h`, `src/mcts/search.cpp`
Test: Cache miss profiling. ASAN memory safety.

### 5. KLD Gain Stopper (+30-50 Elo)

Stop search early when root visit distribution converges (KL divergence per new node below threshold).

Files: `src/mcts/stoppers.h`, `src/mcts/stoppers.cpp`, `src/mcts/search.cpp`
Test: Easy positions terminate early. Complex positions don't stop prematurely.

### 6. Prefetching into NN Cache (+20-40 Elo)

Fill unused GPU batch slots with positions likely needed in future iterations by walking the tree via PUCT scores.

Files: `src/mcts/search.cpp`
Test: Cache hit rate improvement. GPU batch utilization.

### 7. Syzygy Tablebase Probing in MCTS (+40-60 Elo)

Probe WDL tablebases during node expansion for positions with ≤6 pieces, no castling, rule50=0. Root DTZ probing for move filtering.

Files: `src/mcts/search.cpp`, `src/mcts/search.h`
Test: Endgame positions (KRK, KQKR) solved immediately.

### 8. Terminal Handling Improvements (+30-50 Elo)

Two-fold draw correction on tree reuse. Depth-aware terminal reverts. Robust sticky bounds propagation.

Files: `src/mcts/node.h`, `src/mcts/search.cpp`
Test: Repetition-heavy positions. Tree reuse correctness.

### 9. Background Garbage Collection (+10-20 Elo)

Background thread frees pruned subtrees asynchronously (lc0's NodeGarbageCollector pattern, 100ms wake interval).

Files: `src/mcts/node.h`, `src/mcts/node.cpp`
Test: Latency profiling. No memory leaks under ASAN.

### 10. Temperature-Based Move Selection + Contempt (+10-15 Elo)

Temperature-based final move selection with win-percentage cutoff. Contempt (draw score bias).

Files: `src/mcts/search.cpp`, `src/mcts/search_params.h`
Test: Move selection variety in self-play. Draw rate change with contempt.

### 11. WDL Rescaling (+10-15 Elo)

Port lc0's WDL rescaling parameters for better calibrated evaluations.

Files: `src/mcts/search.cpp`, `src/mcts/search_params.h`
Test: WDL output calibration against known positions.

### 12. Safety Stoppers (+5-10 Elo)

Memory watching stopper (prevent OOM). Depth stopper. Mate stopper (early termination on forced mate).

Files: `src/mcts/stoppers.h`, `src/mcts/stoppers.cpp`
Test: Large search with memory limits. Mate-in-N positions.

## Total Estimated Impact: ~300-500 Elo

This should close or exceed the 411 Elo gap to lc0, bringing MetalFish-MCTS to ~3600-3700 Elo range.

## Testing Strategy

Each section follows TDD:
1. Write a test that captures the expected behavior
2. Implement the feature
3. Run regression (BK suite, self-play)
4. Commit on green
5. Run tournament game set for Elo measurement after major milestones (sections 1, 5, 7, 12)
