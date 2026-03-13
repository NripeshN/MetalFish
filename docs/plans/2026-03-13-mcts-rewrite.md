# MCTS Full Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite MetalFish's MCTS run_iteration to match Lc0's pattern exactly: ONE NN call per leaf, no heuristic fallbacks, no policy blending, correct PUCT parameters. Make MCTS as strong as Lc0 with the same network.

**Architecture:** Simplify run_iteration to: select leaf → call nn_evaluator_->Evaluate() → apply policy via ApplyNNPolicy → use NN value directly → backpropagate. Remove GatherBatchEvaluator, expand_node heuristic path, evaluate_position fallback chain. Fix PUCT params to match Lc0 (cpuct_base=38739, cpuct_factor=3.894). Disable Dirichlet noise for competitive play.

**Tech Stack:** C++ (tree.cpp, tree.h), Metal/MPSGraph (unchanged), NEON SIMD (unchanged)

---

### Task 1: Fix PUCT Parameters and Disable Noise

**Files:**
- Modify: `src/mcts/tree.h:378-393`

**Step 1: Update ThreadSafeMCTSConfig defaults to match Lc0**

```cpp
// BEFORE (wrong):
float cpuct_base = 19652.0f;
float cpuct_factor = 2.5f;
bool add_dirichlet_noise = true;

// AFTER (matches Lc0):
float cpuct_base = 38739.0f;
float cpuct_factor = 3.894f;
bool add_dirichlet_noise = false;  // Off for competitive play (Lc0 default)
```

**Step 2: Verify by running MCTS and checking bestmove quality**

```bash
cd build && { printf "uci\nsetoption name UseMCTS value true\nsetoption name NNWeights value ../networks/BT4-1024x15x32h-swa-6147500.pb\nisready\nposition startpos\ngo movetime 10000\n"; sleep 14; printf "quit\n"; sleep 1; } | timeout 18 ./metalfish 2>&1 | grep -E "bestmove|NPS"
```

**Step 3: Commit**

```bash
git add src/mcts/tree.h
git commit -m "mcts: fix PUCT params to match Lc0 (cpuct_base=38739, factor=3.894), disable noise"
```

---

### Task 2: Remove GatherBatchEvaluator

**Files:**
- Modify: `src/mcts/tree.h` -- remove GatherBatchEvaluator class declaration and gather_eval_ member
- Modify: `src/mcts/tree.cpp` -- remove GatherBatchEvaluator implementation, remove from start_search/wait

**Step 1: In tree.h, delete the GatherBatchEvaluator class (lines ~486-530) and the `gather_eval_` member**

**Step 2: In tree.cpp, delete the GatherBatchEvaluator implementation (constructor, cancel, evaluate, execute_batch)**

**Step 3: In start_search, remove the gather_eval_ creation block**

**Step 4: In wait(), remove the gather_eval_->cancel() call**

**Step 5: Commit**

```bash
git commit -m "mcts: remove GatherBatchEvaluator (caused empty results, deadlocks)"
```

---

### Task 3: Rewrite run_iteration -- Single NN Call, No Fallbacks

**Files:**
- Modify: `src/mcts/tree.cpp:1248-1400` (the run_iteration method)

**Step 1: Rewrite run_iteration to this clean pattern:**

```cpp
void ThreadSafeMCTS::run_iteration(WorkerContext &ctx) {
    ctx.reset_to_cached_root();
    
    // 1. SELECT leaf
    ThreadSafeNode *leaf = select_leaf(ctx);
    if (!leaf) return;
    
    // 2. CHECK TERMINAL
    MoveList<LEGAL> moves(ctx.pos);
    if (moves.size() == 0) {
        bool in_check = ctx.pos.checkers() != 0;
        float value = in_check ? -1.0f : 0.0f;
        float draw = in_check ? 0.0f : 1.0f;
        leaf->set_terminal(ThreadSafeNode::Terminal::EndOfGame, value);
        backpropagate(leaf, value, draw, 0.0f);
        return;
    }
    
    // 3. EVALUATE with NN (ONE call, no fallbacks)
    float value = 0.0f;
    float draw = 0.0f;
    float moves_left_val = 30.0f;
    
    if (!leaf->has_children() && nn_evaluator_) {
        try {
            auto nn_result = nn_evaluator_->Evaluate(ctx.pos);
            
            // Apply NN policy directly (no heuristic blending)
            {
                std::lock_guard<std::mutex> lock(leaf->mutex());
                if (!leaf->has_children()) {
                    leaf->create_edges(moves);
                    ApplyNNPolicy(leaf, nn_result);
                    if (config_.add_dirichlet_noise && leaf == tree_->root()) {
                        add_dirichlet_noise(leaf);
                    }
                }
            }
            
            // Use NN value directly (no NNUE/heuristic fallback)
            value = nn_result.value;
            draw = nn_result.has_wdl ? nn_result.wdl[1] : 0.0f;
            moves_left_val = nn_result.has_moves_left ? nn_result.moves_left : 30.0f;
            
            stats_.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
            // NN failed -- treat as uncertain (draw)
            value = 0.0f;
            draw = 1.0f;
        }
    }
    
    // 4. BACKPROPAGATE
    backpropagate(leaf, value, draw, moves_left_val);
    ctx.iterations++;
}
```

Key changes:
- ONE `nn_evaluator_->Evaluate()` call per leaf (not 3)
- Policy applied via `ApplyNNPolicy` ONLY (no `expand_node` heuristic blending)
- Value from NN ONLY (no `evaluate_position` / NNUE fallback)
- If NN fails, value = 0 (draw), not heuristic eval
- No gather_eval_ reference

**Step 2: Test**

```bash
cd build && cmake --build . --target metalfish -j$(sysctl -n hw.ncpu) && \
{ printf "uci\nsetoption name UseMCTS value true\nsetoption name NNWeights value ../networks/BT4-1024x15x32h-swa-6147500.pb\nisready\nposition startpos\ngo movetime 10000\n"; sleep 14; printf "quit\n"; sleep 1; } | timeout 18 ./metalfish 2>&1 | grep -E "bestmove|NPS|NN evals"
```

Expected: ~100 NPS, bestmove d4/e4/Nf3 (strong opening move), NN evals = node count

**Step 3: Compare with Lc0**

```bash
{ printf "uci\nisready\nposition startpos\ngo movetime 10000\n"; sleep 14; printf "quit\n"; sleep 1; } | timeout 18 reference/lc0/build/release/lc0 --weights=networks/BT4-1024x15x32h-swa-6147500.pb --backend=metal 2>&1 | grep -E "bestmove|nodes.*nps"
```

MetalFish should choose the same or similar bestmove as Lc0.

**Step 4: Commit**

```bash
git commit -m "mcts: rewrite run_iteration - single NN call, no heuristic fallbacks"
```

---

### Task 4: Remove Dead Code

**Files:**
- Modify: `src/mcts/tree.cpp` -- remove expand_node, evaluate_position, evaluate_position_direct, evaluate_position_batched
- Modify: `src/mcts/tree.h` -- remove declarations

**Step 1: Delete these methods from tree.cpp:**
- `expand_node()` (~150 lines of heuristic scoring + policy blending)
- `evaluate_position()` (the dispatcher)
- `evaluate_position_batched()` (uses BatchedGPUEvaluator)
- `evaluate_position_direct()` (NNUE/simple_eval fallback)

**Step 2: Remove declarations from tree.h**

**Step 3: Build and verify no compile errors**

**Step 4: Commit**

```bash
git commit -m "mcts: remove dead code (expand_node heuristics, eval fallbacks)"
```

---

### Task 5: Verify Against Lc0 on Multiple Positions

**Step 1: Test on 5 positions and compare bestmoves**

Test positions:
1. startpos
2. `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1` (after 1.e4)
3. `rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2` (French Defense)
4. `r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2` (after 1.e4 Nc6)
5. `rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2` (after 1.e4 Nf6)

Run both MetalFish MCTS and Lc0 on each for 10 seconds, compare bestmoves.

**Step 2: Commit verification results as a comment in tree.cpp**

---

### Task 6: Verify Hybrid Engine Uses Fixed MCTS

**Step 1: Test hybrid engine**

```bash
{ printf "uci\nsetoption name UseHybridSearch value true\nsetoption name NNWeights value ../networks/BT4-1024x15x32h-swa-6147500.pb\nisready\nposition startpos\ngo movetime 10000\n"; sleep 14; printf "quit\n"; sleep 1; } | timeout 18 ./metalfish 2>&1 | grep -E "bestmove|Final"
```

Expected: Hybrid should show non-zero MCTS nodes with proper NN evaluations.

**Step 2: Run quick cutechess tournament (2 games)**

```bash
reference/cutechess/build/cutechess-cli -engine cmd=./metalfish name=MetalFish-AB proto=uci option.Threads=4 option.Hash=256 -engine cmd=./metalfish name=MetalFish-Hybrid proto=uci option.Threads=4 option.Hash=256 option.UseHybridSearch=true "option.NNWeights=../networks/BT4-1024x15x32h-swa-6147500.pb" -each tc=60+1 -games 2 -repeat -recover -pgnout /tmp/verify_hybrid.pgn
```

**Step 3: Commit**

```bash
git commit -m "mcts: verified MCTS and hybrid work correctly after rewrite"
```
