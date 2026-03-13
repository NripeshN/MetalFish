# Paper Evaluation & Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run scientifically rigorous experiments evaluating MetalFish's three engines (AB, MCTS, Hybrid), then rewrite the paper with Hybrid as the central contribution, including all experimental data.

**Architecture:** Create a Python evaluation harness that runs 4 experiments (puzzle suites, best-move agreement, scaling analysis, head-to-head games), saves results as JSON/CSV, then update the paper LaTeX with real data tables and charts.

**Tech Stack:** Python 3 (python-chess, subprocess), LaTeX (Springer svproc), cutechess-cli for games

---

### Task 1: Download and Prepare Test Suites

**Files:**
- Create: `tests/suites/wac.epd` (Win At Chess - 300 positions)
- Create: `tests/suites/sts.epd` (Strategic Test Suite subset - 100 positions)
- Create: `tools/run_evaluation.py` (main evaluation harness)

**Step 1: Download WAC (Win At Chess) EPD**

WAC is a standard 300-position tactical test suite. Each line is: FEN + "bm" (best move) + "id" (position name).

```bash
cd /Users/nripeshn/Documents/PythonPrograms/metalfish
mkdir -p tests/suites
curl -L "https://raw.githubusercontent.com/fsmosca/STS-Rating/master/epd/wac_300.epd" -o tests/suites/wac.epd 2>/dev/null || echo "Download manually"
```

If download fails, create a small curated set of 50 tactical positions with known best moves from standard chess literature (Bratko-Kopec, WAC subset, etc.) directly in EPD format.

**Step 2: Create a curated evaluation position set**

Create `tests/suites/metalfish_eval.epd` with 50 positions covering:
- 15 opening positions (known theory moves)
- 15 tactical positions (forks, pins, sacrifices) 
- 10 strategic/positional positions
- 10 endgame positions

Each line format: `FEN bm BEST_MOVE; id "POSITION_NAME";`

**Step 3: Commit**

```bash
git add tests/suites/ && git commit -m "tests: add evaluation position suites"
```

---

### Task 2: Create Evaluation Harness

**Files:**
- Create: `tools/run_evaluation.py`

**Step 1: Write the evaluation harness**

The script should:

1. **EPD Solver**: Given an EPD file, run each position through an engine at a fixed time/depth/nodes, check if bestmove matches the EPD's `bm` field. Report: total solved / total positions.

2. **Agreement Test**: Run same positions through two engines, count how often they agree on the best move.

3. **Scaling Benchmark**: Run same 20 positions at different thread counts (1, 2, 4, 8, 12), record NPS, depth reached, and best move.

4. **Output**: JSON results file + formatted table to stdout.

Key function signatures:
```python
def solve_epd(engine_cmd, engine_opts, epd_file, time_per_pos=10) -> dict
def agreement_test(engine1, opts1, engine2, opts2, positions, nodes=500) -> dict  
def scaling_benchmark(engine_cmd, positions, thread_counts=[1,2,4,8,12]) -> dict
```

Engine communication: UCI protocol (send position, go movetime/nodes, read bestmove).

**Step 2: Test with a quick dry run**

```bash
python3 tools/run_evaluation.py --test  # Runs 3 positions as sanity check
```

**Step 3: Commit**

```bash
git add tools/run_evaluation.py && git commit -m "tools: add evaluation harness (EPD solver, agreement, scaling)"
```

---

### Task 3: Run Experiment A -- Puzzle Suite Solve Rates

**Step 1: Run all engines on the puzzle suite (10s per position)**

Engines to test:
- `MetalFish-AB` (Threads=12, Hash=512)
- `MetalFish-Hybrid` (Threads=12, Hash=512, UseHybridSearch=true, NNWeights=BT4)
- `MetalFish-Hybrid` (Threads=8, Hash=512, UseHybridSearch=true, NNWeights=BT4)
- `MetalFish-MCTS` (UseMCTS=true, NNWeights=BT4)

```bash
python3 tools/run_evaluation.py --experiment puzzle --suite tests/suites/metalfish_eval.epd --time 10
```

**Step 2: Save results**

Results saved to `results/eval_puzzle_TIMESTAMP.json`

**Step 3: Commit results**

```bash
git add results/ && git commit -m "results: puzzle suite solve rates for all engines"
```

---

### Task 4: Run Experiment B -- MCTS vs Lc0 Agreement

**Step 1: Run agreement test at fixed nodes**

Both MetalFish-MCTS and Lc0 get `go nodes 500` on 50 positions.

```bash
python3 tools/run_evaluation.py --experiment agreement --nodes 500
```

Report:
- Agreement rate (% same bestmove)
- Each engine's match rate vs EPD best move

**Step 2: Save and commit results**

---

### Task 5: Run Experiment C -- Scaling Analysis

**Step 1: Run MetalFish-AB and Hybrid at multiple thread counts**

20 positions, 10s each, at Threads=1, 2, 4, 8, 12:

```bash
python3 tools/run_evaluation.py --experiment scaling --time 10
```

Record per config: average NPS, average depth, solve rate.

Also test Hybrid at Threads=8+GPU and Threads=12+GPU.

**Step 2: Save and commit results**

---

### Task 6: Run Experiment D -- Head-to-Head Games

**Step 1: AB vs Hybrid (12 threads, 300+3, 10 games)**

```bash
python3 tools/run_evaluation.py --experiment games --tc "300+3" --games 10
```

**Step 2: MCTS vs Lc0 (fixed nodes 500, 10 games)**

```bash
python3 tools/run_evaluation.py --experiment games-mcts-lc0 --nodes 500 --games 10
```

**Step 3: Save PGN and results**

---

### Task 7: Rewrite Paper with Real Data

**Files:**
- Modify: `paper/Latex/template/metalfish.tex`

**Step 1: Update Section 5 (Experimental Evaluation) with real data**

Replace placeholder tables with actual measured values:

- Table 2: Per-engine performance → use real NPS from scaling benchmark
- Table 3: Tournament results → use real game results from Experiment D
- Add NEW Table: Puzzle suite solve rates (Experiment A)
- Add NEW Table: MCTS-Lc0 agreement rates (Experiment B)
- Add NEW Figure: Thread scaling chart showing NPS vs threads for AB and Hybrid

**Step 2: Update Section 6 (Discussion) with data-driven insights**

- Frame hybrid advantage based on actual puzzle solve improvements
- Discuss MCTS-Lc0 equivalence based on measured agreement rate
- Update scaling discussion with real numbers

**Step 3: Update Abstract and Conclusion with real numbers**

Replace all "190+ plies" style claims with exact measured values.

**Step 4: Compile and verify**

```bash
cd paper/Latex/template && pdflatex metalfish.tex && pdflatex metalfish.tex
```

**Step 5: Commit**

```bash
git add paper/ results/ && git commit -m "paper: update with real experimental data from all 4 experiments"
```

---

### Task 8: Final Review

**Step 1: Verify all reviewer comments are still addressed**

Run the same 15-point checklist from the previous review.

**Step 2: Verify all tables reference real data files**

**Step 3: Final compile and PDF copy**

```bash
cd paper/Latex/template && pdflatex metalfish.tex && pdflatex metalfish.tex
cp metalfish.pdf PDF/
cp metalfish.pdf ../../PDF/metalfish_revised_v2.pdf
```

**Step 4: Commit and push**

```bash
git add -A && git commit -m "paper: final version with all experimental data" && git push origin paper
```
