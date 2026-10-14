# 03 — Codebase changes required for the paper

> Status 2026-10-12: tooling items C7, C8, C9, C10, C11a are implemented in `tools/paper/`
> (see `tools/paper/README.md`; tests in `tests/test_paper_tools.py`). The AB core is being
> ported to Stockfish 19 (see §P0 C0 below), which supersedes the paper's "AB = Stockfish
> dev-2026-01-03" baseline; the protocol's `SFdev` parity cell becomes a parity check against
> Stockfish 19 itself (bench signature 2497913).

Principle: **no strength tuning before the protocol freeze.** The system under test is
MetalFish 1.0.x as shipped, plus correctness fixes and instrumentation. The only
behaviour-affecting additions are explicit *ablation switches* that default to the
shipped behaviour.

## 0. As-built summary (replaces §3 of the paper)

Verified against `b935e66` on 2026-10-07 (full audit in `01_published_paper_audit.md` §C):

- **Modes**: selected by UCI (`UseHybridSearch`, `UseMCTS`); one binary; 118 options.
- **AB**: Stockfish dev-20260103 search + NNUE (`FullThreats+HalfKAv2_hm` 108.9 MB big net,
  `HalfKAv2_hm` 3.5 MB small net), eight small deltas (root-order hints, keep-last-PV,
  QoS, NEON `SqrClippedReLU`, …). Defaults `Threads=1`, `Hash=16`.
- **MCTS**: Lc0-style PUCT (cpuct 1.745, base 38739, factor 3.894; FPU reduction 0.33;
  policy softmax 1.359, root 1.6), 128-byte `alignas(128)` nodes with `os_unfair_lock`,
  2M-entry direct-mapped NNCache keyed on the current position, Apple default
  minibatch 1 and one worker, smooth time manager for pure MCTS only.
- **NN backend**: BT4 as an MPSGraph (`MetalNetworkGraph`), FP16 body by default
  (`NNMetalFP16=true`), FP32 I/O, one eval at a time under `gpu_mutex_`; batch-1 latency
  7.6 ms (FP16) / 8.8 ms (FP32) on M2 Max; CoreML/ANE path exists, off by default.
- **Hybrid**: three control threads (MCTS controller, AB, coordinator) + Stockfish pool +
  MCTS worker(s). `alignas(128)` shared root structs with mostly relaxed atomics. Thread
  split on a clock with T≥8: MCTS 1, AB T−2. MCTS→AB root-order hints (top-8) and a 240 ms
  AB candidate-verification sub-search are on; AB→MCTS policy boost and shared TT are off.
  Final decision: AB's move unless one of ~30 gated MCTS-override predicates fires and
  `HybridABRootRejectsMCTS` does not veto; agreement early-stop exists but never fired in
  telemetry; the position classifier only lengthens budgets and sets a node minimum.
  Transformer disabled per move when budget < 400 ms or time left < 3 s.

## P0 — Before the protocol freeze

| ID | Change | Where | Acceptance |
| --- | --- | --- | --- |
| C0 | **Port the AB core to Stockfish 19** (tag sf_19): 3-way merge of all Stockfish-derived files with the eight local deltas preserved, single SFNNv16 net `nn-1a298aa575a0.nnue` (small net removed), new `attacks.*`, `pp_3wide.*`, `nnz_helper.h`, `thread_native.h`, `shm_unix.h`, `tune.*`. | `src/core`, `src/search`, `src/eval`, `src/syzygy`, `src/uci`, `CMakeLists.txt`, tools/docs referencing the small net | `bench` reports exactly 2,497,913 nodes (identical to SF19); unit suite, perft/UCI smoke, hybrid smoke pass; MF-AB vs SF19 single-thread sanity match ≈ 0 Elo with zero time losses. |
| C1 | **Build environment hardening.** Homebrew's protobuf 36.1 / abseil 20260817 upgrades broke the dynamically linked binary (`libprotobuf.35.1.0.dylib` missing) and the stale CMake cache pointed at a removed Xcode SDK. Document the clean-reconfigure recipe in README (`rm -rf build/CMakeCache.txt build/CMakeFiles`), record `brew list --versions protobuf abseil` in the manifest, and evaluate static linking of protobuf/abseil (`-DProtobuf_USE_STATIC_LIBS=ON`) so a frozen binary keeps running across brew upgrades. | `CMakeLists.txt`, README | Frozen binary runs after `brew upgrade`; `otool -L build/metalfish` shows no Homebrew dylibs, or the manifest pins them. |
| C2 | **Fix the `MCTSEvals` parser artifact** in `tests/paper_benchmarks.py` (lines ~285–320 never read the hybrid's `MCTSEvals=` field, producing `nn_evals=0`), and delete the README sentence (lines ~113–116) that concluded "the GPU performed zero evaluations" from it. | `tests/paper_benchmarks.py`, `README.md` | Hybrid rows in `results/paper_tactical.json` show non-zero `nn_evals`; README claim removed. |
| C3 | **Encoder en-passant plane bug** (`src/nn/encoder.cpp:~321`): the canonical EP auxiliary plane is not vertically mirrored for black to move. Fix; add a parity test comparing MetalFish policy/value against Lc0 on a set of EP positions for both colours (Lc0 `--verbose-move-stats` at 1 node, or `tools/compare_nn_backend_outputs.py` with an Lc0 reference dump). | `src/nn/encoder.cpp`, `tests/` | Max |Δpolicy| and |Δvalue| vs Lc0 within FP16 tolerance on EP positions; existing parity probes unchanged. |
| C4 | **NNCache default.** 2,000,000 entries ≈ 1.65 GB value-initialised per hybrid/MCTS process. Decide: keep (document memory) or reduce to 500,000 (≈0.4 GB). Either way the manifest records it; the paper reports the footprint. | `src/uci/engine.cpp` (`MCTSNNCacheSize`) | Decision recorded in protocol §2; memory measured in M4. |
| C5 | **Structured per-move telemetry.** Today the hybrid emits `info string HybridTrace: reason=… key=value …` only on true hybrid moves and a separate `Time safety: …` line on fallback moves. Add (a) a trace line for *every* move the hybrid engine plays, including fallbacks (`reason=ab_fallback_time_safety budget_ms=… time_left_ms=…`), (b) explicit `gpu_active=0/1`, `mcts_evals`, `mcts_playouts`, `mcts_evals_per_s`, `ab_depth`, `ab_threads`, `mcts_threads`, `elapsed_ms` fields, and (c) a parser `tools/paper/parse_hybrid_trace.py` → JSONL per move, robust to field order. | `src/hybrid/hybrid_search.cpp` (trace emit), `src/uci/uci.cpp` (fallback path), `tools/paper/` | 100% of hybrid-engine moves in a fastchess log parse to exactly one record; counts match PGN ply counts. |
| C6 | **Uniform thread-split rule and off-by-one.** Make the clock split rule explicit and uniform (`MCTS=1, AB=T−2` for T≥4) or keep the current rule but document it; fix the `Threads<3` fallback off-by-one (coordinator is outside the worker budget, so the guard should be `<2`, `src/uci/uci.cpp:~1675`). The protocol sets `HybridMCTSThreads`/`HybridABThreads` explicitly, so this is about honesty of the default description. | `src/uci/uci.cpp` | `Hybrid thread split:` line matches the documented rule for T ∈ {3..12}; unit test for `compute_hybrid_thread_split`. |
| C7 | **Runner and manifest tooling** (`tools/paper/run_cell.py`): reads `docs/paper_v2/experiments.json`, builds the fastchess command (§3 of the protocol), creates `results/paper_v2/<cell>/`, writes `manifest.json` (git SHA, `shasum -a 256` of binaries/networks/book, tool versions, hardware via `sysctl`, macOS version, options, seed, timestamps), starts the thermal sampler, runs fastchess in blocks, and writes `summary.json`. | `tools/paper/` | Pilot cell runs end to end; `summary.json` fields match fastchess's printout. |
| C8 | **Statistics** (`tools/paper/penta_stats.py`): pentanomial counts from PGN pairs, score mean/variance, Elo + 95% CI, nElo, LOS, draw rate, trinomial for comparison; unit-tested against fastchess output on the smoke PGN and against the `vdbergh` reference formulas. | `tools/paper/`, `tests/` | Agreement with fastchess to ±0.1 Elo on three PGNs. |
| C9 | **Oracle probe** (`tools/paper/oracle_probe.py`): from PGN + parsed traces, select disagreement positions (stratified), run the two judges (SF19 60 s MultiPV; Lc0 30k nodes), store per-candidate cp/WDL, compute summaries with paired bootstrap. Replaces the lost `/tmp/movequal.py`, `judge.py`, `thick.py`. | `tools/paper/` | Reproduces the June 2026 finding on the 90-position sample within noise; outputs JSON + CSV. |
| C10 | **Suite runner** (`tools/paper/suite_runner.py`): EPD `bm`/`am` suites and Lichess puzzle CSV, any UCI engine, per-engine option sets, movetime, Wilson CIs, McNemar pairs; reuse `tools/lichess_puzzle_runner.py` logic for puzzles. | `tools/paper/` | Runs all five suites for SF19 and MetalFish-AB with correct solve counts on a 10-position subset. |
| C11a | **Microbenchmarks**: `tools/paper/nn_latency_sweep.py` (nn_probe + lc0 benchmark across batch sizes and precision), `tools/paper/ab_scaling.py` (fixed-depth bench vs threads), `tools/paper/make_figures.py` (all figures from JSON). | `tools/paper/` | Figures M1/M2 regenerate from JSON with one command. |
| C12 | **Minibatch on a clock.** Verify whether the hybrid forces `MCTSMinibatchSize=1` when the per-move budget is < 1000 ms (`src/uci/uci.cpp:~1187-1191` forces 1 for `movetime < 1000`). For RQ4 the requested minibatch must be honoured on a clock budget; add `MCTSMinibatchSize` precedence over the auto rule when set explicitly, keeping auto behaviour unchanged. | `src/uci/uci.cpp`, `src/mcts/search.cpp` | Trace shows the configured minibatch and ~3× evals/s at mb16 in a 30+0.3 pilot game. |

## P0b — Engine-strength changes made under the strength programme (07)

The author's 2026-10-12 re-scope ("strongest engine first, then the preprint") supersedes
the "no changes to the search" non-goal below for the items in
`07_engine_strength_program.md`; each is game-gated and recorded there. The paper reports
the hybrid as finally configured and, where a change was measured, the before/after gate.

| # | Change | Files | Status |
| --- | --- | --- | --- |
| C13 | **Alpha-beta pool stable across hybrid moves** (07 L5b): `ThreadPool::set` size override, `Engine::resize_threads(int)`, hybrid resizes only when the size changes, plain searches restore the full pool. | `src/search/thread.{h,cpp}`, `src/uci/{engine,uci}.cpp`, `src/hybrid/hybrid_search.cpp` | PR #73; gate L6b showed the time-manager interaction below. |
| C14 | **Hybrid alpha-beta bounded by the coordinator budget** (07 L5c): `HybridBuildABLimits` (movetime = remaining budget for clock searches). | `src/hybrid/hybrid_search.{h,cpp}`, `tests/test_hybrid.cpp` | PR #73; gate L6c running. |
| C15 | **`HybridFollowABTime`** (07 L5d, opt-in): a clock move ends when alpha-beta's own time manager stops; MCTS stopped with it. | `src/uci/{engine,uci}.cpp`, `src/hybrid/hybrid_search.{h,cpp}`, `tests/test_hybrid.cpp` | branch `hybrid-follow-ab-time`; gate L7b (+74 ± 30 after 100 games vs the main hybrid) → **default on**; the per-move deadline no longer applies to clock searches in this mode (`HybridABTimeManagerOwnsClock`). |
| C16 | **Root tie-break overrides off by default** (`HybridRootPawnLeverTieBreak=false`, gates the pawn-lever and quiet-attack tie-breaks): oracle study of 22 override positions (SF19) showed a 1.2% expected-score cost per override. | `src/uci/engine.cpp`, `src/hybrid/hybrid_search.h`, tool/CI option tables | Gated with C15 in L7b; RQ6c separates them. |
| C17 | **Oversubscription by default** (`HybridOversubscribe=true`): alpha-beta keeps all `Threads`, the MCTS worker is an extra thread. | `src/uci/engine.cpp` | Gate L9 (oversubscribed vs 6+1 split on the adopted hybrid, 20+0.2): see 07. |
| C18 | **Hybrid NN-cache cap** (`kHybridNNCacheEntriesCap = 100000` in `make_hybrid_config`): the pure-MCTS default of 2 M entries pre-allocates ≈1.6 GB that the hybrid never uses, making its process ≈6 GB vs 2.3 GB for plain alpha-beta with a 2 GB hash. | `src/uci/uci.cpp` | Footprint re-measured with `vmmap`; gate L14 (lean hybrid vs AB(8), 30+0.3). |

## P1 — Ablation switch (needed for RQ6c; defaults preserve shipped behaviour)

| ID | Change | Where | Acceptance |
| --- | --- | --- | --- |
| C11 | **`HybridArbitration` option** with values `full` (default, current ladder), `minimal` (AB move unless AB produced none; keep agreement bookkeeping and traces), `mcts_primary` (MCTS move unless MCTS produced none; diagnostic only). Implement as one early branch in `make_final_decision()` with `reason=arbitration_minimal` / `arbitration_mcts_primary`; unit tests in `tests/test_hybrid.cpp`; UCI smoke in `tools/uci_smoke.py`. Byte-identical behaviour at `full`. | `src/hybrid/hybrid_search.{h,cpp}`, `src/uci/{engine,uci}.cpp`, tests | Existing hybrid tests unchanged; new tests pass; fixed-node smokes bit-identical at `full`. |

## P1 — Documentation and hygiene (can land any time before submission)

- Retire `tests/paper_benchmarks.py --tournament` and `tools/run_tournament_live.py`'s
  anchor-average "Estimated Elo Ratings" printout (it is the method that produced the
  4048 figure). Keep the tactical/NPS parts of `paper_benchmarks.py` for regression use.
- Update README "Benchmarking Policy" and "Strength Claims": fastchess + UHO + pentanomial
  + nElo + fixed N; forbid `st=`; require `Threads≥3` for hybrid; state the GPU-active rule.
- Update `tools/run_cutechess_tournament.sh` comment about the removed 2/1 split (stale
  since `863e60d`), or replace the script by `tools/paper/run_cell.py` presets.
- Add `docs/paper_v2/` to the docs index; add the frozen-binary build recipe.

## P2 — After the paper's measurements (not before)

- **Arbitration de-slop guided by RQ6**: if `Hybrid-min` ≈ `Hybrid` within CI, retire the
  position-specific override predicates and the inert classifier weighting; keep the
  measured, documented minimum. Validate by games, never by Bratko-Kopec.
- **Throughput work** only if RQ4 shows a positive slope: MPSGraph op fusion to cut
  per-launch latency; `gpu_mutex_` removal with per-worker command queues; Lc0-style
  multi-worker batching with quality guards.
- **CUDA path parity study** as a separate paper/section (different hardware class).
- Known medium/low correctness items from the June 2026 audit that do not affect the
  measurements: `verify_ab_root_candidates` blocking sub-search stealing wall-clock
  (protocol can set `HybridABCandidateVerifyMs=0` as an ablation); semaphore expansion
  dropping legal moves beyond 96 (rare); dead EP history branches.

## Testing and CI

- Unit: `./build/metalfish_tests hybrid core search mcts eval_gpu` must stay green; add
  tests for C6, C8, C11, C12.
- Python: `python3 tests/testing.py --quick`, `tests/test_hybrid_regression_compare.py`,
  new `tests/test_penta_stats.py`, `tests/test_parse_hybrid_trace.py`.
- `pre-commit run --all-files` before each commit.
- CI: keep `hybrid-regression.yml` as is for PR gating; the paper runs are local (GPU and
  hours needed), driven by `tools/paper/run_cell.py`.

## Non-goals for this paper

- No parameter tuning (cpuct, FPU, contempt, AB params) before the freeze.
- No new arbitration heuristics; only the `minimal` ablation.
- No CUDA/Windows results.
- No changes to the Stockfish-derived search beyond instrumentation.
