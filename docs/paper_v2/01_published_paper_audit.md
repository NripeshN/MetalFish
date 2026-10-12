# 01 — Audit of the published paper

Scope: every quantitative or architectural claim in `paper:metalfish/metalfish.tex`
(the LNNS chapter) checked against (a) the code on `post-v1-tuning` at `b935e66`,
(b) the paper-era commit `ec1dee5` (2026-05-11) where relevant, (c) artifacts under
`results/`, and (d) measurements taken on 2026-10-07 on the Apple M2 Max (see
`06_environment_and_baselines.md`). File:line references are to the current tree.

Verdict scale: **Wrong** (contradicted by evidence), **Unsupported** (no artifact or
method can back it), **Stale** (true at some point, not now), **Overstated**,
**Holds**.

## A. Strength claims

| # | Claim (paper) | Evidence | Verdict |
| --- | --- | --- | --- |
| A1 | MetalFish-AB "estimated 4048 Elo with zero losses in 110 games" (abstract, §3.1, §5.4, Fig. 2) | Method (tools/run_tournament_live.py:990-1004): per-match logistic Elo diff from a 5-game score, then *average of (anchor Elo + diff)* over anchors. Anchors were `expected_elo` constants in `tools/engines_config.json` (Stockfish 3800, Berserk 3722, Patricia 3415, Lc0 3700) and Stockfish skill levels (SF-16 "3584", SF-14 "3505", ... invented). The AB-vs-full-Stockfish cell is "undef." (no decisive result), so the average is driven by crushing weakened opponents. A 5-0 score maps to the ±400 clamp. No PGNs or JSON from the 322-game run exist anywhere in the repo. MetalFish-AB is a renamed Stockfish dev build (see C1); the repo's clean match vs Stockfish dev at 20+0.5, Threads=4: **0W-0L-22D** (`results/ab_vs_sf/match.log`). | **Wrong** |
| A2 | Hybrid "~3724 Elo, competitive with Berserk (~3722) and Lc0 (~3700)" | Same invalid calibration. CCRL ratings are for CCRL hardware/time controls; Lc0 with BT4 on Apple Metal runs at 160–650 nps (measured, `06_…`), far below the GPUs behind its CCRL entries, so "Lc0 ≈ 3700" was never a valid anchor on this hardware. Valid repo measurements of hybrid vs AB on the same box: 40+0.4: 0W-1L-49D (−6.9 ± 13.5); 90+0.9: 0W-1L-39D (−8.7 ± 16.8); 20+0.5 transformer-on: 0W-0L-48D (`results/hybrid_vs_ab_tc/`, `results/tf_on_vs_ab/`). | **Wrong** |
| A3 | MCTS "~3289 Elo" | Same method. The only informative statement is that MetalFish-MCTS lost 5/6 to Lc0 and 5/5 to Stockfish in 5–6 game samples. | **Unsupported** |
| A4 | "Comprehensive evaluation across 322 tournament games" | 33 matches of 5 games; the paper's own Limitations admit ±300 Elo. Not reproducible: no artifacts, no PGNs, no seeds, no engine versions beyond strings. Game runner (`tests/paper_benchmarks.py:1076`) uses fixed `go movetime`, Python-side clock, no pentanomial stats; the production tournament script (`tools/run_tournament_live.py`) uses trinomial Elo. | **Unsupported** |
| A5 | "Internal head-to-head: AB vs Hybrid 3-2-0 → +241 Elo; Hybrid vs MCTS 5-0-0 → >+400" | 5 games each. Current data shows hybrid ≈ AB (A2). | **Wrong** (noise) |
| A6 | "The hybrid approach adds ~435 Elo over pure MCTS" (conclusion) | Derived from A2/A3. Trivially true in direction (AB ≫ thin MCTS) but the number is meaningless. | **Unsupported** |

## B. Tactical / throughput claims

| # | Claim | Evidence | Verdict |
| --- | --- | --- | --- |
| B1 | Bratko-Kopec at 10 s: AB 20/24, SF 20/24, Hybrid 20/24, Lc0 19/24, MCTS 18/24 | Current run at 5 s, 8 threads (`results/paper_summary.md`, 2026-06-30): AB 20, MCTS **24**, Hybrid **24**, Lc0 18, SF 20. The 24-position suite is at ceiling; the "solutions" for BK.03/09/17 are human positional judgements (pawn levers), not ground truth. `results/` holds ~1,500 BK/puzzle tuning runs, so hybrid/MCTS thresholds were tuned on these very positions (test-set leakage). | **Stale** and unfit for ranking |
| B2 | "100% bestmove agreement AB–Stockfish validates our alpha-beta" | Expected: the search is Stockfish's (C1). Not a validation of anything but the port. | **Overstated** |
| B3 | "95.8% hybrid–Lc0 agreement suggests the NN component heavily influences the hybrid" | Game telemetry (`results/tournament_20260701_212250`, 209 true hybrid moves): final move = AB's in **207/209**; MCTS move chosen 2/209. Reasons: engines_agree 131, ab_root_rejects_mcts 49, ab_default 27, root_quiet_attack_tiebreak 2. | **Wrong** inference |
| B4 | MetalFish-MCTS throughput 1.26× Lc0 (avg 172 vs 136 nps), 5.5× in a "complex middlegame" (77 vs 14) | 4 positions, 2 threads, 10 s. Lc0 at 14 nps is an outlier that indicates a harness problem (the repo's BK run also shows Lc0 at 43 nps avg vs MF 163, inconsistent with Lc0's own benchmark). Measured 2026-10-07 on M2 Max: `lc0 benchmark` BT4 Metal minibatch 1 = 154–162 nps; MetalFish `metalfish_nn_probe` batch 1 = 131 pos/s FP16 (7.6 ms), 113 FP32 (8.8 ms). At batch 16: MF 443 pos/s vs Lc0 ≈ 410 nps; Lc0 reaches 600–650 nps at minibatch 64–256. In games the hybrid's MCTS worker ran at ~74 evals/s (13.5 ms/eval) with AB threads contending. | **Wrong** / unsupported |
| B5 | "7.6M NPS on 8 threads (M2 Max), 9.8M on M1 Ultra, 7.24× scaling on 8 threads" | BK run shows AB avg 7.37M NPS on 8 threads (plausible). No thread-scaling artifact exists anywhere; "7.24×" is unsupported. | Plausible, **Unsupported** scaling figure |
| B6 | "os_unfair_lock in Node: 2 NPS → 148 NPS, 74× (2 threads)" (Table 1, abstract, contributions) | A 2 NPS baseline is a pathological lock convoy, not a representative baseline; the "before" code is not in the paper branch and the measurement is unreproducible. Node uses `os_unfair_lock` only for `FinalizeScoreUpdate`/`MakeTerminal` (src/mcts/node.h). Default Apple config runs **one** MCTS worker, so multi-thread MCTS throughput is not exercised in production at all. | **Unsupported**; reframe as a bug-fix anecdote at most |
| B7 | Table 1 other rows (+7% history buffers, 16 KB policy table, NNCache 3.5 GB → 38 MB "92×", pooled BackendComputation +16%) | Single-run micro-measurements without variance; NNCache now 2,000,000 entries × ~824 B ≈ 1.65 GB (B9). | **Unsupported**/**Stale** |

## C. Architecture / implementation claims

| # | Claim | Evidence (file:line on `b935e66`) | Verdict |
| --- | --- | --- | --- |
| C1 | "Our alpha-beta engine derives from the Stockfish codebase, inheriting its search infrastructure while adding GPU integration" | Token-level diff vs `reference/stockfish` (dev c27c1747, 2026-01-03) shows only: `ApplyRootOrderHints` (src/search/search.cpp:54-93), keep-last-PV on unproven fail-high (search.cpp:463-474), aarch64 `yield`, two `[[unlikely]]`, thread QoS, searchmoves dedup, `TT::is_allocated()`, a NEON `SqrClippedReLU` path. There is no GPU integration in the AB search. The paper's "killer move heuristics" do not exist in this search. | **Overstated**; must be stated plainly as a Stockfish port |
| C2 | "512 MB transposition table" | Default `Hash` = 16 MB (src/uci/engine.cpp:71-96); experiments used 4096 MB. | **Wrong** |
| C3 | "Dual NNUE networks (125 MB large, 6 MB small) with HalfKAv2" | `nn-c288c895ea92.nnue` 108,919,594 B (FullThreats + HalfKAv2_hm, L1 1024) and `nn-37f18f62d772.nnue` 3,519,630 B (HalfKAv2_hm, L1 128) (src/eval/evaluate.h:25-26, nnue_architecture.h:28-41). | **Wrong** details |
| C4 | "Position classifier (tactical/strategic/endgame) adjusts decision weighting; tactical → AB preferred, quiet → MCTS carries more weight" | Classifier exists (src/hybrid/classifier.{h,cpp}, 5 classes, endgame is a boolean). Its `mcts_weight`/`ab_weight`/`ab_override_threshold` are computed and **never read** outside tests. `DecisionMode::DYNAMIC` shares a case label with `VOTE_WEIGHTED` (hybrid_search.cpp:5315-5316); no branch depends on position type. The classifier only (i) lengthens the time budget when multiplier > 1 and (ii) sets `min_mcts_nodes` for the agreement stop. | **Wrong** |
| C5 | Algorithm 1: "If AB and MCTS agree on best move for 3+ iterations → terminate early (10–40% time savings)" | Exists (hybrid_search.cpp:4007-4035) but gated on: clock search only, 3 consecutive coordinator polls (~1.5 ms, not iterations), elapsed > 25% budget, MCTS playouts ≥ max(100, min_mcts_nodes), AB depth ≥ 10. In the 20+0.2 telemetry it fired **0 of 209** times (median MCTS playouts 45 < 100). No artifact supports any time saving. | **Unsupported** |
| C6 | "Two cache-line-aligned (128 B) atomic structures ... acquire/release ordering, eliminating lock contention" | `ABSharedState`/`MCTSSharedState` are `alignas(128)` (hybrid_search.h:33-141), ~76 B and ~256 B. Stores are mostly relaxed with release only on `has_result`/`update_counter`/`pv_length`/`num_top_moves`; no consistent-snapshot protocol. Producers take `tree_structure_mutex_` and `ab_root_mutex_` elsewhere. | **Overstated** |
| C7 | "Shrinking MCTS nodes from 128 to 64 bytes ... two nodes per cache line" | `static_assert(sizeof(Node) <= 128)` with `alignas(128)` on Apple (src/mcts/search.cpp:36-45; node.h:54-58): **128 bytes on Apple**, already so at the paper commit. 64 bytes only on non-Apple builds (spin-lock shim). | **Wrong** |
| C8 | "NNCache ... 500K entries, 96 moves/entry, 38 MB" and "0% cache hit rate because full-history key" | Now 2,000,000 entries (`MCTSNNCacheSize`), 96 moves, ≈1.65 GB by layout; key covers the current position only (`MCTSCacheHistoryLength=0`, src/mcts/search.h:234-247) since commit `8a376d2` (2026-05-15). 38 MB is not consistent with 96-move entries even at 500K. | **Stale**/**Wrong** |
| C9 | "Cooperative batching ... semaphore gathering ... virtual loss" | Lc0-style gathering exists (search.cpp:1944-2304) but runs only when minibatch > 1; Apple default resolves `MCTSMinibatchSize=0` to **1** (src/uci/uci.cpp:852-859), and pure MCTS is capped to **one** worker on Apple unless `MCTSParallelSearch=true`. Production path = one blocking eval per playout. | **Stale**/misleading |
| C10 | "Lc0 smooth time manager: μ=45.2, σ=5.93, 52% initial tree reuse capped 73%, 12% piggybank" | μ, σ, 0.73 cap and 12% hold (search.cpp:1479-1630); initial reuse is 0.55. The hybrid does **not** use this manager: MCTS receives a fixed `movetime = budget` (hybrid_search.cpp:586-612). | Partly **Holds**; misattributed |
| C11 | "~382 NPS on M1 Ultra's 64 GPU cores" | Unverifiable (no artifact). Batch-1 inference is launch-latency bound (7.6–8.8 ms/eval on M2 Max); a 2× larger GPU does not halve per-op launch latency, so 382 is doubtful. | **Unsupported** |
| C12 | System figure: "UCI → Position Classifier → {AB, MCTS, Hybrid}" | Mode selection is by UCI option, not by the classifier. The hybrid additionally disables the transformer per move when the move budget < 400 ms or time left < 3 s (`TransformerMinMoveBudgetMs`, `TransformerLowTimeFallbackMs`, uci.cpp:881-935) — 228 of 437 hybrid moves at 20+0.2 ran as plain AB, and in the 2026-10-07 smoke test at 10+0.1 almost every move did. The paper never mentions this. | **Wrong** figure; **omission** |
| C13 | AB→MCTS "root hints"/"candidate verification" described as AB informing MCTS | They flow MCTS→AB: MCTS policy/visit hints pre-sort AB's root moves (`ApplyRootOrderHints`), and a 240 ms AB sub-search verifies the top-5 hints. AB→MCTS prior boost (`HybridABPolicyWeight`) and shared TT are off by default. | **Wrong** direction |

## D. Related work and framing

| # | Claim | Problem |
| --- | --- | --- |
| D1 | "Unlike previous hybrid approaches that use one paradigm to inform the other sequentially (Baier & Winands)" | Baier & Winands embed shallow minimax *inside* MCTS (selection/rollout/backup); "sequential" is a mischaracterization. The actual nearest prior work is omitted: Scorpio's alpha-beta rollouts inside MCTS with NN eval (Shawul, 2018), Komodo/Dragon MCTS mode, the 2017 Stockfish MCTS branch (Nicolet), Huang's rollout formulation of alpha-beta (AAAI 2015), Ramanujan et al. on traps (ICAPS 2010), the lc0-stockfish-hybrid project, Ceres/Allie/Stoofvlees as alternative MCTS stacks, and the Lc0 transformer paper (Monroe & Chalmers 2024). See `05_related_work.md`. |
| D2 | `lanctot2014monte` cited for "general game playing" | That paper (implicit minimax backups) evaluates Kalah, Breakthrough and Lines of Action, not GGP. |
| D3 | `coulom2008elo` (WHR) cited as the Elo methodology | The method used was an ad-hoc anchor average, not WHR. |
| D4 | Keywords/intro: "novel hybrid engine" | Novelty must be re-stated as the *parallel, arbitrated* combination with measurement; concurrent AB+MCTS with final arbitration has precedents in practice (engine ensembles) that must be acknowledged. |
| D5 | No data/code availability statement with versions, no seeds, no hardware thermal state, no statistics section | Required for the new paper (see protocol §9). |

## E. Items that hold and can be reused

- The system *exists* and runs: three modes from one binary; hybrid with 3 threads (MCTS controller, AB, coordinator) plus Stockfish's thread pool and MCTS workers; lock-free-ish shared root state; MCTS→AB root hints; BT4 via MPSGraph (FP16 default); CPU NNUE. The as-built description in the agent audit (summarized in `03_codebase_changes.md` §0) should replace §3 of the paper.
- `os_unfair_lock` is used in `Node`; `alignas(128)` cache-line alignment is real.
- Smooth time manager parameters (μ, σ, cap, piggybank) for *pure MCTS* on a clock.
- Lichess deployment exists (1,058 games logged in `results/lichess_audit/`, ~85% draws, 99W/22L) — usable as a descriptive deployment section, not as a strength claim.
- The GPU work being FP16 MPSGraph with ~7.6 ms batch-1 latency is a genuine systems finding (launch-latency bound; batching helps throughput 3.4× but hurt tactical quality in prior tests) and should be measured and reported properly.

## F. Root causes (why the paper went wrong) — to address in the methodology

1. **Confounded contribution**: the AB component is Stockfish; its strength was presented as the system's achievement.
2. **Invalid rating calibration**: anchor-average Elo with invented anchors and 5-game samples; CCRL numbers imported across hardware.
3. **Ceiling-level, leaked tactical suite** used as a quality metric and as a tuning target.
4. **No artifact retention / no versions / no seeds**.
5. **Architecture described from intent, not from code**; several features are inert or gated off by default.
6. **Draw-death blindness**: balanced books (8moves_v3) at 97–100% draws make hybrid-vs-AB differences unmeasurable; nothing in the paper controls for it.
7. **GPU-active fraction never measured**: at blitz the transformer is mostly disabled by the time-safety rule, so "hybrid" results were largely AB results.
