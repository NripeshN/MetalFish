# 08 — Evidence map (every number with its provenance)

Running ledger of measurements that may enter the paper. Each row names the artifact that
backs it; nothing here is hand-entered from memory. Conditions: Apple M2 Max (8P+4E, 38-core
GPU, 32 GB), macOS 27.2, unless stated. "UHO" = `UHO_Lichess_4852_v1.epd`, random order,
colour-swapped pairs, Fishtest adjudication (draw 34/8/20, resign 3/600 two-sided), `-maxmoves 200`.

## A. Engine strength gates (fastchess, pentanomial; `results/paper_v2/sprt/<name>/summary.json`)

| Name | Comparison | TC / threads | Games | Result | Reading |
| --- | --- | --- | --- | --- | --- |
| `sf19port_vs_preport_ab_10p01_v2` | AB (Stockfish 19 port, `4d7ca23`) vs AB (pre-port, `cc473fb`) | 10+0.1, 8 vs 8, Hash 1024 | 136 (68 pairs) | **+62.0 ± 28.2 Elo**, nElo +131.6, LOS 1.000, Ptnml [0,6,33,28,1]; 1 time forfeit (candidate, 12 ms overrun) | Port adopted. Stopped once conclusive (SPRT [0,10] normalized had not yet fired). |
| `L3_ab_t12_vs_t8_10p01` | AB(12 threads) vs AB(8) | 10+0.1, Hash 1024 | 273 (136 pairs) | **+6.4 ± 16.4 Elo**, nElo +16.1, LOS 0.78, Ptnml [0,19,93,24,0]; 3 time forfeits | Inconclusive; 8 P-cores kept as fair default. |
| `L5_hybrid_oversub8_vs_ab8_20p02` | Hybrid (AB 8 + MCTS 1, `HybridOversubscribe=true`) vs AB(8) | 20+0.2, Hash 1024 | 22 (11 pairs) | **−97.2 ± 74.1 Elo**, Ptnml [0,7,3,1,0]; losses adjudicated at −6…−11 pawns, no time forfeits | Lever dropped; hybrid runtime overhead to be quantified in RQ1a/RQ2. Engine log kept (`engines.log.gz`). |

| `L7b_hybrid_followab_notiebreak_vs_hybrid_main_20p02` | Hybrid (stable pool + Stockfish TM owns the clock + tie-breaks off) vs Hybrid (main dc2854a) | 20+0.2, 8 vs 8, Hash 1024 | 100 (50 pairs) | **+74.1 ± 30.5 Elo**, LOS 100%, pairs ratio 8.0 | 07 L5b–L5e; the `L7` first version (deadline kept) read −62.7 ± 45.3 after 29 games |
| `L8_hybrid_new_vs_ab8_20p02` | Hybrid (6+1 split, adopted TM/tie-break changes) vs AB(8), same binary | 20+0.2, Hash 1024 | 151 | **−11.6 ± 20.7 Elo**, LOS 14% | MCTS move chosen 0 times; 64% fallback; same depth, 3% fewer nodes |
| `L9_hybrid_oversub_vs_split_20p02` | Hybrid oversubscribed (8 AB + MCTS extra) vs Hybrid 6+1 split, same binary | 20+0.2, Hash 1024 | 300 (150 pairs) | **+15.1 ± 17.0 Elo**, LOS 96% | adopted (author decision under the capped-run rule) |
| `RQ1a_pilot` (protocol §10) | Hybrid shipped (oversubscribed, 6 GB process) vs AB(8) | **30+0.3**, Hash 2048 | 100 (50 pairs) | **−41.9 ± 25.1 Elo**, Ptnml [0,14,34,2,0] | GPU active 50%, median 27 MCTS evals, MCTS move chosen 0/1165 disagreements; identical time and depth per phase |
| `L10_hybrid_noverify_nohints_vs_ab8_30p03` | Hybrid with root hints + candidate-verify off vs AB(8) | 30+0.3, Hash 2048 | 100 | −17.4 ± 31.0 Elo | diagnostic |
| `L12_hybrid_nullfallback_vs_ab8_30p03` | Hybrid UCI path, transformer never engaged (network still loaded) vs AB(8) | 30+0.3, Hash 2048 | 100 | −20.9 ± 28.5 Elo | null control; deficit present without any GPU search |
| `L13_hybrid_nullfallback_nopreload_vs_ab8_10p01` | Same without preload (footprint = plain) vs AB(8) | 10+0.1, Hash 1024 | 301 | −10.4 ± 16.8 Elo | bit-identical fixed-node searches (12/12); runner overhead 0 ms both arms |
| `L14_hybrid_lean_vs_ab8_30p03` | Hybrid lean (C18 NN-cache cap, main 365f36a) vs AB(8), same binary | 30+0.3, Hash 2048 | 101 | **+13.9 ± 30.3 Elo**, LOS 82% | GPU active 50%, median 31 evals, MCTS move chosen 0 times |

Interim, RQ1a block 0 (2026-10-14): the detached MUST queue (`queue_must_cells.sh`, started 09:09 UTC)
and its fastchess/engine processes disappeared at 11:37 UTC with RQ1a block 0 at 128 games
(64 complete pairs; 0 forfeits; no thermal warnings; no crash report, kernel memory event or
signal in the system log; the block's exit was never logged, so the shell itself was killed).
Readings at the interruption: 0.0 ± 22.0 Elo (`penta_stats.py`, 128 games). The block was
resumed at 11:43 UTC by `queue_must_cells_resume.sh` with `run_cell.py --resume` (same seeded
book order, next opening = pair 65, 186 pairs left; recorded in `RQ1a/manifest.json` →
`resumes`); no game was deleted or replayed. The resumed queue logs any catchable signal.
Second interruption, 13:13 UTC: a Homebrew upgrade of protobuf on the host removed the dynamic
library the rc1 binary linked; fastchess aborted RQ1a at game 183 ("engine startup failure") and
every later block failed its preflight (03 §C19). The 183 rc1 games (0 forfeits; 0.0 ± 20 Elo
range at the time) are kept as `results/paper_v2/RQ1a_rc1_partial/` and are not part of any
cell; RQ1a restarts from pair 1 with the re-linked rc2 binary (protocol amendment §11.7).
rc2 verification (14:26–14:34 local): 130 objects rebuilt from the same source; `bench` 2,497,913
nodes; unit suite 128 checks passed; UCI/perft smoke passed; `test_nn_comparison` fixed-position BT4
references within tolerance (max value delta 0.0012, max top-policy delta 0.006), batch parity exact
to batch 8, one FP16 policy-logit deviation of 0.0056 at batch 16 (the same class of deviation the
green CI run of main 8c0661d shows at batch 2; the engine evaluates at batch 1); 81 Homebrew dynamic
libraries bundled under `build/lib`; bench, probe and unit suite re-passed after bundling. rc2 sha256
8c534bb6f82fe9ec…; tag `paper-v2-freeze-rc2` on 8c0661d (engine source identical to 365f36a).
Logs: `results/paper_v2/queue_logs/rc2_*.txt`.



Memory footprint (vmmap physical footprint, `Hash=2048`, after `isready`): plain alpha-beta 2.3 GB;
hybrid with the default 2 M-entry NN cache 5.8–6.1 GB; hybrid with the C18 cap 4.4 GB. With
`Hash=16`: 3.8 GB (2 M entries) vs 2.2 GB (1,000 entries); the Metal batch cap and the new-game
re-preload change nothing; GPU share 0.39 GB. Host during the pilot: swap 3.1 of 4.35 GB used,
≈270k page-outs; during L14 (lean) +76 page-outs in 10 min.

Telemetry from the L5 engine log (`tools/paper/parse_hybrid_trace.py`): 1,437 hybrid-engine
moves, 745 true hybrid decisions, 692 time-safety fallbacks (GPU active on 51.8% of moves);
reasons: engines_agree 481, ab_root_rejects_mcts 177, ab_default 82, pawn-lever/quiet-attack
tiebreaks 5; MCTS move selected on 4 of 263 disagreements; median 33 playouts and 32 evals
per hybrid move.

## B. Microbenchmarks

| Measurement | Value | Artifact |
| --- | --- | --- |
| `bench` node count, ported engine | 2,497,913 (= Stockfish 19 upstream signature) | build log `results/paper_v2/calibration/build_final.log`; CHANGELOG |
| `bench` single-thread NPS, ported engine vs SF19 PGO binary | 1.308 M vs 1.337 M (ratio 0.979, 3 runs each, sequential) | port agent report (2026-10-12) |
| PGO vs plain LTO build (pre-port tree, depth 13, best of 3) | 1.263 vs 1.284 Mnps, identical node counts (2,477,446) | `tools/build_pgo.sh` run, 2026-10-12 |
| AB NPS vs threads (pre-port tree, depth 15, median of 3) | 1: 1.37 M, 4: 5.80 M, 8: 7.54 M, 12: 10.72 M (+42% at 12 vs 8) | `results/paper_v2/micro/ab_scaling_preport.json` (re-run on the final engine pending) |
| BT4 eval latency, MetalFish MPSGraph, frozen-day sweep (2026-10-14, machine otherwise idle, `micro/nn_latency.json`) | FP16: b1 7.40 ms (135 pos/s), b8 19.7 ms (407), b16 35.9 ms (446), b64 119 ms (538); FP32: b1 8.74 ms (114), b64 125 ms (511) | `results/paper_v2/micro/nn_latency.json` (built from the pre-C18 branch binary, identical NN code) |
| Lc0 0.32.1 Metal BT4 steady-state nps (`lc0 benchmark`, same sweep) | mb1 165, mb2 163, mb4 189, mb8 294, mb16 606, mb32 725, mb64 530 (median of the last five per-second samples; the all-sample median is biased low by the ramp-up) | `results/paper_v2/micro/nn_latency.json` `nps_steady_median` |
| AB NPS vs threads, frozen-day run (depth 16, 5 repeats, `micro/ab_scaling.json`) | 1: 1.53 M, 2: 3.04 M, 4: 6.27 M, 6: 9.18 M, 8: 11.81 M, 12: 13.52 M | `results/paper_v2/micro/ab_scaling.json` |
| BT4 eval latency, MetalFish MPSGraph (idle machine, 2026-10-07) | FP16 b1 7.61 ms (131 pos/s), b16 36.1 ms (443 pos/s); FP32 b1 8.83 ms (113), b16 38.3 ms (418) | `06_environment_and_baselines.md` |
| Lc0 0.32.1 Metal BT4 nps (idle machine, 2026-10-07) | mb1 ≈160, mb4 ≈215, mb16 ≈410, mb64 ≈600, mb256 ≈650 | `06_environment_and_baselines.md` |
| The 2026-10-12 NN sweep | contaminated by concurrent builds; quarantined | `results/paper_v2/micro/nn_latency_m2max_CONTAMINATED_concurrent_builds.json` |

## C. Prior repository evidence (pre-port engine, balanced book)

| Measurement | Value | Artifact |
| --- | --- | --- |
| Hybrid vs AB, 40+0.4, 8 threads, 8moves_v3 | 0W-1L-49D, −6.9 ± 13.5, 98% draws | `results/hybrid_vs_ab_tc/match.log` |
| Hybrid vs AB, 90+0.9 | 0W-1L-39D, −8.7 ± 16.8 | `results/hybrid_vs_ab_tc/match_tc90.log` |
| Hybrid (transformer on) vs AB, 20+0.5 | 0-0-48 | `results/tf_on_vs_ab/match.log` |
| MetalFish-AB vs Stockfish dev-20260106, 20+0.5, 4 threads | 0-0-22 | `results/ab_vs_sf/match.log` |
| 20+0.2 hybrid telemetry (July 2026) | 437 moves, 228 fallbacks, AB move chosen 207/209, median 45 playouts, 0 agreement stops | `results/tournament_20260701_212250/results.json` via `parse_hybrid_trace.py` |
| Bratko-Kopec, 5 s, 8 threads (2026-06-30) | AB 20/24, MCTS 24/24, Hybrid 24/24, Lc0 18/24, SF 20/24 | `results/paper_summary.md` (ceiling + leakage caveats) |
| Lichess bot corpus | 1,058 games: 99 W / 22 L / ≈85% draws | `results/lichess_audit/` via `tools/analyze_lichess_games.py` |

## D. Calibration of the statistics tool

`tools/paper/penta_stats.py` reproduces fastchess's pentanomial printout: Ptnml [0,10,20,10,0]
→ Elo ±38.22, nElo ±76.13; Ptnml [2,12,19,7,0] → LOS 3.59%, PairsRatio 0.50, WL/DD 1.38;
smoke [0,4,5,1,0] → Elo −52.51 ± 71.59/71.60, nElo −115.10 ± 152.27 (`results/paper_v2/calibration/`,
`tests/test_paper_tools.py`).
