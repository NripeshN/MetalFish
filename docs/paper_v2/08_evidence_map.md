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
