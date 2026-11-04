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

| `L7b_hybrid_followab_notiebreak_vs_hybrid_main_20p02` | Hybrid (stable pool + Stockfish TM owns the clock + tie-breaks off) vs Hybrid (main before #73; base binary sha256 4e497415…, the base of L6–L7) | 20+0.2, 8 vs 8, Hash 1024 | 100 (50 pairs) | **+74.1 ± 30.5 Elo**, LOS 100%, pairs ratio 8.0 | 07 L5b–L5e; the `L7` first version (deadline kept) read −62.7 ± 45.3 after 29 games |
| `L8_hybrid_new_vs_ab8_20p02` | Hybrid (6+1 split, adopted TM/tie-break changes) vs AB(8), same binary | 20+0.2, Hash 1024 | 151 | **−11.6 ± 20.7 Elo**, LOS 14% | MCTS move chosen 0 times; 64% fallback; same depth, 3% fewer nodes |
| `L9_hybrid_oversub_vs_split_20p02` | Hybrid oversubscribed (8 AB + MCTS extra) vs Hybrid 6+1 split, same binary | 20+0.2, Hash 1024 | 300 (150 pairs) | **+15.1 ± 17.0 Elo**, LOS 96% | adopted (author decision under the capped-run rule) |
| `RQ1a_pilot` (protocol §10) | Hybrid shipped (oversubscribed, 6 GB process) vs AB(8) | **30+0.3**, Hash 2048 | 100 (50 pairs) | **−41.9 ± 25.1 Elo**, Ptnml [0,14,34,2,0] | GPU active 50%, median 27 MCTS evals, MCTS move chosen 0/1165 disagreements; identical time and depth per phase |
| `L10_hybrid_noverify_nohints_vs_ab8_30p03` | Hybrid with root hints + candidate-verify off vs AB(8) | 30+0.3, Hash 2048 | 100 | −17.4 ± 31.0 Elo | diagnostic |
| `L12_hybrid_nullfallback_vs_ab8_30p03` | Hybrid UCI path, transformer never engaged (network still loaded) vs AB(8) | 30+0.3, Hash 2048 | 100 | −20.9 ± 28.5 Elo | null control; deficit present without any GPU search |
| `L13_hybrid_nullfallback_nopreload_vs_ab8_10p01` | Same without preload (footprint = plain) vs AB(8) | 10+0.1, Hash 1024 | 301 | −10.4 ± 16.8 Elo | bit-identical fixed-node searches (12/12); runner overhead 0 ms both arms |
| `L14_hybrid_lean_vs_ab8_30p03` | Hybrid lean (C18 NN-cache cap, main 365f36a) vs AB(8), same binary | 30+0.3, Hash 2048 | 101 | **+13.9 ± 30.3 Elo**, LOS 82% | GPU active 50%, median 31 evals, MCTS move chosen 0 times |

Interim, RQ1a block 0 (2026-10-14): the detached MUST queue (`queue_must_cells.sh`, started 09:09 local time, 08:09 UTC; first game 08:09:06 UTC)
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
to batch 8, one FP16 policy-logit deviation of 0.0056 at batch 16, against the check's tolerance of 0.005
(`queue_paper_log.txt` 2026-10-14T14:32:15; `queue_logs/rc2_nn_comparison.txt`) (the same class of deviation the
green CI run of main 8c0661d shows at batch 2; the engine evaluates at batch 1); 81 Homebrew dynamic
libraries bundled under `build/lib`; bench, probe and unit suite re-passed after bundling. rc2 sha256
8c534bb6f82fe9ec…; tag `paper-v2-freeze-rc2` on 8c0661d (engine source identical to 365f36a).
Logs: `results/paper_v2/queue_logs/rc2_*.txt`.
Third interruption, 21:19 UTC: the host rebooted (not by the programme) with RQ1a (rc2) at 283
games; the queue was relaunched at 21:24 UTC and resumed from pair 142 (141 complete pairs; the one
orphan game of round 142 stays in the PGN as an unpaired game and enters no pair). The restart
exposed a pairing defect in `penta_stats.py`: it grouped the two games of a pair by fastchess round
number alone, but fastchess restarts round numbers at 1 in every invocation (every block and every
resume), so games of different invocations collided (the rc1 partial read as 36 pairs and 111
unpaired games instead of 91 pairs and 1). Fixed before any multi-invocation cell was interpreted:
pairs are keyed by round number and opening FEN (unit test added). Re-read of the two RQ1a files
after the fix: rc1 partial 183 games = 91 pairs + 1 unpaired; rc2 283 games = 141 pairs + 1 unpaired.
Host load during RQ1a block 0 (rc2): two time forfeits by the AB arm at games 366 and 385 (60 ms
and 15 ms over the clock; last-move NPS 2.3 M and 5.2 M against the usual 14–22 M), i.e. 0.49% of
410 games, while a user process (`openclaw-gateway`) ran at about one full core and the 15-minute
load average reached 24 on the 12-core host (observed 2026-10-15 00:56 UTC). Forfeits stay in the
record as played. From 01:00 UTC a host-load logger (`results/paper_v2/hostload_logger.sh` →
`hostload.log`, one line per minute: load averages, external CPU%, top external process) runs
alongside the cells as a covariate for the drift analysis.





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
| `bench` node count, ported engine | 2,497,913 (= Stockfish 19 upstream signature) | `results/paper_v2/queue_paper_log.txt` bench lines (25, 60, 98, 105); CHANGELOG |
| `bench` single-thread NPS, ported engine vs SF19 PGO binary | 1.308 M vs 1.337 M (ratio 0.979, 3 runs each, sequential) | port agent report (2026-10-12) |
| PGO vs plain LTO build (pre-port tree, depth 13, best of 3) | 1.263 vs 1.284 Mnps, identical node counts (2,477,446) | `tools/build_pgo.sh` run, 2026-10-12 |
| AB NPS vs threads (pre-port tree, depth 15, median of 3) | 1: 1.37 M, 4: 5.80 M, 8: 7.54 M, 12: 10.72 M (+42% at 12 vs 8) | `results/paper_v2/micro/ab_scaling_preport.json` (superseded by M2 on the frozen build, below) |
| BT4 eval latency, MetalFish MPSGraph, sweep of 2026-10-13 (build 8777b99, whose network code is the frozen one; no game running, host load not yet logged; `micro/nn_latency.json`; date from `queue_paper_log.txt`, corrected 2026-10-27) | FP16: b1 7.40 ms (135 pos/s), b8 19.7 ms (407), b16 35.9 ms (446), b64 119 ms (538); FP32: b1 8.74 ms (114), b64 125 ms (511) | `results/paper_v2/micro/nn_latency.json` (built from the pre-C18 branch binary, identical NN code) |
| Lc0 0.32.1 Metal BT4 steady-state nps (`lc0 benchmark`, same sweep) | mb1 165, mb2 163, mb4 189, mb8 294, mb16 606, mb32 725, mb64 530 (median of the last five per-second samples; the all-sample median is biased low by the ramp-up) | `results/paper_v2/micro/nn_latency.json` `nps_steady_median` |
| AB NPS vs threads, run of 2026-10-13 (build 8777b99, whose search code is the frozen one; depth 16, 5 repeats, `micro/ab_scaling.json`) | 1: 1.53 M, 2: 3.04 M, 4: 6.27 M, 6: 9.18 M, 8: 11.81 M, 12: 13.52 M | `results/paper_v2/micro/ab_scaling.json` |
| BT4 eval latency, MetalFish MPSGraph (idle machine, 2026-10-07) | FP16 b1 7.61 ms (131 pos/s), b16 36.1 ms (443 pos/s); FP32 b1 8.83 ms (113), b16 38.3 ms (418) | `06_environment_and_baselines.md` |
| Lc0 0.32.1 Metal BT4 nps (idle machine, 2026-10-07) | mb1 ≈160, mb4 ≈215, mb16 ≈410, mb64 ≈600, mb256 ≈650 | `06_environment_and_baselines.md` |
| The 2026-10-12 NN sweep | contaminated by concurrent builds; quarantined | `results/paper_v2/micro/nn_latency_m2max_CONTAMINATED_concurrent_builds.json` |
| **M1, frozen build** (2026-11-01, 08:38–08:45 UTC, after the MUST cells, by `queue_after_must_rc2_gated.sh` behind the load gate; `microbench.py nn`, 5 warm-up and 40 timed iterations per batch size; outside CPU logged per minute 151–361%, median 184%: the editor about one core, one minute each of `mediaanalysisd` and a WebKit process above 300%) | MetalFish MPSGraph, median latency (positions/s): FP16 b1 7.21 ms (139), b2 8.79 ms (227), b4 12.2 ms (327), b8 18.9 ms (422), b16 35.8 ms (447), b32 60.1 ms (532), b64 119 ms (538); FP32 b1 8.56 ms (117), b8 20.6 ms (388), b16 38.0 ms (422), b64 125 ms (511) | `results/paper_v2/micro/nn_latency_frozen.json` (`metalfish`) |
| **M1, Lc0 0.32.1 Metal, same run** (`lc0 benchmark`; steady-state median of the last five per-second samples; these are search nodes per second, which include cache hits and terminal nodes, so they are not raw network evaluations; `backendbench` failed with "Failed to import MPS module", so no raw Lc0 latency was measured) | mb1 217, mb2 222, mb4 297, mb8 447, mb16 638, mb32 763, mb64 671 nps | `nn_latency_frozen.json` (`lc0`, `nps_steady_median`; `lc0_backendbench` holds the errors) |
| **M2, frozen build** (2026-11-01, 08:45–08:50 UTC, same chain and gate; `microbench.py ab`, depth 16, `Hash` 1024, 5 repeats per thread count, median; outside CPU logged per minute 145%, 283%, 76%, 16% and 15% in that order, the sweep running from 1 to 12 threads) | 1: 1.47 M, 2: 3.02 M, 4: 6.10 M, 6: 9.15 M, 8: 12.78 M, 12: 15.06 M nodes/s (12 threads +18% over 8; the four extra cores are efficiency cores) | `results/paper_v2/micro/ab_scaling_frozen.json` (`aggregate`) |
| **M4, frozen build** (2026-11-01, 08:50 UTC, same chain and gate; `footprint.py`: physical footprint from `vmmap -summary` after `isready` and after a 20 s search of one position, with the cells' options and `Hash` 2048; vmmap's units, G = 2^30 bytes; outside CPU logged 15% and 58% in the two minutes the run overlapped) | AB(8) 2.3 GB (peak 2.3); Hybrid(8) 4.4 GB (peak 4.7); MCTS(1) 5.6 GB after `isready`, 5.7 GB after the search (peak 5.9) | `results/paper_v2/micro/footprint_frozen.json` |

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

## E. Confirmatory cells — interim readings (rc2 binary, `paper-v2-freeze-rc2`)

Interim readings after each completed block, as the hand-off goal requires; **no decision is
taken on an interim reading** (fixed N, protocol §5 RQ1; the cell runs to its pre-registered size).
Generated by `make_figures.py` (`tables/cells.md`, `blocks.md`, `telemetry.md`, `compute.md`).

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Draws | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1a | Hybrid(8) vs AB(8), 30+0.3 | 1 of 4 (block 0) | 501 (250 pairs + 1 unpaired) | -11.8 ± 12.6 | -28.5 ± 30.5 | 3% | [0, 52, 163, 35, 0] | 47% | 2 (0.4%) | 50%; 31 evals/move; MCTS move chosen 2 of 5919 disagreements | `RQ1a/summary.json` |
| RQ2a | AB(6) vs AB(8), 30+0.3 | 1 of 4 (block 0) | 250 (125 pairs) | -9.7 ± 17.0 | -24.8 ± 43.1 | 13% | [0, 23, 86, 16, 0] | 44% | 0 | — | `RQ2a/summary.json` |
| RQ2b | Hybrid-split(8) (6 AB threads + MCTS worker) vs AB(6), 30+0.3 | 1 of 4 (block 0) | 250 (125 pairs) | -27.9 ± 18.4 | -65.7 ± 43.1 | 0.1% | [0, 34, 77, 14, 0] | 50% | 0 | 55%; 55 evals/move; MCTS move chosen 0 of 3063 disagreements | `RQ2b/summary.json` |
| RQ5c | AB(8) vs Stockfish 19 (8 threads), 30+0.3 | 1 of 4 (block 0) | 250 (125 pairs) | +2.8 ± 17.7 | +6.8 ± 43.1 | 62% | [0, 20, 83, 22, 0] | 49% | 0 | — | `RQ5c/summary.json` |

Block 0 timing (from the PGN `GameDuration` headers): 1.58 min per game in RQ1a and RQ2a, 1.55 in RQ2b. Host-load sensitivity after block 0 (amendment 8; `tables/sensitivity.md`): RQ1a 6% of pairs removed, −11.8 → −8.8; RQ2a 52% removed, −9.7 → −14.5; RQ2b 54% removed, −27.9 → −30.0; RQ5c 65% removed, +2.8 → −11.8 (a shift larger than the primary SE, to be discussed; a subset of 35% of the pairs is expected to differ from the full estimate by about one SE through sampling alone). Interim; no decision is taken. Both RQ1a
forfeits were by the AB arm (games 366 and 385, 60 ms and 15 ms over) during heavy external host
load (see the host-load note in §A). The runner kept only the latest invocation in each manifest
until 2026-10-15; the invocation history of RQ1a (two invocations of block 0: from game 0, and the
resume after the reboot from game 283), RQ2a and RQ2b was backfilled from the queue log
(`manifest.pre_backfill.json` keeps the originals), and from now on `run_cell.py` appends every
invocation to the manifest's `history` and resumes a block from its own recorded start.

Load-gated scheduling from 2026-10-15 12:32 local (protocol amendment 8): the host-load log showed
the host shared with other work (626 one-minute samples from 01:57 to 12:23 local: median external
CPU 175%, 75th percentile 304%, 90th percentile 590%, maximum 1,069%; bursts from a virtual machine
and a compiler). RQ2b block 0 was stopped at a pair boundary (54 games, 27 complete pairs, no
unpaired game) and continues under `queue_must_cells_rc2_gated.sh`: chunks of 10 pairs, each started
only when the 5-minute mean external CPU is below 300%. Every block of every cell reports the primary
estimate (all games) and the pre-registered sensitivity estimate (pairs overlapping a high-load
minute removed; `tables/sensitivity.md`).

Fourth interruption, 2026-10-15 about 16:57 local: a kernel panic in the Thunderbolt/USB-C
controller firmware (`ACIO1 NMI FIQ … failed to transition to state 3`;
`/Library/Logs/DiagnosticReports/panic-full-2026-10-15-172710.0002.panic`) rebooted the host. It
is unrelated to the engines (the host's reset log records six resets from 2026-10-09, before the
measurements started). RQ2b block 0 held 174 games (87 complete pairs, none unpaired) and resumed
from pair 88 when the queue was relaunched at 17:29 local; `results/paper_v2/resume_queue.sh`
restarts the logger, queue and chain idempotently after any further reboot.

Forfeit health, 2026-10-16: RQ1a stands at 6 time forfeits in 534 games (1.1%), above the 0.5%
health limit. Four of them fell in block 1 between 11:19 and 11:30 local, while other processes used
4 to 10 cores (three by the alpha-beta arm, one by the hybrid; the last-move node rates were 0.96–2.2 M
nodes/s against the usual 14–22 M). The gate of amendment 8 only checks before a chunk, so protocol
amendment 9 adds a watchdog (`results/paper_v2/load_watchdog.sh`) that stops a running chunk at the
next completed pair when the 2-minute external CPU exceeds 450%, and shrinks chunks to 4 pairs;
deployed at 13:43 local with the queue idle behind the gate (closed since 11:40 local, external CPU
600–1,000%). The forfeited games stay in the record; the sensitivity estimate removes their pairs.

Bookkeeping fix, 2026-10-17 07:00 local: RQ1a block 0 ended with an unpaired game (501 games), which
inverted the game-count parity the amendment-9 watchdog used to find pair boundaries in block 1; one
watchdog stop (at 616 games) split a pair, and the runner, which counted a block's progress as half
its games, then skipped that pair's opening. Of RQ1a block 1's 149 games so far, 73 form complete
pairs and 3 are unpaired (two from host reboots, whose openings were replayed, and this one, whose
opening was not), so the cell loses one of its 1,000 pairs. Fixed before any cell is interpreted: the
watchdog counts parity from the current chunk's start (`games_before` in the manifest history); the
runner counts a block's progress as the openings it has used (distinct opening positions, less one
when the block's last game is unpaired, which is then replayed); `penta_stats.py` pairs consecutive
games first (falling back to round/opening keys for interleaved games). Recomputed statistics of every
cell and gate are unchanged.

Resumption and forfeits, 2026-10-26: the queue had not run since the host reboot of 2026-10-20 18:48
BST (no one resumed it; further reboots 2026-10-20 18:55, 2026-10-22 07:20 and a clean restart
2026-10-26 10:51 GMT); RQ1a then stood at 834 games. It was resumed at 11:41 GMT, and a login agent
(`~/Library/LaunchAgents/org.metalfish.paper-v2.plist`, opening `resume_queue.sh` through Terminal,
which has access to the project folder) now restarts it after any reboot. RQ1a's seventh time forfeit
(game 835, alpha-beta arm, 0.2 s over) came during a 4.2-core load spike below the amendment-9
trigger: RQ1a stands at 7 forfeits in 874 games (0.8%), six by the alpha-beta arm and one by the
hybrid. Amendment 10 lowers the watchdog trigger to 300%, the gate's threshold (watchdog restarted
15:14 GMT with no chunk running).

Health and host-load tables, 2026-10-26 (no cell outcome read): `make_figures.py` also writes
`tables/health.md` (per cell: complete pairs, unpaired games, time forfeits by arm as amendment 10
requires, how many forfeits with a host-load sample overlapped a minute above 300%, adjudications,
other terminations; `penta_stats.py` now records `time_forfeits_by`) and `tables/hostload.md` with
`tables/hostload_macros.tex` (external CPU over every logged minute and over the minutes that overlap
a game, overall and per scheduling regime taken from the queue log), which the manuscript's methods
and threats sections read. At 878 RQ1a games: 437 pairs and 4 unpaired games; time forfeits hybrid 1,
alpha-beta 6; the 5 forfeits with a load sample all overlapped a minute above 300% (games 366 and 385
predate the logger; see §A). Host load to 2026-10-26: median 9.02 cores over 8,577 logged minutes,
81% of them above 3 cores; over the 2,029 minutes that overlap a game, median 2.35 cores (95th
percentile 7.3) and 38% above 3 cores: 25% (158 of 630) before the gate, 48% (377 of 791) with the
gate alone, 40% (238 of 601) with the 450% watchdog, 3 of 7 so far with the 300% watchdog. The gate
alone did not keep in-game load low (it checks only before a chunk starts), which is what amendments
9 and 10 address; the sensitivity analysis covers the remainder.

Interim, RQ1a after block 1 (2026-10-27; block 1 ended 17:36 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 1,000 pairs). Generated by `make_figures.py` into a scratch
directory so that the paper's tables keep a single cut-off (the other three cells are mid-block 1
and were not read).

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1a | Hybrid(8) vs AB(8), 30+0.3 | 2 of 4 (blocks 0–1) | 1,002 (499 pairs + 4 unpaired) | −9.4 ± 9.1 | −22.3 ± 21.6 | 2.1% | [0, 103, 320, 76, 0] | 7 (0.7%; hybrid 1, AB 6) | GPU active 51%; 32 MCTS evals/move (median); disagreement on 36.5% of 64,533 hybrid moves; MCTS move chosen 3 of 12,071 disagreements | `RQ1a/games.pgn`, `RQ1a/summary.json` |

Per block: block 0 −11.8 ± 12.6 (501 games), block 1 −7.0 ± 13.1 (501 games); draw ratio 47% / 48%.
Host-load sensitivity (amendment 8): 172 of 499 pairs (34%) overlap a minute above 300%; without
them −12.8 ± 11.0, a shift within one standard error. No forfeit since amendment 10 (still 7; the 5
with a load sample all overlapped a minute above 300%). Pace 1.57 min per game over 26.3 h of play;
thermal log: 1,685 samples, no warning. Over the 232 in-game minutes since the 300% watchdog,
external load exceeded 3 cores in 17% (40); the crash-looping `openclaw` gateway, a steady ~1 core of
outside load, was stopped on 2026-10-27 at 16:23 UTC when the author freed the machine for the
measurements.

Interim, RQ2a after block 1 (2026-10-28; block 1 ended 00:25 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 500 pairs). Generated by `make_figures.py` into a scratch directory;
only RQ2a's rows were read (RQ2b and RQ5c are mid-block 1, and RQ1a was read after its block 1).

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2a | AB(6) vs AB(8), 30+0.3 | 2 of 4 (blocks 0–1) | 500 (250 pairs) | −20.9 ± 12.6 | −50.7 ± 30.5 | 0.1% | [0, 59, 162, 29, 0] | 0 | — | `RQ2a/games.pgn`, `RQ2a/summary.json` |

Per block: block 0 −9.7 ± 17.0 (250 games), block 1 −32.1 ± 18.4 (250 games); draw ratio 44% / 50%;
451 of the 500 games were adjudicated. Host-load sensitivity (amendment 8): 75 of 250 pairs (30%)
overlap a minute above 300%; without them −25.9 ± 14.7, a shift within one standard error. Pace 1.59
min per game over 13.2 h of play; thermal log: 822 samples, no warning. Block 1 was played from
2026-10-27 17:36 to 2026-10-28 00:25 UTC, all of it under the 300% watchdog: outside load exceeded 3
cores in 12 of its 410 in-game minutes (3%), mostly a spike from desktop applications at 22:06–22:12
UTC that the watchdog stopped.

Interim, RQ2b after block 1 (2026-10-28; block 1 ended 16:28 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 500 pairs). Generated by `make_figures.py` into a scratch directory;
only RQ2b's rows were read (RQ5c is mid-block 1; RQ1a and RQ2a were read after their block 1).

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2b | Hybrid-split(8) (6 AB threads + MCTS worker) vs AB(6), 30+0.3 | 2 of 4 (blocks 0–1) | 500 (250 pairs) | −20.2 ± 12.8 | −48.1 ± 30.5 | 0.1% | [1, 57, 162, 30, 0] | 1 (0.2%; hybrid-split arm) | transformer enabled on 58% of 32,357 hybrid moves; 65 MCTS evals/move (median); disagreement on 33.9%; MCTS move chosen 0 of 6,403 disagreements | `RQ2b/games.pgn`, `RQ2b/summary.json` |

Per block: block 0 −27.9 ± 18.4 (250 games), block 1 −12.5 ± 17.8 (250 games); draw ratio 50% / 44%;
444 of the 500 games were adjudicated. Decision reasons: time-safety 13,482, agreement 12,472, alpha-beta
verification rejecting the MCTS move 4,947, alpha-beta default 1,456. The forfeit (game 477, the
hybrid-split arm, 32 ms over) came at 14:22–14:24 UTC with 5.3–5.9 cores of outside load, inside a game
the 300% watchdog could not stop before its pair ended. Host-load sensitivity (amendment 8): 89 of 250
pairs (36%) overlap a minute above 300%; without them −19.4 ± 15.9, a shift within one standard error.
Pace 1.56 min per game over 13.0 h of play; thermal log: 840 samples, no warning. Block 1 was played
from 2026-10-28 00:25 to 16:28 UTC, with the gate closed for long stretches while the author's other
work (a virtual machine, test suites, agent jobs) used up to 10 cores: outside load exceeded 3 cores in
43 of its 428 in-game minutes (10%).

Interim, RQ5c after block 1 (2026-10-28; block 1 ended 23:33 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 500 pairs). Generated by `make_figures.py` into a scratch directory;
only RQ5c's rows were read. With this, every primary cell has completed two of its four blocks.

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ5c | AB(8) vs Stockfish 19 (8 threads), 30+0.3 | 2 of 4 (blocks 0–1) | 500 (250 pairs) | +4.9 ± 12.5 | +11.8 ± 30.5 | 77.6% | [0, 39, 165, 46, 0] | 0 | — | `RQ5c/games.pgn`, `RQ5c/summary.json` |

Per block: block 0 +2.8 ± 17.7 (250 games), block 1 +6.9 ± 17.8 (250 games); draw ratio 49% / 48%;
452 of the 500 games were adjudicated. Host-load sensitivity (amendment 8): 99 of 250 pairs (40%)
overlap a minute above 300%; without them +2.3 ± 15.3, a shift within one standard error (after block
0 alone the shift had exceeded it). Pace 1.57 min per game over 13.1 h of play; thermal log: 823
samples, no warning. Block 1 was played on 2026-10-28 from 16:28 to 23:33 UTC: outside load exceeded 3
cores in 27 of its 404 in-game minutes (7%).

Interim, RQ1a after block 2 (2026-10-29; block 2 ended 14:16 UTC): monitoring only, **no decision is
taken** (fixed N; RQ1a is interpreted once, after block 3). Generated by `make_figures.py` into a
scratch directory; only RQ1a's rows were read.

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1a | Hybrid(8) vs AB(8), 30+0.3 | 3 of 4 (blocks 0–2) | 1,502 (749 pairs + 4 unpaired) | −8.4 ± 7.3 | −20.2 ± 17.6 | 1.2% | [0, 147, 491, 111, 0] | 7 (0.47%; hybrid 1, AB 6; none in block 2) | transformer enabled on 52% of 96,989 hybrid moves; 33 MCTS evals/move (median); disagreement on 36.9%; MCTS move chosen 6 of 18,607 disagreements (all `mcts_root_reject_q_gap`) | `RQ1a/games.pgn`, `RQ1a/summary.json` |

Per block: block 0 −11.8 ± 12.6 (501 games), block 1 −7.0 ± 13.1 (501), block 2 −6.3 ± 12.1 (500); draw
ratio 47% / 48% / 49%; 1,363 of the 1,502 games were adjudicated. Decision reasons: time-safety 46,552,
agreement 31,830, alpha-beta verification rejecting the MCTS move 14,422, alpha-beta default 4,179,
`mcts_root_reject_q_gap` 6. Host-load sensitivity (amendment 8): 180 of 749 pairs (24%) overlap a
minute above 300%; without them −9.8 ± 8.2, a shift within one standard error. Pace 1.57 min per game
over 39.4 h of play; thermal log: 2,532 samples, no warning. Block 2 was played from 2026-10-28 23:33
to 2026-10-29 14:16 UTC: outside load exceeded 3 cores in 8 of its 872 in-game minutes (1%).

Interim, RQ2a after block 2 (2026-10-29; block 2 ended 20:56 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 500 pairs). Generated by `make_figures.py` into a scratch directory;
only RQ2a's rows were read.

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2a | AB(6) vs AB(8), 30+0.3 | 3 of 4 (blocks 0–2) | 750 (375 pairs) | −21.8 ± 10.1 | −53.7 ± 24.9 | 0.0% | [0, 88, 246, 41, 0] | 0 | — | `RQ2a/games.pgn`, `RQ2a/summary.json` |

Per block: block 0 −9.7 ± 17.0, block 1 −32.1 ± 18.4, block 2 −23.7 ± 17.0 (250 games each); draw ratio
44% / 50% / 47%; 679 of the 750 games were adjudicated. Host-load sensitivity (amendment 8): 87 of 375
pairs (23%) overlap a minute above 300%; without them −24.2 ± 11.3, a shift within one standard error.
Pace 1.58 min per game over 19.8 h of play; thermal log: 1,233 samples, no warning. Block 2 was played
on 2026-10-29 from 14:17 to 20:56 UTC: outside load exceeded 3 cores in 9 of its 399 in-game minutes
(2%).

Interim, RQ2b after block 2 (2026-10-30; block 2 ended 03:59 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 500 pairs). Generated by `make_figures.py` into a scratch directory;
only RQ2b's rows were read.

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2b | Hybrid-split(8) (6 AB threads + MCTS worker) vs AB(6), 30+0.3 | 3 of 4 (blocks 0–2) | 750 (375 pairs) | −17.6 ± 10.5 | −41.9 ± 24.9 | 0.0% | [1, 84, 242, 48, 0] | 1 (0.1%; hybrid-split arm; none in block 2) | transformer enabled on 59% of 48,825 hybrid moves; 69 MCTS evals/move (median); disagreement on 33.9%; MCTS move chosen 0 of 9,798 disagreements | `RQ2b/games.pgn`, `RQ2b/summary.json` |

Per block: block 0 −27.9 ± 18.4, block 1 −12.5 ± 17.8, block 2 −12.5 ± 18.2 (250 games each); draw ratio
50% / 44% / 49%; 672 of the 750 games were adjudicated. Decision reasons: time-safety 19,921, agreement
19,106, alpha-beta verification rejecting the MCTS move 7,601, alpha-beta default 2,197. Host-load
sensitivity (amendment 8): 91 of 375 pairs (24%) overlap a minute above 300%; without them −16.5 ±
12.0, a shift within one standard error. Pace 1.56 min per game over 19.5 h of play; thermal log: 1,260
samples, no warning. Block 2 was played from 2026-10-29 20:56 to 2026-10-30 03:59 UTC: outside load
exceeded 3 cores in 2 of its 423 in-game minutes.

Interim, RQ5c after block 2 (2026-10-30; block 2 ended 10:39 UTC): monitoring only, **no decision is
taken** (fixed N; the cell runs to 500 pairs). Generated by `make_figures.py` into a scratch directory;
only RQ5c's rows were read. With this, every primary cell has completed three of its four blocks.

| Cell | Comparison | Blocks done | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | Time forfeits | Telemetry | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ5c | AB(8) vs Stockfish 19 (8 threads), 30+0.3 | 3 of 4 (blocks 0–2) | 750 (375 pairs) | +2.3 ± 10.2 | +5.7 ± 24.9 | 67.3% | [0, 60, 250, 65, 0] | 0 | — | `RQ5c/games.pgn`, `RQ5c/summary.json` |

Per block: block 0 +2.8 ± 17.7, block 1 +6.9 ± 17.8, block 2 −2.8 ± 17.2 (250 games each); draw ratio
49% / 48% / 54%; 674 of the 750 games were adjudicated. Host-load sensitivity (amendment 8): 105 of 375
pairs (28%) overlap a minute above 300%; without them −0.6 ± 11.6, a shift within one standard error.
Pace 1.57 min per game over 19.6 h of play; thermal log: 1,238 samples, no warning. Block 2 was played
on 2026-10-30 from 03:59 to 10:39 UTC: outside load exceeded 3 cores in 6 of its 400 in-game minutes
(2%).

**RQ1a final reading** (2026-10-31; block 3 ended 07:34 UTC; the cell is complete and is interpreted
once, by the rule of amendment 12 (a)). Computed from `RQ1a/games.pgn` with `penta_stats.py` and
`robustness.py`; no other cell was read.

| Cell | Comparison | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | W / D / L | Time forfeits | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1a | Hybrid(8) vs AB(8), 30+0.3 | 2,002 (999 pairs + 4 unpaired) | **−4.3 [−10.7, +2.0]** (± 6.3) | −10.5 ± 15.2 | 8.9% | [0, 185, 654, 160, 0] | 504 / 973 / 525 | 7 (0.35%; AB 6, hybrid 1) | `RQ1a/games.pgn`, `RQ1a/summary.json` |

Decision rule (amendment 12 (a)): the 95% interval spans zero, so RQ1a is **inconclusive at this N**:
no effect of the concurrent GPU search was detected at an equal `Threads` setting, so the GPU path does
not pay for itself at this budget on this machine; effects above +2.0 Elo or below −10.7 Elo are
excluded.

Sensitivity analyses (amendment 12 (b), (h); the primary estimate's standard error is 3.2 Elo): zero
outside load (pair-level regression on the fraction of high-load minutes, 794 pairs with samples, 226
exposed; slope −0.001 ± 0.020 per unit fraction) −3.0 ± 8.0; low-load pairs only (amendment 8; 773 of
999 pairs kept) −3.4 ± 7.1; forfeit pairs removed −4.6 ± 6.3; forfeits scored as draws −5.2 ± 6.3;
without the first 92 openings −4.8 ± 6.7. No estimate moves by more than one standard error, so the
result is **not load-sensitive**. Blocks (amendment 12 (g)): −11.8 ± 12.6, −7.0 ± 13.1, −6.3 ± 12.1,
+7.6 ± 12.7 (draw ratio 47–50%); Cochran's Q = 5.03 (3 df), p = 0.17; bootstrap over the runner's 218
chunks, 95% interval [−9.9, +2.2]. Load-log coverage: block 0 45 of 250 pairs (the logger started
during block 0), blocks 1–3 complete. The 183 games set aside by amendment 7 (rc1 binary, first 92
openings): 91 pairs + 1 unpaired, −9.5 ± 18.6.

Telemetry (`summary.json`, 128,952 hybrid moves): transformer enabled on 53.1%; median 36 MCTS
evaluations and 38 playouts per transformer-enabled move; alpha-beta and MCTS disagree on 36.7% of the
traced moves; the MCTS move was played on 7 of 25,138 disagreements, all through
`mcts_root_reject_q_gap`. Decision reasons: time-safety 60,496, agreement 43,318, alpha-beta
verification rejecting the MCTS move 19,579, alpha-beta default 5,552. The interim interval after
three blocks (−8.4 ± 7.3) had excluded zero; as registered, no decision was taken on it.

System update and restart, 2026-10-31 (amendment 13): macOS 27.2 (final release) installed automatically
at 13:40 UTC over the release candidate while RQ2a block 3 was playing; RQ2a game 945 (AB(6) arm) was
lost on time by 873 ms during the installation, and the host restarted at 13:46 UTC before the pair's
second game finished (the game is unpaired and its opening is replayed). The login agent restarted the
logger, the queue, the watchdog and the chain at 13:57 UTC. The `openclaw` gateway had relaunched with
the login session and was crash-looping again (1,825 restarts by 15:05 UTC); it was stopped again at
15:05 UTC, while Spotlight re-indexed after the update. The binary's SHA-256 (`8c534bb6…`) was
re-checked and the engine starts normally on the new build. The host's update history
(`docs/paper_v2/os_builds.json`) shows three earlier automatic updates during the programme: 27.2 beta
(2026-09-23), beta 3 (2026-10-16 09:31 UTC) and the release candidate (2026-10-20 17:39 UTC).

**RQ2a final reading** (2026-10-31; block 3 ended 18:22 UTC; the cell is complete and is interpreted
once, by amendments 12 and 13). Computed from `RQ2a/games.pgn` with `penta_stats.py` and
`robustness.py`; no other cell was read.

| Cell | Comparison | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | W / D / L | Time forfeits | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2a | AB(6) vs AB(8), 30+0.3 | 1,001 (500 pairs + 1 unpaired) | **−19.8 [−28.6, −11.1]** (± 8.8) | −48.8 ± 21.5 | 0.0% | [0, 114, 329, 57, 0] | 230 / 483 / 288 | 1 (0.1%; AB(6), the unpaired game 945 during the system update) | `RQ2a/games.pgn`, `RQ2a/summary.json` |

Reading (02 §5 RQ2; exploratory, amendment 12 (g)): the 95% interval is below zero, so giving two of
the eight alpha-beta threads to the GPU side, as the pre-amendment split did, costs about 20 Elo at
30+0.3 on this machine.

Sensitivity analyses (amendment 12 (b); the primary estimate's standard error is 4.5 Elo): zero outside
load (pair-level regression on the fraction of high-load minutes; all 500 pairs have samples, 135
exposed; slope +0.059 ± 0.026 per unit fraction) −24.7 ± 9.7, which differs from the primary estimate by
4.9 Elo, more than one standard error, so RQ2a is **load-sensitive** and both estimates are reported:
the positive slope means the gap between six and eight threads was smaller in pairs played under outside
load, so the estimated price of the two threads at zero outside load is larger than the all-games
estimate. Low-load pairs only (amendment 8; 365 of 500 kept) −23.8 ± 10.1 (shift −4.0, within one
standard error). The cell's only forfeit is in the unpaired game, so removing forfeit pairs or scoring
forfeits as draws leaves the estimate at −19.8 ± 8.8. In-game minutes with outside CPU above 300%: 199
of 1,629 (12.2%); median outside CPU 128%. Blocks (amendment 12 (g)): −9.7 ± 17.0, −32.1 ± 18.4, −23.7 ±
17.0, −13.9 ± 17.5 (draw ratio 44–52%); Cochran's Q = 3.69 (3 df), p = 0.30; bootstrap over the runner's
113 chunks, 95% interval [−31.4, −11.7]. Pairs exposed to a minute above 300%, by block (descriptive):
65 of 125 (52%; block 0 was played before the load gate of amendment 8), 10, 12 and 48 (38%; block 3 was
played on 2026-10-31, around the system update). Load-log coverage: complete in all four blocks.
Amendment 12 (b)(ii) says that block 0 of RQ2a predates the logger; it does not: the logger's first line
is 2026-10-15 00:57 UTC, during RQ1a's block 0 and 2.5 h before RQ2a's first game (03:24 UTC), so only
RQ1a's block 0 is partly uncovered (45 of 250 pairs).

By macOS build (amendment 13 (b); a pair belongs to the build its first game started on): 27.2 beta,
250 games (125 pairs, exactly block 0), −9.7 ± 17.0; release candidate, 695 games (347 pairs + the
unpaired game), −25.1 ± 10.6; final release, 56 games (28 pairs, below the 100-pair threshold, no
estimate). Release candidate minus beta is −15.3 (standard error 10.2), larger than its standard error,
so it is reported: the beta pairs are block 0, the only block played without the load gate and the
one with the most outside load, so build, scheduling regime and load cannot be separated here, and
Cochran's Q over the blocks does not indicate heterogeneity (p = 0.30).

Health (`make_figures.py`): adjudicated 899 of 1,001 games; no other terminations; 1.59 min per game
over 26.5 h of play; thermal log: 1,666 samples, no warning. GPU and memory covariates (amendment
12 (c); 1,005 in-game minutes logged since 2026-10-27 21:20 UTC): GPU utilisation median 2% (p95 50%;
neither arm uses the GPU), GPU in-use memory median 799 MB, memory available median 62% (minimum 48%),
5,200 page-outs during the cell's games.

RQ1a by macOS build (amendment 13 (c); computed on 2026-10-31 after RQ1a's interpretation, and labelled
as such): 27.2 beta, 514 games (256 pairs + 2 unpaired: block 0 and the first 6 pairs of block 1),
−10.9 ± 12.4; beta 3, 320 games (159 pairs + 2 unpaired, within block 1), +2.2 ± 15.7; release
candidate, 1,168 games (584 pairs: the rest of block 1 and blocks 2–3), −3.3 ± 8.3. Beta 3 minus beta is
+13.0 (standard error 10.2), larger than its standard error; release candidate minus beta (+7.6, standard
error 7.6) and minus beta 3 (−5.5, standard error 9.1) are not. The builds follow the blocks, whose
estimates Cochran's Q does not flag (p = 0.17).

**RQ2b final reading** (2026-11-01; block 3 ended 01:45 UTC; the cell is complete and is interpreted
once, by amendments 12 and 13). Computed from `RQ2b/games.pgn` with `penta_stats.py` and
`robustness.py`; the derived RQ2c also uses `RQ2a/games.pgn` (interpreted on 2026-10-31); no other
cell was read.

| Cell | Comparison | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | W / D / L | Time forfeits | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2b | Hybrid-split(8) (6 AB threads + MCTS worker) vs AB(6), 30+0.3 | 1,000 (500 pairs) | **−17.0 [−26.2, −7.9]** (± 9.1) | −40.3 ± 21.5 | 0.0% | [1, 113, 320, 66, 0] | 230 / 491 / 279 | 1 (0.1%; hybrid-split arm, block 1, 2026-10-28, by 32 ms) | `RQ2b/games.pgn`, `RQ2b/summary.json` |
| RQ2c (derived) | Hybrid-split(8) vs AB(8) = RQ2b + RQ2a (AB(6) vs AB(8)); shared-opening covariance included | 1,000 + 1,001 | **−36.9 [−49.6, −24.2]** (± 12.7) | — | — | — | — | — | `RQ2b/games.pgn`, `RQ2a/games.pgn` |

Reading (02 §5 RQ2; exploratory, amendment 12 (g)): the 95% interval is below zero, so with the same six
alpha-beta threads as its opponent, the split hybrid's GPU search (one MCTS worker on the rest of the
eight-thread budget) cost about 17 Elo at 30+0.3 on this machine instead of adding strength. Every move
the hybrid played was its alpha-beta side's move (the MCTS move was chosen in none of the 13,152
disagreements; telemetry below), so the loss does not come from moves chosen by MCTS. Derived RQ2c: the
split hybrid against AB(8) is −36.9 Elo (95% interval [−49.6, −24.2]), about 20 Elo for the two threads
(RQ2a) and 17 Elo for running the GPU search beside the six (RQ2b). 02 §5 writes RQ2c ≈ RQ2b − RQ2a with
RQ2a scored from AB(8)'s side; RQ2a is scored here from AB(6)'s side, so the two are added. The outcome
02 §5 singles out, a positive RQ2b with a negative RQ2c (the GPU search helps only when CPU threads are
free), did not occur: RQ2b is negative, so the GPU search did not help even with the alpha-beta threads
held equal. The direct RQ2c cell is queued after the oracle (`queue_after_must_rc2_gated.sh`).

Sensitivity analyses (amendment 12 (b); the primary estimate's standard error is 4.7 Elo): zero outside
load (pair-level regression; 499 of 500 pairs have samples, 113 exposed; slope −0.008 ± 0.025 per unit
fraction) −16.3 ± 10.1 (shift +0.7); low-load pairs only (amendment 8; 387 of 500 kept) −16.6 ± 10.3
(shift +0.4); forfeit pairs removed (499 pairs) −16.4 ± 9.0 (shift +0.7); forfeits scored as draws
−16.7 ± 9.0 (shift +0.3). No estimate moves by more than one standard error, so RQ2b is **not
load-sensitive**. In-game minutes with outside CPU above 300%: 221 of 1,697 (13.0%); median outside
CPU 140%. Blocks (amendment 12 (g)): −27.9 ± 18.4, −12.5 ± 17.8, −12.5 ± 18.2, −15.3 ± 18.5 (draw
ratio 44–54%); Cochran's Q = 1.85 (3 df), p = 0.60; bootstrap over the runner's 120 chunks, 95%
interval [−26.5, −7.6]. Pairs exposed to a minute above 300%, by block (descriptive): 67 of 125 (54%;
block 0, played on 2026-10-15 under the gate of amendment 8, then checked only between chunks of 10
pairs, before the in-chunk watchdog of amendments 9 and 10), 22 (18%), 2 (2%) and 22 (18%). Load-log
coverage: 124 of 125 pairs in block 0, complete in blocks 1–3.

By macOS build (amendment 13 (b); a pair belongs to the build its first game started on): 27.2 beta,
250 games (125 pairs, exactly block 0), −27.9 ± 18.4; release candidate, 500 games (250 pairs, blocks
1–2), −12.5 ± 12.7; final release, 250 games (125 pairs, exactly block 3), −15.3 ± 18.5. Release
candidate minus beta is +15.3 (standard error 11.4), larger than its standard error, so it is
reported: the beta pairs are block 0, which had the most outside load, the lowest MCTS evaluation rate
(below) and no in-chunk watchdog, so build, scheduling and load cannot be separated; Cochran's Q over
the blocks does not indicate heterogeneity (p = 0.60). Final release minus beta (+12.6, standard error
13.3) and minus release candidate (−2.8, standard error 11.5) are within their standard errors.

Telemetry (`summary.json`, 65,371 hybrid moves): transformer enabled on 59.1%; median 70 MCTS
evaluations and 76 playouts per transformer-enabled move; alpha-beta and MCTS disagree on 34.0% of the
traced moves; the MCTS move was played on none of the 13,152 disagreements. Decision reasons:
time-safety 26,726, agreement 25,493, alpha-beta verification rejecting the MCTS move 10,166,
alpha-beta default 2,986.

GPU health (amendment 12 (c); MCTS evaluations per second of move time on the transformer-enabled
moves, matched to the PGN by position: 38,604 of 38,640 trace records; the cell's rate is the
six-thread point of M3): median 89.5 (p10 24.4) in block 0, 122.7 (39.7) in block 1, 129.7 (46.9) in
block 2, 118.8 (41.4) in block 3 and 117.6 (35.8) over the cell; total evaluations over total move
time 56.7, 78.3, 83.5, 75.9 and 73.7. Block 0's median is 28% below the mean of blocks 1–3 (123.7), and
so is its total rate (79.2); block 0 is also the beta build and the block with the most outside load.

Health (`make_figures.py`): adjudicated 898 of 1,000 games (390 draws; wins 230 for the hybrid, 278 for
AB(6)), 101 ended normally, 1 time forfeit (in a minute with outside CPU above 300%); no other
terminations; 1.56 min per game over 26.0 h of play; thermal log: 1,684 samples, no warning. GPU and
memory covariates (amendment 12 (c); 1,287 of the cell's 1,697 in-game minutes carry them, blocks 1–3,
logged since 2026-10-27 21:20 UTC): GPU utilisation median 87% (p95 89%; the hybrid's own inference;
block medians 88%, 88%, 84%), GPU in-use memory median 918 MB (block medians 966, 937 and 642 MB),
memory available median 60% (minimum 36%), 9,164 page-outs during the cell's games.

**RQ5c final reading** (2026-11-01; block 3 ended 08:33 UTC; the cell is complete and is interpreted
once, by amendments 12 and 13). Computed from `RQ5c/games.pgn` with `penta_stats.py` and
`robustness.py`; no other cell was read. With this reading the four MUST game cells (RQ1a, RQ2a, RQ2b
and RQ5c) are complete.

| Cell | Comparison | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | W / D / L | Time forfeits | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ5c | AB(8) (frozen build, alpha-beta only) vs Stockfish 19 (8 threads), 30+0.3, `Hash` 2048 MB each; move overhead 50 ms for AB(8), Stockfish 19's default 10 ms | 1,000 (500 pairs) | **+1.4 [−7.3, +10.1]** (± 8.7) | +3.5 ± 21.5 | 62.3% | [0, 79, 338, 83, 0] | 249 / 506 / 245 | 0 | `RQ5c/games.pgn`, `RQ5c/summary.json` |

Reading (02 §5 RQ5c; exploratory, amendment 12 (g); read as a parity interval with no equivalence
claim, amendment 12 (a)): the 95% interval spans zero, so no difference between the frozen build's
alpha-beta engine and Stockfish 19 at equal threads and hash was detected at 30+0.3 on this machine.
The interval excludes differences above +10.1 Elo and below −7.3 Elo; no equivalence margin was
registered, so the reading does not say that the two engines are equally strong. It is consistent
with the expectation registered in 02 §5 (≈ 0 after the 2026-10-12 port). AB(8) is the baseline of
RQ1a and RQ2a, so those comparisons are against an engine within this range of Stockfish 19. Threads
and hash are equal, the move overhead is not: AB(8) uses the 50 ms of every MetalFish arm
(`experiments.json`; the paper's match setup states it) and Stockfish 19 its default 10 ms, which
leaves AB(8) slightly less time per move; the size of that effect is not measured.

Sensitivity analyses (amendment 12 (b); the primary estimate's standard error is 4.4 Elo): zero outside
load (pair-level regression; all 500 pairs have samples, 109 exposed; slope +0.011 ± 0.020 per unit
fraction) +0.3 ± 9.5 (shift −1.1); low-load pairs only (amendment 8; 391 of 500 kept) −0.4 ± 9.6
(shift −1.8); there were no time forfeits, so both forfeit variants equal the primary estimate. No
estimate moves by more than one standard error, so RQ5c is **not load-sensitive**. In-game minutes
with outside CPU above 300%: 248 of 1,619 (15.3%); median outside CPU 146%. Blocks (amendment 12 (g)):
+2.8 ± 17.7, +6.9 ± 17.8, −2.8 ± 17.2, −1.4 ± 16.6 (draw ratio 48–54%); Cochran's Q = 0.73 (3 df),
p = 0.87; bootstrap over the runner's 113 chunks, 95% interval [−8.1, +11.2]. Pairs exposed to a
minute above 300%, by block (descriptive): 81 of 125 (65%; block 0, played on 2026-10-15/16 under the
gate of amendment 8, then checked only between chunks of 10 pairs, before the in-chunk watchdog of
amendments 9 and 10), 18 (14%), 6 (5%) and 4 (3%). Load-log coverage: complete in all four blocks.

By macOS build (amendment 13 (b); a pair belongs to the build its first game started on): 27.2 beta,
250 games (125 pairs, exactly block 0), +2.8 ± 17.7; release candidate, 500 games (250 pairs, blocks
1–2), +2.1 ± 12.4; final release, 250 games (125 pairs, exactly block 3), −1.4 ± 16.6. All three
differences are within their standard errors: release candidate minus beta −0.7 (standard error
11.0), final release minus beta −4.2 (12.4), final release minus release candidate −3.5 (10.6).

Health (`make_figures.py`): adjudicated 894 of 1,000 games (400 draws; wins 249 for AB(8), 245 for
Stockfish 19), 106 ended normally (all draws); no time forfeits and no other terminations; 1.58 min
per game over 26.3 h of play; thermal log: 1,661 samples, no warning. Host covariates (amendment
12 (c); 1,212 of the cell's 1,619 in-game minutes carry them, blocks 1–3, logged since 2026-10-27
21:20 UTC): GPU utilisation median 1% (p95 22%; neither engine uses the GPU), GPU in-use memory
median 674 MB, memory available median 62% (minimum 35%), 6,073 page-outs during the cell's games.

**Rounding corrections** (2026-11-01, before the correction note was filed or the paper used these
values). Four values in the final readings above had been rounded twice: the final-reading helper
prints two decimals, and the records rounded that print again. Checked at full precision (each cell's
`summary.json` and a recomputation from the PGNs) and against the tables of `make_figures.py
--final-analyses` (`robustness.md`, `setaside.md`, `os_builds.md`), they are: RQ1a −4.3 Elo (exactly
−4.348; recorded as −4.4), with its interval [−10.7, +2.0] unchanged; the 183 set-aside games
−9.5 ± 18.6 (−9.547; recorded as −9.6); RQ1a's beta 3 minus beta +13.0 (+13.046; recorded as +13.1);
RQ2a's zero-load estimate −24.7 ± 9.7 (± 9.747; recorded as ± 9.8). The entries above, the
strength-check table (07) and the correction package now carry these values. No reading changes:
RQ1a's interval still spans zero, the build difference is still larger than its standard error, and
RQ2a's zero-load shift (4.9 Elo) is unchanged. Every other value of the four final readings matches
the recomputation.

**Final analyses** (2026-11-01; `make_figures.py --final-analyses` at 09:47–09:50 UTC, after the parser
and labelling fixes of PR #130 and the verification counts of PR #132, on a view of `results/paper_v2`
without RQ1b, which began at 08:50 UTC: the four completed cells, the pilot and the set-aside games,
with `hostload.log` cut after its 08:33:49 UTC sample, the minute in which RQ5c's last game ended; the
tables are reproduced from the deposit by leaving out the cells that were still running and cutting
`hostload.log` there). The tables are written to `docs/paper_v2/tables/` (git-ignored) and copied with
the deposit; the paper's tables and figures are this run's. Values the paper uses that the final
readings above do not already give:

| Cell | Arm (moves) | Moves | Mean s per move | Median depth | Median nodes/s | Mean s by own move number: 1–10, 11–20, 21–30, 31–40, 41–60, 61+ |
| --- | --- | --- | --- | --- | --- | --- |
| RQ1a | AB(8) | 128,863 | 0.735 | 21 | 10.78 M | 1.33, 1.09, 0.86, 0.58, 0.39, 0.31 |
| RQ1a | Hybrid(8), all | 128,877 | 0.735 | 21 | 10.52 M | 1.31, 1.09, 0.87, 0.59, 0.39, 0.30 |
| RQ1a | Hybrid(8), transformer-enabled | 68,408 | 1.048 | 22 | 8.88 M | 1.31, 1.10, 0.95, 0.76, 0.59, 0.47 |
| RQ1a | Hybrid(8), time-safety | 60,469 | 0.380 | 19 | 12.59 M | 1.60, 0.80, 0.62, 0.45, 0.35, 0.30 |
| RQ2a | AB(6) | 66,225 | 0.724 | 21 | 8.79 M | 1.31, 1.09, 0.88, 0.58, 0.39, 0.31 |
| RQ2a | AB(8) | 66,194 | 0.720 | 21 | 11.38 M | 1.29, 1.06, 0.87, 0.59, 0.41, 0.31 |
| RQ2b | AB(6) | 65,328 | 0.722 | 21 | 8.68 M | 1.27, 1.03, 0.88, 0.61, 0.41, 0.31 |
| RQ2b | Hybrid-split(8), all | 65,344 | 0.718 | 21 | 8.97 M | 0.96, 1.06, 1.02, 0.70, 0.42, 0.31 |
| RQ2b | Hybrid-split(8), transformer-enabled | 38,618 | 0.952 | 22 | 7.06 M | 0.96, 1.06, 1.06, 0.85, 0.61, 0.51 |
| RQ2b | Hybrid-split(8), time-safety | 26,726 | 0.379 | 19 | 13.00 M | —, 0.94, 0.69, 0.50, 0.35, 0.31 |
| RQ5c | AB(8) | 65,171 | 0.728 | 21 | 11.01 M | 1.32, 1.10, 0.86, 0.57, 0.39, 0.30 |
| RQ5c | Stockfish 19 (8 threads) | 65,170 | 0.733 | 21 | 11.52 M | 1.38, 1.13, 0.85, 0.55, 0.37, 0.30 |

(`timing.md`: the PGN move comments, with the hybrid's moves split into transformer-enabled and
time-safety moves by the traces; a move counts for its position, so the classes are not matched game
phases: time-safety moves come late in the game. The split hybrid's time-safety moves use the full
eight-thread pool. A hybrid move's nodes per second are the engine's own report: alpha-beta and MCTS
nodes over the hybrid's search time, `365f36a:src/hybrid/hybrid_search.cpp` 4164-4182.) The split hybrid
used less time than AB(6) in its first ten moves (0.96 against 1.27 s) and more in moves 21–40; RQ1a's
hybrid and AB(8) used their clocks alike. Over all moves, RQ1a's hybrid reported a median of 10.52
million nodes per second against AB(8)'s 10.78 million in the same games, and Stockfish 19 searched at
11.52 million against AB(8)'s 11.01 million in RQ5c.

Agreement stop (engine log, "engines agree, stopping early"; `flags.md`): RQ1a 660, 1.0% of its 68,456
transformer-enabled moves (`summary.json`, `moves_hybrid`); RQ2b 6,485, 16.8% of 38,645; pilot 3 of
3,278. The stop requires at least 100 MCTS nodes; the median MCTS evaluations per enabled move were 36
in RQ1a and 70 in RQ2b. "Subsearch stop grace expired": 0 in each cell. Root hints and verification
(`flags.md`, traced moves other than time-safety moves): RQ1a 68,456 moves, 57,308 with hints, 12,225
with the verification sub-search; RQ2b 38,645, 35,353 and 11,695; pilot 3,278, 2,711 and 584. On the
disagreements, RQ1a had hints on 20,140 of 25,138, with the first hint the MCTS move on 14,211 and the
alpha-beta move on 2,429; RQ2b 11,545 of 13,152, 8,149 and 1,318; pilot 935 of 1,165, 610 and 116.

Thread pool of the split hybrid (RQ2b). On a time-safety move the hybrid plays plain alpha-beta on the
full `Threads` pool (`365f36a:src/uci/uci.cpp` 312-317 and 1731-1736), and on a transformer-enabled move
it sizes the pool to its alpha-beta share, only when the size changes (`hybrid_search.cpp` 3813-3820); a
resize rebuilds the workers with empty per-thread histories. In the split configuration
(`HybridOversubscribe=false`, share 6 of 8) the alpha-beta side therefore had six threads on the
transformer-enabled moves (59.1% of its moves) and eight on the time-safety moves (40.9%), and each
change rebuilt the pool. Switches between the two classes in game order (`switches.md`): RQ2b 818 from a
transformer-enabled to a time-safety move over its 1,000 games and none back, so the pool was rebuilt at
the first transformer-enabled move of each game and again at the switch to time-safety moves in 818
games; RQ1a 1,779 and 14 over 2,002 games, without any resize, because its share is the full pool; pilot
93 and 1. This qualifies the RQ2b reading above: the alpha-beta threads were equal to AB(6)'s on the
split hybrid's transformer-enabled moves, while on its time-safety moves it had two more; the cell
measures the split configuration as built, and the derived RQ2c, a difference of measured scores, is
unaffected.

In-game MCTS evaluation rate (M3; amendment 12 (c); `gpu_health.md`; evaluations per second of move
time on the matched transformer-enabled moves, median with 10th percentile, and total evaluations over
total move time): RQ1a, eight alpha-beta threads plus the worker, 68,384 of 68,445 records matched:
58 (19), total 38; by block 53 (16), 55 (16), 64 (20), 78 (26), totals 31, 34, 37, 49. RQ2b, six
threads plus the worker: 118 (36), total 74 (117.6 and 73.7 in its final reading). Pilot: 40 (13), total
27. M1's batch-1 ceiling on the frozen build is 139 evaluations per second (FP16; §B).

RQ1a's GPU and memory covariates (amendment 12 (c); `covariates.md`; 1,750 in-game minutes carry them,
logged since 2026-10-27 21:20 UTC): GPU utilisation median 58% (p95 73%), GPU in-use memory median
834 MB, memory available median 56% (minimum 45%), 8,062 page-outs during the cell's games. RQ1a's
adjudications (`adjudications.md`): 789 draws; wins 498 for the hybrid, 524 for AB(8).

Host load (`hostload.md`, `outside_processes.md`): 16,256 minutes logged from 2026-10-15 to 2026-11-01
08:33 UTC, median outside load 4.20 cores (95th percentile 11.6), above three cores in 56% (9,036); the
7,723 minutes that overlap a game of these cells: median 1.44 cores (95th percentile 5.6), 14% (1,076)
above three cores; by scheduling regime: before the gate 25% of 630 minutes, gate only 48% of 791, gate
with the watchdog at 450% 40% of 601, gate with the watchdog at 300% 5% of 5,701. Largest outside
process in the 1,076 minutes above three cores: the virtual machine (Virtualization.framework) 251,
Homebrew Python 3.11 229, Python 3.14 127 (the interpreter of this study's runner and analysis scripts,
among others), the editor (Kiro) 100, openclaw-gateway 86, mediaanalysisd 44, Python 3.12 41, Spotlight
indexing 38, openclaw 33, WebKit web content 20.
Host-load logger extended, 2026-10-27 21:20 UTC (amendment 12c): every line of `hostload.log` now also
carries `gpu_util` (IOAccelerator "Device Utilization %", an instantaneous sample that includes the
engines' own GPU use), `gpu_mem_mb` (GPU in-use system memory), `mem_level` (`kern.memorystatus_level`,
% of memory available) and the cumulative `pageouts` and `swapouts` since boot. First extended line:
21:20:37, gpu_util 2%, gpu_mem_mb 315, mem_level 80%, pageouts 78,012, swapouts 12,986,612. The
logger was restarted under the same name, so the gate and the watchdog are unchanged; one minute
went unsampled during the swap.

Trace of the frozen build's gate L14, parsed 2026-10-27 for the manuscript (`parse_hybrid_trace.py` on
`sprt/L14_hybrid_lean_vs_ab8_30p03/engines.log.gz` → `moves.jsonl`): 6,855 hybrid decisions;
3,413 time-safety fallbacks, 2,098 agreements, 1,344 disagreements with the transformer enabled
(1,026 rejected by the alpha-beta root verification, 318 alpha-beta default), MCTS move chosen 0 times.
The RQ1a pilot (`RQ1a_pilot/moves.jsonl`): 6,549 decisions, 1,165 such disagreements, MCTS move chosen 0
times. Protocol time stamps (GitHub server side): frozen protocol merged 2026-10-14 06:46:26 UTC (PR #77,
tag `paper-v2-protocol-freeze`); the first confirmatory game started 08:09:06 UTC (RQ1a rc1, first
`GameStartTime`). The pilot used RQ1a's seed (`-srand 20261007`, first 50 openings); the gates used
`-srand 20261012`.

Arbiter and timing facts of the frozen source (manuscript §3.3 and the arbiter appendix):
`docs/paper_v2/arbiter_reference.md`, compiled from `git show 365f36a` with `path:line` references and
checked against the source for the verification budget (min(240 ms, budget/8), budget >= 1 s), the clock
restart after verification, the single-candidate checks (80-160 ms, queen on the board), the reported
score, the veto, the join grace (budget/4 within 1-5 s), the agreement stop and the preload before
`readyok`; the dynamic ladder writes 37 reasons (the veto and 36 predicates, 31 reachable under the
arm's settings: 24 lift the veto, 2 lift it in a narrower form, 4 are blocked by it, 1 needs depth < 10).
L14 log counts (`sprt/L14_hybrid_lean_vs_ab8_30p03/engines.log.gz`, counted 2026-10-27): 3,442
`HybridTrace` lines and 3,413 time-safety lines; `subsearch stop grace expired` 0; `engines agree,
stopping early` 0; each game's first `isready` answered `readyok` after the preload (2.5 s in the first
game) and before the first `go`, which carried the full clock.

**RQ1b final reading** (2026-11-03; block 3 ended 18:24 UTC; the cell is complete and was interpreted
once, at 18:25 UTC, by amendments 12 (f) and 13). There was no interim reading of RQ1b: amendment 11's
per-block readings cover the four MUST cells, and RQ1b's block boundaries (2026-11-01 22:53,
2026-11-02 11:58 and 2026-11-03 03:24 UTC) were checked for health only (games, terminations, the
binary and commit of every chunk, thermal log, memory), without computing a score. Computed from
`RQ1b/games.pgn` with `penta_stats.py` and `robustness.py`. The tables cited below come from the table
and analysis functions of `make_figures.py --final-analyses` (without the plots and the diagram export),
run at main `02eff6b` on 2026-11-03 18:26:52–18:28:49 UTC on the full `results/paper_v2` into a scratch
directory; in that run every row of the four version-1 cells reproduced the paper's tables exactly. No
other cell was read. RQ1b used RQ1a's seed (`-srand 20261007`) and book, so its 500 pairs played, in
order, the first 500 openings that RQ1a played.

| Cell | Comparison | Games | Elo (95% CI) | nElo | LOS | Ptnml [0..2] | W / D / L | Time forfeits | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1b | Hybrid(8) (shipped, oversubscribed) vs AB(8), **60+0.6**, `Hash` 2048 MB each, move overhead 50 ms each | 1,000 (500 pairs) | **−40.5 [−49.6, −31.4]** (± 9.1) | −96.9 ± 21.5 | 0.0% | [0, 158, 300, 42, 0] | 203 / 478 / 319 | 0 | `RQ1b/games.pgn`, `RQ1b/summary.json` |

Reading (02 §5 RQ1; exploratory, amendment 12 (g), no alpha of its own): the 95% interval lies below
zero. At 60+0.6, the time control with the most GPU participation, the shipped hybrid was weaker than
plain alpha-beta with the same `Threads` setting on this machine, by about 40 Elo (31 to 50 Elo at 95%).
The transformer was enabled on 83.1% of the hybrid's moves (53.1% in RQ1a at 30+0.3), and the arbiter
played the MCTS move on 1 of 17,338 disagreements, so the deficit does not come from moves chosen by
MCTS. RQ1a's interval at 30+0.3 was [−10.7, +2.0]; the two cells are separate matches at different time
controls, and no contrast between them was registered.

Sensitivity analyses (amendment 12 (b); the primary estimate's standard error is 4.6 Elo): zero outside
load (pair-level regression; all 500 pairs have samples, 116 exposed; slope +0.025 ± 0.044 per unit
fraction) −41.6 ± 9.9 (shift −1.1); low-load pairs only (amendment 8; 384 of 500 kept) −39.1 ± 10.6
(shift +1.4); there were no time forfeits, so both forfeit variants equal the primary estimate. No
estimate moves by more than one standard error, so RQ1b is **not load-sensitive**. In-game minutes with
outside CPU above 300%: 229 of 3,208 (7.1%); median outside CPU 155%. Largest outside process in those
minutes: the editor (Kiro) 114, Homebrew Python 3.12 30, Spotlight indexing 21, Python 3.14 18 (the
interpreter of the runner and the analysis scripts; 11 of these minutes lie within about a minute of a
chunk start, when the runner does its bookkeeping), WindowServer 9, WebKit web content 7, `top` 6,
mediaanalysisd 5 (counted over RQ1b's games with `make_figures.parse_hostload_full` and
`in_game_indices`, the helpers behind `outside_processes.md`; exposure by block with
`robustness.pair_load_fraction`). Blocks
(amendment 12 (g)): −27.9 ± 18.8, −54.6 ± 18.7, −33.5 ± 16.8, −46.1 ± 18.0 (draw ratio 42–52%);
Cochran's Q = 4.98 (3 df), p = 0.17; bootstrap over the runner's 160 chunks, 95% interval
[−49.3, −31.5]. Pairs exposed to a minute above 300%, by block (descriptive): 24 (19%), 25 (20%),
31 (25%) and 36 (29%). Load-log coverage: complete in all four blocks. By macOS build (amendment 13 (b)):
all 500 pairs started on the final release of macOS 27.2.

Health (`health.md`, `compute.md`, `therm.md`): adjudicated 932 of 1,000 games (410 draws; wins 203 for
the hybrid, 319 for AB(8)), 68 ended normally (all draws); no time forfeits and no other terminations;
3.06 min per game over 51.0 h of play (2026-11-01 08:50 to 2026-11-03 18:24 UTC, with the gate's
pauses); thermal log: 3,200 samples, no warning; every chunk's binary was the frozen one (SHA-256
`8c534bb6…`, both arms) and every chunk's commit was on main with `src/` identical to `365f36a`. Host
covariates (amendment 12 (c); `covariates.md`; all 3,208 in-game minutes carry them): GPU utilisation
median 68% (p95 75%), GPU in-use memory median 952 MB, memory available median 51% (minimum 36%),
65,165 page-outs during the cell's games. On 2026-11-02 from 15:00 to 15:22 UTC outside processes used
nine to ten cores and available memory fell to 34%; the gate was closed from 14:44 to 15:55 UTC, so no
game was played in that interval.

Telemetry (M5, RQ6a; `telemetry.md`, `summary.json`): 63,835 hybrid moves; transformer enabled on 83.1%
(53.1% in RQ1a); median 92 MCTS evaluations and 101 playouts per enabled move (36 and 38 in RQ1a);
disagreement on 32.7% of the traced moves (17,338); MCTS move chosen on 1 of the 17,338 disagreements
(`mcts_ultra_low_node_root_confidence`). Decision reasons: agreement 35,682, alpha-beta rejects the MCTS
move 13,362, time-safety 10,815, alpha-beta default 3,975, the MCTS move 1. In-game MCTS evaluation rate
(M3; amendment 12 (c); `gpu_health.md`): 52,988 of 52,988 transformer-enabled records matched by
position; evaluations per second of move time, median 87 (10th percentile 31), total 57; by block
87 (32), 91 (35), 87 (31) and 80 (26), totals 57, 61, 57 and 52.

Time per move and clock use (`timing.md`, `timing_profile.tex`; PGN move comments): AB(8) 63,789 moves,
mean 1.472 s (p99 9.977 s), median depth 24, median 10.25 million nodes per second; the hybrid 63,835
moves, mean 1.410 s (p99 9.770 s), median depth 22, median 8.64 million: on its 53,020
transformer-enabled moves mean 1.579 s, median depth 23, median 7.91 million; on its 10,815 time-safety
moves mean 0.581 s, median depth 17, median 11.57 million. Mean time per move by the arm's own move
number (1–10, 11–20, 21–30, 31–40, 41–60, 61+): AB(8) 2.66, 2.15, 1.67, 1.17, 0.81, 0.61 s; the hybrid
1.58, 1.73, 1.91, 1.61, 1.05, 0.69 s, against matching profiles in RQ1a (1.31 against 1.33 s in moves
1–10). Switches between the two classes in game order (`switches.md`): 518 from a time-safety to a
transformer-enabled move and 912 the other way, over 1,000 games.

Channels (`flags.md`; traced moves other than time-safety moves, and the engine log): root hints on
47,585 of the 53,020 transformer-enabled moves (89.7%; RQ1a 83.7%), the verification sub-search on
37,020 (69.8%; RQ1a 17.9%), and the agreement rule ("engines agree, stopping early") ended 25,774 of
them early (48.6%; RQ1a 1.0%, RQ2b 16.8%); its condition includes at least 100 MCTS nodes (the median
enabled move ran 101 playouts, against 38 in RQ1a); "subsearch stop grace expired" 0. On the
disagreements: hints on 15,007 of 17,338, with the first hint the MCTS move on 10,484 and the
alpha-beta move on 1,610. These logs show how the hybrid's search differed from AB(8)'s at 60+0.6
without separating the causes: over all moves it reported a median of 8.64 million nodes per second
against AB(8)'s 10.25 million in the same games (RQ1a: 10.52 against 10.78), nearly half of its
transformer-enabled moves ended early by the agreement rule, and it spent less time than AB(8) in its
first ten moves. No cell of the frozen build switches the agreement rule off, and RQ6d (hints and
verification off) is played at 30+0.3.

## F. Runner cells (GitHub-hosted Apple M1 runners, protocol amendment 15)

**Runner probe** (2026-11-04; runs 41832147565, 41837556525 and 41846444311 of `.github/workflows/paper-runner-cells.yml`,
one fresh runner per job; job logs and artifacts under `results/paper_v2/runner_probe/run_<id>/`). Not a
cell: the probe fixed the runner settings of amendment 15 before any runner cell started, and none of its
games or judgments enters an estimate. Values are given for the two probe runners in turn.

| Quantity | Value | Artifact |
| --- | --- | --- |
| Runner | Apple M1 (Virtual), `VirtualMac2,1`, 3 cores, 7 GB (7,516,192,768 bytes); macOS 26.6.2 (25G83), Xcode 26.6, Apple clang 21.0.0 | `job_probe_engine.log`, "Runner inventory" |
| GPU | Apple Paravirtual device (Metal, unified memory, 4,778 MB recommended working set); FP16 1024×1024×1024 MPSGraph product 7.09 / 4.97 ms (0.30 / 0.43 TFLOPS) | same |
| Frozen engine built on the runner | tag `paper-v2-freeze-rc2` (no `src` difference from rc1); bench 2,497,913 nodes, the signature, at 719,859 / 765,526 nodes/s | same, build step |
| BT4 latency, batch 1 (M1 on the runner) | FP16 21.35 / 19.65 ms; FP32 32.09 / 71.15 ms (workstation: 7.6 / 8.8 ms); FP16 at batch 32: 324.0 ms, 98.8 positions/s (first runner) | `runner_micro/nn_latency.json` |
| AB scaling (M2 on the runner; depth 16, Hash 1024, median of 3) | 0.59 / 0.47 M nodes/s at 1 thread, 0.91 / 0.92 M at 2, 1.44 / 1.27 M at 3 | `runner_micro/ab_scaling.json` |
| Hybrid(3) footprint at Hash 1024 | 2.75 / 3.20 GB maximum resident size (one traced search) | `job_probe_engine.log`, traced search |
| Game pace, Hybrid(3) vs AB(3), 30+0.3, Hash 1024 | 4 games in 7.8 min on the first runner (1.95 min per game with engine restarts), game durations 1:20–1:45; no forfeit | `PROBE/games.pgn`, job log |
| Syzygy 3-4-5 | 290 files from the lichess mirror, each matching the mirror's SHA-256 list, as do the workstation's 290 files | `results/paper_v2/runner_probe/lichess_sha256.txt`, `local_syzygy.txt` |
| Oracle judges on the runner | Stockfish 19 (3 threads, 60 s): 84.6 M nodes, depth 36 on one middlegame position; Lc0 0.32.1 `backendbench`: 31.2 / 24.0 positions/s at batch 1, 63.8 / 62.1 at batch 32; the oracle's first two positions were unfinished after 61 min (each position: two 60 s Stockfish searches and three 30,000-node Lc0 searches) | `run_41846444311/job_probe_judges.log`, `run_41837556525/job_probe_judges_cancelled.log` |

The two probe runners differed by 10–40% in GPU and CPU throughput, which is why amendment 15 (c)
interleaves the blocks of the cells it compares.
