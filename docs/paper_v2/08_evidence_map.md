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
to batch 8, one FP16 policy-logit deviation of 0.0056 at batch 16 (the same class of deviation the
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
| AB NPS vs threads (pre-port tree, depth 15, median of 3) | 1: 1.37 M, 4: 5.80 M, 8: 7.54 M, 12: 10.72 M (+42% at 12 vs 8) | `results/paper_v2/micro/ab_scaling_preport.json` (re-run on the final engine pending) |
| BT4 eval latency, MetalFish MPSGraph, sweep of 2026-10-13 (build 8777b99, whose network code is the frozen one; no game running, host load not yet logged; `micro/nn_latency.json`; date from `queue_paper_log.txt`, corrected 2026-10-27) | FP16: b1 7.40 ms (135 pos/s), b8 19.7 ms (407), b16 35.9 ms (446), b64 119 ms (538); FP32: b1 8.74 ms (114), b64 125 ms (511) | `results/paper_v2/micro/nn_latency.json` (built from the pre-C18 branch binary, identical NN code) |
| Lc0 0.32.1 Metal BT4 steady-state nps (`lc0 benchmark`, same sweep) | mb1 165, mb2 163, mb4 189, mb8 294, mb16 606, mb32 725, mb64 530 (median of the last five per-second samples; the all-sample median is biased low by the ramp-up) | `results/paper_v2/micro/nn_latency.json` `nps_steady_median` |
| AB NPS vs threads, run of 2026-10-13 (build 8777b99, whose search code is the frozen one; depth 16, 5 repeats, `micro/ab_scaling.json`) | 1: 1.53 M, 2: 3.04 M, 4: 6.27 M, 6: 9.18 M, 8: 11.81 M, 12: 13.52 M | `results/paper_v2/micro/ab_scaling.json` |
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
