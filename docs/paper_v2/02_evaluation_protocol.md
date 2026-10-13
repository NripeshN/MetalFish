# 02 — Pre-registered evaluation protocol

Status: DRAFT for author approval. Once approved, this file is frozen by tagging the
commit (`paper-v2-protocol-freeze`); any later change is recorded in §11 (amendments)
with a date and reason, never silently.

## 0. Principles

1. **Measure the marginal value, not the component.** The AB component is a Stockfish
   port, so its strength is not the contribution. Every primary comparison is
   *same binary, same machine, same wall-clock, same total CPU-thread budget*, with
   the GPU search switched on or off.
2. **Outcome-neutral.** Hypotheses, sample sizes, statistics and decision rules are
   fixed before data collection; the paper is written for any outcome (§5.1 has the
   three pre-written framings).
3. **Fishtest-grade methodology.** Game pairs with colour swap, unbalanced human
   openings (UHO), pentanomial statistics with normalised Elo (nElo), fixed-N designs
   for estimates (no optional stopping), engine restart per game, no pondering,
   concurrency 1.
4. **Everything retained.** PGNs, engine logs, traces, JSON summaries, manifests with
   hashes and seeds. Figures are generated from JSON by scripts.
5. **Tactical suites are secondary and are not strength.** They are reported as
   accuracies with confidence intervals and with the test-set-leakage caveat (§7.5).

## 1. Fixed environment (to be recorded in every manifest)

| Item | Value (2026-10-07) |
| --- | --- |
| Machine | Apple M2 Max, 12 CPU cores (8 performance + 4 efficiency), 38-core GPU, 32 GB unified memory |
| OS | macOS 27.2 (Darwin 27.2.0), Xcode-beta toolchain, Apple clang 21.0.0 |
| Engine | MetalFish built from the frozen tag (Release, `-DUSE_METAL=ON`, LTO, `-mcpu=native`); `id name MetalFish 1.0.0`; binary SHA-256 recorded |
| Networks | `BT4-1024x15x32h-swa-6147500.pb` (382,616,086 B); `nn-c288c895ea92.nnue` (108,919,594 B); `nn-37f18f62d772.nnue` (3,519,630 B); SHA-256 recorded |
| Game manager | fastchess v1.8.2-alpha (`reference/fastchess/fastchess`) |
| Rating | Ordo 1.2.6 (`reference/ordo/ordo`) for multi-engine tables |
| References | Stockfish 19 (`reference/stockfish-sf19/src/stockfish`, tag sf_19, PGO build, bench 2,497,913 nodes); Stockfish dev-20260106-c27c1747 (`reference/stockfish/src/stockfish`, same base as MetalFish-AB); Lc0 v0.32.1 Metal (`/opt/homebrew/bin/lc0`); Berserk and Patricia rebuilt from their latest releases (Berserk 14, Patricia 5.1) before RQ8 |
| Books | `reference/books/unzipped/UHO_Lichess_4852_v1.epd` (2,632,036 positions; Fishtest's book) primary; `UHO_4060_v4.epd` fallback band; `reference/books/8moves_v3.pgn` (34,700 games) balanced-book control |
| Suites | STS v3 (1,500), ERET (111), Arasan 2023 (200), WAC (300), Lichess puzzle DB (6,157,341 puzzles) under `reference/testsuites/`, `reference/lichess_puzzles/` |
| Tablebases | Syzygy 3-4-5 (`syzygy/`, 290 files). Not given to engines in primary cells (both arms are the same binary); used only by the oracle judge |

## 2. Configurations under test

All MetalFish configurations use the same frozen binary. `T` is the UCI `Threads`
value. Unless stated, all other options are the shipped defaults.

| Name | Options | Notes |
| --- | --- | --- |
| `AB(T)` | `UseHybridSearch=false UseMCTS=false Threads=T Hash=2048` | Pure CPU alpha-beta + NNUE (Stockfish-derived) |
| `Hybrid(T)` | `UseHybridSearch=true NNWeights=<BT4> Threads=T Hash=2048 HybridTrace=true` | On a game clock with T≥8: 1 MCTS worker + (T−2) AB threads + coordinator. For T<8 the engine uses T−1 AB threads; to keep the rule uniform across RQ3 cells set `HybridMCTSThreads=1 HybridABThreads=T-2` explicitly for every hybrid cell and verify via the `Hybrid thread split:` log line |
| `Hybrid-min(T)` | `Hybrid(T)` + `HybridArbitration=minimal` (new option, see 03 §C11) | AB move by default; MCTS move only when AB has no result. Isolates the override ladder |
| `Hybrid-mb{k}(T)` | `Hybrid(T)` + `MCTSMinibatchSize=k` (k ∈ {4,16}) | MCTS throughput knob for RQ4 (needs 03 §C12) |
| `MCTS(1)` | `UseMCTS=true NNWeights=<BT4> Threads=1 MCTSMaxThreads=1 MCTSMinibatchSize=1 TransformerLowTimeFallbackMs=0` | Pure transformer MCTS as shipped (batch 1, one worker) |
| `SF19(T)` | Stockfish 19, `Threads=T Hash=2048` | Current release reference |
| `SFdev(T)` | Stockfish dev-20260106, `Threads=T Hash=2048` | Historical base of MetalFish 1.0.0's AB (the 2026-10-12 port moved the AB core to Stockfish 19; the parity reference is now `SF19` and `bench` reproduces its 2,497,913-node signature) |
| `Lc0-mb1` | lc0 0.32.1 `--backend=metal`, `option.Threads=1 option.MinibatchSize=1`, BT4 weights | Matched-settings reference for MCTS(1) |
| `Lc0-def` | lc0 0.32.1 `--backend=metal`, defaults (`Threads` auto, `MinibatchSize` auto), BT4 weights | Lc0 as users run it |

Memory note: a `Hybrid` process holds BT4 (0.38 GB) + NNCache (2,000,000 entries ≈ 1.65 GB
by layout) + Hash 2 GB; an `AB` process holds Hash 2 GB. Two processes fit in 32 GB with
margin. If 03 §C4 changes the NNCache default before the freeze, the frozen value is
what is evaluated and reported.

Known hybrid gotchas that invalidated earlier measurements (all must be checked in the
pilot): `Threads<3` silently falls back to AB; cutechess/fastchess options must be
separate `option.NAME=VALUE` tokens; never use fixed time per move (`st=`) with this
engine (massive time forfeits); `go movetime < 3000` and clock budgets `< 400 ms` or
`time left < 3000 ms` disable the transformer for that move (`Time safety:` log line).

## 3. Match protocol (fastchess)

Reference command for one cell (paths relative to the repo root; `<OUT>` is the cell's
results directory under `results/paper_v2/<cell_id>/`):

```bash
./reference/fastchess/fastchess \
  -engine cmd=build/metalfish name=MF-Hybrid-T8 \
      option.UseHybridSearch=true option.UseMCTS=false \
      option.NNWeights=$PWD/networks/BT4-1024x15x32h-swa-6147500.pb \
      option.HybridMCTSThreads=1 option.HybridABThreads=6 option.HybridTrace=true \
  -engine cmd=build/metalfish name=MF-AB-T8 \
      option.UseHybridSearch=false option.UseMCTS=false \
  -each tc=30+0.3 option.Threads=8 option.Hash=2048 \
  -rounds 1000 -games 2 -repeat -concurrency 1 -recover \
  -openings file=reference/books/unzipped/UHO_Lichess_4852_v1.epd format=epd order=random \
  -srand 20261007 \
  -draw movenumber=34 movecount=8 score=20 \
  -resign movecount=3 score=600 twosided=true \
  -maxmoves 200 \
  -pgnout file=<OUT>/games.pgn nodes=true nps=true seldepth=true timeleft=true \
  -log file=<OUT>/engines.log engine=true \
  -config outname=<OUT>/fastchess_config.json \
  -report penta=true -ratinginterval 50
```

Verified on 2026-10-07 with a 2-game smoke run: `-engine ... option.X=Y`, `-each tc=`,
`-rounds/-games/-repeat/-concurrency/-recover`, `-openings file= format=epd order=random`,
`-pgnout file= nodes=true nps=true`, `-log file= engine=true`, `-config outname=`,
`-report penta=true -ratinginterval`. **To verify in the pilot (P0.6):** `-srand`,
`-draw/-resign/-maxmoves` syntax, `seldepth=true timeleft=true`, and `-sprt ... model=normalized`
(used only for exploratory screens, never for the confirmatory cells).

Rules:

- Adjudication follows Fishtest defaults (draw: after move 34, 8 consecutive moves within
  ±20 cp; resign: 3 consecutive moves beyond ±600 cp, two-sided). Counts of adjudicated
  games, time forfeits, illegal moves, crashes and disconnects are reported per cell.
- `-recover` restarts a crashed engine; the game is scored per fastchess rules and the
  event is logged. No game is excluded post hoc.
- Engine restart per game is fastchess's default (process restarted each game); BT4 load
  time (~1–2 s) happens before `readyok` and is not charged to the clock.
- The same book, order and seed are used for every cell within a research question so
  arms and cells see the same opening sample; `-repeat` plays each opening with colours
  swapped.
- Each cell is split into 4 blocks of equal pair count. Blocks of different cells are
  interleaved round-robin over the calendar so thermal and background drift are balanced
  across cells (§7.1).
- Machine state during runs: lid open/plugged in (if a laptop), display off, `caffeinate -dims`,
  no other user processes, Spotlight indexing of `results/` disabled. `pmset -g therm`
  is sampled every 60 s into `<OUT>/therm.log`.

## 4. Statistics

- **Primary estimator**: pentanomial (per game pair: 0, ½, 1, 1½, 2 points) score mean
  and variance → logistic Elo difference with 95% normal-approximation CI, LOS, and
  normalised Elo (nElo) as reported by fastchess (`-report penta=true`) and re-computed
  independently by `tools/paper/penta_stats.py` from the PGN (Fishtest formulas; the two
  must agree to rounding).
- **Multi-engine tables (RQ8)**: Ordo with `-a 0 -A "Stockfish 19"` (anchor = SF19 at 0),
  simultaneous estimation, error bars from Ordo's simulation (`-s 1000`). Reported as
  "relative to SF19 on this hardware"; no CCRL-style absolute ratings are claimed.
- **Trends (RQ3, RQ4)**: weighted least squares of Elo difference on the covariate with
  per-cell variances; slope with 95% CI; also plotted per cell with CIs.
- **Oracle study (RQ6)**: per-position cp loss and expected-score loss (via the engine's
  WDL model) for the AB move and the MCTS move; paired bootstrap over positions (10,000
  resamples) for differences and for the oracle-arbiter upper bound.
- **Suites (RQ7)**: solve proportions with Wilson 95% intervals; paired comparisons via
  McNemar where two engines are compared on the same positions.
- **Confirmatory vs exploratory**: RQ1 at 30+0.3 is the single confirmatory test. All other
  cells are reported as estimates with CIs and labelled exploratory; no family-wise
  correction is applied, and all planned cells are reported whether or not run to
  completion (with reasons).
- **Sample-size reasoning** (equal-strength engines, trinomial normal approximation, draw
  rate D; pentanomial on UHO tightens these by roughly 1.4× — to be confirmed from the
  pilot's actual pair-score variance):

| Games | D = 0.5 | D = 0.6 | D = 0.9 (balanced book) |
| --- | --- | --- | --- |
| 400 | ±24 | ±22 | ±11 |
| 800 | ±17 | ±15 | ±8 |
| 1,000 | ±15 | ±14 | ±7 |
| 2,000 | ±11 | ±10 | ±5 |
| 4,000 | ±8 | ±7 | ±3 |

  Elo compression at high draw rates is why the balanced-book cell is a *control*, not a
  measurement: nElo (Elo / √(1−D) in the simple case) is the book-independent quantity.

## 5. Research questions, hypotheses, designs

### RQ1 — Marginal value of the GPU search at equal budget (confirmatory)

*Does running the BT4 transformer MCTS alongside the alpha-beta search change playing
strength, on one machine, at equal wall-clock and equal total CPU-thread budget?*

- **H1**: Elo(Hybrid(8) − AB(8)) ≠ 0 at 30+0.3. Prior evidence (40+0.4: −6.9 ± 13.5;
  90+0.9: −8.7 ± 16.8; balanced book, 97–100% draws) suggests |Δ| < 10.
- **Design**: cell `RQ1a`: Hybrid(8) vs AB(8), 30+0.3, UHO_Lichess, **1,000 pairs**
  (2,000 games), fixed N, no peeking beyond fastchess's periodic printout.
- **Secondary cells**: `RQ1b` 60+0.6, 500 pairs (thicker MCTS trees); `RQ1c` 10+0.1,
  1,000 pairs (blitz: the transformer is disabled for most moves by the time-safety rule;
  GPU-active fraction is the key covariate); `RQ1d` balanced-book control: 8moves_v3.pgn
  at 30+0.3, 500 pairs (expected ≥90% draws; demonstrates why the earlier measurements
  could not resolve anything).
- **Covariates recorded from traces**: GPU-active fraction of moves, MCTS playouts/move,
  MCTS evals/s, AB depth, decision-reason histogram, time usage.
- **Decision rule** (RQ1a, 95% CI): lower bound > 0 → "the GPU search adds strength";
  upper bound < 0 → "the GPU search costs strength"; otherwise → "no detectable effect at
  the achieved resolution (±X Elo)". Pre-written framings for the three outcomes are in
  `04_paper_blueprint.md` §2.

### RQ2 — Isolation: cost of threads vs value of MCTS

- `RQ2a`: AB(6) vs AB(8), 30+0.3, 500 pairs → the price the hybrid pays for giving two
  threads to MCTS and the coordinator (expect −10 to −25 Elo).
- `RQ2b`: Hybrid(8) vs AB(6), 30+0.3, 500 pairs → value of the GPU search given the *same*
  AB resources (expect ≥ 0 if the arbiter never picks a worse move).
- Interpretation: RQ1a ≈ RQ2b − RQ2a. A positive RQ2b with a negative RQ1a means the GPU
  helps but not enough to pay for its CPU threads on this core count.

### RQ3 — CPU-budget crossover

*Does the hybrid's relative value grow as CPU budget shrinks (laptop-class parts)?*

- Cells `RQ3-T4`, `RQ3-T6` (T8 from RQ1a): Hybrid(T) vs AB(T) with explicit
  `HybridMCTSThreads=1 HybridABThreads=T-2`, 30+0.3, 400 pairs each. Optional `RQ3-T12`
  (all cores including efficiency cores) if budget allows.
- **H3**: slope of Elo difference vs T is negative (hybrid relatively better at low T).
- Output: Fig. "Elo difference vs CPU threads" with CIs and WLS slope.

### RQ4 — MCTS throughput scaling (projection to faster GPUs)

*If the GPU search were k× faster, would the hybrid gain?* Batching multiplies evals/s
(measured 2026-10-07: batch 1 = 131 pos/s, batch 16 = 443 pos/s FP16) at the cost of
staleness; it is the only clean throughput knob available on this hardware.

- Cells `RQ4-mb4`, `RQ4-mb16`: Hybrid-mb{4,16}(8) vs AB(8), 30+0.3, 400 pairs each.
  Optional `RQ4-2w`: Hybrid(8) with `HybridMCTSThreads=2 HybridABThreads=5`, 400 pairs.
- Covariate: measured MCTS evals/s and playouts/move per cell (from traces).
- **H4**: Elo difference increases with MCTS throughput. A flat curve means the arbiter or
  the thin-tree move quality, not throughput, is the limit.
- Prerequisite: 03 §C12 (the hybrid must honour `MCTSMinibatchSize` on a clock).

### RQ5 — Component checks (positioning, not contribution)

- `RQ5a`: MCTS(1) vs Lc0-mb1, 30+0.3, 300 pairs → is the MCTS implementation on par with
  Lc0's search at matched settings and the same network?
- `RQ5b` (optional): MCTS(1) vs Lc0-def, 300 pairs → vs Lc0 as users run it.
- `RQ5c`: AB(8) vs SF19(8), 30+0.3, 500 pairs → parity of the ported AB core with the current
  Stockfish release (expect ≈ 0 after the 2026-10-12 port; the port measured +62 ± 28 Elo over
  the previous dev-20260103 base at 10+0.1).
- `RQ5d`: AB(8) vs SF19(8) at *equal settings* is the parity check after the Stockfish 19
  port (expect ≈ 0; `bench` already matches upstream node-for-node). `RQ5c` (AB vs SF19)
  therefore doubles as the parity cell; the SFdev cell is kept only to quantify how far the
  published v1.0.0 engine was behind (optional).

### RQ6 — Arbitration analysis and oracle study

- `RQ6a` (free): from all hybrid PGNs/traces: decision reasons, disagreement rate between
  AB and MCTS best moves, MCTS visits at decision time, how often the MCTS move is chosen,
  agreement-stop firing rate, candidate-verification firing rate.
- `RQ6b` oracle: sample 600 positions where AB and MCTS disagreed (stratified by game
  phase and by MCTS visit count) from RQ1a/RQ1b games. Judge each candidate move with two
  independent oracles: Stockfish 19 (8 threads, 60 s/position, MultiPV covering both
  moves, Syzygy enabled) and Lc0 0.32.1 (Metal, 30,000 nodes, BT4). Record cp loss and
  expected-score loss of each candidate vs the oracle's best. Report: fraction where AB's
  move is better / MCTS's is better / tie (|Δ| ≤ 10 cp); mean loss; the *oracle-arbiter
  upper bound* (always pick the better of the two) in expected-score units and projected
  Elo; and the realised arbiter's accuracy on these positions.
- `RQ6c` ablation: Hybrid-min(8) vs Hybrid(8), 30+0.3, 500 pairs → what the ~30-predicate
  override ladder buys over "AB unless AB has nothing". Prerequisite 03 §C11.
- `RQ6d` (optional): Hybrid(8) with `HybridMCTSABRootHints=false HybridABCandidateVerifyMs=0`
  vs AB(8), 500 pairs → does the MCTS→AB hint channel matter?

### RQ7 — Positional/tactical suites (secondary, with leakage caveat)

Engines: AB(8), Hybrid(8) with `TransformerLowTimeFallbackMs=0 TransformerMinMoveBudgetMs=0`
(so the GPU is active at short movetimes — report that this differs from shipped defaults),
MCTS(1), SF19(8), Lc0-def. Suites and budgets: STS v3 1,500 @ 2 s; ERET 111 @ 15 s; Arasan
2023 200 @ 30 s; WAC 300 @ 1 s; Lichess puzzles: 1,000 sampled per seed × 3 seeds, rating
1200–2800 stratified, popularity ≥ 70, full solution line required, 1 s/move. Report solve
rates with Wilson CIs and solve-time distributions. Bratko-Kopec is reported once (24
positions) only to connect with the earlier paper, with the ceiling and leakage caveats.

### RQ8 — Relative gauntlet (descriptive)

MetalFish {AB(8), Hybrid(8), MCTS(1)} vs {SF19(8), Lc0-def, Berserk 14 (8 threads)}:
9 pairings × 100 pairs at 30+0.3 (1,800 games). Rated with Ordo anchored at SF19 = 0.
Optional extension to a full round robin including Patricia 5.1 if budget allows. All
numbers are "relative, this hardware, this TC"; no claim is made about CCRL ratings.

### RQ9 — Deployment (descriptive)

Summarise the Lichess bot corpus (`results/lichess_audit/`, 1,058 games as of
2026-10-07: 99 W / 22 L / ≈85% draws; `tools/analyze_lichess_games.py`): opponents,
speeds, draw structure, time losses, and the earlier finding that drawn games were
genuinely equal (Stockfish re-analysis of final positions). Optionally run 200 fresh
games with the frozen binary. Presented as deployment evidence, not strength.

## 6. Microbenchmarks (systems section)

| ID | Measurement | Tool |
| --- | --- | --- |
| M1 | BT4 eval latency and throughput vs batch {1,2,4,8,16,32,64} × {FP16, FP32}; MetalFish MPSGraph vs Lc0 Metal | `build/metalfish_nn_probe`, `lc0 benchmark --minibatch-size` |
| M2 | AB NPS and time-to-depth vs threads {1,2,4,6,8,12}, fixed-depth bench on 20 positions × 5 repeats | engine `bench` |
| M3 | In-game MCTS eval rate vs number of AB threads (CPU contention curve) | traces from RQ1/RQ3 |
| M4 | Resident memory per configuration | `ps`/`footprint` sampled during pilot |
| M5 | GPU-active fraction and MCTS playouts/move vs TC | traces from RQ1a–c |
| M6 | Thermal state over long runs | `pmset -g therm` logs |

## 7. Controls and threats to validity

1. **Drift/thermal**: interleaved blocks (§3), thermal log, report per-block results.
2. **Single hardware platform**: acknowledged; RQ3/RQ4 give the scaling direction; the
   CUDA path is explicitly out of scope for this paper (future work).
3. **Opening book dependence**: UHO primary + balanced-book control + nElo reporting; the
   UHO critique (Elo spread exaggeration; book-dependent engine behaviour) is discussed.
4. **GPU-active fraction**: measured per cell; the shipped time-safety rule is part of the
   system under test; an always-on variant at 10+0.1 is an optional ablation (`RQ1c-on`).
5. **Tuning leakage**: hybrid/MCTS heuristics were tuned on Bratko-Kopec and Lichess puzzle
   samples held in `results/`; suite results for MetalFish are therefore optimistic and are
   labelled as such. Game results are unaffected.
6. **Reference version drift**: the AB base is Stockfish dev-2026-01-03; SF19 is newer
   and stronger; RQ5c/RQ5d quantify both.
7. **Adjudication**: Fishtest defaults; counts reported; a sensitivity check re-runs one
   block of RQ1a without adjudication.
8. **Statistical**: one confirmatory test, fixed N, all cells reported.
9. **Engine correctness**: unit suite, perft, UCI smoke, NN parity gate (`metalfish_nn_probe`
   CPU vs Metal) pass on the frozen binary; the EP-plane fix (03 §C3) is validated against
   Lc0 outputs before the freeze.

## 8. Compute budget and prioritisation

Measured: 2 games at 10+0.1 took 98 s wall-clock including restarts (≈0.8 min/game).
Estimates: 30+0.3 ≈ 1.9 min/game, 60+0.6 ≈ 3.7 min/game (concurrency 1).

| Priority | Cells | Games | Est. hours |
| --- | --- | --- | --- |
| MUST | RQ1a, RQ2a, RQ2b, RQ5c, RQ6b (oracle), M1–M6 | 5,000 + oracle | ≈185 (7.7 days) |
| SHOULD | RQ1b, RQ1c, RQ3-T4/T6, RQ4-mb4/mb16, RQ5a, RQ5d, RQ6c, RQ7, RQ8 | ≈9,600 + suites | ≈371 (15.5 days) |
| COULD | RQ1d, RQ4-2w, RQ5b, RQ6d, RQ1c-on, RQ3-T12 | ≈4,600 | ≈160 (6.7 days) |

Total ≈ 716 machine-hours (30 days) at concurrency 1 on the M2 Max. Options to shorten:
(a) a second Apple Silicon machine (e.g., an EC2 `mac2-m1ultra.metal` dedicated host) runs
the SHOULD cells with its own manifest (results are per-machine and are *not* pooled);
(b) halve N on SHOULD cells (CIs widen by √2); (c) drop COULD cells.

## 9. Reporting and artifact standards

- `results/paper_v2/<cell_id>/` contains: `manifest.json` (git SHA, binary and network
  SHA-256s, tool versions, book SHA-256, seed, TC, full option lists, hardware, OS, start
  and end timestamps, thermal summary), `games.pgn`, `engines.log` (hybrid traces),
  `fastchess_config.json`, `summary.json` (pentanomial counts, Elo, CI, nElo, LOS, draw
  rate, adjudication/forfeit counts, GPU-active fraction, MCTS playouts statistics).
- Figures and tables are produced by `tools/paper/make_figures.py` from `summary.json`
  files; no hand-entered numbers in the paper.
- The complete artifact set (PGNs, JSON, scripts, frozen binary hashes) is deposited on
  Zenodo with a DOI and cited from the paper; the repository tag is cited for code.

## 10. Pilot and freeze checklist (P0.6)

1. Land 03 §P0 fixes; build; run unit suite, perft/UCI smoke, NN parity probe.
2. Tag `paper-v2-freeze-rc1`; record hashes.
3. Pilot: 50 pairs of RQ1a with the exact command of §3. Check: GPU-active fraction ≥ 80%
   of hybrid moves; time forfeits ≤ 0.5% of games; draw rate within 45–65%; mean game
   duration; peak memory; fastchess vs `penta_stats.py` agreement; `-srand` reproducibility
   (same seed → same openings).
4. If the draw rate > 65%, switch the book to a higher UHO band (`UHO_4060_v4` or the next
   UHO 2024 band) for *all* cells and record the amendment. If time forfeits > 0.5%, raise
   `Move Overhead` for both arms equally and record.
5. Freeze: tag `paper-v2-protocol-freeze`; start MUST cells.

## 11. Amendments

(none yet)
