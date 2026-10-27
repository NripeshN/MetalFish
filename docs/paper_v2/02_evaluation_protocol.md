# 02 — Pre-registered evaluation protocol

Status: frozen on 2026-10-14 (tag `paper-v2-protocol-freeze`). Changes to the design after
the freeze are recorded in §11 (amendments) with a date and reason, never silently; §10 also
records the outcome of the pilot and the freeze, and this header was updated on 2026-10-27.

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
| `Hybrid(T)` | `UseHybridSearch=true NNWeights=<BT4> Threads=T Hash=2048 HybridTrace=true` | Shipped configuration (amendment §11.3, 2026-10-13): on a game clock alpha-beta keeps all T threads and the single MCTS worker is an extra thread that mostly blocks on GPU inference (`HybridOversubscribe=true`), Stockfish's time manager owns the move (`HybridFollowABTime=true`), root tie-break overrides off. `Hybrid-split(T)` = `Hybrid(T)` + `HybridOversubscribe=false HybridABThreads=T−2` (1 MCTS worker + (T−2) AB threads), the pre-amendment arm, used as the ablation in RQ2b/RQ2c. |
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
- Engines are restarted for every game (`-each ... restart=on`; fastchess's default keeps
  processes alive across games). BT4 load time (~1–2 s) happens before `readyok` and is not
  charged to the clock.
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

- `RQ2a`: AB(6) vs AB(8), 30+0.3, 500 pairs → the price of giving two of eight CPU threads
  to the GPU side (the pre-amendment split).
- `RQ2b`: Hybrid-split(8) vs AB(6), 30+0.3, 500 pairs → value of the GPU search given the
  *same* six alpha-beta threads.
- `RQ2c`: Hybrid-split(8) vs AB(8), 30+0.3, 500 pairs → the pre-amendment headline;
  RQ2c ≈ RQ2b − RQ2a.
- Interpretation: RQ1a (shipped, oversubscribed hybrid vs AB(8)) measures the marginal value
  of the GPU side at the full CPU budget; RQ2a–c decompose the old split design. A positive
  RQ2b with a negative RQ2c means the GPU search helps only when CPU threads are free.

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

**Pilot record (2026-10-13, build 8777b99 = tree of PR #75, 50 pairs of RQ1a at 30+0.3):**
draw rate 46% (within 45–65%), time forfeits 0, `-srand` reproducible, fastchess and
`penta_stats.py` agree (−41.89 ± 25.07 both), mean game 1.95 min, GPU-active fraction
**50.1%** of the hybrid arm's moves (median 27 MCTS evaluations per move) — below the 80%
target; see amendment §11.4. The pilot's strength reading (−41.9 ± 25.1 Elo for the hybrid)
triggered diagnostic gates before the freeze (07, L10–L14). Outcome: the deficit came from
the hybrid process's memory footprint (an unused 1.6 GB MCTS NN cache made it ≈6 GB vs 2.3 GB
and the host paged); with the cache capped (C18) the hybrid reads +13.9 ± 30.3 Elo against
AB(8) at 30+0.3 over 101 games. **Freeze: main 365f36a (2026-10-14)**; the MUST cells run on
`build/` rebuilt from that commit, whose SHA-256 is recorded in each cell manifest.

## 11. Amendments

1. **2026-10-13 — shipped hybrid configuration changed before the freeze** (strength
   programme 07, L5b–L5e; codebase changes C13–C16): the alpha-beta pool persists across
   hybrid moves, Stockfish's time manager owns the clock in hybrid mode
   (`HybridFollowABTime=true`, the coordinator's per-move budget bounds only the MCTS side),
   and the root tie-break overrides are off (`HybridRootPawnLeverTieBreak=false`). The
   "Hybrid" arm in §2 and `experiments.json` means this configuration; the v1.0 hybrid is
   kept as a secondary arm where the budget allows (RQ1d). Gate evidence: L7b, +74 ± 30 Elo
   over the previous hybrid at 20+0.2 (100 games).
2. **2026-10-13 — engines restart per game** (`restart=on`), already stated in §3; fastchess
   keeps processes alive by default, which the first runs did not override.
3. **2026-10-13 — shipped hybrid is oversubscribed** (programme 07 L9; codebase change C17):
   `HybridOversubscribe=true` by default, so `Hybrid(T)` keeps all T alpha-beta threads and
   runs the MCTS worker as an extra thread. Hybrid cells no longer pin `HybridABThreads`;
   the 6+1 split becomes `Hybrid-split(T)` and is the ablation in RQ2b/RQ2c (RQ2c repurposed
   from "oversubscribed vs AB(8)", which RQ1a now covers). Gate evidence: L9 (oversubscribed
   vs split, 20+0.2) — see 07.
4. **2026-10-13 — GPU-active target not met at 30+0.3.** The pilot measured the GPU active on
   50% of the hybrid's moves at the primary time control (the time-safety fallback runs plain
   alpha-beta once the estimated budget is below 400 ms or the clock below 3 s, and
   Stockfish-style time use reaches that regime by mid-game). The 80% criterion of §10 was a
   feasibility target, not a property the engine guarantees; RQ1a keeps 30+0.3 as pre-registered
   and reports the fraction as a covariate (M5), and RQ1b (60+0.6) is promoted to the first
   SHOULD cell because it is the regime with the most GPU participation.
5. **2026-10-14 — RQ6d promoted from COULD to SHOULD.** The 30+0.3 pilot of the shipped
   configuration read −41.9 ± 25.1 Elo against AB(8) and the L10 diagnostic with the MCTS root
   hints and candidate-verify pre-searches off read −17.4 ± 31.0; the intervals overlap, so the
   question is left to RQ6d at its pre-registered sample size rather than decided on 100-game
   gates. The shipped defaults are unchanged. A null control (hybrid UCI path with the
   transformer never engaged, L12) and the 6+1 split control (L11) are reported in 07.
6. **2026-10-14 — RQ6b sampling strata defined** (before any oracle run). A disagreement
   position is a hybrid decision record whose alpha-beta and MCTS best moves differ and that
   was not a time-safety fallback. Game phase: *opening* = fullmove number ≤ 15; *endgame* =
   at most 6 non-pawn, non-king pieces on the board (both sides together); otherwise
   *middlegame*. MCTS visit count = root visits at decision time (`mcts_root_visits`, falling
   back to playouts, then evaluations), binned at the terciles of the pooled RQ1a+RQ1b
   disagreement population. Allocation proportional to stratum size (largest-remainder
   rounding), seeded shuffle within strata (`--seed 1`), 600 positions; the population and
   sample counts per stratum and the tercile cut points are recorded in
   `results/paper_v2/RQ6b/oracle.jsonl.summary.json`. Judges as in §5; the exact invocation is
   `results/paper_v2/run_oracle_rq6b.sh`, chained after RQ1b and the microbenchmark re-run in
   `results/paper_v2/queue_after_must.sh`. This fixes the stratification that §5 left open; the
   estimands are unchanged.
7. **2026-10-14 — binary re-linked after a host library upgrade; RQ1a restarted.** During RQ1a
   block 0 (183 games played with the rc1 binary) Homebrew upgraded the host's protobuf library,
   after which the frozen binary could not start new engine processes (03 §C19). The source is
   unchanged; the same commit was rebuilt against the new library with its dynamic libraries
   bundled, re-verified (bench signature, unit suite, NN probe) and re-frozen as
   `paper-v2-freeze-rc2`. Because the protocol requires one binary for every game of a cell and
   the rc1 binary can no longer be executed for an equivalence check, RQ1a restarts from its
   first opening pair with rc2; the 183 rc1 games are kept as `RQ1a_rc1_partial` for the record
   and enter no cell. No cell result had been read beyond the 100-game diagnostic checkpoint.
8. **2026-10-15 — shared host: load-gated chunked scheduling, host-load covariate, sensitivity
   analysis.** §3 assumed an otherwise idle machine. From 2026-10-15 01:57 local time a logger
   (`results/paper_v2/hostload_logger.sh` → `hostload.log`, one sample per minute of the CPU used by
   processes other than the engines and the game manager) shows the host shared with the author's
   other work: a median of 1.75 cores and bursts up to 10.7 cores (virtual machines, compilers) over
   626 samples. The engines need about 9 of the 12 cores (8 alpha-beta threads, plus the MCTS worker
   in the hybrid arm), so external load above 3 cores oversubscribes the CPU, and it can bias
   comparisons whose arms need different numbers of free cores (RQ1a, RQ2a–c). Decisions, taken
   before any cell is interpreted and without examining any outcome by load:
   (a) **Scheduling**: from RQ2b block 0 onward the confirmatory queue plays each block in chunks of
   10 pairs (`run_cell.py --resume --max-pairs 10`; same openings, seed, order and binary), and a
   chunk starts only when the mean external CPU over the last 5 samples is below 300% (3 cores);
   gate waits are logged. Games already played and games in progress are unaffected.
   (b) **Primary estimates** include every game as played, as pre-registered.
   (c) **Sensitivity analysis** (reported for every cell next to the primary estimate): the same
   statistics after removing every game pair in which either game overlaps a logged minute with
   external CPU above 300%; games played before the logger existed carry no samples and are kept.
   The fraction of pairs removed is reported; a difference between the two estimates larger than the
   primary estimate's standard error is reported and discussed as a threat to validity.
9. **2026-10-16 — the load gate also acts during a chunk.** Amendment 8 checks the host only
   before a chunk starts; a spike that begins inside a chunk is not stopped. Between 11:19 and
   11:30 local on 2026-10-16, with other processes using 4 to 10 cores, four games of RQ1a block 1
   were lost on time (three by the alpha-beta arm, one by the hybrid), taking the cell's forfeit
   rate to 1.1%, above the 0.5% health limit. From the next chunk on: chunks are 4 pairs instead of
   10, and a watchdog (`results/paper_v2/load_watchdog.sh`) stops the running chunk at the next
   completed game pair (no game is cut short and no pair is split) when the external CPU averaged
   over the last 2 minutes exceeds 450%; the gate then waits as before. Games, openings, seed,
   order and binary are unchanged; every game played stays in the record as played (forfeits
   included), the primary estimate keeps every game, and the sensitivity estimate of amendment 8
   removes the pairs that overlapped high load, which covers these four games. Fixed before any
   cell is interpreted.
10. **2026-10-26 — the in-chunk watchdog uses the gate's threshold.** Amendment 9 stopped a chunk
    when the 2-minute external CPU exceeded 450%. On 2026-10-26 game 835 of RQ1a was lost on time
    (by 0.2 s) during a spike to 418%, below that trigger; across RQ1a six of the seven time forfeits
    so far were by the alpha-beta arm, so load-induced forfeits can bias the primary estimate in the
    hybrid's favour as well as add noise. The watchdog now stops a chunk at the next completed pair
    when the 2-minute external CPU exceeds 300%, the same threshold as the gate and the sensitivity
    analysis (12 cores less the engines' ~9). Everything else in amendments 8 and 9 is unchanged; the
    forfeited games stay in the record and the per-arm forfeit counts are reported with each cell.
    Fixed before any cell is interpreted.
11. **2026-10-27 — interim readings recorded after each block (disclosure).** §5 RQ1 allows no
    peeking beyond fastchess's periodic printout. Since 2026-10-15 an interim reading of each MUST
    cell has been recorded in `08_evidence_map.md` §E after each completed block, as the
    programme's hand-off instructions (`09_handoff_goal.md`) require for monitoring: the
    pentanomial estimate fastchess also prints, plus the per-block split, the host-load
    sensitivity estimate of amendment 8, the hybrid telemetry and the health counts. No decision
    rule is attached to these readings: no cell has been or will be stopped, extended,
    re-scheduled or analysed differently because of them, and the final analysis is the one
    pre-registered here. Amendments 8–10, which change scheduling only, were written after the
    first of these readings (block 0 of RQ1a and RQ2a). The paper reports the readings as such.
12. **2026-10-27 — pre-reading clarifications after a pre-submission review (written after the
    interim readings of amendment 11).** A five-seat review of the manuscript's design and methods,
    with every result withheld, asked for the following to be fixed before any cell is read. None
    changes a cell's games, size, seed, order, binary or primary statistic.
    (a) **Reading of RQ1a.** The decision rule stays the sign of the 95% interval. If the interval
    spans zero, RQ1a is reported as inconclusive at the planned N, together with the range of
    effects the interval excludes; no equivalence margin is introduced, because a margin chosen
    after interim readings could not be justified independently of them. What each outcome is
    taken to mean is fixed now: an interval above zero means the concurrent GPU search adds
    strength at equal `Threads` on this machine, and the telemetry and RQ6 say through which
    channel; an interval spanning zero means no effect was detected at this N, so the GPU path
    does not pay for itself at this budget on this machine; an interval below zero means the hybrid
    costs strength, and RQ2, the telemetry and the host-load analyses bound how much of that is
    thread or contention cost. RQ5c is read the same way as a parity interval, with no equivalence
    claim.
    (b) **Host load and forfeits.** Besides the exclusion analysis of amendment 8, every cell is
    reported (i) with a pair-level regression of the pair score on the fraction of the pair's
    logged minutes above 300% of outside load, giving the slope and the estimate at zero load;
    (ii) with the load-log coverage per block (block 0 of RQ1a and RQ2a predates the logger);
    (iii) with the pairs that contain a time forfeit removed, and with forfeits re-scored as draws.
    If the primary estimate and any of these differ by more than the primary standard error, both
    are reported and the result is called load-sensitive.
    (c) **GPU and memory covariates.** From 2026-10-27 21:20 UTC the host-load logger also records,
    every minute, the GPU's device utilisation and in-use memory (`ioreg`, engines included), the
    kernel's memory-status level and the cumulative page-outs and swap-outs. They are reported per
    cell as covariates. The gate is not extended to them, because during a chunk the engines' own
    GPU use cannot be separated from outside use. The hybrid's in-game MCTS evaluation rate from
    the traces is reported per block as a measure of GPU health.
    (d) **Oracle study (RQ6b).** Both judges stay (Stockfish 19, and Lc0 with the BT4 network), and
    their lineage overlaps are stated: the Stockfish judge runs the alpha-beta candidate's search
    with more time, and the Lc0 judge uses the MCTS candidate's network, so each judge is expected
    to favour one candidate. Results are reported per judge, with the judges' agreement, and the
    arbiter bound is also computed on the positions where both judges agree, disagreements
    counted as undecided. The Lc0 judge scores each candidate with equal effort, by evaluating the
    position after the candidate move with the same node budget, instead of one `searchmoves`
    search whose visits favour one move. Rates over all disagreements are reweighted by the
    recorded stratum population counts. Expected score uses the logistic conversion with 250 cp
    per unit of log-odds, and the projected Elo of the bound is the logistic Elo of the
    expected-score difference and is labelled as such. Positions are selected by stratum only,
    never by a judge's verdict.
    (e) **RQ6d** switches off the hints and the verification sub-search together, so it measures
    that combined channel, against AB(8) at 30+0.3 over 500 pairs; it is described as such, and its
    comparison with RQ1a is an across-match contrast reported with propagated uncertainty.
    (f) **"Interpreted"** in §11 means read for any purpose other than the health monitoring of
    amendment 11; each cell is interpreted once, after its last block.
