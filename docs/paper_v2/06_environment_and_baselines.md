# 06 — Environment, tooling and baseline measurements (2026-10-07)

## Hardware and OS

- Apple M2 Max: 12 CPU cores (8 performance, 4 efficiency), 38-core GPU (Metal 4), 32 GB
  unified memory; 252 GB free disk.
- macOS 27.2 (Darwin 27.2.0, build 26B5091g); Xcode-beta toolchain; Apple clang 21.0.0.
- Python 3.14.7 with python-chess 1.11.2, numpy 2.4.4; CMake, meson, ninja, zstd, wget,
  pandoc, MacTeX (`pdflatex`, `latexmk`, `bibtex`, `biber`) present.

## Engine build note (fixed today)

`build/metalfish` (built 2026-07-09) no longer ran: Homebrew had upgraded protobuf to 36.1
(the binary linked `libprotobuf.35.1.0.dylib`), and the CMake cache referenced a removed
`Xcode.app` SDK path. Fix: `rm -rf build/CMakeCache.txt build/CMakeFiles`, then
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON` and
`cmake --build build --target metalfish metalfish_tests metalfish_nn_probe -j12`.
The rebuilt engine answers `uci`/`isready` in ~2 s (`id name MetalFish 1.0.0`). The same
ABI break affects `reference/lc0/build/release/lc0` (abseil); Homebrew `lc0` 0.32.1 is used
instead. See `03_codebase_changes.md` C1.

## Installed evaluation tooling (details and command log in `reference/TOOLING.md`)

| Tool | Version | Path |
| --- | --- | --- |
| fastchess | v1.8.2-alpha (2026-07-26) | `reference/fastchess/fastchess` |
| cutechess-cli | 1.4.0 (Qt 6.11.2) | `reference/cutechess/build/cutechess-cli` |
| Ordo | 1.2.6 (17eec774) | `reference/ordo/ordo` |
| Stockfish 19 | tag sf_19 (edb0d9db), PGO `ARCH=apple-silicon`; bench 2,497,913 nodes, 1,306,439 nodes/s (bench is single-threaded) | `reference/stockfish-sf19/src/stockfish`; also `brew` Stockfish 19 at `/opt/homebrew/bin/stockfish` |
| Stockfish dev-20260106-c27c1747 | pre-existing | `reference/stockfish/src/stockfish` |
| Lc0 | 0.32.1 (Homebrew; Metal backend default) | `/opt/homebrew/bin/lc0` |
| Berserk / Patricia | pre-existing Jan-2026 builds; rebuild from releases 14 / 5.1 before RQ8 | `reference/berserk/src/berserk`, `reference/Patricia/engine/patricia` |
| Books | UHO_Lichess_4852_v1 (2,632,036), UHO_4060_v4 (241,670), UHO_4060_v3 (242,201), 8mvs_big_+80_+109 (25,857), noob_3moves (150,932), popularpos_lichess_v3 (200,000), endgames (157,846), Drawkiller_balanced_big (15,962), UHO_XXL_+0.90_+1.19.pgn (223,070 games), 8moves_v3.pgn (34,700 games) | `reference/books/unzipped/`, `reference/books/` |
| Suites | Arasan 2023 (200) and 2026 (200), WAC-new (300), BT2630 (30), ERET (111), STS v3 (1,500; also v4/v5/v6), TCEC15 (418) | `reference/testsuites/` |
| Lichess puzzles | 6,157,341 puzzles (CSV 1.14 GB) | `reference/lichess_puzzles/lichess_db_puzzle.csv` |
| Syzygy | 3-4-5 complete (145 rtbw + 145 rtbz, 939 MB) | `syzygy/` |
| Networks | BT4-1024x15x32h-swa-6147500.pb; nn-c288c895ea92.nnue; nn-37f18f62d772.nnue; t1-512x15x8h-distilled (ANE experiments only) | `networks/` |

## Baseline measurements

### BT4 evaluation latency (MetalFish `metalfish_nn_probe`, Metal/MPSGraph, start position, 5 warm-up + 40 iterations)

| Precision | Batch | Median ms / batch | Positions / s |
| --- | --- | --- | --- |
| FP16 (default) | 1 | 7.61 | 131 |
| FP16 | 16 | 36.1 | 443 |
| FP32 | 1 | 8.83 | 113 |
| FP32 | 16 | 38.3 | 418 |

Reading: batch-1 cost is dominated by fixed per-eval overhead (kernel launches and sync),
not arithmetic — FP16 saves 14% at batch 1, and batch 16 costs 4.7× the time for 16× the
work. This is the structural ceiling that limits MCTS node rates on this platform.

### Lc0 0.32.1 Metal, same BT4 weights (`lc0 benchmark`, 3 positions, 20 s each, 2 threads)

| minibatch-size | nps (3 positions) |
| --- | --- |
| 1 | 154, 161, 162 |
| 4 | 198, 217, 215 |
| 16 | 402, 411, 413 |
| 64 | 562, 589, 607 |
| 256 | 436, 561, 656 (still ramping at 20 s) |

Reading: at matched batch 1, Lc0 (≈160 nps) is slightly *faster* than MetalFish's probe
(131 pos/s); at batch 16 the two are comparable (≈410 vs 443). The chapter's "1.26× faster
than Lc0" and "5.5× in a complex middlegame" are not reproducible.

### fastchess end-to-end smoke test (MetalFish hybrid vs alpha-beta)

Command: fastchess, `tc=10+0.1`, `Threads=8`, `Hash=1024`, UHO_4060_v4 openings, 1 round
× 2 games with colour swap, `HybridTrace=true`, engine logs on. Result: 1 win, 1 draw for
the hybrid; 98 s wall-clock for 2 games including engine restarts; pentanomial summary
printed (`Ptnml(0-2): [0, 0, 0, 1, 0]`).

Telemetry from the engine log: `Hybrid thread split: MCTS=1 AB=6 (search workers=7, +1
coordinator, total=8)`; only 8 `HybridTrace` lines for the whole match (the remaining
hybrid moves ran as plain AB under the time-safety rule: `Time safety: estimated move
budget 388ms; using Alpha-Beta without transformer MCTS`); on the 8 true hybrid moves the
transformer performed 21 evaluations per move on average (max 36); reasons: `engines_agree`
×3, `ab_root_rejects_mcts` ×1. Conclusion for the protocol: at 10+0.1 the hybrid is
effectively the AB engine; the primary time control must be ≥ 30+0.3 and the GPU-active
fraction must be reported per cell.

### Repository artifacts reused as prior evidence

- `results/hybrid_vs_ab_tc/`: hybrid vs AB, 40+0.4: 0W-1L-49D (−6.9 ± 13.5, 98% draws);
  90+0.9: 0W-1L-39D (−8.7 ± 16.8). Balanced book, 8 threads each.
- `results/tf_on_vs_ab/`: hybrid (transformer forced on) vs AB, 20+0.5: 0W-0L-48D.
- `results/ab_vs_sf/`: MetalFish-AB vs Stockfish dev, 20+0.5, Threads=4: 0W-0L-22D.
- `results/tournament_20260701_212250/`: 20+0.2 hybrid telemetry — 437 hybrid-engine moves,
  228 time-safety fallbacks; on 209 true hybrid moves the AB move was selected 207 times;
  median 45 MCTS playouts/move; ≈74 evals/s with AB threads contending; 0 agreement stops.
- `results/paper_summary.md` (2026-06-30, 5 s/position, 8 threads): Bratko-Kopec AB 20/24,
  MCTS 24/24, Hybrid 24/24, Lc0 18/24, Stockfish 20/24; AB avg 7.37M NPS, Stockfish 9.98M.
- `results/lichess_audit/`: 1,058 games (99 W, 22 L, ≈85% draws); 38,461 ponderhits;
  0 time losses attributable to the engine.
