# MetalFish

MetalFish is a UCI chess engine for Apple Silicon ARM Macs. The v0.1.0-alpha
release is focused on the hybrid engine: CPU alpha-beta with NNUE and GPU MCTS
with an Lc0-compatible transformer running through Metal/MPSGraph.

This alpha is intentionally Apple Silicon first. Linux, Windows, CUDA, and x86
builds are not release targets.

## Project Status

MetalFish is experimental engine research, not a Stockfish replacement claim.
The standalone alpha-beta search is Stockfish-family search engineering; the
standalone MCTS path is an Lc0-compatible transformer search implementation.
The main research contribution is the hybrid runtime: CPU NNUE alpha-beta and
GPU transformer MCTS search the same position concurrently, exchange root-level
signals, and use a coordinator to select the final move.

## Engine Modes

| Mode | UCI option | Purpose |
| --- | --- | --- |
| Hybrid | `UseHybridSearch true` | Primary engine for play |
| Alpha-Beta | `UseHybridSearch false`, `UseMCTS false` | CPU fallback and diagnostics |
| MCTS | `UseMCTS true` | Transformer search diagnostics |

Hybrid runs AB and MCTS in parallel. AB uses CPU NNUE and the transposition
table; MCTS uses the BT4 transformer on the Apple GPU. The two searches share
root signals and the coordinator chooses the final move from both engines.

## Benchmarking Policy

Benchmark results in this repository must use a fair resource policy:

- One shared worker-thread budget per engine.
- On Apple Silicon, `auto` uses the performance-core count.
- CPU engines receive the same `Threads` and `Hash` settings where supported.
- Lc0 and MetalFish MCTS receive the same `Threads` value.
- MetalFish Hybrid receives the same total worker budget split between MCTS and
  AB workers.
- Tactical-suite scores are reported as tactical accuracy only, not Elo.

Run the fair Bratko-Kopec tactical sweep on the current machine:

```bash
python3 tests/paper_benchmarks.py --tactical --movetime 5000 \
  --threads auto --hash auto \
  --engines metalfish-ab,metalfish-mcts,metalfish-hybrid,lc0,stockfish
```

The script writes machine-readable results to `results/paper_tactical.json` and
a report to `results/paper_summary.md`. Any headline strength claims should be
based on those generated artifacts or on a larger cutechess tournament, not on
stale README tables.

Latest local fair tactical run, M2 Max, 2026-05-18, 5 seconds per position,
8 performance-core workers, 4096 MB hash where supported:

| Engine | Score | Completed | Notes |
| --- | ---: | ---: | --- |
| MetalFish AB | 21/24 | 24/24 | CPU NNUE alpha-beta |
| MetalFish Hybrid | 21/24 | 24/24 | 2 MCTS workers + 6 AB workers |
| Stockfish reference | 20/24 | 24/24 | same `Threads`/`Hash` budget |
| Lc0 with BT4 weights | 17/24 | 24/24 | Metal backend, `Threads=8` |
| MetalFish MCTS | 12/24 | 23/24 | timed out on BK.24 with full-worker MCTS |

This is a tactical-suite result only. It is useful for regression tracking, but
it is not an Elo estimate.

For game-strength testing, prefer cutechess matches with the same resource
policy:

```bash
THREADS=8 HASH=4096 ./tools/run_cutechess_tournament.sh \
  --games=20 --tc=300+0.1 --concurrency=1
```

## Apple Silicon Optimizations

- Metal/MPSGraph transformer inference
- Unified-memory CPU/GPU search architecture
- Accelerate/vDSP policy softmax
- NEON and ARM dot-product NNUE kernels
- 128-byte aligned MCTS nodes for Apple Silicon cache lines
- Native Apple CPU tuning and Release LTO by default
- Local Polyglot opening books for the Lichess bot
- Syzygy 3-4-5 tablebase support

## Requirements

- Apple Silicon Mac, arm64
- macOS 13 or newer
- Xcode Command Line Tools
- CMake 3.20 or newer
- Homebrew packages: `protobuf zlib abseil`
- Python 3.11+ for bot and benchmark tooling

```bash
brew install cmake protobuf zlib abseil
python3 -m pip install -r tests/requirements.txt
```

## Network Files

MetalFish expects network files under `networks/`.

```bash
mkdir -p networks
curl -L -o networks/nn-c288c895ea92.nnue \
  https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -L -o networks/nn-37f18f62d772.nnue \
  https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
curl -L -o networks/BT4-1024x15x32h-swa-6147500.pb.gz \
  https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz
gzip -dc networks/BT4-1024x15x32h-swa-6147500.pb.gz \
  > networks/BT4-1024x15x32h-swa-6147500.pb
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
cmake --build build --target metalfish -j"$(sysctl -n hw.ncpu)"
```

## Run Hybrid

```bash
./build/metalfish
```

Recommended UCI setup:

```text
setoption name UseHybridSearch value true
setoption name NNWeights value networks/BT4-1024x15x32h-swa-6147500.pb
setoption name Threads value 8
setoption name Hash value 4096
setoption name Ponder value true
isready
position startpos
go movetime 5000
```

## Lichess Bot

The bot is configured around the Hybrid engine, local opening books, Syzygy,
pondering, crash recovery, and resource-aware thread/hash sizing.

Download the included local Polyglot book set:

```bash
python3 tools/opening_book_manager.py download --book all
```

Run a rated seek with an Elo floor:

```bash
python3 tools/lichess_bot.py --seek --rotate --no-casual \
  --accept-rated --elo-seek --min-rated-opponent-elo 2200
```

The bot will load `books/*.bin` automatically. Online Explorer fallback is off
by default; enable it only when needed with `METALFISH_BOOK_ALLOW_ONLINE=true`.

## Tests

```bash
cmake --build build --target metalfish_tests -j"$(sysctl -n hw.ncpu)"
./build/metalfish_tests
python3 tests/testing.py --quick
python3 tests/test_lichess_bot.py
python3 tests/test_lichess_puzzle_runner.py
python3 tests/test_ponder_stress.py --smoke
python3 tests/paper_benchmarks.py --tactical --movetime 5000 \
  --threads auto --hash auto
```

## Strength Claims

Use precise language when discussing results:

- Valid: "MetalFish Hybrid solved X/24 Bratko-Kopec positions at 5s/position on
  an M2 Max using the fair benchmark script."
- Valid: "MetalFish MCTS loads the same BT4 Lc0 weights and uses compatible
  input/policy mapping, but has its own Metal runtime and search controls."
- Valid: "The hybrid engine combines CPU alpha-beta and GPU MCTS with live
  root-signal sharing and evidence-based final arbitration."
- Invalid without larger testing: "MetalFish is stronger than Stockfish" or
  "MetalFish has higher Elo than Lc0."

## Repository Layout

```text
src/core/      Board representation and move generation
src/eval/      NNUE evaluation and Apple Silicon CPU kernels
src/nn/        BT4 transformer loader, encoder, Metal/MPSGraph backend
src/search/    Alpha-beta search
src/mcts/      Lc0-style MCTS search
src/hybrid/    Parallel Hybrid coordinator
src/uci/       UCI protocol
tests/         Unit, UCI, ponder, and benchmark tests
tools/         Lichess bot, puzzle runner, books, Syzygy, tournaments
```

## License

MetalFish is released under GPL-3.0. See [LICENSE](LICENSE).
