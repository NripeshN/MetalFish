# MetalFish

MetalFish is a UCI chess engine for Apple Silicon ARM Macs. The v0.1.0-alpha
release is focused on the hybrid engine: CPU alpha-beta with NNUE and GPU MCTS
with an Lc0-compatible transformer running through Metal/MPSGraph.

This alpha is intentionally Apple Silicon first. Linux, Windows, CUDA, and x86
builds are not release targets.

## Engine Modes

| Mode | UCI option | Purpose |
| --- | --- | --- |
| Hybrid | `UseHybridSearch true` | Primary engine for play |
| Alpha-Beta | `UseHybridSearch false`, `UseMCTS false` | CPU fallback and diagnostics |
| MCTS | `UseMCTS true` | Transformer search diagnostics |

Hybrid runs AB and MCTS in parallel. AB uses CPU NNUE and the transposition
table; MCTS uses the BT4 transformer on the Apple GPU. The two searches share
root signals and the coordinator chooses the final move from both engines.

## Current Alpha Results

Bratko-Kopec tactical suite, 5 seconds per position:

| Engine | Score |
| --- | --- |
| MetalFish Hybrid | 22/24 |
| MetalFish AB | 22/24 |
| MetalFish MCTS | 19/24 |
| Lc0 with same BT4 weights | 19/24 |
| Stockfish reference | 20/24 |

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
python3 tests/paper_benchmarks.py --tactical --movetime 5000 --threads 1
```

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
