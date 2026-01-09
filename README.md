# MetalFish

A high-performance UCI chess engine optimized for Apple Silicon, featuring Metal GPU acceleration and unified memory architecture.

## Overview

MetalFish is a chess engine designed to leverage Apple Silicon's unified memory architecture and Metal compute capabilities. The engine implements state-of-the-art search algorithms combined with NNUE neural network evaluation, delivering competitive performance on modern Apple hardware.

## Features

### Search

- Alpha-Beta search with iterative deepening
- Principal Variation Search (PVS)
- Transposition table with aging
- Late Move Reductions (LMR)
- Null Move Pruning
- Futility Pruning
- Razoring
- ProbCut
- Singular Extensions
- Check Extensions
- Internal Iterative Reductions (IIR)
- Aspiration Windows
- MultiPV support

### Move Ordering

- Hash move priority
- MVV-LVA for captures
- Killer moves
- Counter moves
- History heuristics (butterfly, capture, continuation)
- Low ply history
- Pawn history

### Evaluation

- NNUE (Efficiently Updatable Neural Network)
- HalfKAv2 architecture
- Incremental accumulator updates
- NEON SIMD acceleration on Apple Silicon

### Additional Features

- Syzygy tablebase support
- Skill level adjustment
- UCI Elo rating limit
- Pondering
- Time management with sudden death and increment support

### GPU Acceleration (Metal)

- Unified memory for zero-copy CPU/GPU data sharing
- Metal compute shaders for batch operations
- GPU-accelerated NNUE evaluation infrastructure

## Project Structure

```
metalfish/
├── src/
│   ├── core/           # Bitboard, position, move generation
│   ├── search/         # Search algorithms, history tables, TT
│   ├── eval/           # NNUE evaluation
│   │   └── nnue/       # Neural network implementation
│   ├── uci/            # UCI protocol
│   ├── metal/          # Metal GPU acceleration
│   └── syzygy/         # Tablebase probing
├── shaders/            # Metal compute shaders
├── external/           # External dependencies
├── tests/              # Test suite
└── reference/          # Reference implementations
```

## Building

### Requirements

- macOS 12.0 or later
- Xcode Command Line Tools
- CMake 3.20 or later
- Apple Silicon (M1/M2/M3/M4) recommended

### Build Instructions

```bash
cd metalfish
mkdir build && cd build
cmake ..
make -j8
```

### NNUE Network Files

The NNUE network files are required but not included in the repository due to size. Download them and place in the `src/` directory:

```bash
cd src
curl -LO https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -LO https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
```

### Running Tests

```bash
# C++ unit tests
./metalfish_tests

# Python integration tests
python3 ../tests/testing.py
```

## Usage

MetalFish implements the Universal Chess Interface (UCI) protocol and is compatible with standard chess GUIs:

- Arena
- Cute Chess
- Banksia GUI
- lichess-bot
- SCID

### Command Line

```bash
./metalfish
```

Example UCI session:

```
uci
isready
position startpos
go depth 20
```

### UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Threads | spin | 1 | Number of search threads |
| Hash | spin | 16 | Transposition table size (MB) |
| MultiPV | spin | 1 | Number of principal variations |
| Skill Level | spin | 20 | Playing strength (0-20) |
| UCI_Elo | spin | 3190 | Elo rating limit |
| SyzygyPath | string | | Path to Syzygy tablebase files |
| SyzygyProbeDepth | spin | 1 | Minimum depth for tablebase probing |
| Syzygy50MoveRule | check | true | Use 50-move rule in tablebase probing |
| SyzygyProbeLimit | spin | 7 | Maximum pieces for tablebase probing |

## Performance

Benchmark results on Apple Silicon:

| Metric | Value |
|--------|-------|
| Nodes/second | ~1.5M (single thread) |
| Benchmark nodes | 2,351,490 |
| SIMD | NEON with dot product |

The engine produces identical search results to reference implementations, ensuring correctness of the search algorithm and evaluation function.

## Testing

The test suite includes:

- Bitboard operations
- Position management
- Move generation (perft verified)
- Search components
- Metal GPU functionality
- UCI protocol compliance

All 30 standard perft positions pass with correct node counts.

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

## Author

Nripesh Niketan
