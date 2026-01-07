# MetalFish

A GPU-accelerated UCI chess engine for Apple Silicon, implementing Stockfish-style search with Metal GPU acceleration.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

MetalFish is a chess engine that combines traditional alpha-beta search techniques from Stockfish with Apple Metal GPU acceleration on Apple Silicon's unified memory architecture.

## Features

### Search (32+ Stockfish Features Implemented)

#### Move Ordering
- **ButterflyHistory** - Quiet move success tracking by from/to squares
- **KillerMoves** - Refutation moves per ply
- **CounterMoveHistory** - Moves that refute the previous move
- **CapturePieceToHistory** - Capture move success tracking
- **PawnHistory** - Pawn structure-aware history (indexed by pawn key)
- **LowPlyHistory** - Extra weight for moves near root (first 5 plies)
- **ContinuationHistory** - Move sequence success (1, 2, 4, 6 ply lookback)

#### Search Extensions
- **Check Extension** - Extend when giving check
- **Singular Extension** - Extend clearly best moves
- **Multi-Cut Pruning** - Within singular extension framework
- **Passed Pawn Extension** - Extend for pawns reaching 7th rank
- **Recapture Extension** - Via LMR reduction decrease
- **Upcoming Repetition Detection** - Proactive repetition avoidance

#### Pruning Techniques
- **Null Move Pruning** - With verification search
- **Futility Pruning** - For quiet moves and captures
- **SEE-based Pruning** - Static Exchange Evaluation pruning
- **Late Move Pruning (LMP)** - Skip late quiet moves at shallow depths
- **Late Move Reductions (LMR)** - 14+ adjustment factors
- **ProbCut** - Prune with shallow capture search
- **Mate Distance Pruning** - Prune when short mate found
- **Internal Iterative Reductions (IIR)** - Reduce depth without TT move
- **History-based Pruning** - Skip moves with very negative history

#### Evaluation
- **NNUE Support** - Stockfish .nnue file loading with GPU acceleration
- **Rule50 Dampening** - Linear eval reduction as 50-move rule approaches
- **Correction History** - Adjust static eval based on search results
- **Draw Randomization** - Prevent 3-fold repetition blindness

#### Search Infrastructure
- **Transposition Table** - With aging and generation tracking
- **Aspiration Windows** - With averaging and fail-high tracking
- **Best Move Stability** - For time management decisions
- **Dynamic Time Management** - Adjust based on stability and score changes
- **Iterative Deepening** - Progressive deepening with info output
- **Quiescence Search** - Tactical resolution at leaf nodes
- **MultiPV** - Multiple principal variation search
- **Pondering** - Think on opponent's time
- **Thread Pool** - Infrastructure for multi-threaded search
- **Syzygy Tablebases** - Endgame tablebase probing interface

### GPU Acceleration (Metal)
- GPU-accelerated batch position evaluation
- Metal compute shaders for NNUE forward pass
- **GPU incremental accumulator updates** - Efficient NNUE updates
- Unified memory (zero-copy) on Apple Silicon
- GPU-accelerated move generation helpers
- GPU-accelerated SEE calculation

### Move Generation
- Magic bitboards for sliding pieces
- Legal move generation with pin detection
- Perft verification (all standard positions pass)

## Not Yet Implemented (Major Stockfish Features)

- **Lazy SMP** - Multi-threaded parallel search (infrastructure ready)
- **Optimism Blending** - Material-scaled optimism in eval
- **Full Syzygy TB Loading** - Currently interface only, file loading TBD

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1/M2/M3/M4)
- CMake 3.20+
- Xcode Command Line Tools

## Building

```bash
cd metalfish
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

## Usage

```bash
./metalfish
```

UCI commands:
```
uci
setoption name MultiPV value 3
position startpos
go depth 15
go movetime 5000
go ponder
ponderhit
stop
quit
```

## Testing

```bash
# C++ unit tests
./build/metalfish_tests

# Python perft tests
python3 tests/testing.py
```

## Architecture

```
metalfish/
├── src/
│   ├── core/          # Bitboards, position, move generation
│   ├── search/        # Alpha-beta search with all pruning techniques
│   ├── eval/          # NNUE loader, GPU evaluation
│   ├── metal/         # Metal device, GPU operations
│   └── uci/           # UCI protocol
├── shaders/           # Metal compute shaders
├── include/           # Header files
└── tests/             # Unit and integration tests
```

## Performance Notes

MetalFish uses GPU acceleration primarily for batch evaluation scenarios. For single-position evaluation during search, the overhead of GPU dispatch (~10-50μs) often exceeds the computational benefit. The engine automatically falls back to CPU evaluation for single positions while using GPU for batch operations where parallelism provides net benefit.

## Benchmark Results

*Last updated: 2025-01-07*

### Performance

| Metric | Value |
|--------|-------|
| Perft(6) Nodes | 119,060,324 |
| All Perft Tests | 30/30 Passing |
| Unit Tests | 5/5 Passing |

### Notes
- Benchmarks run on Apple Silicon (M-series)
- Hash size: 64 MB (default), Threads: 1 (single-threaded)
- GPU acceleration via Metal for batch evaluation
- Classical evaluation used when NNUE network not loaded

## License

GPL-3.0 - Same as Stockfish

## Credits

**Inspired by:**
- [Stockfish](https://github.com/official-stockfish/Stockfish) - Search algorithms, NNUE architecture
- [MLX](https://github.com/ml-explore/mlx) - Metal programming patterns

**NNUE Training Data:**
- Networks compatible with this engine use training data from [Leela Chess Zero](https://lczero.org/) (ODbL license)

## Author

**Nripesh Niketan** (2025)

Contributions and feedback welcome!
