# MetalFish

A GPU-accelerated UCI chess engine for Apple Silicon, implementing Stockfish-style search with Metal GPU acceleration.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

MetalFish is a chess engine that combines traditional alpha-beta search techniques from Stockfish with Apple Metal GPU acceleration on Apple Silicon's unified memory architecture.

## Features

### Search (60+ Stockfish Features Implemented)

#### Move Ordering

- **ButterflyHistory** - Quiet move success tracking by from/to squares (initialized to 68)
- **KillerMoves** - Refutation moves per ply
- **CounterMoveHistory** - Moves that refute the previous move
- **CapturePieceToHistory** - Capture move success tracking (initialized to -689)
- **PawnHistory** - Pawn structure-aware history indexed by pawn key (initialized to -1238)
- **LowPlyHistory** - Extra weight for moves near root (first 5 plies, filled with 97)
- **ContinuationHistory** - Move sequence success with Stockfish weights (1133, 683, 312, 582, 149, 474 for plies 1-6, initialized to -529)
- **Staged Move Generation** - Efficient MovePicker with capture/quiet phases
- **SearchedList** - Fixed-size list for tracking searched moves (32 capacity)

#### Search Extensions

- **Check Extension** - Extend when giving check
- **Singular Extension** - Extend clearly best moves (with double and triple extension)
- **Shuffling Detection** - Avoid over-extending in repetitive positions
- **Multi-Cut Pruning** - Within singular extension framework
- **Passed Pawn Extension** - Extend for pawns reaching 7th rank
- **Recapture Extension** - Via LMR reduction decrease
- **Upcoming Repetition Detection** - Proactive repetition avoidance (search and qsearch)
- **Negative Extensions** - Reduce ttMove when not singular

#### Pruning Techniques

- **Null Move Pruning** - With verification search at high depths
- **Futility Pruning** - For quiet moves and captures
- **SEE-based Pruning** - Static Exchange Evaluation pruning
- **Late Move Pruning (LMP)** - Skip late quiet moves at shallow depths
- **Late Move Reductions (LMR)** - Multiple adjustment factors including cutoffCnt
- **ProbCut** - Prune with shallow capture search
- **Small ProbCut** - TT-based pruning (beta + 418 threshold)
- **Razoring** - Drop to qsearch for low eval positions
- **Mate Distance Pruning** - Prune when short mate found
- **Internal Iterative Reductions (IIR)** - Reduce depth without TT move
- **History-based Pruning** - Skip moves with very negative history

#### Evaluation

- **NNUE Support** - Stockfish .nnue file loading with GPU acceleration
- **Classical Evaluation** - Material + piece-square tables fallback
- **Rule50 Dampening** - Linear eval reduction as 50-move rule approaches
- **Full Correction History** - Pawn, minor piece, non-pawn (white/black), continuation (initialized to 8)
- **Draw Randomization** - Prevent 3-fold repetition blindness
- **Optimism Blending** - Material-scaled optimism formula

#### Search Infrastructure

- **Transposition Table** - With aging, generation tracking, and rule50 handling
- **Proper TT Value Handling** - value_to_tt/value_from_tt with rule50 adjustment
- **TT Cutoff with History Bonus** - Update quiet histories on TT hit
- **Graph History Interaction Workaround** - Avoid TT cutoffs at high rule50
- **Aspiration Windows** - With meanSquaredScore-based delta sizing
- **Best Move Stability** - For time management decisions
- **Dynamic Time Management** - Adjust based on stability and score changes
- **Effort Tracking** - Nodes per root move for time allocation
- **Hindsight Depth Adjustment** - priorReduction-based depth changes
- **opponentWorsening Flag** - For improved pruning decisions
- **allNode Flag** - For LMR scaling on ALL nodes
- **evalDiff History Update** - Static eval difference improves quiet ordering
- **Iterative Deepening** - Progressive deepening with info output
- **Quiescence Search** - Tactical resolution at leaf nodes with repetition check
- **MultiPV** - Multiple principal variation search
- **Pondering** - Think on opponent's time
- **CutoffCnt Tracking** - For LMR adjustment based on child node behavior
- **update_all_stats** - Comprehensive history updates
- **Fail-Low Countermove Bonuses** - For quiet and capture countermoves
- **ttMoveHistory Updates** - Track TT move success for singular extension
- **ttPv Propagation** - Propagate PV status on fail low
- **Depth Reduction After Alpha Improvement** - Reduce depth after finding good move
- **bestValue Adjustment** - For fail high cases
- **Lazy SMP** - Multi-threaded parallel search with per-thread Position copies
- **Skill Level** - Playing strength handicap (0-20, Elo 1320-3190)
- **statScore** - History-based LMR adjustment
- **Post-LMR Continuation History Updates** - Update history after LMR re-search
- **Syzygy Tablebases** - Endgame tablebase probing interface

### GPU Acceleration (Metal)

- GPU-accelerated batch position evaluation
- Metal compute shaders for NNUE forward pass
- GPU incremental accumulator updates
- GPU move scoring kernel (MVV-LVA + history)
- GPU SEE calculation kernel
- Unified memory (zero-copy) on Apple Silicon
- GPU-accelerated move generation helpers
- GPU perft for verification

### Move Generation

- Magic bitboards for sliding pieces
- Legal move generation with pin detection
- Staged move generation (captures, killers, quiets)
- Perft verification (all standard positions pass)

## Not Yet Implemented

- **Full Syzygy TB Loading** - Currently interface only, file loading pending

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
# C++ unit tests (63 tests)
./build/metalfish_tests

# Python UCI and perft tests (39 tests)
python3 tests/testing.py --quick
```

### Test Coverage

| Category           | Tests   | Status          |
| ------------------ | ------- | --------------- |
| C++ Unit Tests     | 63      | Passing         |
| UCI Protocol Tests | 9       | Passing         |
| Perft Tests        | 30      | Passing         |
| **Total**          | **102** | **All Passing** |

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

MetalFish uses GPU acceleration primarily for batch evaluation scenarios. For single-position evaluation during search, the overhead of GPU dispatch can exceed the computational benefit. The engine automatically falls back to CPU evaluation for single positions while using GPU for batch operations where parallelism provides net benefit.

## Benchmark Results

### Performance

| Metric          | Value         |
| --------------- | ------------- |
| Perft(6) Nodes  | 119,060,324   |
| All Perft Tests | 30/30 Passing |
| C++ Unit Tests  | 63/63 Passing |
| UCI Tests       | 9/9 Passing   |

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

Contributions and feedback welcome.
