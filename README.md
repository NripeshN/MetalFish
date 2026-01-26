# MetalFish

A high-performance UCI chess engine optimized for Apple Silicon, featuring Metal GPU-accelerated NNUE evaluation and multiple search algorithms including advanced MCTS.

## Overview

MetalFish is a chess engine that leverages Apple Silicon's unified memory architecture and Metal compute capabilities. It implements three distinct search approaches:

| Search Mode | Description | UCI Command |
|-------------|-------------|-------------|
| **Alpha-Beta** | Traditional minimax with pruning | `go` |
| **MCTS** | Monte Carlo Tree Search with GPU batching | `mctsmt` |
| **Hybrid** | Parallel MCTS + Alpha-Beta with dynamic integration | `parallel_hybrid` |

## Search Algorithms

### Alpha-Beta Search (MetalFish-AB)

The primary search algorithm featuring:

- Principal Variation Search (PVS) with aspiration windows
- Iterative deepening with transposition table
- Late Move Reductions (LMR) and Late Move Pruning
- Null Move Pruning, Futility Pruning, Razoring
- Singular Extensions and Check Extensions
- History heuristics (butterfly, capture, continuation, pawn)
- Killer moves and counter moves
- MVV-LVA move ordering
- GPU-accelerated NNUE evaluation

### Monte Carlo Tree Search (MetalFish-MCTS)

A multi-threaded MCTS implementation optimized for Apple Silicon GPU evaluation:

#### Core Algorithms

- **PUCT Selection**: Logarithmic exploration bonus with configurable cpuct
  ```
  cpuct = init + factor * log((parent_N + base) / base)
  ```
- **First Play Urgency (FPU)**: Reduction strategy for unvisited nodes
  ```
  fpu = -parent_Q - reduction * sqrt(visited_policy)
  ```
- **Moves Left Head (MLH)**: Utility adjustment for preferring shorter wins
- **WDL Rescaling**: Win/Draw/Loss probability rescaling for contempt
- **Dirichlet Noise**: Root exploration with configurable alpha and epsilon
- **Policy Temperature**: Softmax temperature for move selection
- **Solid Tree Optimization**: Cache-locality improvements for large trees

#### Multi-Threading

- Virtual loss for concurrent tree traversal
- Lock-free atomic statistics updates using `std::memory_order_relaxed`
- Thread-local position management
- Collision detection and handling

#### Apple Silicon Optimizations

- SIMD-accelerated policy softmax using Accelerate framework (`vDSP_*`)
- 128-byte cache-line aligned node statistics
- GPU-resident evaluation batches in unified memory
- Asynchronous GPU dispatch with completion handlers
- Zero-copy CPU/GPU data sharing

### Hybrid MCTS-Alpha-Beta (MetalFish-Hybrid)

A parallel hybrid search that runs MCTS and Alpha-Beta simultaneously:

#### Architecture

- **Parallel Execution**: MCTS and AB run in separate threads concurrently
- **Shared State**: Lock-free communication via atomic variables
- **Coordinator Thread**: Manages time allocation and final decision

#### Integration Strategy

- MCTS provides broad exploration and move ordering
- Alpha-Beta provides tactical verification and precise evaluation
- Dynamic weighting based on:
  - Search depth achieved
  - Score agreement between searches
  - Position characteristics (tactical vs. strategic)

#### Decision Logic

1. If both searches agree on best move, use it immediately
2. If AB finds a significantly better move (threshold-based), prefer AB
3. Otherwise, weight by search confidence and depth

## GPU Acceleration

### Metal Backend

MetalFish implements comprehensive GPU acceleration using Apple's Metal framework:

- Metal compute shaders for neural network inference
- Zero-copy CPU/GPU data sharing via unified memory
- Persistent command buffers to minimize dispatch overhead
- Batch processing for efficient GPU utilization
- Thread-safe GPU access with mutex protection

### GPU-Accelerated Operations

- Feature extraction (HalfKAv2_hm architecture)
- Feature transformer with sparse input handling
- Dual-perspective accumulator updates
- Fused forward pass for output layers
- Batch evaluation for MCTS (up to 256 positions per batch)
- SIMD policy softmax computation

### Performance (Apple M2 Max)

| Metric | Value |
|--------|-------|
| GPU Batch Throughput | 3.3M positions/second |
| Single Position Latency | ~285 microseconds |
| Dispatch Overhead | ~148 microseconds |
| Unified Memory Bandwidth | 52.7 GB/s |

## Project Structure

```
metalfish/
├── src/
│   ├── core/              # Bitboard, position, move generation
│   │   ├── bitboard.*     # Bitboard operations and magic bitboards
│   │   ├── position.*     # Board representation and state
│   │   ├── movegen.*      # Legal move generation
│   │   └── types.h        # Core type definitions
│   ├── search/            # Alpha-Beta search implementation
│   │   ├── search.*       # Main search loop
│   │   ├── movepick.*     # Move ordering
│   │   └── thread.*       # Thread pool management
│   ├── eval/              # NNUE evaluation
│   │   └── nnue/          # Neural network architecture
│   ├── mcts/              # MCTS and hybrid search
│   │   ├── mcts_core.h            # Core MCTS algorithms
│   │   ├── thread_safe_mcts.*     # Multi-threaded MCTS
│   │   ├── parallel_hybrid_search.* # Parallel hybrid search
│   │   ├── hybrid_search.*        # Hybrid MCTS-AB integration
│   │   ├── apple_silicon_mcts.*   # Apple Silicon optimizations
│   │   ├── mcts_tt.*              # MCTS transposition table
│   │   └── ab_integration.*       # Alpha-Beta bridge
│   ├── gpu/               # GPU acceleration framework
│   │   ├── gpu_nnue_integration.* # GPU NNUE manager
│   │   ├── gpu_accumulator.*      # Feature extraction
│   │   ├── gpu_mcts_backend.*     # MCTS GPU backend
│   │   ├── backend.h              # GPU backend interface
│   │   └── metal/                 # Metal implementation
│   │       ├── metal_backend.mm   # Metal backend
│   │       └── nnue.metal         # Metal shaders
│   ├── uci/               # UCI protocol implementation
│   └── syzygy/            # Tablebase probing
├── tests/                 # Comprehensive test suite
├── tools/                 # Tournament and analysis scripts
│   ├── elo_tournament.py  # Automated Elo tournament
│   └── engines_config.json # Engine configuration
├── reference/             # Reference engines (gitignored)
└── external/              # Dependencies (metal-cpp)
```

## Building

### Requirements

- macOS 12.0 or later
- Xcode Command Line Tools
- CMake 3.20 or later
- Ninja (recommended)
- Apple Silicon (M1/M2/M3/M4) recommended

### Build Instructions

```bash
cd metalfish
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja metalfish
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| USE_METAL | ON (macOS) | Enable Metal GPU acceleration |
| BUILD_TESTS | ON | Build test suite |
| BUILD_GPU_BENCHMARK | ON | Build GPU benchmark utility |

### NNUE Network Files

Download the required network files:

```bash
cd src
curl -LO https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -LO https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
```

## Usage

MetalFish implements the Universal Chess Interface (UCI) protocol.

### Quick Start

```bash
./build/metalfish
```

### Example UCI Session

```
uci
isready
position startpos
go depth 20
```

### Search Commands

| Command | Description |
|---------|-------------|
| `go depth N` | Alpha-Beta search to depth N |
| `go movetime M` | Alpha-Beta search for M milliseconds |
| `go wtime W btime B` | Alpha-Beta with time management |
| `mctsmt movetime M` | Multi-threaded MCTS search |
| `parallel_hybrid movetime M` | Parallel hybrid MCTS+AB search |

### UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Threads | spin | 1 | Number of search threads |
| Hash | spin | 16 | Transposition table size (MB) |
| MultiPV | spin | 1 | Number of principal variations |
| Skill Level | spin | 20 | Playing strength (0-20) |
| UseGPU | check | true | Enable GPU NNUE evaluation |
| SyzygyPath | string | | Path to Syzygy tablebase files |
| Ponder | check | false | Enable pondering |

## Testing

### Running Tests

```bash
# Build and run all tests
cd build
ninja metalfish_tests
./metalfish_tests
```

### Test Coverage

The test suite validates:

- Bitboard operations and magic bitboards
- Position management and FEN parsing
- Move generation (perft verified)
- Alpha-Beta search correctness
- MCTS components (nodes, tree, statistics, PUCT)
- Hybrid search integration
- GPU shader compilation and execution
- GPU NNUE correctness vs CPU reference
- Alpha-Beta integration bridge

## Elo Tournament

MetalFish includes an automated tournament system for Elo estimation.

### Tournament Configuration

| Setting | Value |
|---------|-------|
| Opening Book | 8moves_v3.pgn (16 plies, CCRL-style) |
| Games per Opening | 2 (color swap) |
| Time Control | 10+0.1 |
| Ponder | OFF |

### Running Locally

```bash
python3 tools/elo_tournament.py --games 20 --time "10+0.1"
```

### CI Tournament

The GitHub Actions workflow runs a comprehensive tournament against various open-source engines at different strength levels.

## Performance Summary

### Alpha-Beta Search

- ~1.5M nodes/second (single thread)
- High-quality search with GPU-accelerated NNUE evaluation

### MCTS Search

| Threads | NPS | Scaling |
|---------|-----|---------|
| 1 | 333K | 1.0x |
| 2 | 405K | 1.2x |
| 4 | 706K | 2.1x |

### GPU NNUE

- Batch evaluation: 3.3M positions/second
- Speedup over sequential: 11.6x at batch size 4096

## Compatibility

MetalFish is compatible with standard chess GUIs:

- Cute Chess
- Arena
- Banksia GUI
- SCID
- lichess-bot

## License

GNU General Public License v3.0. See LICENSE file for details.

## Author

Nripesh Niketan

## Acknowledgments

MetalFish builds upon research and techniques from the open-source chess engine community:
- Advanced search algorithms and evaluation techniques
- Monte Carlo Tree Search research and implementations
- Apple's Metal framework and unified memory architecture
- The computer chess community for research and testing methodologies
