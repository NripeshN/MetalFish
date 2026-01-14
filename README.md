# MetalFish

A high-performance UCI chess engine optimized for Apple Silicon, featuring Metal GPU-accelerated NNUE evaluation and multiple search algorithms.

## Overview

MetalFish is a chess engine that leverages Apple Silicon's unified memory architecture and Metal compute capabilities. It implements three distinct search approaches, all utilizing GPU-accelerated NNUE evaluation:

1. **Alpha-Beta Search** - Traditional minimax with pruning (Stockfish-derived)
2. **MCTS (Monte Carlo Tree Search)** - Tree search with neural network guidance
3. **Hybrid MCTS-Alpha-Beta** - Combines MCTS exploration with Alpha-Beta tactical verification

## Search Algorithms

### Alpha-Beta Search

The primary search algorithm, derived from Stockfish, featuring:

- Principal Variation Search (PVS) with aspiration windows
- Iterative deepening with transposition table
- Late Move Reductions (LMR) and Late Move Pruning
- Null Move Pruning, Futility Pruning, Razoring
- Singular Extensions and Check Extensions
- History heuristics (butterfly, capture, continuation, pawn)
- Killer moves and counter moves
- MVV-LVA move ordering

**UCI Command:** `go depth N` or `go movetime M`

### Monte Carlo Tree Search (MCTS)

A multi-threaded MCTS implementation optimized for GPU evaluation:

- PUCT selection with policy priors
- Virtual loss for multi-threaded tree traversal
- Lock-free atomic statistics updates
- Thread-local position management
- Dirichlet noise at root for exploration
- Heuristic policy priors (captures, checks, promotions)

**UCI Command:** `mctsmt threads=N movetime M`

**Performance:** ~700K nodes/second with 4 threads on M2 Max

### Hybrid MCTS-Alpha-Beta

Combines both approaches with dynamic strategy selection:

- Position classifier (tactical vs. strategic)
- MCTS for strategic exploration
- Alpha-Beta verifier for tactical positions
- Configurable verification depth and override threshold

**UCI Command:** `mcts movetime M`

## GPU Acceleration

MetalFish implements comprehensive GPU acceleration for NNUE evaluation:

### Architecture

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
- Batch evaluation for MCTS

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
│   ├── core/           # Bitboard, position, move generation
│   ├── search/         # Alpha-Beta search implementation
│   ├── eval/           # NNUE evaluation
│   │   └── nnue/       # Neural network architecture
│   ├── mcts/           # MCTS and hybrid search
│   │   ├── thread_safe_mcts.*     # Multi-threaded MCTS
│   │   ├── hybrid_search.*        # Hybrid MCTS-AB
│   │   ├── position_classifier.*  # Tactical/strategic detection
│   │   └── ab_integration.*       # Alpha-Beta integration
│   ├── gpu/            # GPU acceleration framework
│   │   ├── gpu_nnue_integration.* # GPU NNUE manager
│   │   ├── persistent_pipeline.*  # Command buffer optimization
│   │   └── metal/      # Metal backend
│   ├── uci/            # UCI protocol implementation
│   └── syzygy/         # Tablebase probing
├── tests/              # Test suite
├── tools/              # Tournament and analysis scripts
├── paper/              # Academic paper (LaTeX)
└── external/           # Dependencies (metal-cpp)
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
cmake .. -DUSE_METAL=ON
make -j8
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
| `mctsmt threads=N movetime M` | Multi-threaded MCTS search |
| `mcts movetime M` | Hybrid MCTS-Alpha-Beta search |

### UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Threads | spin | 1 | Number of Alpha-Beta search threads |
| Hash | spin | 16 | Transposition table size (MB) |
| MultiPV | spin | 1 | Number of principal variations |
| Skill Level | spin | 20 | Playing strength (0-20) |
| UseGPU | check | true | Enable GPU NNUE evaluation |
| SyzygyPath | string | | Path to Syzygy tablebase files |

### Benchmarks

```bash
# Standard benchmark
./build/metalfish bench

# GPU benchmark
./build/metalfish_gpu_bench

# Hybrid search validation
./build/metalfish hybridbench
```

## Testing

```bash
# Run all tests
./build/metalfish_tests

# Run specific test categories
./build/metalfish_tests  # Includes bitboard, position, movegen, search, MCTS, GPU tests
```

The test suite validates:
- Bitboard operations
- Position management and FEN parsing
- Move generation (perft verified)
- Search correctness
- MCTS components (nodes, tree, statistics)
- GPU shader compilation and execution
- GPU NNUE correctness vs CPU

## Performance Summary

### Alpha-Beta Search
- ~1.5M nodes/second (single thread, CPU NNUE)
- Standard Stockfish-level search quality

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

- Stockfish team for the search and evaluation framework
- Apple for Metal compute and unified memory architecture
- The computer chess community for research and testing methodologies
