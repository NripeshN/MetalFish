# MetalFish

A high-performance UCI chess engine built for Apple Silicon, featuring Metal GPU-accelerated neural network evaluation and three distinct search engines.

## Overview

MetalFish exploits Apple Silicon's unified memory architecture and Metal GPU compute to deliver competitive chess analysis. It ships three search modes selectable at runtime via standard UCI options:

| Engine | Description | UCI Option |
|--------|-------------|------------|
| **Alpha-Beta** | Classical minimax with CPU NNUE (~4M NPS) | Default |
| **MCTS** | GPU-batched Monte Carlo Tree Search with transformer | `setoption name UseMCTS value true` |
| **Hybrid** | Parallel MCTS + AB with real-time PV injection | `setoption name UseHybridSearch value true` |

## Search Engines

### Alpha-Beta (search/)

A full-featured iterative-deepening PVS search with CPU NNUE evaluation:

- Aspiration windows with gradual widening
- Null move pruning, futility pruning, razoring
- Late Move Reductions (LMR) and Late Move Pruning
- Singular extensions and check extensions
- History heuristics (butterfly, capture, continuation, pawn)
- Killer moves and counter moves
- Static Exchange Evaluation (SEE) for capture ordering
- Transposition table with cluster-based replacement
- Syzygy tablebase probing at root and in-search
- CPU NNUE with NEON SIMD (~80ns per eval, ~4M NPS)

### MCTS (mcts/)

A multi-threaded Monte Carlo Tree Search engine with GPU transformer evaluation:

- **PUCT selection** with logarithmic exploration growth
- **First Play Urgency** reduction strategy for unvisited nodes
- **Moves Left Head** utility for shorter-win preference
- **WDL rescaling** with configurable draw contempt
- **Dirichlet noise** at the root for exploration
- **Policy softmax temperature** with vDSP SIMD acceleration
- **Virtual loss** for lock-free parallel tree traversal
- **Arena-based allocation** with 128-byte cache-line aligned nodes
- **Batched GPU evaluation** with adaptive timeout and double-buffering
- **O(1) policy lookup** via pre-built move index table

### Hybrid (hybrid/)

Runs MCTS and Alpha-Beta in true parallel, combining their strengths via real-time PV injection:

- **CPU (AB)** and **GPU (MCTS)** run simultaneously at full throughput
- AB uses `search_with_callbacks()` for native iterative deepening with per-iteration PV publishing
- MCTS reads AB PV from shared state (zero-copy unified memory) and boosts those edges in the tree
- **Agreement-based early stopping** -- when both engines agree on the same move for 3+ checks, search stops early (saves ~40-50% time)
- Position classifier tunes decision weights (tactical vs strategic)
- Lock-free atomic communication between threads

## Neural Networks

MetalFish uses two complementary networks:

### NNUE (eval/nnue/)

Efficiently Updatable Neural Network for the Alpha-Beta engine:

- Dual-network architecture (big: 1024, small: 128 hidden dimensions)
- HalfKAv2_hm feature set (45,056 king-relative piece-square features)
- Incremental accumulator updates on make/unmake
- 8 layer stacks with PSQT buckets
- NEON SIMD with dot product instructions on Apple Silicon

### Transformer (nn/)

Attention-based network for the MCTS engine:

- 112-plane input encoding (8 history positions + auxiliary planes)
- Multi-head attention encoder layers with FFN
- Attention-based policy head (1858-move output)
- WDL value head and optional moves-left head
- Input canonicalization with board transforms
- Supports `.pb` and `.pb.gz` weight files (float32, float16, bfloat16, linear16 encodings)

## Apple Silicon Optimizations

| Optimization | Detail |
|-------------|--------|
| **FP16 weights** | Transformer weights stored as float16 on GPU for 2x memory bandwidth |
| **Unified memory** | Zero-copy CPU/GPU data sharing, no transfer overhead |
| **Buffer pooling** | Pre-allocated I/O buffers with `os_unfair_lock` avoid per-inference allocation |
| **Sub-batch parallelism** | Large batches split across parallel Metal command buffers |
| **Actual batch eval** | GPU evaluates only the real batch size, not the padded maximum |
| **vDSP softmax** | Accelerate framework SIMD for policy softmax in MCTS |
| **Fast math** | Bit-hack `FastLog`, `FastTanh`, `FastExp`, `FastSqrt` for PUCT |
| **128-byte alignment** | Node structures aligned to Apple Silicon cache lines |
| **Metal compute** | Custom Metal shaders for NNUE sparse inference |
| **MPSGraph** | Apple's graph API for transformer encoder/attention/FFN |
| **ARM yield** | `__builtin_arm_yield()` in spin-wait loops |
| **NEON dot product** | `-march=armv8.2-a+dotprod` for NNUE feature transforms |

## Project Structure

```
metalfish/
  src/
    main.cpp                 Entry point
    core/                    Bitboard, position, move generation, types
    eval/                    NNUE evaluation + Metal GPU acceleration
      nnue/                  Network layers, features, accumulator
      metal/                 Metal compute shaders for NNUE
    nn/                      Transformer network for MCTS
      metal/                 MPSGraph backend
        mps/                 Network graph builder
        tables/              Policy mapping tables
      proto/                 Protobuf weight format
    search/                  Alpha-Beta search engine
    mcts/                    MCTS search engine
    hybrid/                  Hybrid MCTS+AB search engine
    uci/                     UCI protocol, engine, options
    syzygy/                  Syzygy tablebase probing
  tests/                     Test suite (5 modules, 100+ assertions)
  tools/                     Tournament scripts
  networks/                  Network weight files
```

## Building

### Requirements

- macOS 13.0 or later
- Xcode Command Line Tools (with Metal support)
- CMake 3.20+
- Protobuf 3.0+
- Apple Silicon (M1/M2/M3/M4) recommended

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_METAL` | ON (macOS) | Metal GPU acceleration |
| `BUILD_TESTS` | ON | Build test suite |

### Network Files

Place network files in the `networks/` directory:

```bash
# NNUE networks (for Alpha-Beta) -- auto-loaded on startup
networks/nn-c288c895ea92.nnue
networks/nn-37f18f62d772.nnue

# Transformer network (for MCTS/Hybrid) -- set via UCI option
networks/BT4-1024x15x32h-swa-6147500.pb
```

## Usage

MetalFish speaks the Universal Chess Interface (UCI) protocol and is compatible with all standard chess GUIs.

### Quick Start

```bash
./build/metalfish
```

### Example Session

```
uci
setoption name Threads value 4
setoption name Hash value 256
isready
position startpos
go depth 20
```

### Engine Modes

All three engines are accessed via the standard `go` command. Set the mode with UCI options before searching:

```
# Alpha-Beta (default)
setoption name UseMCTS value false
setoption name UseHybridSearch value false
go movetime 5000

# MCTS (requires transformer network)
setoption name UseMCTS value true
setoption name NNWeights value /path/to/BT4-network.pb
go nodes 800

# Hybrid (requires transformer network)
setoption name UseHybridSearch value true
setoption name NNWeights value /path/to/BT4-network.pb
go movetime 5000
```

### Key UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `Threads` | spin | 1 | Search threads |
| `Hash` | spin | 16 | Transposition table (MB) |
| `MultiPV` | spin | 1 | Principal variations |
| `Skill Level` | spin | 20 | Strength (0-20) |
| `UseMCTS` | check | false | Use MCTS engine |
| `UseHybridSearch` | check | false | Use Hybrid engine |
| `UseGPU` | check | true | Enable GPU NNUE for MCTS batching |
| `NNWeights` | string | | Transformer network path |
| `SyzygyPath` | string | | Tablebase directory |
| `Ponder` | check | false | Pondering |

## Testing

```bash
cd build

# Run all unit tests (core, search, eval/gpu, mcts, hybrid)
./metalfish_tests

# Run a specific test module
./metalfish_tests mcts

# Run NN comparison test (requires METALFISH_NN_WEIGHTS env var)
METALFISH_NN_WEIGHTS=/path/to/BT4-network.pb ./test_nn_comparison

# Python integration tests (UCI protocol, perft)
cd .. && python3 tests/testing.py
```

### Test Coverage

| Module | Tests | What it covers |
|--------|-------|----------------|
| core | 29 | Bitboard, position, move generation, FEN, castling, en passant |
| search | 21 | History tables, limits, root moves, skill, stack, values |
| eval_gpu | 1031 | Metal detection, buffer alloc/read/write, unified memory, NNUE manager |
| mcts | 27 | Node creation, edges, policy, tree structure, PUCT, thread safety |
| hybrid | 22 | Config, shared state, classifier, position adapter, strategy |

## Compatibility

MetalFish works with any UCI-compatible chess GUI:

- Cute Chess
- Arena
- Banksia GUI
- SCID
- lichess-bot

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Author

Nripesh Niketan
