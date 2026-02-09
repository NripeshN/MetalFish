# MetalFish

A high-performance UCI chess engine built for Apple Silicon, featuring Metal GPU-accelerated neural network evaluation and three distinct search engines.

## Overview

MetalFish exploits Apple Silicon's unified memory architecture and Metal GPU compute to deliver competitive chess analysis. It ships three search modes selectable at runtime:

| Engine | Description | UCI Command |
|--------|-------------|-------------|
| **Alpha-Beta** | Classical minimax with modern pruning | `go` |
| **MCTS** | GPU-batched Monte Carlo Tree Search | `mctsmt` |
| **Hybrid** | Parallel MCTS + Alpha-Beta fusion | `parallel_hybrid` |

## Search Engines

### Alpha-Beta (search/)

A full-featured iterative-deepening PVS search:

- Aspiration windows with gradual widening
- Null move pruning, futility pruning, razoring
- Late Move Reductions (LMR) and Late Move Pruning
- Singular extensions and check extensions
- Multi-cut pruning
- History heuristics (butterfly, capture, continuation, pawn)
- Killer moves and counter moves
- Static Exchange Evaluation (SEE) for capture ordering
- Transposition table with cluster-based replacement
- Syzygy tablebase probing at root and in-search
- GPU-accelerated NNUE evaluation

### MCTS (mcts/)

A multi-threaded Monte Carlo Tree Search engine tuned for GPU batch evaluation:

- **PUCT selection** with logarithmic exploration growth
- **First Play Urgency** reduction strategy for unvisited nodes
- **Moves Left Head** utility for shorter-win preference
- **WDL rescaling** with configurable draw contempt
- **Dirichlet noise** at the root for training-style exploration
- **Policy softmax temperature** for move sharpening
- **Virtual loss** for lock-free parallel tree traversal
- **Arena-based allocation** with 128-byte cache-line aligned nodes
- **Batched GPU evaluation** with adaptive timeout and double-buffering
- **vDSP-accelerated softmax** via Apple's Accelerate framework

### Hybrid (hybrid/)

Runs MCTS and Alpha-Beta in parallel, combining their strengths:

- Lock-free atomic communication between search threads
- Position classifier selects MCTS/AB weighting per position
- AB tactical results update MCTS policy priors in real time
- MCTS exploration guides AB move ordering
- Coordinator thread manages time and produces the final decision

## Neural Networks

MetalFish uses two complementary networks:

### NNUE (eval/nnue/)

Efficiently Updatable Neural Network for the Alpha-Beta engine:

- Dual-network architecture (big: 1024, small: 128 hidden dimensions)
- HalfKAv2_hm feature set (45,056 king-relative piece-square features)
- Incremental accumulator updates on make/unmake
- 8 layer stacks with PSQT buckets
- GPU-accelerated inference via Metal compute shaders

### Transformer (nn/)

Attention-based network for the MCTS engine:

- 112-plane input encoding (8 history positions + auxiliary planes)
- Multi-head attention encoder layers with FFN
- Attention-based policy head (1858-move output)
- WDL value head and optional moves-left head
- Input canonicalization with board transforms
- Supports `.pb` and `.pb.gz` weight files (float32, float16, bfloat16, linear16 encodings)

## Apple Silicon Optimizations

MetalFish is purpose-built for Apple Silicon:

| Optimization | Detail |
|-------------|--------|
| **FP16 weights** | Transformer weights stored as float16 on GPU for 2x memory bandwidth |
| **Unified memory** | Zero-copy CPU/GPU data sharing, no transfer overhead |
| **Buffer pooling** | Pre-allocated I/O buffers with `os_unfair_lock` avoid per-inference allocation |
| **Sub-batch parallelism** | Large batches split across parallel Metal command buffers |
| **Actual batch eval** | GPU evaluates only the real batch size, not the padded maximum |
| **vDSP softmax** | Accelerate framework SIMD for policy softmax in MCTS |
| **Fast math** | Bit-hack `FastLog`, `FastTanh`, `FastExp` for PUCT computation |
| **128-byte alignment** | Node structures aligned to Apple Silicon cache lines |
| **Metal compute** | Custom Metal shaders for NNUE sparse inference |
| **MPSGraph** | Apple's graph API for transformer encoder/attention/FFN |

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
  tests/                     Test suite
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

The NNUE network files should be placed in the `networks/` directory or the working directory:

```bash
# NNUE networks (for Alpha-Beta)
networks/nn-c288c895ea92.nnue
networks/nn-37f18f62d772.nnue

# Transformer network (for MCTS)
networks/BT4-1024x15x32h-swa-6147500.pb
```

## Usage

MetalFish speaks the Universal Chess Interface (UCI) protocol.

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

### Search Commands

| Command | Engine | Description |
|---------|--------|-------------|
| `go depth N` | Alpha-Beta | Search to depth N |
| `go movetime M` | Alpha-Beta | Search for M milliseconds |
| `go wtime W btime B` | Alpha-Beta | Tournament time control |
| `mctsmt movetime M` | MCTS | Multi-threaded MCTS for M ms |
| `mctsmt nodes N` | MCTS | MCTS with N node budget |
| `parallel_hybrid movetime M` | Hybrid | Parallel MCTS+AB for M ms |

### Key UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `Threads` | spin | 1 | Search threads |
| `Hash` | spin | 16 | Transposition table (MB) |
| `MultiPV` | spin | 1 | Principal variations |
| `Skill Level` | spin | 20 | Strength (0-20) |
| `UseGPU` | check | true | GPU NNUE evaluation |
| `UseMCTS` | check | false | Use MCTS instead of AB |
| `NNWeights` | string | | Transformer network path |
| `SyzygyPath` | string | | Tablebase directory |
| `Ponder` | check | false | Pondering |

## Testing

```bash
cd build
make metalfish_tests test_nn_comparison
./metalfish_tests
./test_nn_comparison    # requires METALFISH_NN_WEIGHTS env var
```

The test suite validates:

- Bitboard operations and magic bitboards
- Position management and FEN parsing
- Move generation correctness
- Alpha-Beta search behavior
- MCTS tree construction and PUCT selection
- Hybrid search integration
- GPU shader compilation and inference
- Neural network output comparison against reference

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
