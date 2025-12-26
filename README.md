# MetalFish

A GPU-accelerated UCI chess engine using Apple Metal, designed for Apple Silicon.

[![CI](https://github.com/yourusername/metalfish/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/metalfish/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

MetalFish is a chess engine that leverages Apple Silicon's unified memory architecture to accelerate neural network evaluation using Metal GPU shaders. It implements Stockfish-level search algorithms with GPU-accelerated NNUE (Efficiently Updatable Neural Network) evaluation.

**Key Innovation**: By utilizing Apple Silicon's unified memory (zero-copy CPU-GPU sharing), MetalFish batches position evaluations to the GPU while maintaining the sophisticated alpha-beta search that makes Stockfish the world's strongest chess engine.

## Features

### Search
- **Alpha-Beta Search**: Full Stockfish-style principal variation search
- **Iterative Deepening**: Progressive depth search with time management
- **Transposition Table**: 16MB default hash table with replacement strategy
- **Move Ordering**: MVV-LVA, killer moves, history heuristics, counter moves
- **Quiescence Search**: Extended tactical search at leaf nodes
- **Check Extensions**: Deeper search when in check

### Pruning & Reductions
- **Late Move Reductions (LMR)**: Depth reduction for late moves
- **Null Move Pruning**: Skip move to prove beta cutoff
- **Futility Pruning**: Prune hopeless moves near leaf
- **Razoring**: Reduce depth on weak positions
- **Static Exchange Evaluation (SEE)**: Prune bad captures

### Evaluation
- **GPU NNUE**: HalfKAv2_hm architecture (45056 → 1024 → 16 → 32 → 1)
- **Batched GPU Inference**: Multiple positions evaluated in parallel
- **Unified Memory**: Zero-copy data transfer between CPU and GPU
- **Classical Fallback**: Material + PST evaluation when GPU unavailable

### UCI Protocol
- Full UCI compliance for GUI integration
- Configurable hash size, threads, NNUE path
- Position analysis, benchmark, and perft commands

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1/M2/M3/M4) - **Metal GPU required**
- CMake 3.20+
- Xcode Command Line Tools
- Python 3.8+ (for perft tests)

## Building

```bash
cd metalfish
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_METAL_SHADERS` | OFF | Compile Metal shaders to metallib |
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release) |

### With Metal Shader Compilation

```bash
cmake .. -DBUILD_METAL_SHADERS=ON
make -j$(sysctl -n hw.ncpu)
```

## Usage

### UCI Mode (Default)

```bash
./metalfish
```

### Common UCI Commands

```
uci                         # Initialize UCI protocol
setoption name Hash value 256   # Set hash table size (MB)
setoption name Threads value 4  # Set thread count
position startpos           # Set up starting position
position fen <fen>          # Set up custom position
go depth 20                 # Search to depth 20
go movetime 5000            # Search for 5 seconds
go wtime 60000 btime 60000  # Tournament time control
quit                        # Exit
```

### Benchmark

```bash
echo "bench 16" | ./metalfish
```

### Perft Testing

```bash
echo "position startpos\nperft 6\nquit" | ./metalfish
```

### Run All Tests

```bash
# C++ unit tests
./metalfish_tests

# Python integration tests (perft + UCI)
python3 tests/testing.py
```

## Architecture

```
metalfish/
├── include/
│   ├── core/          # types.h, bitboard.h, position.h, movegen.h, zobrist.h
│   ├── metal/         # device.h, allocator.h
│   ├── search/        # search.h, tt.h, movepick.h
│   ├── eval/          # evaluate.h, nnue.h, gpu_nnue.h, nnue_loader.h
│   └── uci/           # uci.h
├── src/
│   ├── core/          # Core chess logic
│   ├── metal/         # Metal device, allocator, kernels
│   ├── search/        # Alpha-beta search, transposition table
│   ├── eval/          # NNUE (CPU + GPU), classical eval
│   └── uci/           # UCI protocol handler
├── shaders/
│   ├── nnue_stockfish.metal  # Full NNUE GPU implementation
│   └── nnue.metal            # Optimized NNUE kernels
├── tests/
│   ├── test_*.cpp     # C++ unit tests
│   └── testing.py     # Python integration tests
└── external/
    └── metal-cpp/     # Apple's Metal C++ bindings
```

## GPU Optimization

### NNUE on Metal

The NNUE neural network is implemented as Metal compute shaders:

1. **Feature Transformer**: Sparse input → 1024-dim accumulator
2. **FC Layers**: ClippedReLU activations, quantized weights
3. **Batch Processing**: Up to 256 positions per GPU dispatch

### Unified Memory Advantage

```cpp
// Zero-copy buffer creation on Apple Silicon
MTL::Buffer* buffer = device->newBuffer(
    data_ptr, 
    size, 
    MTL::ResourceStorageModeShared
);
// CPU and GPU access the same physical memory!
```

This eliminates the copy overhead that limits GPU usage in traditional discrete GPU architectures.

### Batching Strategy

```
CPU Search Thread:
  └─> Collect leaf positions
      └─> When batch full (64-256 positions)
          └─> Dispatch GPU NNUE evaluation
              └─> Wait for GPU
                  └─> Continue search with results
```

## Performance

| Metric | Value |
|--------|-------|
| Perft(6) | 119,060,324 nodes |
| Perft Speed | ~10M nodes/sec |
| Search | ~500K nodes/sec (depth 20) |
| NNUE Batch | ~100K evals/sec (GPU) |

*Benchmarked on Apple M2 Max*

## UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 16 | Transposition table size (MB) |
| Threads | spin | 1 | Number of search threads |
| NNUE_Path | string | "" | Path to NNUE weights file |
| MultiPV | spin | 1 | Number of principal variations |

## License

MetalFish is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Based on:
- [Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0) - Search algorithms, NNUE architecture
- [MLX](https://github.com/ml-explore/mlx) (MIT) - Metal compute patterns and optimizations

## Author

**Nripesh Niketan** (2025)

## Acknowledgments

- The Stockfish team for the world's strongest open-source chess engine
- Apple for the MLX framework and Metal-cpp bindings
- The computer chess community for decades of research and the NNUE revolution
