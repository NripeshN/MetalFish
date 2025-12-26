# MetalFish

A GPU-accelerated UCI chess engine using Apple Metal, designed for Apple Silicon.

## Overview

MetalFish is a chess engine that leverages the unified memory architecture of Apple Silicon to accelerate chess computations using Metal GPU shaders. It is based on the algorithms and code from [Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0) with GPU optimization inspired by [MLX](https://github.com/ml-explore/mlx).

## Features

- **GPU-Accelerated NNUE Evaluation**: Neural network inference runs on the GPU for batched position evaluation
- **Unified Memory**: Zero-copy data sharing between CPU and GPU on Apple Silicon
- **Parallel Move Generation**: GPU kernels for generating and validating moves in bulk
- **Hybrid Search**: Alpha-beta search on CPU with batched GPU leaf evaluation
- **UCI Protocol**: Standard UCI interface for compatibility with chess GUIs

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1/M2/M3/M4) recommended
- CMake 3.20+
- Xcode Command Line Tools

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

## Usage

```bash
# Start in UCI mode
./metalfish

# Show device information
./metalfish --info

# Run benchmark
./metalfish --bench
```

## Architecture

```
metalfish/
├── include/           # Header files
│   ├── core/          # Chess types, bitboard, position
│   ├── metal/         # Metal device abstraction
│   ├── search/        # Search algorithms
│   └── eval/          # Evaluation functions
├── src/               # Source files
│   ├── core/          # Core chess logic
│   ├── metal/         # Metal backend (device, allocator, kernels)
│   ├── search/        # Search implementation
│   ├── eval/          # Evaluation (NNUE, classical)
│   └── uci/           # UCI protocol
├── shaders/           # Metal compute shaders
│   ├── nnue.metal     # NNUE neural network kernels
│   ├── eval.metal     # Static evaluation kernels
│   └── movegen.metal  # Move generation kernels
└── reference/         # Reference implementations (Stockfish, MLX)
```

## GPU Optimization Strategy

### Batched Evaluation
Instead of evaluating positions one at a time, MetalFish collects leaf nodes during search and evaluates them in batches on the GPU. This amortizes kernel launch overhead and maximizes GPU utilization.

### Unified Memory
Apple Silicon's unified memory architecture allows CPU and GPU to share memory without explicit copies. MetalFish uses `MTL::ResourceStorageModeShared` buffers for:
- Transposition table entries
- NNUE network weights
- Position data

### NNUE on GPU
The NNUE neural network layers are implemented as Metal compute shaders:
- Feature transformer with sparse input handling
- Fully connected layers with ClippedReLU activation
- Batched inference for multiple positions

## License

MetalFish is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Based on:
- Stockfish (GPL-3.0) - https://github.com/official-stockfish/Stockfish
- MLX (MIT) - https://github.com/ml-explore/mlx

## Author

Nripesh Niketan (2025)

## Acknowledgments

- The Stockfish team for the world's strongest open-source chess engine
- Apple for the MLX framework demonstrating Metal compute patterns
- The computer chess community for decades of research

