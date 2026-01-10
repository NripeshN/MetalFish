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

### GPU Acceleration

MetalFish includes a comprehensive GPU acceleration framework with support for multiple backends:

**Supported Backends:**
- **Metal** - Apple Silicon (production ready)
- **ROCm/HIP** - AMD GPUs (in development)
- **CUDA** - NVIDIA GPUs (planned)

**Architecture:**
- Backend-agnostic GPU interface supporting multiple platforms
- Zero-copy CPU/GPU data sharing via unified memory (Metal, ROCm APUs)
- Runtime shader compilation for flexibility
- Batch processing for efficient GPU utilization

**GPU-Accelerated Operations:**
- NNUE batch evaluation infrastructure
- Batch SEE (Static Exchange Evaluation)
- Feature transformer kernels for sparse input
- Fused forward pass for neural network inference

**Performance (Apple M2 Max with Metal):**
- GPU compute bandwidth: up to 52.7 GB/s
- Unified memory write: 10.9 GB/s from CPU
- Unified memory read: 4.2 GB/s from CPU

## Project Structure

```
metalfish/
├── src/
│   ├── core/           # Bitboard, position, move generation
│   ├── search/         # Search algorithms, history tables, TT
│   ├── eval/           # NNUE evaluation
│   │   └── nnue/       # Neural network implementation
│   ├── uci/            # UCI protocol
│   ├── gpu/            # GPU acceleration framework
│   │   ├── backend.h   # Abstract GPU interface
│   │   ├── nnue_eval   # GPU NNUE evaluation
│   │   ├── batch_ops   # Batch GPU operations
│   │   ├── metal/      # Metal backend implementation
│   │   │   └── kernels/# Metal compute shaders
│   │   └── rocm/       # ROCm/HIP backend implementation
│   │       └── kernels/# HIP compute kernels
│   ├── metal/          # Legacy Metal device management
│   └── syzygy/         # Tablebase probing
├── external/           # External dependencies (metal-cpp)
├── tests/              # Test suite
└── reference/          # Reference implementations
```

## Building

### Requirements

**Common Requirements:**
- CMake 3.20 or later
- C++20 compatible compiler

**Platform-Specific:**
- **macOS**: macOS 12.0 or later, Xcode Command Line Tools, Apple Silicon (M1/M2/M3/M4) recommended
- **Linux (ROCm)**: ROCm 5.0 or later, AMD GPU with GCN 3.0+ architecture
- **Windows**: Visual Studio 2019 or later (CUDA/ROCm support planned)

### Build Instructions

**Metal (macOS):**
```bash
cd metalfish
cmake -B build -DUSE_METAL=ON
cmake --build build -j8
```

**ROCm/HIP (Linux):**
```bash
cd metalfish
cmake -B build -DUSE_ROCM=ON
cmake --build build -j8
```

**CPU Only:**
```bash
cd metalfish
cmake -B build -DUSE_METAL=OFF -DUSE_ROCM=OFF
cmake --build build -j8
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| USE_METAL | ON (macOS) | Enable Metal GPU acceleration |
| USE_ROCM | OFF | Enable ROCm/HIP GPU acceleration |
| USE_CUDA | OFF | Enable CUDA GPU acceleration (future) |
| BUILD_TESTS | ON | Build test suite |
| BUILD_GPU_BENCHMARK | OFF | Build GPU benchmark utility |

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
./build/metalfish_tests

# Python integration tests
python3 tests/testing.py

# GPU benchmark (if built)
./build/metalfish_gpu_bench
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
./build/metalfish
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
| Benchmark nodes | 2,477,446 |
| SIMD | NEON with dot product |
| GPU Bandwidth | 52.7 GB/s |

The engine produces identical search results to reference implementations, ensuring correctness of the search algorithm and evaluation function.

## GPU Development Roadmap

Current GPU acceleration status:

| Feature | Metal | ROCm | CUDA |
|---------|-------|------|------|
| GPU Backend Abstraction | ✓ Complete | ✓ Complete | Planned |
| Unified Memory Support | ✓ Complete | ✓ Complete | Planned |
| Runtime Shader Compilation | ✓ Complete | In Progress | Planned |
| Batch SEE Infrastructure | ✓ Complete | Planned | Planned |
| NNUE Batch Evaluation | In Progress | Planned | Planned |
| Search Integration | Planned | Planned | Planned |

## Testing

The test suite includes:

- Bitboard operations
- Position management
- Move generation (perft verified)
- Search components
- Metal GPU functionality
- GPU shader compilation and execution
- UCI protocol compliance

All 30 standard perft positions pass with correct node counts.

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

## Author

Nripesh Niketan
