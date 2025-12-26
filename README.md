# MetalFish

An experimental GPU-accelerated chess engine exploring Apple Metal on Apple Silicon.

[![CI](https://github.com/nripeshn/metalfish/actions/workflows/ci.yml/badge.svg)](https://github.com/nripeshn/metalfish/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## ⚠️ Important Disclaimer

**MetalFish is NOT a Stockfish clone.** It is an experimental research project exploring whether GPU acceleration can benefit chess engine search on Apple Silicon's unified memory architecture.

### Current Status: Experimental / Work in Progress

- **Search**: Basic alpha-beta with some pruning techniques. Missing many sophisticated features that make Stockfish strong (singular extensions, full history heuristics, proper LMR tuning, etc.)
- **NNUE**: Can load Stockfish network files, but does **full O(n) recomputation** - not "efficiently updatable" like CPU NNUE
- **Performance**: Currently **slower than Stockfish CPU** due to GPU dispatch overhead and lack of incremental updates
- **Strength**: Significantly weaker than Stockfish due to simplified search

### The GPU Challenge

As noted by experienced engine developers: _"Chess code is very sequential and branchy which is the worst case for GPUs"_

The fundamental challenge:

1. **Alpha-beta search is inherently sequential** - each node depends on previous results
2. **GPU dispatch overhead** (~10-50μs) exceeds the benefit for single position evaluation
3. **NNUE's "Efficiently Updatable" property** relies on incremental CPU updates that are hard to replicate on GPU
4. **Stockfish CPU NNUE** evaluates in ~1μs vs ~50-100μs for GPU dispatch

## What This Project Actually Is

An exploration of:

- Apple Metal GPU programming for chess
- Unified memory (zero-copy) benefits on Apple Silicon
- Whether batched GPU evaluation could work for specific use cases (training data generation, MCTS-style search)
- Learning exercise in chess programming

## Features (Implemented)

### Search (Basic)

- Alpha-Beta with Principal Variation Search
- Iterative Deepening with aspiration windows
- Transposition Table
- Basic Late Move Reductions (2 adjustment factors vs Stockfish's 15+)
- Basic Null Move Pruning
- Basic Futility Pruning
- Basic Razoring
- Quiescence Search

### Search (NOT Implemented - Major Missing Features)

- ❌ Singular Extensions
- ❌ Check Extensions
- ❌ History Extensions
- ❌ Multi-Cut
- ❌ ProbCut
- ❌ Proper continuation history (4-ply)
- ❌ Capture history
- ❌ Counter move history (used)
- ❌ Sophisticated LMR adjustments
- ❌ IIR (Internal Iterative Reductions)
- ❌ Lazy SMP
- ❌ Syzygy tablebase support

### Evaluation

- NNUE network loading (Stockfish .nnue format)
- GPU forward pass (Metal shaders)
- **No incremental updates** - full recomputation each position
- Classical material fallback

### Move Ordering (Partial)

- TT move first
- MVV-LVA for captures
- Basic SEE pruning
- Killer moves (structure exists, not fully utilized)
- History tables (structure exists, not fully utilized)

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
position startpos
go depth 10
quit
```

## Performance Comparison

| Metric              | MetalFish         | Stockfish 17 CPU       |
| ------------------- | ----------------- | ---------------------- |
| Search NPS          | ~100-500K         | ~2-5M                  |
| NNUE eval/sec       | ~100K (GPU batch) | ~2M+ (CPU incremental) |
| Single eval latency | ~50-100μs         | ~0.5-1μs               |
| Strength (Elo)      | Unknown (weak)    | ~3600+                 |

The GPU approach is fundamentally disadvantaged for traditional alpha-beta search.

## Architecture

```
metalfish/
├── src/
│   ├── core/          # Bitboards, position, move generation
│   ├── search/        # Basic alpha-beta search
│   ├── eval/          # NNUE loader, GPU evaluation
│   ├── metal/         # Metal device, allocator
│   └── uci/           # UCI protocol
├── shaders/           # Metal compute shaders
└── tests/             # Unit and integration tests
```

## Future Exploration

Areas that might actually benefit from GPU:

1. **Training data generation** - batch evaluate millions of positions
2. **MCTS-style search** - batch leaf evaluations
3. **Analysis mode** - deep analysis where latency matters less
4. **Neural network training** - obvious GPU benefit

## License

GPL-3.0 - Same as Stockfish

**Inspired by:**

- [Stockfish](https://github.com/official-stockfish/Stockfish) - Search concepts, NNUE architecture
- [MLX](https://github.com/ml-explore/mlx) - Metal programming patterns

**NNUE Training Data:**

- Networks compatible with this engine are trained on data from [Leela Chess Zero](https://lczero.org/) (ODbL license)

## Author

**Nripesh Niketan** (2025)

This is a learning/research project. Feedback and contributions welcome!
