# Neural Network Infrastructure for MetalFish

This directory contains the neural network inference infrastructure for MetalFish's MCTS search, designed to be compatible with transformer-based networks (specifically BT4 architecture).

## Overview

This implementation provides:
1. **Position Encoding** - 112-plane input format compatible with training data
2. **Weight Loading** - Protobuf-based network weight loading (.pb/.pb.gz)
3. **Policy Mapping** - UCI move to policy index conversion (1858 outputs)
4. **MCTS Integration** - Bridge between neural network and MCTS search
5. **Network Backend** - Abstract interface for inference (stub implementation provided)

## Directory Structure

```
src/nn/
├── proto/
│   ├── net.proto          # Protobuf definition for network weights
│   ├── net.pb.h          # Generated protobuf header
│   └── net.pb.cc         # Generated protobuf implementation
├── encoder.h/cpp         # Position to 112-plane encoding (✓ Full implementation)
├── loader.h/cpp          # Load network weights from .pb files (✓ Complete)
├── policy_map.h/cpp      # Move to policy index mapping (✓ Full 1858 tables)
├── network.h/cpp         # Abstract network interface (✓ Complete)
└── metal/                # Metal backend (✓ Complete)
    ├── metal_network.h   # Metal network class
    ├── metal_network.mm  # Metal/MPSGraph implementation (~1010 LOC)
    └── README.md         # Metal backend documentation
```

## Current Status

### ✅ Fully Implemented
- Protobuf weight format parsing (all formats: FLOAT32/16, BFLOAT16, LINEAR16)
- Full 8-position history encoding with canonicalization transforms
- Complete 1858-element policy mapping tables
- Metal/MPSGraph transformer backend with full architecture
- MCTS evaluator integration
- Comprehensive test framework with 15 benchmark positions

### 🎯 Production Ready
- Position encoder with flip/mirror/transpose canonicalization
- Policy tables with O(1) bidirectional lookup
- Weight loader with gzip decompression
- Metal backend optimized for Apple Silicon unified memory
- Batch processing support for efficient inference

## Usage

### Basic Example

```cpp
#include "nn/network.h"
#include "nn/encoder.h"
#include "mcts/nn_mcts_evaluator.h"

// Set environment variable or provide path directly
// export METALFISH_NN_WEIGHTS=/path/to/network.pb

// Load network (auto-detects Metal backend on macOS)
auto network = NN::CreateNetwork("/path/to/network.pb", "auto");

// Encode position
Position pos;
StateInfo si;
pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);
NN::InputPlanes input = NN::EncodePositionForNN(
    pos, MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);

// Evaluate
NN::NetworkOutput output = network->Evaluate(input);
// output.policy contains 1858 move probabilities
// output.value contains position evaluation (-1 to 1)
// output.wdl contains [win, draw, loss] probabilities (if network supports it)
```

### MCTS Integration

```cpp
#include "mcts/nn_mcts_evaluator.h"

// Create evaluator
MCTS::NNMCTSEvaluator evaluator("/path/to/network.pb");

// Evaluate position
Position pos;
// ... initialize position ...
auto result = evaluator.Evaluate(pos);

// result.value: position evaluation
// result.policy_priors: map of Move → probability for all legal moves
// result.wdl: [win, draw, loss] probabilities
```

## Technical Details

### Input Format

The network expects 112 input planes (8×8×112):
- **Planes 0-103**: Position history (8 positions × 13 planes each)
  - 6 planes for our pieces (P, N, B, R, Q, K)
  - 6 planes for opponent pieces
  - 1 plane for repetition count
- **Planes 104-111**: Auxiliary planes
  - Castling rights (4 planes: us kingside, us queenside, them kingside, them queenside)
  - En passant or side-to-move (1 plane, format-dependent)
  - Rule50 counter (1 plane, normalized)
  - Move count or zero plane (1 plane)
  - All ones plane (1 plane, for edge detection)

### Canonicalization

The encoder supports canonicalization transforms to reduce the input space:
- **Flip**: Horizontal flip (if king on left half of board)
- **Mirror**: Vertical mirror (if no pawns and king on top half)
- **Transpose**: Diagonal transpose (for certain symmetric positions)

These transforms are applied when using canonical input formats:
- `INPUT_112_WITH_CANONICALIZATION`
- `INPUT_112_WITH_CANONICALIZATION_V2`
- Armageddon variants

### Policy Mapping

The 1858 policy outputs represent:
- **Queen-like moves**: All queen moves from each square (up to 56 per square)
- **Knight moves**: All 8 knight moves from each square
- **Underpromotions**: N/B/R promotions in 3 directions (forward, diagonal-left, diagonal-right)
- **Queen promotions**: Similar structure to underpromotions

Use `MoveToNNIndex()` and `IndexToNNMove()` for conversion.

### Metal Backend Architecture

The Metal implementation uses MPSGraph to build a transformer network:
1. **Input embedding**: 112×8×8 → embedding_size (typically 1024)
2. **Transformer encoder**: Configurable layers (typically 15) with:
   - Multi-head self-attention (typically 32 heads)
   - Feed-forward network (typically 4× expansion)
   - Layer normalization
   - Residual connections
3. **Output heads**:
   - Policy: embedding_size → 1858 (move probabilities)
   - Value: embedding_size → 1 (position evaluation)
   - WDL: embedding_size → 3 (win/draw/loss)
   - Moves-left: embedding_size → 1 (game length prediction)

The implementation is optimized for Apple Silicon:
- Unified memory (zero-copy between CPU/GPU)
- Pre-compiled MPSGraph executables
- Efficient batch processing
  - Color to move or en passant
  - Rule50 counter
  - Move count
  - Constant plane (all 1s)

### Policy Output

The network outputs 1858 move probabilities:
- Queen moves: 56 directions × 64 squares
- Knight moves: 8 directions × 64 squares
- Underpromotions: 9 types × 64 squares

### Network Format

Supports networks in protobuf format (.pb or .pb.gz):
- Transformer-based architectures (BT4)
- Attention-based policy and value heads
- WDL (Win/Draw/Loss) output support

## Building

The neural network infrastructure is automatically built as part of MetalFish:

```bash
mkdir build && cd build
cmake ..
make metalfish
make test_nn_comparison  # Build tests
```

### Dependencies

- **Protobuf** (>= 3.0): For weight file parsing
- **zlib**: For .gz decompression
- **Metal** (macOS only): For GPU inference

## Testing

Run the test suite:

```bash
./build/test_nn_comparison
```

This tests:
1. Position encoding to 112 planes
2. Network weight loading
3. MCTS evaluator integration

## TODO: Metal Backend Implementation

The Metal backend needs to implement:

1. **MPSGraph construction** for transformer architecture:
   - Embedding layer
   - Multi-head attention blocks
   - Feed-forward networks
   - Layer normalization
   - Policy and value heads

2. **Batching** for efficient inference:
   - Batch multiple positions
   - Optimize for unified memory
   - Pipeline with MCTS search

3. **Optimization**:
   - FP16 inference
   - Metal Performance Shaders Graph
   - Zero-copy unified memory access
   - Async compute

## References

- Network architecture: Based on transformer design
- Input format: Compatible with standard training pipeline
- Policy encoding: UCI move space (1858 moves)

## License

GPL-3.0 - See LICENSE file

## Copyright

Copyright (C) 2025 Nripesh Niketan
