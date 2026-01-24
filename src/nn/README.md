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
├── encoder.h/cpp         # Position to 112-plane encoding
├── loader.h/cpp          # Load network weights from .pb files
├── policy_map.h/cpp      # Move to policy index mapping
├── network.h/cpp         # Abstract network interface
└── metal/                # Metal backend (TODO)
    └── metal_network.mm  # Metal/MPSGraph implementation (TODO)
```

## Current Status

### ✅ Implemented
- Protobuf weight format parsing
- 112-plane position encoding
- Basic policy mapping infrastructure
- MCTS evaluator integration points
- Test framework

### ⚠️ Partial Implementation
- Position encoder (simplified, no canonicalization transforms)
- Policy tables (simplified mapping, not full 1858-move table)
- Weight loader (basic decompression, needs validation)

### ❌ Not Implemented
- Metal backend for transformer inference
- Full policy mapping tables
- Canonicalization transforms
- Batch optimization
- Network weight validation
- Performance benchmarking

## Usage

### Basic Example

```cpp
#include "nn/network.h"
#include "nn/encoder.h"
#include "mcts/nn_mcts_evaluator.h"

// Load network
auto network = NN::CreateNetwork("path/to/network.pb.gz");

// Encode position
Position pos;
StateInfo si;
pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);
NN::InputPlanes input = NN::EncodePositionForNN(pos);

// Evaluate
NN::NetworkOutput output = network->Evaluate(input);
// output.policy contains 1858 move probabilities
// output.value contains position evaluation (-1 to 1)
```

### MCTS Integration

```cpp
#include "mcts/nn_mcts_evaluator.h"

// Create evaluator
MCTS::NNMCTSEvaluator evaluator("path/to/network.pb.gz");

// Evaluate position
Position pos;
// ... initialize position ...
MCTS::NNEvaluation result = evaluator.Evaluate(pos);
```

## Technical Details

### Input Format

The network expects 112 input planes (8×8×112):
- **Planes 0-103**: Position history (8 positions × 13 planes each)
  - 6 planes for our pieces (P, N, B, R, Q, K)
  - 6 planes for opponent pieces
  - 1 plane for repetition count
- **Planes 104-111**: Auxiliary planes
  - Castling rights (4 planes)
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
