# Neural Network Inference Module

This module implements Lc0-compatible neural network inference for MetalFish's MCTS search.

## Architecture

```
src/nn/
├── network.h           - Neural network interface (from lc0)
├── encoder.h/cpp       - Position to 112-plane encoding (from lc0)
├── loader.h/cpp        - Protobuf weight loading (from lc0)
├── policy_tables.h/cpp - Policy index mapping (from lc0)
├── metal/              - Metal GPU backend
│   ├── metal_backend.h/mm
│   └── network_graph.h/mm
└── mcts_evaluator.h/cpp - MCTS integration

proto/
└── net.proto           - Network weight format (from lc0, adapted)
```

## Implementation Status

### ✅ Completed
- [x] Project structure created
- [x] Protobuf schema copied and adapted (pbmetalfish namespace)
- [x] Protobuf C++ code generated
- [x] Initial files copied: network.h, encoder.h/cpp, loader.h/cpp

### 🚧 In Progress
- [ ] Fix namespace issues in copied files
- [ ] Adapt encoder.cpp for MetalFish Position class
- [ ] Copy and adapt policy mapping tables
- [ ] Implement Metal backend

### ❌ Not Started
- [ ] Position history tracking for MCTS
- [ ] Complete Metal MPSGraph implementation (~86KB NetworkGraph.mm)
- [ ] MCTS evaluator integration
- [ ] Verification tests with 15 benchmark positions
- [ ] Network weight loading and decompression

## Key Requirements

From lc0 reference implementation:

### Input Format
- **112 planes total**:
  - 8 move history × 13 planes/board = 104 planes
  - 8 auxiliary planes (castling rights, en passant, etc.)
- Each plane: 8×8 board representation

### Policy Output
- **1858 possible moves** in standard chess
- Attention-based policy mapping for transformers
- Move index to UCI move conversion

### Network Formats Supported
- Big Transformer 4 (BT4): 1024 embedding, 15 layers, 32 attention heads
- Protobuf format (.pb, .pb.gz)

## Dependencies

- **Protobuf**: For weight file parsing
- **Metal Performance Shaders Graph** (macOS): GPU inference
- **MetalFish core**: Position, Move, Bitboard classes

## Scope Estimate

This is a substantial implementation requiring:
- ~5,000 lines: Chess position/board representation compatible with lc0 encoding
- ~650 lines: Position encoder with transformations
- ~500 lines: Weight loader with decompression
- ~2,000 lines: Policy mapping tables (generated data)
- ~86,000 lines: Metal backend with MPSGraph (NetworkGraph.mm alone)
- ~1,000 lines: MCTS integration
- ~500 lines: Testing and verification

**Total: ~95,000+ lines of code** to copy, adapt, and integrate from lc0.

## Next Steps

1. **Create adapter layer** between MetalFish::Position and lc0 encoder requirements
2. **Copy policy tables** from lc0 (attention_policy_map.h, policy_map.h)
3. **Implement Metal backend** by adapting lc0's Metal implementation
4. **Add position history** tracking for 8-ply encoding
5. **Create MCTS evaluator** that replaces NNUE evaluation
6. **Verify with benchmarks** to achieve 100% match with lc0 reference

## References

- lc0 source: /home/runner/work/MetalFish/MetalFish/reference/lc0
- Network weights: networks/BT4-1024x15x32h-swa-6147500.pb
- Documentation: https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/
