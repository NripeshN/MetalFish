# Lc0-Compatible Neural Network Implementation

## Overview

This document describes the Lc0-compatible neural network inference implementation for MetalFish's MCTS search. The implementation aims to produce identical results to Lc0 for the same positions, optimized for Apple Silicon's unified memory architecture.

## Implementation Status

### ✅ Completed Components

#### 1. Neural Network Infrastructure (`src/nn/`)

**Position Encoder** (`encoder.h`, `encoder.cpp`)
- ✅ 112-plane Lc0-compatible input format
- ✅ 8 history positions × 13 planes (pieces + repetitions)
- ✅ 8 auxiliary planes (side to move, rule50, castling, etc.)
- ✅ Board flipping for black-to-move positions
- ✅ Validation tests passing

**Policy Tables** (`policy_tables.h`, `policy_tables.cpp`)
- ✅ UCI move ↔ NN policy index mapping interface
- ✅ 1858 policy output structure
- ✅ Queen-like move encoding
- ⚠️ Knight move encoding (needs refinement)
- ⚠️ Underpromotion encoding (simplified)

**Weight Loader** (`loader.h`, `loader.cpp`)
- ✅ Network weight data structures defined
- ✅ BT4 transformer configuration (1024x15x32h)
- ⚠️ Protobuf parsing (placeholder - requires libprotobuf)
- ⚠️ Gzip decompression (placeholder - requires zlib)

**MCTS Evaluator** (`nn_mcts_evaluator.h`, `nn_mcts_evaluator.cpp`)
- ✅ Interface between NN and MCTS defined
- ✅ Transposition table for caching evaluations
- ✅ WDL (Win-Draw-Loss) output handling
- ⚠️ Inference backend (placeholder - requires Metal implementation)

#### 2. Test Suite (`tests/test_lc0_comparison.cpp`)

- ✅ Test framework with 15 benchmark positions
- ✅ Position encoding validation tests
- ✅ Policy table validation tests
- ✅ Network loading tests (skipped without network file)
- ✅ Lc0 comparison framework (awaiting inference implementation)

#### 3. Build System

- ✅ CMakeLists.txt updated with NN module
- ✅ All code compiles successfully
- ✅ Tests integrated into test suite
- ✅ 8/9 test suites passing

### 🚧 Pending Implementation

#### Critical Components Requiring Significant Development

**1. Protobuf Integration**
```bash
# Required: Add protobuf dependency to CMakeLists.txt
find_package(Protobuf REQUIRED)

# Need to:
- Define lc0.proto schema (from Lc0 repository)
- Generate protobuf code (protoc)
- Implement weight extraction in loader.cpp
- Add zlib for .pb.gz decompression
```

**2. Metal Transformer Backend** (`src/nn/metal/`)

This is the largest remaining component, requiring:

- **MPSGraph Implementation**
  - Multi-head self-attention layers
  - Layer normalization
  - Feed-forward networks (MLP with GELU activation)
  - Embedding layer
  - Residual connections

- **Unified Memory Optimization**
  - Zero-copy buffer sharing between CPU/GPU
  - MTLResourceStorageModeShared for weights
  - 128-byte cache line alignment for M-series chips
  - Async command queue management

- **Inference Pipeline**
  - Batch processing support
  - Dynamic batch size optimization
  - Policy head (1858 outputs + softmax)
  - Value head (3 outputs for WDL)
  - Moves-left head (single output)

**3. Policy Table Refinement**

Current implementation uses simplified knight move indexing. Need to:
- Match Lc0's exact policy encoding for knight moves
- Verify underpromotion indices
- Add comprehensive validation tests

**4. MCTS Integration**

Connect NN evaluator to existing MCTS in `thread_safe_mcts.cpp`:
- Replace NNUE evaluation with NN evaluation for leaf nodes
- Apply NN policy to edge selection
- Use NN Q-value for backpropagation
- Maintain Lc0's Q-value sign conventions

## Usage (When Complete)

### Building with Neural Network Support

```bash
cd metalfish
mkdir build && cd build

# Configure (will require protobuf)
cmake .. -DUSE_METAL=ON

# Build
cmake --build . --target metalfish

# Run tests
./metalfish_tests
```

### Network File Setup

Download the BT4 network:
```bash
mkdir networks
cd networks
wget https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz
gunzip BT4-1024x15x32h-swa-6147500.pb.gz
```

### Running Lc0 Comparison Tests

```bash
cd build
./metalfish_tests lc0_comparison
```

Expected output (when complete):
```
=== Lc0 Comparison Tests ===
Position 1/15: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
  Lc0 best move: e2e4
  MetalFish best move: e2e4
  ✓ MATCH

...

Results: 15/15 positions match (100%)
```

## Architecture

### Data Flow

```
Position → Encoder → 112-plane input
                          ↓
                    Metal Backend
                    (Transformer)
                          ↓
             Policy (1858) + Value (WDL) + MLH
                          ↓
                   MCTS Evaluator
                          ↓
              ThreadSafeMCTS Search
                          ↓
                     Best Move
```

### Key Design Decisions

1. **Lc0 Compatibility**
   - Exact input encoding (112 planes)
   - Identical policy indexing (1858 outputs)
   - WDL value representation
   - Q-value sign conventions

2. **Apple Silicon Optimization**
   - Unified memory for zero-copy access
   - 128-byte cache line alignment
   - Metal Performance Shaders Graph (MPSGraph)
   - Async GPU command submission

3. **Modularity**
   - Clean separation between NN and MCTS
   - Pluggable inference backend
   - Testable components

## Testing

### Current Test Results

```
Position Encoding: ✅ PASSING
  - 112-plane format correct
  - Board flipping working
  - Auxiliary planes correct

Policy Tables: ⚠️ PARTIAL
  - Queen-like moves: Working
  - Knight moves: Needs refinement
  - Promotions: Simplified

Network Loading: ⏭️ SKIPPED
  - Awaiting protobuf integration
  
Lc0 Comparison: ⏭️ SKIPPED
  - Framework ready
  - Awaiting inference implementation
```

### Benchmark Positions

The test suite uses 15 diverse positions:
- Starting position
- Kiwipete (tactical complexity)
- Endgames (pawn, rook, queen)
- Complex middlegames
- Tactical positions

## Implementation Roadmap

### Phase 1: Dependencies ⏱️ ~1-2 days
- [ ] Add protobuf to CMakeLists.txt
- [ ] Add zlib for compression
- [ ] Define lc0.proto schema
- [ ] Generate protobuf code

### Phase 2: Weight Loading ⏱️ ~2-3 days
- [ ] Implement protobuf parsing
- [ ] Extract transformer weights
- [ ] Extract policy/value/MLH heads
- [ ] Add weight validation

### Phase 3: Metal Backend ⏱️ ~2-3 weeks
- [ ] Implement attention mechanism
- [ ] Implement layer normalization  
- [ ] Implement MLP layers
- [ ] Integrate with MPSGraph
- [ ] Optimize for unified memory
- [ ] Add batch processing

### Phase 4: Integration ⏱️ ~3-5 days
- [ ] Connect NN to MCTS
- [ ] Verify Q-value conventions
- [ ] Policy application to edges
- [ ] Performance tuning

### Phase 5: Validation ⏱️ ~2-3 days
- [ ] Fix policy table issues
- [ ] Run all benchmark positions
- [ ] Verify 100% match with Lc0
- [ ] Performance benchmarks

**Total Estimated Time: 4-5 weeks for full implementation**

## Dependencies

### Required
- CMake 3.20+
- C++20 compiler
- Metal (macOS) or CUDA (Linux/Windows)
- Protobuf 3.x
- zlib

### Optional
- Lc0 for comparison testing
- Network weights (365MB)

## Performance Targets

On Apple Silicon M2/M3:
- Batch inference: <10ms for 256 positions
- Single position: <1ms latency
- MCTS nodes/second: >10,000 NPS
- Memory usage: <1GB (including weights)

## References

- Lc0 Source: https://github.com/LCZero/lc0
- BT4 Network: https://lczero.org/dev/wiki/
- Transformer Paper: "Attention is All You Need"
- MetalFish MCTS: `src/mcts/thread_safe_mcts.cpp`

## Contributing

When implementing remaining components:

1. **Maintain Lc0 Compatibility**
   - Match exact output format
   - Preserve numerical precision
   - Use identical conventions

2. **Test Thoroughly**
   - Add unit tests for new components
   - Verify against Lc0 output
   - Benchmark performance

3. **Document Changes**
   - Update this README
   - Add inline documentation
   - Explain design decisions

## License

GPL-3.0 (same as MetalFish and Lc0)
