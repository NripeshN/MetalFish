# Implementing Lc0-Compatible Neural Network Inference

## Overview

This document provides a roadmap for completing the Lc0-compatible neural network inference implementation in MetalFish. The infrastructure is in place; this guide details the remaining work.

## Current State

### ✅ Completed Infrastructure
- Directory structure created (`src/nn/`, `proto/`)
- Protobuf schema adapted (`proto/net.proto` with `pbmetalfish` namespace)
- Protobuf C++ code generated (`proto/net.pb.h`, `proto/net.pb.cc`)
- Stub NN evaluator interface (`src/mcts/nn_mcts_evaluator.h/cpp`)
- Test framework with 15 benchmark positions (`tests/test_nn_comparison.cpp`)
- Build system integration (CMakeLists.txt updated, all tests pass)
- Reference lc0 repository cloned (`reference/lc0/`)

### ❌ Remaining Work (~95k lines)

## Implementation Phases

### Phase 1: Position Encoding (Week 1)
**Goal**: Encode chess positions into 112-plane format compatible with lc0

#### 1.1 Copy Chess Representation (~1000 lines)
```bash
# From reference/lc0/src/chess/
cp board.h board.cc src/nn/
cp bitboard.h src/nn/
```

**Adapt for MetalFish:**
- Replace `lczero::` with `MetalFish::NN::`
- Adapt to use MetalFish's `Position` class where possible
- Create adapter functions between MetalFish and lc0 representations

#### 1.2 Complete Position Encoder (~650 lines)
File: `src/nn/encoder.cpp` (currently copied but not adapted)

**Key functions to implement:**
```cpp
InputPlanes EncodePositionForNN(
    pbmetalfish::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, 
    int history_planes,
    FillEmptyHistory fill_empty_history, 
    int* transform_out);
```

**Encoding format:**
- 8 history positions × 13 planes = 104 planes
  - Per board: our pieces (6 types) + opponent pieces (6 types) + repetitions (1)
- 8 auxiliary planes:
  - Color (all 1s or 0s)
  - Total move count
  - Castling rights (4 planes: K, Q, k, q)
  - No-progress count
  - (Additional planes for variants)

**Dependencies:**
- `PositionHistory` class (track 8 previous positions)
- Board transformation functions (flip, mirror, transpose)
- Bitboard manipulation utilities

**Test strategy:**
```cpp
// Verify encoding matches lc0 exactly
Position pos;
pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);
auto planes = EncodePositionForNN(/* ... */);
// Compare with lc0 reference output
```

### Phase 2: Policy Mapping (~2000 lines)

#### 2.1 Copy Policy Tables
```bash
mkdir -p src/nn/tables
cp reference/lc0/src/neural/tables/policy_map.h src/nn/tables/
cp reference/lc0/src/neural/tables/attention_policy_map.h src/nn/tables/
```

**Files to adapt:**
- `policy_map.h`: Maps 1858 policy indices to chess moves
- `attention_policy_map.h`: For transformer networks (BT4)

**Key data structures:**
```cpp
// 1858-element lookup table
extern const uint16_t kConvPolicyMap[];

// Move encoding/decoding
uint16_t MoveToNNIndex(Move move, int transform);
Move MoveFromNNIndex(int idx, int transform);
```

#### 2.2 Create Policy Tables (~2000 lines generated data)
The policy tables are large arrays of indices. Most of this is generated data.

**Test strategy:**
```cpp
// Verify policy mapping
Move e2e4 = make_move(SQ_E2, SQ_E4);
uint16_t idx = MoveToNNIndex(e2e4, 0);
Move decoded = MoveFromNNIndex(idx, 0);
assert(e2e4 == decoded);
```

### Phase 3: Weight Loading (~500 lines)

#### 3.1 Complete loader.cpp
File: `src/nn/loader.cpp` (currently copied but not adapted)

**Key functions:**
```cpp
std::unique_ptr<WeightsFile> LoadWeightsFromFile(const std::string& path);
```

**Implementation steps:**
1. Detect file format (.pb or .pb.gz)
2. Decompress if needed (gzip support)
3. Parse protobuf using `proto/net.pb.h`
4. Validate network format
5. Extract weights into usable C++ structures

**Dependencies:**
- zlib for .gz decompression
- Protobuf library (already linked)

**Test strategy:**
```cpp
auto weights = LoadWeightsFromFile("networks/BT4-1024x15x32h-swa-6147500.pb");
assert(weights->format().network_format().network() == NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT);
// Verify transformer parameters
assert(weights->format().network_format().transformer_depth() == 15);
```

### Phase 4: Metal Backend (~86,000 lines!!!)

This is the largest component by far.

#### 4.1 Copy Metal Backend
```bash
mkdir -p src/nn/metal
cp reference/lc0/src/neural/backends/metal/*.h src/nn/metal/
cp reference/lc0/src/neural/backends/metal/*.cc src/nn/metal/
cp reference/lc0/src/neural/backends/metal/*.mm src/nn/metal/
mkdir -p src/nn/metal/mps
cp reference/lc0/src/neural/backends/metal/mps/*.h src/nn/metal/mps/
cp reference/lc0/src/neural/backends/metal/mps/*.mm src/nn/metal/mps/
```

**Key files:**
- `network_metal.h/cc` (~500 lines): Main Metal backend
- `metal_common.h` (~100 lines): Shared Metal utilities
- `mps/NetworkGraph.mm` (~86,000 lines!!!): MPSGraph implementation
- `mps/MetalNetworkBuilder.h/mm` (~2,000 lines): Graph builder

#### 4.2 Adapt Metal Backend
**Massive adaptation effort required:**
1. Replace all `lczero::` with `MetalFish::NN::`
2. Update include paths
3. Integrate with MetalFish's Metal infrastructure
4. Connect to encoder and decoder

**Key interfaces:**
```cpp
class MetalBackend {
    void AddInputs(const InputPlanes& planes);
    void ComputeBlocking();
    std::vector<float> GetPolicyOutput();  // 1858 values
    float GetValueOutput();
    // For WDL heads:
    std::array<float, 3> GetWDLOutput();
};
```

**Optimization opportunities:**
- Leverage unified memory (zero-copy CPU/GPU)
- Use Metal Performance Shaders Graph for transformers
- Batch inference for MCTS parallelism

**Test strategy:**
```cpp
// Load network
auto weights = LoadWeightsFromFile("networks/BT4-1024x15x32h-swa-6147500.pb");
MetalBackend backend(weights);

// Encode position
Position pos;
pos.set("startpos", false, &si);
auto planes = EncodePositionForNN(/* ... */);

// Run inference
backend.AddInputs(planes);
backend.ComputeBlocking();

// Check outputs
auto policy = backend.GetPolicyOutput();  // Should be 1858 values
auto value = backend.GetValueOutput();     // Should be in [-1, 1]
```

### Phase 5: MCTS Integration (~1000 lines)

#### 5.1 Implement Lc0NNEvaluator
Update `src/mcts/nn_mcts_evaluator.cpp` from stub to real implementation.

**Key changes:**
```cpp
std::unique_ptr<Lc0NNEvaluator> Lc0NNEvaluator::create(const std::string& weights_path) {
    auto evaluator = std::unique_ptr<Lc0NNEvaluator>(new Lc0NNEvaluator());
    
    // Load weights
    evaluator->weights_ = LoadWeightsFromFile(weights_path);
    
    // Create Metal backend
    evaluator->backend_ = std::make_unique<MetalBackend>(evaluator->weights_);
    
    // Create encoder/decoder
    evaluator->encoder_ = std::make_unique<PositionEncoder>();
    evaluator->decoder_ = std::make_unique<PolicyDecoder>();
    
    evaluator->ready_ = true;
    return evaluator;
}

NNEvaluation Lc0NNEvaluator::evaluate(const Position& pos) {
    // Encode position to 112 planes
    auto planes = encoder_->Encode(pos);
    
    // Run inference
    backend_->AddInputs(planes);
    backend_->ComputeBlocking();
    
    // Get outputs
    auto policy_logits = backend_->GetPolicyOutput();  // 1858 values
    auto value = backend_->GetValueOutput();
    
    // Decode policy to legal moves
    NNEvaluation result;
    result.value = value;
    
    MoveList<LEGAL> moves(pos);
    for (const auto& m : moves) {
        int idx = decoder_->MoveToIndex(m, transform);
        float prob = softmax(policy_logits[idx]);
        result.policy.emplace_back(static_cast<Move>(m), prob);
    }
    
    return result;
}
```

#### 5.2 Update ThreadSafeMCTS
File: `src/mcts/thread_safe_mcts.cpp`

**Add NN evaluation option:**
```cpp
class ThreadSafeMCTS {
    // Add NN evaluator member
    std::unique_ptr<NN::NNEvaluator> nn_eval_;
    
    // Use in leaf evaluation
    float EvaluateLeaf(ThreadSafeNode* node, Position& pos) {
        if (nn_eval_ && nn_eval_->is_ready()) {
            auto eval = nn_eval_->evaluate(pos);
            ApplyPolicyToEdges(node, eval.policy);
            return eval.value;
        } else {
            // Fallback to NNUE
            return EvaluateWithNNUE(pos);
        }
    }
};
```

### Phase 6: Verification (~500 lines)

#### 6.1 Complete test_nn_comparison.cpp
Update from stub to real verification:

```cpp
bool test_nn_comparison() {
    // Load network
    auto evaluator = NN::Lc0NNEvaluator::create("networks/BT4-1024x15x32h-swa-6147500.pb");
    assert(evaluator->is_ready());
    
    int matches = 0;
    for (size_t i = 0; i < kBenchmarkPositions.size(); ++i) {
        Position pos;
        StateInfo si;
        pos.set(kBenchmarkPositions[i], false, &si);
        
        // Evaluate with MetalFish
        auto eval = evaluator->evaluate(pos);
        
        // Find best move
        Move best_move = GetBestMove(eval.policy);
        std::string move_str = move_to_uci(best_move);
        
        // Compare with expected
        if (move_str == kExpectedMoves[i]) {
            std::cout << "✓ MATCH: " << move_str << "\n";
            matches++;
        } else {
            std::cout << "✗ MISMATCH: got " << move_str 
                      << ", expected " << kExpectedMoves[i] << "\n";
        }
    }
    
    // Require 100% match
    return matches == kBenchmarkPositions.size();
}
```

#### 6.2 Reference Comparison
**Generate reference outputs from lc0:**
```bash
# Build lc0 reference
cd reference/lc0
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Generate reference outputs
for pos in "${kBenchmarkPositions[@]}"; do
    echo "position fen $pos" | ./lc0 --weights=../../networks/BT4-1024x15x32h-swa-6147500.pb \
        --nodes=800 --verbose-move-stats > reference_output_$i.txt
done
```

**Parse reference outputs** and compare:
- Best move must match
- Policy logits should match within tolerance (1e-5)
- Value should match within tolerance (1e-5)

## File Checklist

### Must Implement
- [ ] `src/nn/encoder.cpp` - Position encoding (642 lines)
- [ ] `src/nn/loader.cpp` - Weight loading (200 lines)
- [ ] `src/nn/tables/policy_map.h` - Policy mapping (2000+ lines)
- [ ] `src/nn/metal/network_metal.cc` - Metal backend (500 lines)
- [ ] `src/nn/metal/mps/NetworkGraph.mm` - MPSGraph (86,000 lines!)
- [ ] `src/nn/metal/mps/MetalNetworkBuilder.mm` - Graph builder (2,000 lines)
- [ ] `src/mcts/nn_mcts_evaluator.cpp` - Full implementation (replace stub)
- [ ] `tests/test_nn_comparison.cpp` - Full verification (replace stub)

### Must Copy & Adapt from lc0
- [ ] `src/chess/board.h/cc` - Chess board representation
- [ ] `src/chess/bitboard.h` - Bitboard utilities
- [ ] `src/neural/decoder.h/cc` - Policy decoding

### Already Done ✅
- [x] `proto/net.proto` - Network format
- [x] `src/nn/README.md` - Documentation
- [x] `src/mcts/nn_mcts_evaluator.h` - Interface
- [x] Infrastructure (CMakeLists.txt, build system)

## Estimated Effort

| Phase | Lines of Code | Estimated Time |
|-------|--------------|----------------|
| Position Encoding | ~1,650 | 3-5 days |
| Policy Mapping | ~2,000 | 2-3 days |
| Weight Loading | ~500 | 1-2 days |
| Metal Backend | ~88,000 | 2-3 weeks |
| MCTS Integration | ~1,000 | 2-3 days |
| Testing & Verification | ~500 | 2-3 days |
| **Total** | **~93,650** | **3-5 weeks** |

## Critical Success Factors

1. **100% Match Requirement**: All 15 benchmark positions must return identical best moves to lc0
2. **Numerical Precision**: Policy logits and values must match within 1e-5
3. **Performance**: Must leverage Apple Silicon's unified memory effectively
4. **Maintainability**: Clean separation between encoding, inference, and MCTS layers

## References

- lc0 source: `/home/runner/work/MetalFish/MetalFish/reference/lc0`
- Network weights: `networks/BT4-1024x15x32h-swa-6147500.pb` (download separately)
- lc0 documentation: https://lczero.org/dev/wiki/
- Metal Performance Shaders Graph: https://developer.apple.com/documentation/metalperformanceshadersgraph

## Getting Help

If you get stuck:
1. Check lc0 implementation in `reference/lc0/`
2. Review lc0 encoder test: `reference/lc0/src/neural/encoder_test.cc`
3. Study Metal backend: `reference/lc0/src/neural/backends/metal/`
4. Ask in MetalFish discussions with specific questions

Good luck! This is a substantial but achievable project.
