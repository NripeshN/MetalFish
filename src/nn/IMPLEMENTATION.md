# Lc0 Neural Network Quick Reference

## File Structure

```
src/nn/
├── README.md                    # This file
├── loader.h/cpp                # Network weight loading (protobuf)
├── encoder.h/cpp               # Position → 112-plane encoding
├── policy_tables.h/cpp         # UCI move ↔ Policy index mapping
├── nn_mcts_evaluator.h/cpp     # NN ↔ MCTS bridge
└── metal/                      # Metal inference backend (TODO)
    ├── transformer.h/cpp       # Transformer implementation
    ├── attention.h/cpp         # Multi-head attention
    ├── mlp.h/cpp              # Feed-forward network
    └── metal_buffers.h/cpp    # Unified memory management
```

## Key Interfaces

### Position Encoding

```cpp
#include "src/nn/encoder.h"

using namespace MetalFish::NN;

// Encode a position
Lc0PositionEncoder encoder;
EncodedPosition encoded;
encoder.encode(position, encoded);

// Access planes
float* plane_data = encoded.get_plane(plane_idx); // 0-111
```

### Policy Mapping

```cpp
#include "src/nn/policy_tables.h"

// Initialize once at startup
PolicyTables::initialize();

// Convert move to policy index
int idx = move_to_policy_index(move, position);

// Get all legal moves with indices
auto moves = get_legal_moves_with_indices(position);
```

### NN Evaluation (when complete)

```cpp
#include "src/nn/nn_mcts_evaluator.h"

// Create evaluator
auto nn = std::make_shared<Lc0NNEvaluator>();
nn->load_network("networks/BT4-1024x15x32h-swa-6147500.pb");

// Evaluate position
NNEvaluation result;
nn->evaluate(position, result);

// Access results
float q = result.q_value;              // Q from side-to-move perspective
float w = result.win_prob;             // Win probability
float d = result.draw_prob;            // Draw probability  
float l = result.loss_prob;            // Loss probability
float m = result.moves_left;           // Moves left estimate

// Policy distribution over legal moves
for (auto& [move, prob] : result.policy) {
  // move: UCI move
  // prob: Policy probability (post-softmax)
}
```

### MCTS Integration (when complete)

```cpp
#include "src/nn/nn_mcts_evaluator.h"

// Create MCTS evaluator
auto evaluator = std::make_shared<NNMCTSEvaluator>(nn);

// Evaluate for MCTS
std::vector<std::pair<Move, float>> policy;
float draw_prob, moves_left;
float q = evaluator->evaluate_for_mcts(position, policy, draw_prob, moves_left);

// Apply to MCTS node
for (auto& [move, p] : policy) {
  // Add edge to MCTS node with policy p
}
```

## Lc0 Input Encoding (112 planes)

### History Planes (0-103): 8 positions × 13 planes

For each of 8 history positions (current, -1 ply, -2 ply, ..., -7 ply):

**Planes per position (13):**
- 0-5: White pieces (P, N, B, R, Q, K)
- 6-11: Black pieces (P, N, B, R, Q, K)
- 12: Repetition count (0.0 = none, 0.5 = once, 1.0 = twice+)

**Encoding:**
- Each plane is 64 squares (8×8 board)
- 1.0 = piece present, 0.0 = absent
- Board from side-to-move perspective (flipped for black)

### Auxiliary Planes (104-111): 8 planes

- **104**: Side to move (1.0 = white, 0.0 = black)
- **105**: Rule50 counter / 99 (normalized)
- **106**: Zeroed plane (legacy)
- **107**: All 1.0s plane
- **108**: Kingside castling rights (us)
- **109**: Queenside castling rights (us)
- **110**: Kingside castling rights (them)
- **111**: Queenside castling rights (them)

## Policy Output Encoding (1858 indices)

### Queen-like moves (3584 indices)
- 64 squares × 56 queen moves
- 56 = 8 directions × 7 distances
- Directions: N, NE, E, SE, S, SW, W, NW
- Distances: 1-7 squares

**Formula:** `from_sq * 56 + direction * 7 + (distance - 1)`

### Knight moves (512 indices)
- 64 squares × 8 knight moves
- 8 L-shaped moves from each square

**Formula:** `3584 + from_sq * 8 + direction`

### Underpromotions (762 indices)
- 3 piece types (N, B, R) × 2 directions × 127 squares
- Queen promotions counted as queen moves

**Formula:** `4096 + piece_type * 254 + from_sq * 2 + direction`

## Network Architecture (BT4)

```
Input: 112 planes × 64 squares = 7168 values

↓ Input Embedding
  Weights: [7168, 1024]
  Output: [1024] embedding

↓ Transformer Encoder (15 layers)
  Each layer:
    ↓ Multi-Head Attention (32 heads, 32 dim each)
      Q, K, V: [1024, 1024]
      Output: [1024, 1024]
    ↓ Layer Norm 1
    ↓ Residual Connection
    ↓ MLP (Feed-Forward)
      FC1: [1024, 2816] + GELU
      FC2: [2816, 1024]
    ↓ Layer Norm 2
    ↓ Residual Connection

↓ Policy Head
  Weights: [1024, 1858]
  Output: [1858] logits → softmax → probabilities

↓ Value Head
  Weights: [1024, 3]
  Output: [3] for W/D/L → softmax → probabilities

↓ Moves Left Head
  Weights: [1024, 1]
  Output: [1] moves left estimate
```

## Metal Implementation Notes

### Unified Memory Strategy

```cpp
// Allocate shared memory (CPU/GPU zero-copy)
MTLResourceOptions options = MTLResourceStorageModeShared;
id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];

// Direct CPU access (no memcpy needed)
float* cpu_ptr = (float*)buffer.contents;
// ... write data ...

// GPU can read same data immediately (no copy)
[encoder setBuffer:buffer offset:0 atIndex:0];
```

### MPSGraph Example

```cpp
// Create graph for transformer layer
MPSGraph* graph = [[MPSGraph alloc] init];

// Input tensor
MPSGraphTensor* input = [graph placeholderWithShape:@[@1024] name:@"input"];

// Matrix multiply: Q = input @ W_q
MPSGraphTensor* q = [graph matrixMultiplicationWithPrimaryTensor:input
                                              secondaryTensor:W_q
                                                         name:@"query"];

// Multi-head attention
// ... (split heads, scale, softmax, etc.)

// Feed through MPSGraph
MPSGraphExecutable* executable = [graph compileWithDevice:device 
                                                  feeds:inputs
                                          targetTensors:outputs];
```

## Testing

### Run Lc0 Comparison Tests

```bash
cd build
./metalfish_tests

# Output shows:
# - Position Encoding: PASS
# - Policy Tables: PARTIAL (knight moves need fix)
# - Network Loading: SKIP (needs protobuf)
# - Lc0 Comparison: SKIP (needs inference)
```

### Add New Tests

```cpp
// In tests/test_lc0_comparison.cpp

bool test_my_feature() {
  std::cout << "Testing my feature...\n";
  
  // ... test code ...
  
  return success;
}

// Add to run_all_lc0_tests()
Test tests[] = {
  // ... existing tests ...
  {"My Feature", test_my_feature},
};
```

## Common Patterns

### Batch Processing

```cpp
// Encode multiple positions
std::vector<EncodedPosition> batch(batch_size);
for (size_t i = 0; i < batch_size; ++i) {
  encoder.encode(positions[i], batch[i]);
}

// Batch inference
std::vector<NNEvaluation> results;
nn->evaluate_batch(positions, results);
```

### Caching

```cpp
// MCTS evaluator has built-in cache
auto evaluator = std::make_shared<NNMCTSEvaluator>(nn);

// Automatic caching by position key
float q = evaluator->evaluate_for_mcts(pos, ...);

// Clear cache between searches
evaluator->clear_cache();
evaluator->new_search(); // Increment age
```

## Debugging Tips

1. **Position Encoding**
   ```cpp
   // Verify plane sums
   for (int p = 0; p < 112; ++p) {
     float sum = 0;
     for (int sq = 0; sq < 64; ++sq) {
       sum += encoded.get_plane(p)[sq];
     }
     std::cout << "Plane " << p << " sum: " << sum << "\n";
   }
   ```

2. **Policy Indices**
   ```cpp
   // Check move mapping
   for (Move m : legal_moves) {
     int idx = move_to_policy_index(m, pos);
     std::cout << UCIEngine::move(m, false) 
               << " → index " << idx << "\n";
   }
   ```

3. **Metal Debugging**
   ```bash
   # Enable Metal validation
   export METAL_DEVICE_WRAPPER_TYPE=1
   
   # Check GPU usage
   # Activity Monitor → GPU History
   ```

## Next Steps for Implementation

See `src/nn/README.md` for:
- Detailed roadmap
- Time estimates
- Dependencies
- References
