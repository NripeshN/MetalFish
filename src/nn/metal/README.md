# Metal Neural Network Backend

This directory contains the Metal/MPSGraph implementation for transformer-based neural network inference on Apple Silicon.

## Overview

The Metal backend uses Apple's MetalPerformanceShadersGraph (MPSGraph) framework to execute transformer neural networks on the GPU. It provides high-performance inference for chess position evaluation using modern attention-based architectures.

## Architecture

### Files

- `metal_network.h` - Public C++ interface following the Network base class
- `metal_network.mm` - Objective-C++ implementation using MPSGraph

### Network Structure

The implementation supports transformer-based neural networks with the following architecture:

```
Input (112 planes × 8×8 board)
  ↓
Flatten (7168 values)
  ↓
Embedding Layer (7168 → embedding_size)
  ↓
Layer Normalization (optional)
  ↓
Transformer Encoder Stack (repeat for num_layers):
  ├─ Layer Normalization
  ├─ Multi-Head Self-Attention
  ├─ Residual Connection
  ├─ Layer Normalization
  ├─ Feed-Forward Network
  └─ Residual Connection
  ↓
Output Heads:
  ├─ Policy Head → 1858 move probabilities
  └─ Value Head → 1 value or 3 WDL probabilities
```

## Features

### Supported Network Types

- Pure transformer architecture
- Configurable embedding size (typically 256-512)
- Variable number of encoder layers (1-24)
- Configurable attention heads (4-16)
- Multiple activation functions (ReLU, Swish, Mish, SELU, etc.)

### Output Formats

- **Policy**: 1858-dimensional probability distribution over legal moves
- **Value**: Single scalar evaluation (-1 to 1)
- **WDL**: Win/Draw/Loss probabilities (3 values)
- **Moves Left**: Predicted moves until game end (infrastructure ready)

### Performance Optimizations

1. **Graph Compilation**: MPSGraph built once at initialization, reused for all inferences
2. **Unified Memory**: Uses shared memory mode for efficient CPU↔GPU transfers
3. **Batch Processing**: Native support for evaluating multiple positions in parallel
4. **Automatic Optimization**: Metal runtime optimizes graph execution

## Usage

### From C++

```cpp
#include "nn/metal/metal_network.h"

using namespace MetalFish::NN;

// Load weights
auto weights = LoadWeights("weights.pb.gz");

// Create Metal network
auto network = std::make_unique<Metal::MetalNetwork>(weights.value());

// Encode position
InputPlanes input = EncodePositionForNN(position);

// Evaluate
NetworkOutput output = network->Evaluate(input);

// Access results
float value = output.value;
std::vector<float> policy = output.policy;  // 1858 move probabilities
if (output.has_wdl) {
  float win = output.wdl[0];
  float draw = output.wdl[1];
  float loss = output.wdl[2];
}
```

### Batch Evaluation

```cpp
std::vector<InputPlanes> batch;
for (const auto& pos : positions) {
  batch.push_back(EncodePositionForNN(pos));
}

auto outputs = network->EvaluateBatch(batch);
```

## Implementation Details

### Weight Loading

Weights are loaded from protobuf format (`.pb` or `.pb.gz` files) and converted to Metal buffers. The implementation supports multiple encoding formats:

- FLOAT32 (standard)
- FLOAT16 (half precision)
- BFLOAT16 (brain float)
- LINEAR16 (quantized)

### Activation Functions

Configurable activation functions detected from network weights:

- **ReLU**: max(0, x)
- **ReLU²**: max(0, x)²
- **Swish**: x * sigmoid(x)
- **Mish**: x * tanh(softplus(x))
- **SELU**: Scaled exponential linear unit
- **Tanh**: Hyperbolic tangent
- **Sigmoid**: 1 / (1 + e^(-x))

### Multi-Head Attention

The current implementation uses a simplified attention mechanism that can be extended to true multi-head attention. The key components are:

1. **Query, Key, Value Projections**: Linear transformations of input
2. **Scaled Dot-Product Attention**: softmax(Q·K^T / √d_k) · V
3. **Output Projection**: Linear transformation of attention output

### Layer Normalization

Standard layer normalization with learnable scale (gamma) and shift (beta):

```
y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
```

### Feed-Forward Network

Two-layer MLP with configurable activation:

```
FFN(x) = activation(x·W1 + b1)·W2 + b2
```

## Memory Management

The implementation uses RAII and smart pointers for automatic resource management:

- `std::unique_ptr` for PIMPL pattern
- `@autoreleasepool` for Metal object lifecycle
- Automatic buffer allocation and deallocation

## Error Handling

The network throws exceptions on:

- Metal device not available
- Failed to create command queue
- Missing required weights
- Invalid weight dimensions

## Performance Characteristics

Expected performance on Apple Silicon:

- **M1/M2**: ~20-40ms per position (single)
- **M1 Pro/Max**: ~15-30ms per position (single)
- **Batch size 256**: ~30-60ms total (0.12-0.24ms per position)

Performance scales well with:
- Larger batch sizes
- Unified memory architecture
- Neural Engine acceleration (automatic in some operations)

## Future Enhancements

### Planned

1. **True Multi-Head Attention**: Reshape tensors for parallel head computation
2. **Position Encoding**: Support learned and fixed position embeddings
3. **Smolgen**: Dynamic weight generation for policy head
4. **Relative Position Encoding**: RPE for improved spatial reasoning

### Optimization Opportunities

1. **MPSGraphExecutable**: Pre-compile graphs for faster execution
2. **Mixed Precision**: FP16 operations where appropriate
3. **Memory Pooling**: Reuse input/output buffers
4. **Graph Caching**: Cache compiled graphs for different batch sizes

## Testing

The Metal backend is tested as part of the main test suite:

```bash
cd build
./metalfish_tests
```

Network-specific tests:

```bash
./test_nn_comparison  # Compare Metal vs. stub backends
```

## Requirements

- macOS 12.0 or later
- Apple Silicon (M1/M2/M3) or Intel with AMD GPU
- MetalPerformanceShadersGraph framework
- Metal-cpp headers (automatically downloaded by CMake)

## Troubleshooting

### Metal not available

If Metal is not available, the backend will throw an exception and fall back to the CPU stub. Check:

```bash
system_profiler SPDisplaysDataType | grep Metal
```

### Out of memory

Reduce batch size or use smaller network. Metal has limited GPU memory:

- M1: 8GB shared
- M1 Pro: 16GB shared
- M1 Max: 32-64GB shared

### Slow inference

Check that:
1. Network is compiled in Release mode (`-O3`)
2. Graph is reused (not rebuilt per inference)
3. Batch size is reasonable (powers of 2 work well)

## License

GPL-3.0 - See LICENSE file for details

## Copyright

Copyright (C) 2025 Nripesh Niketan
