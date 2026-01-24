# CUDA Backend Optimization Guide

## Overview

The MetalFish CUDA backend has been optimized to achieve parity with the Metal backend, leveraging NVIDIA GPU capabilities including:

- **Tensor Cores**: Hardware-accelerated FP16 and INT8 matrix operations (Volta SM 7.0+)
- **Warp-Level Primitives**: Efficient shuffle, ballot, and reduction operations  
- **Unified Memory**: Optimized with access hints and prefetching
- **Advanced Profiling**: NVTX markers, kernel timing, and occupancy analysis
- **Architecture Detection**: Runtime capability detection for optimal code paths

## Building with CUDA Optimizations

### Basic CUDA Build

```bash
cmake -DUSE_CUDA=ON ..
make
```

### Enable All Optimizations

```bash
cmake -DUSE_CUDA=ON \
      -DCUDA_TENSOR_CORES=ON \
      -DCUDA_WARP_PRIMITIVES=ON \
      -DCUDA_PROFILING=ON \
      -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
      ..
make
```

## Key Features

### 1. Tensor Core Acceleration
- FP16 WMMA API for 16x16x16 tiles
- INT8 support on Turing+ (SM 7.5+)
- Up to 8x speedup vs standard CUDA cores

### 2. Warp-Level Primitives
- Shuffle-based reductions (no shared memory)
- Cooperative groups for flexible synchronization
- Up to 3x faster than traditional approaches

### 3. Unified Memory Optimization
- `cudaMemAdvise` hints for optimal placement
- Asynchronous prefetching
- Read-mostly hints for weights

### 4. Advanced Profiling
- NVTX markers for Nsight
- Kernel timing infrastructure
- Occupancy calculator
- Bandwidth testing tools

For complete documentation, see the implementation files:
- `src/gpu/cuda/kernels/nnue_simd.cu` - Warp primitives
- `src/gpu/cuda/kernels/nnue_tensor_core.cu` - Tensor cores
- `src/gpu/cuda/cuda_memory.cu` - Memory management
- `src/gpu/cuda/cuda_profiling.h` - Profiling tools
