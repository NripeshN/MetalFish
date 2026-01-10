# CUDA Backend Architecture

This document describes the CUDA GPU acceleration backend for MetalFish.

## Overview

The CUDA backend provides GPU acceleration for NVIDIA GPUs, mirroring the architecture of the existing Metal backend. It implements the same abstract `Backend` interface defined in `src/gpu/backend.h`, ensuring consistent API across different GPU platforms.

## File Structure

```
src/gpu/cuda/
├── cuda_backend.cu          # CUDA implementation of GPU::Backend interface
└── kernels/
    └── nnue_full.cu         # CUDA kernels for NNUE evaluation
```

## Components

### 1. CUDA Backend (`cuda_backend.cu`)

Implements the GPU backend interface using CUDA APIs:

- **CUDABuffer**: Manages GPU memory using `cudaMalloc` and `cudaMallocManaged`
  - Supports both unified memory (when available) and discrete GPU memory
  - Provides CPU-accessible pointers for data transfer
  
- **CUDAKernel**: Wraps CUDA kernel functions (`CUfunction`)
  - Created from compiled PTX or loaded from libraries
  - Provides kernel attributes and launch configuration
  
- **CUDACommandEncoder**: Records and executes GPU commands
  - Sets kernel parameters and buffers
  - Manages kernel launches with grid/block dimensions
  - Provides synchronization primitives
  
- **CUDABackend**: Main singleton backend instance
  - Initializes CUDA runtime and driver API
  - Provides buffer and kernel management
  - Supports runtime kernel compilation via NVRTC

### 2. NNUE Kernels (`kernels/nnue_full.cu`)

CUDA kernels for neural network evaluation:

- **Feature extraction**: `extract_halfka_features`
  - Extracts chess position features for NNUE
  
- **Feature transformer**: `feature_transformer`
  - Applies weights to sparse feature indices
  
- **Incremental updates**: `incremental_update`
  - Efficiently updates accumulators for move changes
  
- **Network layers**: 
  - `affine_transform_relu`: Linear layer with ClippedReLU activation
  - `affine_transform_sqr_relu`: Linear layer with SqrClippedReLU activation
  - `output_layer`: Final evaluation layer
  
- **Fused forward pass**: `nnue_forward_pass`
  - Complete NNUE inference in a single kernel
  - Optimized for low-latency evaluation

## Building with CUDA

### Requirements

- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- CMake 3.20 or later
- C++ compiler with C++20 support

### Build Instructions

```bash
cmake -B build -DUSE_CUDA=ON
cmake --build build -j8
```

### Supported GPU Architectures

The backend is compiled for the following CUDA architectures:
- Pascal: 6.0, 6.1
- Volta: 7.0
- Turing: 7.5
- Ampere: 8.0, 8.6
- Ada: 8.9
- Hopper: 9.0

## Implementation Details

### Memory Management

The CUDA backend supports three memory modes:

1. **Shared (Unified Memory)**: Uses `cudaMallocManaged` for zero-copy access
   - Automatically migrates between CPU and GPU
   - Best for systems with unified memory support
   
2. **Private (GPU-only)**: Uses `cudaMalloc` for device memory
   - Fastest for GPU-only data
   - Requires explicit synchronization
   
3. **Managed**: System-managed CPU/GPU synchronization
   - Falls back to unified memory on most systems

### Runtime Compilation

The backend supports runtime kernel compilation via NVRTC:

```cpp
gpu.compile_library("my_kernels", kernel_source);
auto kernel = gpu.create_kernel("my_function", "my_kernels");
```

This enables dynamic kernel generation and optimization.

### Kernel Execution

Kernels are launched using the command encoder pattern:

```cpp
auto encoder = gpu.create_encoder();
encoder->set_kernel(kernel.get());
encoder->set_buffer(buffer.get(), 0);
encoder->dispatch_threads(1024);  // Launch 1024 threads
gpu.submit_and_wait(encoder.get());
```

## Testing

CUDA functionality is tested in `tests/test_cuda.cpp`:

- Backend initialization and device detection
- Buffer creation and memory access
- Kernel compilation and execution
- Unified memory verification

Tests gracefully skip when CUDA is not available.

## Future Work

- Integration with NNUE evaluation pipeline
- Performance optimization for batch inference
- Multi-GPU support
- Stream-based asynchronous execution
- Tensor Core utilization for matrix operations

## Comparison with Metal Backend

| Feature | Metal | CUDA |
|---------|-------|------|
| Unified Memory | Always available (Apple Silicon) | Depends on GPU/driver |
| Kernel Language | Metal Shading Language | CUDA C++ |
| Runtime Compilation | MSL source → Metal IR | CUDA C++ → PTX → SASS |
| Thread Groups | threadgroups | blocks |
| Thread Execution | threads | threads |
| Synchronization | Memory fences | `__syncthreads()`, stream sync |

Both backends implement the same abstract interface, ensuring portable code.
