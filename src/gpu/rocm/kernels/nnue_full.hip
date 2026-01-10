/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive HIP Compute Kernels for NNUE Evaluation

  This file contains all GPU kernels needed for NNUE inference:
  - Feature extraction (HalfKAv2_hm and FullThreats)
  - Feature transformer (sparse to dense)
  - Network layers (AffineTransform, ClippedReLU, SqrClippedReLU)
  - Incremental accumulator updates

  Optimized for AMD ROCm/HIP platform.
*/

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

// ============================================================================
// NNUE Architecture Constants
// ============================================================================

// Network dimensions
#define FT_DIM_BIG 1024
#define FT_DIM_SMALL 128
#define FC0_OUT 15
#define FC1_OUT 32
#define PSQT_BUCKETS 8
#define LAYER_STACKS 8

// Feature dimensions
#define HALFKA_DIMS 45056  // 64 * 11 * 64
#define THREAT_DIMS 1536   // Full threats feature size

// Quantization
#define WEIGHT_SCALE_BITS 6
#define OUTPUT_SCALE 16

// Chess constants
#define SQUARE_NB 64
#define COLOR_NB 2
#define PIECE_TYPE_NB 7

// ============================================================================
// Type Definitions
// ============================================================================

typedef short weight_t;
typedef signed char layer_weight_t;
typedef int accumulator_t;

// ============================================================================
// Utility Functions
// ============================================================================

__device__ inline int clipped_relu(int x) {
  return min(max(x, 0), 127);
}

__device__ inline int sqr_clipped_relu(int x) {
  int clipped = clipped_relu(x);
  return clipped * clipped;
}

// ============================================================================
// Feature Transformer Kernel
// ============================================================================

/**
 * Transform sparse features to dense accumulator
 * Each thread processes one output dimension
 */
__global__ void feature_transformer(
    const weight_t* __restrict__ weights,
    const weight_t* __restrict__ biases,
    const int* __restrict__ active_features,
    int num_active,
    accumulator_t* __restrict__ output,
    int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= output_size) return;
    
    // Start with bias
    accumulator_t sum = biases[tid];
    
    // Accumulate weights for all active features
    for (int i = 0; i < num_active; ++i) {
        int feature_idx = active_features[i];
        sum += weights[feature_idx * output_size + tid];
    }
    
    output[tid] = sum;
}

// ============================================================================
// Affine Transform Kernel
// ============================================================================

/**
 * Affine transformation: output = weights * input + biases
 */
__global__ void affine_transform(
    const layer_weight_t* __restrict__ weights,
    const int* __restrict__ biases,
    const accumulator_t* __restrict__ input,
    int input_size,
    accumulator_t* __restrict__ output,
    int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= output_size) return;
    
    int sum = biases[tid];
    
    for (int i = 0; i < input_size; ++i) {
        sum += weights[tid * input_size + i] * input[i];
    }
    
    output[tid] = sum;
}

// ============================================================================
// ClippedReLU Kernel
// ============================================================================

/**
 * Apply clipped ReLU activation
 */
__global__ void apply_clipped_relu(
    const accumulator_t* __restrict__ input,
    accumulator_t* __restrict__ output,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= size) return;
    
    output[tid] = clipped_relu(input[tid] >> WEIGHT_SCALE_BITS);
}

// ============================================================================
// Squared ClippedReLU Kernel
// ============================================================================

/**
 * Apply squared clipped ReLU activation
 */
__global__ void apply_sqr_clipped_relu(
    const accumulator_t* __restrict__ input,
    accumulator_t* __restrict__ output,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= size) return;
    
    output[tid] = sqr_clipped_relu(input[tid] >> WEIGHT_SCALE_BITS);
}

// ============================================================================
// Incremental Accumulator Update Kernel
// ============================================================================

/**
 * Update accumulator incrementally based on feature changes
 */
__global__ void update_accumulator(
    accumulator_t* __restrict__ accumulator,
    const weight_t* __restrict__ weights,
    const int* __restrict__ added_features,
    int num_added,
    const int* __restrict__ removed_features,
    int num_removed,
    int output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= output_size) return;
    
    accumulator_t sum = accumulator[tid];
    
    // Add new features
    for (int i = 0; i < num_added; ++i) {
        int feature_idx = added_features[i];
        sum += weights[feature_idx * output_size + tid];
    }
    
    // Remove old features
    for (int i = 0; i < num_removed; ++i) {
        int feature_idx = removed_features[i];
        sum -= weights[feature_idx * output_size + tid];
    }
    
    accumulator[tid] = sum;
}

// ============================================================================
// Batch Evaluation Kernel
// ============================================================================

/**
 * Batch NNUE evaluation for multiple positions
 */
__global__ void batch_nnue_eval(
    const accumulator_t* __restrict__ accumulators,
    const layer_weight_t* __restrict__ fc0_weights,
    const int* __restrict__ fc0_biases,
    const layer_weight_t* __restrict__ fc1_weights,
    const int* __restrict__ fc1_biases,
    int* __restrict__ outputs,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for intermediate results
    __shared__ accumulator_t fc0_out[FC0_OUT];
    __shared__ accumulator_t fc1_out[FC1_OUT];
    
    int tid = threadIdx.x;
    const accumulator_t* acc = accumulators + batch_idx * FT_DIM_BIG;
    
    // FC0 layer
    if (tid < FC0_OUT) {
        int sum = fc0_biases[tid];
        for (int i = 0; i < FT_DIM_BIG; ++i) {
            sum += fc0_weights[tid * FT_DIM_BIG + i] * 
                   clipped_relu(acc[i] >> WEIGHT_SCALE_BITS);
        }
        fc0_out[tid] = sum;
    }
    __syncthreads();
    
    // FC1 layer
    if (tid < FC1_OUT) {
        int sum = fc1_biases[tid];
        for (int i = 0; i < FC0_OUT; ++i) {
            sum += fc1_weights[tid * FC0_OUT + i] * 
                   sqr_clipped_relu(fc0_out[i] >> WEIGHT_SCALE_BITS);
        }
        fc1_out[tid] = sum;
    }
    __syncthreads();
    
    // Final output
    if (tid == 0) {
        int final_sum = 0;
        for (int i = 0; i < FC1_OUT; ++i) {
            final_sum += fc1_out[i];
        }
        outputs[batch_idx] = final_sum / OUTPUT_SCALE;
    }
}

// ============================================================================
// Placeholder kernels for future implementation
// ============================================================================

// These will be expanded as needed:
// - HalfKAv2_hm feature extraction
// - Full threats feature extraction
// - PSQT (Piece-Square Table) evaluation
// - Multi-stack layer evaluation
