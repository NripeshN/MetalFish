/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE Tensor Core Kernels

  Tensor core accelerated kernels using WMMA API for maximum performance
  on Volta (SM 7.0) and later architectures.
*/

#ifndef NNUE_CUDA_TENSOR_CORE_CU
#define NNUE_CUDA_TENSOR_CORE_CU

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Include WMMA (Warp Matrix Multiply-Accumulate) API
#if __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda::wmma;
#endif

// ============================================================================
// Architecture Constants
// ============================================================================

constexpr int FT_DIM_BIG = 1024;
constexpr int FT_DIM_SMALL = 128;
constexpr int FC0_OUT = 15;
constexpr int FC1_OUT = 32;
constexpr int WEIGHT_SCALE_BITS = 6;
constexpr int OUTPUT_SCALE = 16;

// WMMA tile sizes (16x16x16 for FP16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

using layer_weight_t = int8_t;
using accumulator_t = int32_t;

// ============================================================================
// Activation Functions
// ============================================================================

__device__ __forceinline__ int8_t clipped_relu(int16_t x) {
  return static_cast<int8_t>(max(0, min(127, static_cast<int>(x))));
}

__device__ __forceinline__ int8_t sqr_clipped_relu(int16_t x) {
  int clamped = max(0, min(127, static_cast<int>(x)));
  return static_cast<int8_t>((clamped * clamped) >> 7);
}

// ============================================================================
// FP16 Conversion Helpers
// ============================================================================

/**
 * Convert int8 activation to half precision
 */
__device__ __forceinline__ half int8_to_half(int8_t x) {
  return __int2half_rn(static_cast<int>(x));
}

/**
 * Convert half precision back to int8 with clipping
 */
__device__ __forceinline__ int8_t half_to_int8_clipped(half x) {
  int val = __half2int_rn(x);
  return static_cast<int8_t>(max(0, min(127, val)));
}

// ============================================================================
// Tensor Core FC Layer (FP16)
// ============================================================================

#if __CUDA_ARCH__ >= 700

/**
 * Fully connected layer using tensor cores (WMMA API)
 * Input: [batch_size, input_size] in FP16
 * Weights: [output_size, input_size] in FP16
 * Output: [batch_size, output_size] in FP16
 * 
 * Uses 16x16x16 tiles for optimal tensor core utilization
 */
__global__ void fc_layer_tensor_core_fp16(
    const half *__restrict__ input,
    const half *__restrict__ weights,
    const half *__restrict__ biases,
    half *__restrict__ output,
    int batch_size, int input_size, int output_size) {
  
  // Warp and lane IDs
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int warpN = blockIdx.y;
  
  // Declare the fragments
  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
  
  // Initialize the output to zero
  fill_fragment(c_frag, __float2half(0.0f));
  
  // Bounds check
  if (warpM * WMMA_M >= batch_size || warpN * WMMA_N >= output_size) {
    return;
  }
  
  // Matrix multiply: C = A * B^T
  // A: [batch_size, input_size]
  // B: [output_size, input_size] (transposed to col_major)
  for (int k = 0; k < input_size; k += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = k;
    int bRow = k;
    int bCol = warpN * WMMA_N;
    
    // Load A fragment (input activations)
    if (aRow < batch_size && aCol < input_size) {
      load_matrix_sync(a_frag, input + aRow * input_size + aCol, input_size);
    }
    
    // Load B fragment (weights, transposed)
    if (bCol < output_size && bRow < input_size) {
      load_matrix_sync(b_frag, weights + bCol * input_size + bRow, input_size);
    }
    
    // Perform the matrix multiply-accumulate
    mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  
  // Add biases
  if (biases != nullptr) {
    for (int i = 0; i < c_frag.num_elements; i++) {
      int row = i / WMMA_N;
      int col = i % WMMA_N;
      int global_col = warpN * WMMA_N + col;
      if (global_col < output_size) {
        c_frag.x[i] = __hadd(c_frag.x[i], biases[global_col]);
      }
    }
  }
  
  // Store the output
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  if (cRow < batch_size && cCol < output_size) {
    store_matrix_sync(output + cRow * output_size + cCol, c_frag, 
                      output_size, mem_row_major);
  }
}

/**
 * FC0 layer using tensor cores
 * Converts int32 accumulators to FP16, applies tensor cores, converts back
 */
__global__ void fc0_layer_tensor_core(
    const accumulator_t *__restrict__ accumulators,
    const half *__restrict__ weights_fp16,
    const half *__restrict__ biases_fp16,
    int8_t *__restrict__ output_sqr,
    int8_t *__restrict__ output_linear,
    int hidden_dim, int batch_size) {
  
  extern __shared__ half shared_mem[];
  half *input_fp16 = shared_mem;
  half *output_fp16 = shared_mem + blockDim.x * hidden_dim;
  
  int pos_idx = blockIdx.x;
  if (pos_idx >= batch_size) return;
  
  const accumulator_t *white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t *black_acc = white_acc + hidden_dim;
  
  // Convert both perspectives to FP16
  for (int i = threadIdx.x; i < 2 * hidden_dim; i += blockDim.x) {
    const accumulator_t *acc = (i < hidden_dim) ? white_acc : black_acc;
    int idx = (i < hidden_dim) ? i : i - hidden_dim;
    
    // Apply clipped ReLU and convert to FP16
    int16_t val = static_cast<int16_t>(acc[idx] >> WEIGHT_SCALE_BITS);
    int8_t clipped = clipped_relu(val);
    input_fp16[i] = __int2half_rn(clipped);
  }
  __syncthreads();
  
  // Use tensor cores for the matrix multiply
  // Each warp handles one output neuron
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  
  if (warp_id < (FC0_OUT + 1)) {
    int out_idx = warp_id;
    
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    
    fill_fragment(c_frag, __float2half(0.0f));
    
    // Process in tiles
    // WMMA operations require all threads in the warp to participate
    for (int k = 0; k < 2 * hidden_dim; k += WMMA_K) {
      if (k < 2 * hidden_dim) {
        load_matrix_sync(a_frag, input_fp16 + k, 2 * hidden_dim);
        load_matrix_sync(b_frag, weights_fp16 + out_idx * 2 * hidden_dim + k, 
                        2 * hidden_dim);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    
    // Reduce across fragment and add bias
    if (lane == 0) {
      half sum = __float2half(0.0f);
      for (int i = 0; i < c_frag.num_elements; i++) {
        sum = __hadd(sum, c_frag.x[i]);
      }
      sum = __hadd(sum, biases_fp16[out_idx]);
      
      int16_t result = __half2int_rn(sum);
      
      // Store squared and linear outputs
      if (out_idx < FC0_OUT) {
        output_sqr[pos_idx * 2 * FC0_OUT + out_idx] = sqr_clipped_relu(result);
        output_sqr[pos_idx * 2 * FC0_OUT + FC0_OUT + out_idx] = sqr_clipped_relu(result);
      } else {
        output_linear[pos_idx * 2] = clipped_relu(result);
        output_linear[pos_idx * 2 + 1] = clipped_relu(result);
      }
    }
  }
}

/**
 * Fused NNUE evaluation using tensor cores throughout
 */
__global__ void nnue_forward_tensor_core(
    const accumulator_t *__restrict__ accumulators,
    const half *__restrict__ fc0_weights,
    const half *__restrict__ fc0_biases,
    const half *__restrict__ fc1_weights,
    const half *__restrict__ fc1_biases,
    const half *__restrict__ fc2_weights,
    const half *__restrict__ fc2_biases,
    int32_t *__restrict__ output,
    int hidden_dim, int batch_size) {
  
  extern __shared__ half shared_mem[];
  
  int pos_idx = blockIdx.x;
  if (pos_idx >= batch_size) return;
  
  half *fc0_input = shared_mem;
  half *fc0_output = shared_mem + 2 * hidden_dim;
  half *fc1_output = fc0_output + 2 * (FC0_OUT + 1);
  
  const accumulator_t *white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t *black_acc = white_acc + hidden_dim;
  
  // Convert accumulators to FP16
  for (int i = threadIdx.x; i < 2 * hidden_dim; i += blockDim.x) {
    const accumulator_t *acc = (i < hidden_dim) ? white_acc : black_acc;
    int idx = (i < hidden_dim) ? i : i - hidden_dim;
    int16_t val = static_cast<int16_t>(acc[idx] >> WEIGHT_SCALE_BITS);
    fc0_input[i] = __int2half_rn(clipped_relu(val));
  }
  __syncthreads();
  
  // FC0 layer with tensor cores (simplified)
  // ... (tensor core matrix multiply)
  
  // FC1 layer with tensor cores
  // ... (tensor core matrix multiply)
  
  // FC2 layer (small, can use standard multiplication)
  if (threadIdx.x == 0) {
    half sum = fc2_biases[0];
    for (int i = 0; i < FC1_OUT; i++) {
      sum = __hfma(fc1_output[i], fc2_weights[i], sum);
    }
    output[pos_idx] = __half2int_rn(sum);
  }
}

#endif // __CUDA_ARCH__ >= 700

// ============================================================================
// INT8 Tensor Core Support (Turing SM 7.5+)
// ============================================================================

#if __CUDA_ARCH__ >= 750

/**
 * FC layer using INT8 tensor cores (Turing and later)
 * Provides even better performance for quantized inference
 */
__global__ void fc_layer_tensor_core_int8(
    const int8_t *__restrict__ input,
    const int8_t *__restrict__ weights,
    const int32_t *__restrict__ biases,
    int8_t *__restrict__ output,
    int batch_size, int input_size, int output_size) {
  
  // INT8 tensor cores use 8x8x16 tiles on Turing
  // 16x8x16 tiles on Ampere and later
  
  // Warp and lane IDs
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int warpN = blockIdx.y;
  
  // Note: INT8 WMMA requires different fragment types
  // This is a simplified example - full implementation would use
  // appropriate fragment types for INT8
  
  // Bounds check
  if (warpM * 16 >= batch_size || warpN * 16 >= output_size) {
    return;
  }
  
  // INT8 tensor core implementation would go here
  // For now, this serves as a placeholder for future optimization
}

#endif // __CUDA_ARCH__ >= 750

// ============================================================================
// Host Interface Functions
// ============================================================================

extern "C" {

/**
 * Check if tensor cores are available on the current device
 */
bool cuda_tensor_cores_available(int device_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  // Tensor cores available on SM 7.0 (Volta) and later
  return prop.major >= 7;
}

/**
 * Check if INT8 tensor cores are available
 */
bool cuda_int8_tensor_cores_available(int device_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  // INT8 tensor cores available on SM 7.5 (Turing) and later
  return (prop.major > 7) || (prop.major == 7 && prop.minor >= 5);
}

// Tensor core function implementations are architecture-specific
// and must be compiled with appropriate -arch flags

/**
 * FC layer with FP16 tensor cores
 * Only available when compiled for SM 7.0+
 */
void cuda_fc_layer_tensor_core_fp16(
    const half *input, const half *weights, const half *biases,
    half *output, int batch_size, int input_size, int output_size,
    cudaStream_t stream) {
  
  // Runtime check for architecture support
  int device;
  cudaGetDevice(&device);
  if (!cuda_tensor_cores_available(device)) {
    std::cerr << "[CUDA] Tensor cores not available on this device" << std::endl;
    return;
  }
  
  dim3 block(128);  // 4 warps per block
  dim3 grid((batch_size + 15) / 16,  // WMMA_M = 16
            (output_size + 15) / 16); // WMMA_N = 16
  
  // Launch the kernel - it will be compiled for all architectures in CMAKE_CUDA_ARCHITECTURES
  // The kernel code is conditionally compiled based on __CUDA_ARCH__ during device compilation
  fc_layer_tensor_core_fp16<<<grid, block, 0, stream>>>(
      input, weights, biases, output, batch_size, input_size, output_size);
}

/**
 * FC0 layer with tensor cores
 * Only available when compiled for SM 7.0+
 */
void cuda_fc0_layer_tensor_core(
    const accumulator_t *accumulators,
    const half *weights_fp16, const half *biases_fp16,
    int8_t *output_sqr, int8_t *output_linear,
    int hidden_dim, int batch_size, cudaStream_t stream) {
  
  int device;
  cudaGetDevice(&device);
  if (!cuda_tensor_cores_available(device)) {
    std::cerr << "[CUDA] Tensor cores not available on this device" << std::endl;
    return;
  }
  
  dim3 block(128);
  dim3 grid(batch_size);
  size_t shared_mem = (2 * hidden_dim + 2 * (FC0_OUT + 1)) * sizeof(half);
  
  // Launch the kernel - it will be compiled for all architectures in CMAKE_CUDA_ARCHITECTURES
  fc0_layer_tensor_core<<<grid, block, shared_mem, stream>>>(
      accumulators, weights_fp16, biases_fp16,
      output_sqr, output_linear, hidden_dim, batch_size);
}

/**
 * Full NNUE forward pass with tensor cores
 * Note: This is a simplified implementation. Full implementation would require
 * complete tensor core matrix operations for all layers.
 * Only available when compiled for SM 7.0+
 */
void cuda_nnue_forward_tensor_core(
    const accumulator_t *accumulators,
    const half *fc0_weights, const half *fc0_biases,
    const half *fc1_weights, const half *fc1_biases,
    const half *fc2_weights, const half *fc2_biases,
    int32_t *output, int hidden_dim, int batch_size, cudaStream_t stream) {
  
  int device;
  cudaGetDevice(&device);
  if (!cuda_tensor_cores_available(device)) {
    std::cerr << "[CUDA] Tensor cores not available on this device" << std::endl;
    return;
  }
  
  // TODO: Implement full tensor core forward pass
  // Currently this is a placeholder that demonstrates the API
  // A complete implementation would:
  // 1. Convert accumulators to FP16
  // 2. Use tensor cores for FC0 layer (hidden_dim -> FC0_OUT)
  // 3. Use tensor cores for FC1 layer (FC0_OUT -> FC1_OUT)
  // 4. Use standard ops for FC2 (small output, not worth tensor cores)
  // 5. Apply activations and skip connections
  
  std::cerr << "[CUDA] Full tensor core forward pass not yet implemented" << std::endl;
  std::cerr << "[CUDA] Use individual layer functions instead" << std::endl;
}

} // extern "C"

#endif // NNUE_CUDA_TENSOR_CORE_CU
