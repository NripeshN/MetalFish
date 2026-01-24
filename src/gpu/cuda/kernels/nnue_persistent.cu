/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Persistent Kernels for Small Batches

  Implements persistent kernels that stay resident on the GPU,
  reducing launch overhead for small batch evaluations.
*/

#ifndef NNUE_PERSISTENT_KERNELS_CU
#define NNUE_PERSISTENT_KERNELS_CU

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

using weight_t = int16_t;
using layer_weight_t = int8_t;
using accumulator_t = int32_t;

constexpr int FC0_OUT = 15;
constexpr int FC1_OUT = 32;
constexpr int WEIGHT_SCALE_BITS = 6;
constexpr int OUTPUT_SCALE = 16;

// ============================================================================
// Work Queue for Persistent Kernels
// ============================================================================

/**
 * Work item for NNUE evaluation
 */
struct NNUEWorkItem {
  const accumulator_t *accumulators;
  int32_t *output;
  int hidden_dim;
  bool valid;
};

/**
 * Persistent kernel for small batch NNUE evaluation
 * Stays resident and processes work items as they arrive
 */
__global__ void persistent_nnue_evaluator(
    const layer_weight_t *fc0_weights,
    const int32_t *fc0_biases,
    const layer_weight_t *fc1_weights,
    const int32_t *fc1_biases,
    const layer_weight_t *fc2_weights,
    const int32_t *fc2_biases,
    NNUEWorkItem *work_queue,
    volatile int *queue_head,
    volatile int *queue_tail,
    int max_queue_size,
    volatile bool *shutdown_flag) {
  
  __shared__ int8_t fc0_sqr[2 * 16];
  __shared__ int8_t fc0_linear[2];
  __shared__ int8_t fc1_out[32];
  
  auto grid = cg::this_grid();
  int work_idx = blockIdx.x;
  
  while (true) {
    // Check for shutdown
    if (*shutdown_flag) {
      break;
    }
    
    // Try to get work
    if (*queue_tail <= *queue_head) {
      // No work available, wait briefly
      __nanosleep(1000);  // Sleep 1 microsecond
      continue;
    }
    
    // Get work item atomically
    int item_idx = atomicAdd(const_cast<int*>(queue_head), 1);
    if (item_idx >= *queue_tail) {
      // Missed it, try again
      continue;
    }
    
    item_idx = item_idx % max_queue_size;
    NNUEWorkItem work = work_queue[item_idx];
    
    if (!work.valid) {
      continue;
    }
    
    // Process the work item
    const accumulator_t *white_acc = work.accumulators;
    const accumulator_t *black_acc = white_acc + work.hidden_dim;
    
    // FC0 layer - simplified version for persistent kernel
    int tid = threadIdx.x;
    
    // Process each perspective
    for (int p = 0; p < 2; p++) {
      const accumulator_t *acc = (p == 0) ? white_acc : black_acc;
      
      for (int out = tid; out <= FC0_OUT; out += blockDim.x) {
        int32_t sum = fc0_biases[out];
        
        for (int i = 0; i < work.hidden_dim; i++) {
          int16_t val = static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS);
          int8_t clipped = static_cast<int8_t>(max(0, min(127, static_cast<int>(val))));
          sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
        }
        
        int16_t result = static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS);
        if (out < FC0_OUT) {
          int clamped = max(0, min(127, static_cast<int>(result)));
          fc0_sqr[p * FC0_OUT + out] = static_cast<int8_t>((clamped * clamped) >> 7);
        } else {
          fc0_linear[p] = static_cast<int8_t>(max(0, min(127, static_cast<int>(result))));
        }
      }
    }
    __syncthreads();
    
    // FC1 layer
    if (tid < FC1_OUT) {
      int32_t sum = fc1_biases[tid];
      for (int i = 0; i < 2 * FC0_OUT; i++) {
        sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + tid];
      }
      fc1_out[tid] = static_cast<int8_t>(
          max(0, min(127, static_cast<int>(sum >> WEIGHT_SCALE_BITS))));
    }
    __syncthreads();
    
    // FC2 layer with skip connection
    if (tid == 0) {
      int32_t sum = fc2_biases[0];
      for (int i = 0; i < FC1_OUT; i++) {
        sum += fc1_out[i] * fc2_weights[i];
      }
      
      int32_t skip_val = ((fc0_linear[0] + fc0_linear[1]) * 600 * OUTPUT_SCALE) /
                         (2 * 127 * (1 << WEIGHT_SCALE_BITS));
      *work.output = sum + skip_val;
    }
    
    // Mark work as complete
    work_queue[item_idx].valid = false;
    __syncthreads();
  }
}

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

/**
 * Launch persistent kernel
 * This kernel stays resident and processes work from a queue
 */
void cuda_launch_persistent_evaluator(
    const layer_weight_t *fc0_weights,
    const int32_t *fc0_biases,
    const layer_weight_t *fc1_weights,
    const int32_t *fc1_biases,
    const layer_weight_t *fc2_weights,
    const int32_t *fc2_biases,
    NNUEWorkItem *work_queue,
    volatile int *queue_head,
    volatile int *queue_tail,
    int max_queue_size,
    volatile bool *shutdown_flag,
    cudaStream_t stream) {
  
  // Launch with moderate block size
  dim3 block(128);
  dim3 grid(4);  // 4 blocks for better latency hiding
  
  persistent_nnue_evaluator<<<grid, block, 0, stream>>>(
      fc0_weights, fc0_biases,
      fc1_weights, fc1_biases,
      fc2_weights, fc2_biases,
      work_queue, queue_head, queue_tail,
      max_queue_size, shutdown_flag);
}

} // extern "C"

#endif // USE_CUDA
#endif // NNUE_PERSISTENT_KERNELS_CU
