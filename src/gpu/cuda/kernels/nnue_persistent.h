/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Persistent Kernels Header
*/

#ifndef NNUE_PERSISTENT_KERNELS_H
#define NNUE_PERSISTENT_KERNELS_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

using layer_weight_t = int8_t;
using accumulator_t = int32_t;

/**
 * Work item for NNUE evaluation
 */
struct NNUEWorkItem {
  const accumulator_t *accumulators;
  int32_t *output;
  int hidden_dim;
  bool valid;
};

extern "C" {

/**
 * Launch persistent kernel for small batch processing
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
    cudaStream_t stream);

} // extern "C"

#endif // USE_CUDA
#endif // NNUE_PERSISTENT_KERNELS_H
