/*
  MetalFish - GPU-accelerated chess engine - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  This file provides Metal GPU acceleration for NNUE evaluation,
  leveraging Apple Silicon's unified memory for zero-copy access.
*/

#ifndef GPU_NNUE_H_INCLUDED
#define GPU_NNUE_H_INCLUDED

#ifdef __APPLE__

#include <cstdint>
#include <memory>
#include <vector>

namespace MetalFish {

// Forward declarations
class MetalDevice;

// GPU-accelerated NNUE evaluation
class GPUNNUEEvaluator {
public:
  GPUNNUEEvaluator();
  ~GPUNNUEEvaluator();

  // Initialize with NNUE network weights
  bool initialize(const void *weights, size_t size);

  // Batch evaluate multiple positions
  // Returns scores for each position
  std::vector<int32_t>
  batch_evaluate(const std::vector<const uint8_t *> &transformed_features,
                 size_t batch_size);

  // Single position evaluation (falls back to CPU for small batches)
  int32_t evaluate(const uint8_t *transformed_features);

  // Check if GPU acceleration is available
  static bool is_available();

private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;
};

// GPU-accelerated feature transformer
class GPUFeatureTransformer {
public:
  GPUFeatureTransformer();
  ~GPUFeatureTransformer();

  // Initialize with transformer weights
  bool initialize(const void *weights, size_t size);

  // Transform features for a batch of positions
  void batch_transform(const std::vector<const int16_t *> &accumulators,
                       std::vector<uint8_t *> &output, size_t batch_size);

private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;
};

} // namespace MetalFish

#endif // __APPLE__

#endif // GPU_NNUE_H_INCLUDED
