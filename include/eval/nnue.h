/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include "metal/allocator.h"
#include "metal/device.h"
#include <memory>
#include <string>
#include <vector>

namespace MetalFish {

namespace NNUE {

// NNUE architecture constants (matching Stockfish's architecture)
constexpr int HALF_DIMENSIONS = 1024;
constexpr int FT_IN_DIMS = 64 * 11 * 64; // King bucket * piece types * squares
constexpr int FT_OUT_DIMS = HALF_DIMENSIONS;

// Layer sizes
constexpr int L1_SIZE = 2 * HALF_DIMENSIONS; // Two perspectives
constexpr int L2_SIZE = 16;
constexpr int L3_SIZE = 32;
constexpr int OUTPUT_SIZE = 1;

// Accumulator structure for incremental updates
struct Accumulator {
  alignas(64) int16_t accumulation[COLOR_NB][HALF_DIMENSIONS];
  bool computed[COLOR_NB] = {false, false};

  void reset() { computed[WHITE] = computed[BLACK] = false; }
};

// Feature index calculation
int feature_index(Square ksq, Square psq, Piece pc, Color perspective);

// Network class - handles NNUE evaluation using Metal GPU
class Network {
public:
  Network();
  ~Network();

  // Load network from file
  bool load(const std::string &path);
  bool load_from_embedded();

  // Single position evaluation
  Value evaluate(const Position &pos, Accumulator &acc);

  // Batch evaluation for GPU parallelism
  void batch_evaluate(const Position *positions, Accumulator *accumulators,
                      Value *outputs, size_t count);

  // Update accumulator incrementally
  void update_accumulator(const Position &pos, Accumulator &acc);

  // Refresh accumulator from scratch
  void refresh_accumulator(const Position &pos, Accumulator &acc);

  bool is_loaded() const { return loaded; }
  std::string info() const { return networkInfo; }

private:
  // GPU buffers for network weights
  Metal::Buffer featureWeights;
  Metal::Buffer featureBiases;
  Metal::Buffer l1Weights;
  Metal::Buffer l1Biases;
  Metal::Buffer l2Weights;
  Metal::Buffer l2Biases;
  Metal::Buffer outputWeights;
  Metal::Buffer outputBias;

  // Working buffers for GPU computation
  Metal::Buffer inputBuffer;
  Metal::Buffer accumulatorBuffer;
  Metal::Buffer outputBuffer;

  // Metal compute pipelines
  MTL::ComputePipelineState *featureTransformKernel = nullptr;
  MTL::ComputePipelineState *affineTransformKernel = nullptr;
  MTL::ComputePipelineState *clippedReluKernel = nullptr;

  bool loaded = false;
  std::string networkInfo;

  void allocate_buffers();
  void compile_kernels();
};

// Global network instance
extern std::unique_ptr<Network> network;

// Initialize NNUE system
void init();

// Evaluate a position
Value evaluate(const Position &pos);

} // namespace NNUE

} // namespace MetalFish
