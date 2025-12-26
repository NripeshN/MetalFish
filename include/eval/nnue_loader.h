/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  NNUE Network Loader - Loads Stockfish .nnue files
  ==================================================

  This module handles loading of Stockfish NNUE network files (.nnue)
  and prepares weights for GPU evaluation.

  Stockfish NNUE Architecture (Big Network):
  - Feature Transformer: 45056 inputs -> 1024 outputs
  - FC0: 2048 -> 16 (2x perspectives concatenated)
  - FC1: 30 -> 32
  - FC2: 32 -> 1 + skip connection
  - PSQT: 45056 -> 8 buckets
*/

#pragma once

#include "core/types.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <istream>
#include <memory>
#include <string>
#include <vector>

namespace MetalFish {
namespace Eval {
namespace NNUE {

// NNUE dimensions (matching Stockfish big network)
constexpr int FT_INPUT_DIMS = 45056;      // HalfKAv2_hm features
constexpr int FT_HIDDEN_DIMS_BIG = 1024;  // Big network hidden
constexpr int FT_HIDDEN_DIMS_SMALL = 128; // Small network hidden
constexpr int FC0_OUTPUTS = 16;           // 15 + 1 (skip)
constexpr int FC1_INPUTS = 30;            // FC0-1 * 2
constexpr int FC1_OUTPUTS = 32;
constexpr int FC2_OUTPUTS = 1;
constexpr int PSQT_BUCKETS = 8;
constexpr int LAYER_STACKS = 8;

// Weight types
using FTWeight = int16_t;
using FTBias = int16_t;
using FCWeight = int8_t;
using FCBias = int32_t;

// Network weights structure
struct NNUEWeights {
  // Feature transformer weights
  std::vector<FTWeight> ft_weights;  // [FT_INPUT_DIMS x HalfDimensions]
  std::vector<FTBias> ft_biases;     // [HalfDimensions]
  std::vector<int16_t> psqt_weights; // [FT_INPUT_DIMS x PSQT_BUCKETS]

  // FC layers (for each layer stack)
  std::vector<FCWeight>
      fc0_weights;                // [HalfDims*2 x FC0_OUTPUTS x LAYER_STACKS]
  std::vector<FCBias> fc0_biases; // [FC0_OUTPUTS x LAYER_STACKS]
  std::vector<FCWeight>
      fc1_weights;                // [FC1_INPUTS x FC1_OUTPUTS x LAYER_STACKS]
  std::vector<FCBias> fc1_biases; // [FC1_OUTPUTS x LAYER_STACKS]
  std::vector<FCWeight> fc2_weights; // [FC1_OUTPUTS x LAYER_STACKS]
  std::vector<FCBias> fc2_biases;    // [1 x LAYER_STACKS]

  int half_dimensions = 0;
  uint32_t hash = 0;
  bool loaded = false;

  void resize(int halfDims);
};

// Helper to read little-endian values
template <typename T> inline T read_le(std::istream &stream) {
  T value;
  stream.read(reinterpret_cast<char *>(&value), sizeof(T));
  return value;
}

// NNUE file loader
class NNUELoader {
public:
  // Load from file path
  bool load_from_file(const std::string &path, NNUEWeights &weights);

  // Load from memory
  bool load_from_memory(const uint8_t *data, size_t size, NNUEWeights &weights);

  // Load from stream
  bool load_from_stream(std::istream &stream, NNUEWeights &weights);

  // Get last error message
  const std::string &error() const { return error_; }

private:
  std::string error_;

  bool read_header(std::istream &stream, uint32_t &version, uint32_t &hash,
                   std::string &arch);
  bool read_feature_transformer(std::istream &stream, NNUEWeights &weights);
  bool read_network(std::istream &stream, NNUEWeights &weights);
};

// Default embedded network (can be embedded at compile time)
extern const uint8_t *EmbeddedNNUEData;
extern const size_t EmbeddedNNUESize;

// Load default network
bool load_default_network(NNUEWeights &weights);

// Load network from UCI option
bool load_network(const std::string &path, NNUEWeights &weights);

} // namespace NNUE
} // namespace Eval
} // namespace MetalFish
