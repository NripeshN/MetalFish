/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "eval/nnue_loader.h"
#include <cstring>
#include <iostream>
#include <sstream>

namespace MetalFish {
namespace Eval {
namespace NNUE {

// No embedded network by default
const uint8_t *EmbeddedNNUEData = nullptr;
const size_t EmbeddedNNUESize = 0;

void NNUEWeights::resize(int halfDims) {
  half_dimensions = halfDims;

  ft_weights.resize(FT_INPUT_DIMS * halfDims);
  ft_biases.resize(halfDims);
  psqt_weights.resize(FT_INPUT_DIMS * PSQT_BUCKETS);

  fc0_weights.resize(halfDims * 2 * FC0_OUTPUTS * LAYER_STACKS);
  fc0_biases.resize(FC0_OUTPUTS * LAYER_STACKS);
  fc1_weights.resize(FC1_INPUTS * FC1_OUTPUTS * LAYER_STACKS);
  fc1_biases.resize(FC1_OUTPUTS * LAYER_STACKS);
  fc2_weights.resize(FC1_OUTPUTS * LAYER_STACKS);
  fc2_biases.resize(LAYER_STACKS);
}

bool NNUELoader::load_from_file(const std::string &path, NNUEWeights &weights) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    error_ = "Cannot open file: " + path;
    return false;
  }
  return load_from_stream(stream, weights);
}

bool NNUELoader::load_from_memory(const uint8_t *data, size_t size,
                                  NNUEWeights &weights) {
  std::string str(reinterpret_cast<const char *>(data), size);
  std::istringstream stream(str, std::ios::binary);
  return load_from_stream(stream, weights);
}

bool NNUELoader::load_from_stream(std::istream &stream, NNUEWeights &weights) {
  // Read NNUE file header
  // Format: version (4 bytes), hash (4 bytes), size (4 bytes), arch string

  uint32_t version = read_le<uint32_t>(stream);

  // Stockfish NNUE magic/version
  // Check for valid NNUE file signature
  if (version != 0x7AF32F16 && version != 0x7AF32F17) {
    // Try reading as newer format with different header
    stream.seekg(0);

    // Read header hash
    uint32_t headerHash = read_le<uint32_t>(stream);
    (void)headerHash;

    // Read description length
    uint32_t descLen = read_le<uint32_t>(stream);
    if (descLen > 1024) {
      error_ = "Invalid NNUE file format";
      return false;
    }

    // Skip description
    stream.seekg(descLen, std::ios::cur);
  }

  // Try to read feature transformer
  if (!read_feature_transformer(stream, weights)) {
    return false;
  }

  // Try to read network layers
  if (!read_network(stream, weights)) {
    return false;
  }

  weights.loaded = true;
  std::cout << "[NNUE] Loaded network with " << weights.half_dimensions
            << " hidden dimensions" << std::endl;

  return true;
}

bool NNUELoader::read_header(std::istream &stream, uint32_t &version,
                             uint32_t &hash, std::string &arch) {
  version = read_le<uint32_t>(stream);
  hash = read_le<uint32_t>(stream);

  uint32_t size = read_le<uint32_t>(stream);
  if (size > 256) {
    error_ = "Invalid architecture string size";
    return false;
  }

  arch.resize(size);
  stream.read(arch.data(), size);

  return stream.good();
}

bool NNUELoader::read_feature_transformer(std::istream &stream,
                                          NNUEWeights &weights) {
  // Read feature transformer hash
  uint32_t hash = read_le<uint32_t>(stream);
  weights.hash = hash;

  // Determine hidden dimensions from hash or remaining file size
  // For now, try to detect based on common patterns
  // Stockfish big: 1024, small: 128

  // Check current position
  auto pos = stream.tellg();
  stream.seekg(0, std::ios::end);
  auto fileSize = stream.tellg();
  stream.seekg(pos);

  auto remaining = fileSize - pos;

  // Estimate half dimensions based on file size
  // Feature transformer takes most of the space
  // ft_weights: FT_INPUT_DIMS * HalfDims * 2 bytes
  // ft_biases: HalfDims * 2 bytes
  // psqt: FT_INPUT_DIMS * 8 * 2 bytes
  // FC layers are small in comparison

  int halfDims;
  if (remaining > 100 * 1024 * 1024) {
    halfDims = FT_HIDDEN_DIMS_BIG;
  } else if (remaining > 5 * 1024 * 1024) {
    halfDims = FT_HIDDEN_DIMS_BIG;
  } else {
    halfDims = FT_HIDDEN_DIMS_SMALL;
  }

  weights.resize(halfDims);

  // Read biases
  stream.read(reinterpret_cast<char *>(weights.ft_biases.data()),
              weights.ft_biases.size() * sizeof(FTBias));

  // Read weights
  stream.read(reinterpret_cast<char *>(weights.ft_weights.data()),
              weights.ft_weights.size() * sizeof(FTWeight));

  // Read PSQT weights
  stream.read(reinterpret_cast<char *>(weights.psqt_weights.data()),
              weights.psqt_weights.size() * sizeof(int16_t));

  if (!stream.good()) {
    error_ = "Failed to read feature transformer";
    return false;
  }

  return true;
}

bool NNUELoader::read_network(std::istream &stream, NNUEWeights &weights) {
  // Read FC layers for each layer stack
  // Stockfish has 8 output buckets with shared feature transformer

  for (int stack = 0; stack < LAYER_STACKS; ++stack) {
    // FC0 hash
    uint32_t fc0_hash = read_le<uint32_t>(stream);
    (void)fc0_hash;

    // FC0 biases
    int fc0_bias_offset = stack * FC0_OUTPUTS;
    stream.read(reinterpret_cast<char *>(&weights.fc0_biases[fc0_bias_offset]),
                FC0_OUTPUTS * sizeof(FCBias));

    // FC0 weights
    int fc0_weight_size = weights.half_dimensions * 2 * FC0_OUTPUTS;
    int fc0_weight_offset = stack * fc0_weight_size;
    stream.read(
        reinterpret_cast<char *>(&weights.fc0_weights[fc0_weight_offset]),
        fc0_weight_size * sizeof(FCWeight));

    // FC1 hash
    uint32_t fc1_hash = read_le<uint32_t>(stream);
    (void)fc1_hash;

    // FC1 biases
    int fc1_bias_offset = stack * FC1_OUTPUTS;
    stream.read(reinterpret_cast<char *>(&weights.fc1_biases[fc1_bias_offset]),
                FC1_OUTPUTS * sizeof(FCBias));

    // FC1 weights
    int fc1_weight_size = FC1_INPUTS * FC1_OUTPUTS;
    int fc1_weight_offset = stack * fc1_weight_size;
    stream.read(
        reinterpret_cast<char *>(&weights.fc1_weights[fc1_weight_offset]),
        fc1_weight_size * sizeof(FCWeight));

    // FC2 hash
    uint32_t fc2_hash = read_le<uint32_t>(stream);
    (void)fc2_hash;

    // FC2 bias
    stream.read(reinterpret_cast<char *>(&weights.fc2_biases[stack]),
                sizeof(FCBias));

    // FC2 weights
    int fc2_weight_offset = stack * FC1_OUTPUTS;
    stream.read(
        reinterpret_cast<char *>(&weights.fc2_weights[fc2_weight_offset]),
        FC1_OUTPUTS * sizeof(FCWeight));
  }

  if (!stream.good() && !stream.eof()) {
    error_ = "Failed to read network layers";
    return false;
  }

  return true;
}

bool load_default_network(NNUEWeights &weights) {
  if (EmbeddedNNUEData == nullptr || EmbeddedNNUESize == 0) {
    return false;
  }

  NNUELoader loader;
  return loader.load_from_memory(EmbeddedNNUEData, EmbeddedNNUESize, weights);
}

bool load_network(const std::string &path, NNUEWeights &weights) {
  NNUELoader loader;
  bool success = loader.load_from_file(path, weights);
  if (!success) {
    std::cerr << "[NNUE] " << loader.error() << std::endl;
  }
  return success;
}

} // namespace NNUE
} // namespace Eval
} // namespace MetalFish
