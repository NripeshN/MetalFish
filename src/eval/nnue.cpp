/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

*/

#include "eval/nnue.h"
#include "core/bitboard.h"
#include <cstring>
#include <fstream>
#include <iostream>

namespace MetalFish {

namespace NNUE {

std::unique_ptr<Network> network;

// Feature index calculation
int feature_index(Square ksq, Square psq, Piece pc, Color perspective) {
  // Simplified feature index calculation
  // In a full implementation, this would use king buckets and piece indices
  int pcIndex = type_of(pc) - 1 + (color_of(pc) == perspective ? 0 : 6);
  if (pcIndex < 0 || pcIndex > 11)
    return -1; // Skip kings

  Square orientedKsq = perspective == WHITE ? ksq : flip_rank(ksq);
  Square orientedPsq = perspective == WHITE ? psq : flip_rank(psq);

  int kingBucket = orientedKsq / 8; // Simple king bucket based on rank

  return kingBucket * 64 * 11 + pcIndex * 64 + orientedPsq;
}

Network::Network() {
  // Constructor - buffers will be allocated when network is loaded
}

Network::~Network() {
  // Destructor - Metal resources will be freed automatically by Buffer
  // destructor
  if (featureTransformKernel)
    featureTransformKernel->release();
  if (affineTransformKernel)
    affineTransformKernel->release();
  if (clippedReluKernel)
    clippedReluKernel->release();
}

bool Network::load(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open NNUE network file: " << path << std::endl;
    return false;
  }

  // Read and validate header
  uint32_t magic;
  file.read(reinterpret_cast<char *>(&magic), sizeof(magic));

  // For now, we'll use a simplified network structure
  // A full implementation would parse the Stockfish NNUE format

  file.close();

  // Mark as loaded for classical fallback
  // Full NNUE GPU implementation would allocate Metal buffers here
  networkInfo = "Classical Evaluation (NNUE GPU implementation pending)";
  loaded = false; // Set to true when GPU NNUE is fully implemented

  return loaded;
}

bool Network::load_from_embedded() {
  // Embedded networks not supported - use load() with file path
  return false;
}

void Network::allocate_buffers() {
  auto &allocator = Metal::get_allocator();

  // Allocate buffers for network weights
  // Sizes based on Stockfish's NNUE architecture
  featureWeights = allocator.malloc(FT_IN_DIMS * FT_OUT_DIMS * sizeof(int16_t));
  featureBiases = allocator.malloc(FT_OUT_DIMS * sizeof(int16_t));

  l1Weights = allocator.malloc(L1_SIZE * L2_SIZE * sizeof(int8_t));
  l1Biases = allocator.malloc(L2_SIZE * sizeof(int32_t));

  l2Weights = allocator.malloc(L2_SIZE * L3_SIZE * sizeof(int8_t));
  l2Biases = allocator.malloc(L3_SIZE * sizeof(int32_t));

  outputWeights = allocator.malloc(L3_SIZE * sizeof(int8_t));
  outputBias = allocator.malloc(sizeof(int32_t));

  // Working buffers
  inputBuffer = allocator.malloc(FT_IN_DIMS * sizeof(int8_t));
  accumulatorBuffer = allocator.malloc(2 * HALF_DIMENSIONS * sizeof(int16_t));
  outputBuffer = allocator.malloc(sizeof(int32_t));
}

void Network::compile_kernels() {
  // This would compile Metal compute kernels for NNUE inference
  // The actual kernel code is in shaders/nnue.metal
}

Value Network::evaluate(const Position &pos, Accumulator &acc) {
  // GPU-accelerated NNUE - when network not loaded, returns 0
  // Classical eval is used as primary evaluation in evaluate.cpp
  (void)pos;
  (void)acc;
  return VALUE_ZERO;
}

void Network::batch_evaluate(const Position *positions,
                             Accumulator *accumulators, Value *outputs,
                             size_t count) {
  // GPU batch evaluation
  for (size_t i = 0; i < count; ++i) {
    outputs[i] = evaluate(positions[i], accumulators[i]);
  }
}

void Network::update_accumulator(const Position &pos, Accumulator &acc) {
  // Incremental accumulator update
  (void)pos;
  acc.computed[WHITE] = acc.computed[BLACK] = false;
}

void Network::refresh_accumulator(const Position &pos, Accumulator &acc) {
  // Full accumulator refresh from scratch
  acc.reset();

  // Would compute accumulator from position features here
  (void)pos;
}

void init() { network = std::make_unique<Network>(); }

Value evaluate(const Position &pos) {
  if (network && network->is_loaded()) {
    Accumulator acc;
    return network->evaluate(pos, acc);
  }

  // Classical evaluation fallback
  Value score = VALUE_ZERO;

  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece pc = pos.piece_on(s);
    if (pc == NO_PIECE)
      continue;

    Value value = PieceValue[pc];
    score += (color_of(pc) == pos.side_to_move()) ? value : -value;
  }

  return score;
}

} // namespace NNUE

} // namespace MetalFish
