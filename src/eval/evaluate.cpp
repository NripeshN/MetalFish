/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "eval/evaluate.h"
#include "core/bitboard.h"
#include "eval/gpu_nnue.h"
#include "eval/nnue.h"
#include "eval/nnue_gpu.h"
#include <iostream>
#include <sstream>

namespace MetalFish {

namespace Eval {

namespace {

// Material piece values for classical evaluation (used as fallback)
constexpr int PieceValues[PIECE_TYPE_NB] = {0, 100, 320, 330, 500, 900, 20000};

// Piece-square tables for classical evaluation
constexpr int PawnTable[SQUARE_NB] = {
    0,  0,  0,  0,   0,   0,  0,  0,  50, 50, 50,  50, 50, 50,  50, 50,
    10, 10, 20, 30,  30,  20, 10, 10, 5,  5,  10,  25, 25, 10,  5,  5,
    0,  0,  0,  20,  20,  0,  0,  0,  5,  -5, -10, 0,  0,  -10, -5, 5,
    5,  10, 10, -20, -20, 10, 10, 5,  0,  0,  0,   0,  0,  0,   0,  0};

constexpr int KnightTable[SQUARE_NB] = {
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   0,   0,
    0,   -20, -40, -30, 0,   10,  15,  15,  10,  0,   -30, -30, 5,
    15,  20,  20,  15,  5,   -30, -30, 0,   15,  20,  20,  15,  0,
    -30, -30, 5,   10,  15,  15,  10,  5,   -30, -40, -20, 0,   5,
    5,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50};

constexpr int BishopTable[SQUARE_NB] = {
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0,   0,   0,   0,
    0,   0,   -10, -10, 0,   5,   10,  10,  5,   0,   -10, -10, 5,
    5,   10,  10,  5,   5,   -10, -10, 0,   10,  10,  10,  10,  0,
    -10, -10, 10,  10,  10,  10,  10,  10,  -10, -10, 5,   0,   0,
    0,   0,   5,   -10, -20, -10, -10, -10, -10, -10, -10, -20};

constexpr int RookTable[SQUARE_NB] = {
    0,  0, 0, 0, 0, 0, 0, 0,  5,  10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5, -5, 0,  0,  0,  0,  0,  0,  -5,
    -5, 0, 0, 0, 0, 0, 0, -5, -5, 0,  0,  0,  0,  0,  0,  -5,
    -5, 0, 0, 0, 0, 0, 0, -5, 0,  0,  0,  5,  5,  0,  0,  0};

constexpr int QueenTable[SQUARE_NB] = {
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0,   0,   0,  0,  0,   0,   -10,
    -10, 0,   5,   5,  5,  5,   0,   -10, -5,  0,   5,   5,  5,  5,   0,   -5,
    0,   0,   5,   5,  5,  5,   0,   -5,  -10, 5,   5,   5,  5,  5,   0,   -10,
    -10, 0,   5,   0,  0,  0,   0,   -10, -20, -10, -10, -5, -5, -10, -10, -20};

constexpr int KingMiddleGameTable[SQUARE_NB] = {
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50,
    -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
    -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30,
    -20, -10, -20, -20, -20, -20, -20, -20, -10, 20,  20,  0,   0,
    0,   0,   20,  20,  20,  30,  10,  0,   0,   10,  30,  20};

// Get piece-square value
int psq_value(Piece pc, Square s) {
  Color c = color_of(pc);
  PieceType pt = type_of(pc);
  Square rs = c == WHITE ? s : flip_rank(s);

  switch (pt) {
  case PAWN:
    return PawnTable[rs];
  case KNIGHT:
    return KnightTable[rs];
  case BISHOP:
    return BishopTable[rs];
  case ROOK:
    return RookTable[rs];
  case QUEEN:
    return QueenTable[rs];
  case KING:
    return KingMiddleGameTable[rs];
  default:
    return 0;
  }
}

// Classical evaluation (fallback when NNUE is not loaded)
Value classical_eval(const Position &pos) {
  Value score = VALUE_ZERO;

  // Material and piece-square tables
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece pc = pos.piece_on(s);
    if (pc == NO_PIECE)
      continue;

    Value pieceValue = PieceValues[type_of(pc)] + psq_value(pc, s);
    score += (color_of(pc) == WHITE) ? pieceValue : -pieceValue;
  }

  // Mobility bonus
  Color us = pos.side_to_move();
  Bitboard occupied = pos.pieces();

  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece pc = pos.piece_on(s);
    if (pc == NO_PIECE)
      continue;

    Color c = color_of(pc);
    PieceType pt = type_of(pc);

    if (pt == KNIGHT || pt == BISHOP || pt == ROOK || pt == QUEEN) {
      Bitboard attacks = attacks_bb(pt, s, occupied);
      int mobility = popcount(attacks & ~pos.pieces(c));
      Value bonus = Value(mobility * (pt == QUEEN ? 1 : (pt == ROOK ? 2 : 3)));
      score += (c == WHITE) ? bonus : -bonus;
    }
  }

  // King safety (simplified)
  Square wKing = pos.square<KING>(WHITE);
  Square bKing = pos.square<KING>(BLACK);

  Bitboard wKingZone = KingAttacks[wKing];
  Bitboard bKingZone = KingAttacks[bKing];

  int wKingAttackers = popcount(pos.pieces(BLACK, QUEEN, ROOK) & wKingZone);
  int bKingAttackers = popcount(pos.pieces(WHITE, QUEEN, ROOK) & bKingZone);

  score -= wKingAttackers * 50;
  score += bKingAttackers * 50;

  // Tempo
  score += (us == WHITE) ? 15 : -15;

  return us == WHITE ? score : -score;
}

} // anonymous namespace

void init() {
  // Initialize NNUE if available
  NNUE::init();

  // Initialize GPU NNUE evaluator (the primary evaluator)
  auto &gpu = gpu_nnue();
  if (gpu.is_ready()) {
    std::cout << "[Eval] GPU NNUE acceleration ready (unified memory)"
              << std::endl;
  }
}

bool load_network(const std::string &path) {
  return NNUE::network && NNUE::network->load(path);
}

bool is_network_loaded() { return NNUE::network && NNUE::network->is_loaded(); }

std::string network_info() {
  return NNUE::network ? NNUE::network->info() : "No network loaded";
}

Value evaluate(const Position &pos) {
  Value v;

  // For single-position evaluation during search, classical eval is faster
  // due to GPU command buffer overhead. GPU is used for batch evaluation.

  // Use CPU NNUE if loaded
  if (NNUE::network && NNUE::network->is_loaded()) {
    v = NNUE::evaluate(pos);
  } else {
    // Classical evaluation (fast for single positions)
    v = classical_eval(pos);
  }

  // Rule 50 dampening: linearly reduce eval as 50-move rule approaches
  // This helps avoid draws when winning and correctly assess draw-ish positions
  v -= v * pos.rule50_count() / 199;

  return v;
}

// GPU-accelerated evaluation - use for batch processing
Value evaluate_gpu(const Position &pos) {
  auto &gpu = gpu_nnue();
  if (gpu.is_ready()) {
    return gpu.evaluate(pos);
  }
  return evaluate(pos);
}

void batch_evaluate(const Position *positions, Value *scores, size_t count) {
  // Try GPU batch evaluation first
  auto &gpu_nnue = NNUE::get_gpu_nnue();
  if (gpu_nnue.is_gpu_available()) {
    std::vector<const Position *> pos_ptrs(count);
    for (size_t i = 0; i < count; ++i) {
      pos_ptrs[i] = &positions[i];
    }
    std::vector<Value> results;
    gpu_nnue.evaluate_batch(pos_ptrs, results);
    for (size_t i = 0; i < count; ++i) {
      scores[i] = results[i];
    }
  } else {
    // Fallback to sequential evaluation
    for (size_t i = 0; i < count; ++i) {
      scores[i] = evaluate(positions[i]);
    }
  }
}

std::string trace(const Position &pos) {
  std::ostringstream ss;
  ss << "Evaluation: " << evaluate(pos) << "\n";

  if (is_network_loaded()) {
    ss << "Using NNUE: " << network_info() << "\n";
  } else {
    ss << "Using classical evaluation\n";
  }

  return ss.str();
}

} // namespace Eval

} // namespace MetalFish
