/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "core/zobrist.h"
#include "core/bitboard.h"
#include <random>

namespace MetalFish {

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side;
Key noPawns;

void init() {
  // Use a fixed seed for reproducibility
  std::mt19937_64 rng(1070372ULL);

  auto rand64 = [&rng]() { return rng(); };

  for (int pc = NO_PIECE; pc < PIECE_NB; ++pc) {
    for (int sq = SQ_A1; sq < SQUARE_NB; ++sq) {
      psq[pc][sq] = rand64();
    }
  }

  for (int f = FILE_A; f < FILE_NB; ++f) {
    enpassant[f] = rand64();
  }

  for (int cr = NO_CASTLING; cr < CASTLING_RIGHT_NB; ++cr) {
    castling[cr] = 0;
    Bitboard b = cr;
    while (b) {
      Key k = castling[1ULL << pop_lsb(b)];
      castling[cr] ^= k ? k : rand64();
    }
  }

  side = rand64();
  noPawns = rand64();
}

} // namespace Zobrist

} // namespace MetalFish
