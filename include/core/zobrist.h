/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

*/

#pragma once

#include "types.h"
#include <array>

namespace MetalFish {

namespace Zobrist {

// Zobrist key arrays
extern Key psq[PIECE_NB][SQUARE_NB];
extern Key enpassant[FILE_NB];
extern Key castling[CASTLING_RIGHT_NB];
extern Key side;
extern Key noPawns;

void init();

} // namespace Zobrist

} // namespace MetalFish
