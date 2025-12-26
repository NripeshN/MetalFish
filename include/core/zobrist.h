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
