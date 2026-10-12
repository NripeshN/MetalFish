/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "core/bitboard.h"

namespace MetalFish {

// Returns an ASCII representation of a bitboard suitable
// to be printed to standard output. Useful for debugging.
std::string Bitboards::pretty(Bitboard b) {

  std::string s = "+---+---+---+---+---+---+---+---+\n";

  for (Rank r = RANK_8;; --r) {
    for (File f = FILE_A; f <= FILE_H; ++f)
      s += b & make_square(f, r) ? "| X " : "|   ";

    s += "| " + std::to_string(1 + r) + "\n+---+---+---+---+---+---+---+---+\n";

    if (r == RANK_1)
      break;
  }
  s += "  a   b   c   d   e   f   g   h\n";

  return s;
}

} // namespace MetalFish
