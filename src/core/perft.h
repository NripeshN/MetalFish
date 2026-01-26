/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file perft.h
 * @brief MetalFish source file.
 */

  Licensed under GPL-3.0
*/

#ifndef PERFT_H_INCLUDED
#define PERFT_H_INCLUDED

#include <cstdint>

#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "uci/uci.h"

namespace MetalFish::Benchmark {

// Utility to verify move generation. All the leaf nodes up
// to the given depth are generated and counted, and the sum is returned.
template <bool Root> uint64_t perft(Position &pos, Depth depth) {

  StateInfo st;

  uint64_t cnt, nodes = 0;
  const bool leaf = (depth == 2);

  for (const auto &m : MoveList<LEGAL>(pos)) {
    if (Root && depth <= 1)
      cnt = 1, nodes++;
    else {
      pos.do_move(m, st);
      cnt = leaf ? MoveList<LEGAL>(pos).size() : perft<false>(pos, depth - 1);
      nodes += cnt;
      pos.undo_move(m);
    }
    if (Root)
      sync_cout << UCIEngine::move(m, pos.is_chess960()) << ": " << cnt
                << sync_endl;
  }
  return nodes;
}

inline uint64_t perft(const std::string &fen, Depth depth, bool isChess960) {
  StateInfo st;
  Position p;
  p.set(fen, isChess960, &st);

  return perft<true>(p, depth);
}
} // namespace MetalFish::Benchmark

#endif // PERFT_H_INCLUDED