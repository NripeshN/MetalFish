/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

*/

#pragma once

#include "types.h"
#include <algorithm>
#include <cstddef>

namespace MetalFish {

class Position;

enum GenType { CAPTURES, QUIETS, EVASIONS, NON_EVASIONS, LEGAL };

struct ExtMove : public Move {
  int value;

  void operator=(Move m) { static_cast<Move &>(*this) = m; }

  operator float() const = delete;
};

inline bool operator<(const ExtMove &f, const ExtMove &s) {
  return f.value < s.value;
}

template <GenType> Move *generate(const Position &pos, Move *moveList);

// Explicit template declarations
template <> Move *generate<CAPTURES>(const Position &pos, Move *moveList);
template <> Move *generate<QUIETS>(const Position &pos, Move *moveList);
template <> Move *generate<EVASIONS>(const Position &pos, Move *moveList);
template <> Move *generate<NON_EVASIONS>(const Position &pos, Move *moveList);
template <> Move *generate<LEGAL>(const Position &pos, Move *moveList);

// MoveList wraps generate() for convenient iteration
template <GenType T> struct MoveList {
  explicit MoveList(const Position &pos) : last(generate<T>(pos, moveList)) {}

  const Move *begin() const { return moveList; }
  const Move *end() const { return last; }
  size_t size() const { return last - moveList; }
  bool contains(Move move) const {
    return std::find(begin(), end(), move) != end();
  }

private:
  Move moveList[MAX_MOVES], *last;
};

} // namespace MetalFish
