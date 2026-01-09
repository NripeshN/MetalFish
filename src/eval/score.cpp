/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "eval/score.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "uci/uci.h"

namespace MetalFish {

Score::Score(Value v, const Position &pos) {
  assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);

  if (!is_decisive(v)) {
    score = InternalUnits{UCIEngine::to_cp(v, pos)};
  } else if (std::abs(v) <= VALUE_TB) {
    auto distance = VALUE_TB - std::abs(v);
    score = (v > 0) ? Tablebase{distance, true} : Tablebase{-distance, false};
  } else {
    auto distance = VALUE_MATE - std::abs(v);
    score = (v > 0) ? Mate{distance} : Mate{-distance};
  }
}

} // namespace MetalFish