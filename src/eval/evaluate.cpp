/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "eval/evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "core/position.h"
#include "core/types.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_accumulator.h"
#include "eval/nnue/nnue_misc.h"
#include "uci/uci.h"

namespace MetalFish {

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the side to move. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position &pos) {
  Color c = pos.side_to_move();
  return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c)) +
         pos.non_pawn_material(c) - pos.non_pawn_material(~c);
}

bool Eval::use_smallnet(const Position &pos) {
  return std::abs(simple_eval(pos)) > 962;
}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks &networks, const Position &pos,
                     Eval::NNUE::AccumulatorStack &accumulators,
                     Eval::NNUE::AccumulatorCaches &caches, int optimism) {

  assert(!pos.checkers());

  bool smallNet = use_smallnet(pos);
  auto [psqt, positional] =
      smallNet ? networks.small.evaluate(pos, accumulators, caches.small)
               : networks.big.evaluate(pos, accumulators, caches.big);

  Value nnue = (125 * psqt + 131 * positional) / 128;

  // Re-evaluate the position when higher eval accuracy is worth the time spent
  if (smallNet && (std::abs(nnue) < 277)) {
    std::tie(psqt, positional) =
        networks.big.evaluate(pos, accumulators, caches.big);
    nnue = (125 * psqt + 131 * positional) / 128;
    smallNet = false;
  }

  // Blend optimism and eval with nnue complexity
  int nnueComplexity = std::abs(psqt - positional);
  optimism += optimism * nnueComplexity / 476;
  nnue -= nnue * nnueComplexity / 18236;

  int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
  int v = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;

  // Damp down the evaluation linearly when shuffling
  v -= v * pos.rule50_count() / 199;

  // Guarantee evaluation does not hit the tablebase range
  v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

  return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position &pos, const Eval::NNUE::Networks &networks) {

  if (pos.checkers())
    return "Final evaluation: none (in check)";

  auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
  auto caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);

  std::stringstream ss;
  ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
  ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

  ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2)
     << std::setw(15);

  auto [psqt, positional] =
      networks.big.evaluate(pos, *accumulators, caches->big);
  Value v = psqt + positional;
  v = pos.side_to_move() == WHITE ? v : -v;
  ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos)
     << " (white side)\n";

  v = evaluate(networks, pos, *accumulators, *caches, VALUE_ZERO);
  v = pos.side_to_move() == WHITE ? v : -v;
  ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos)
     << " (white side)";
  ss << " [with scaled NNUE, ...]";
  ss << "\n";

  return ss.str();
}

} // namespace MetalFish
