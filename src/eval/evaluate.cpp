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

#include "core/misc.h"
#include "core/position.h"
#include "core/types.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_accumulator.h"
#include "eval/nnue/nnue_misc.h"
#include "uci/uci.h"

namespace MetalFish {

void Eval::set_use_apple_silicon_nnue(bool) {}

bool Eval::use_apple_silicon_nnue() { return false; }

// Always uses CPU NNUE with incremental accumulator updates. On Apple Silicon
// this keeps AB latency low and leaves the GPU free for transformer MCTS.
// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Network &network, const Position &pos,
                     Eval::NNUE::AccumulatorStack &accumulators,
                     Eval::NNUE::AccumulatorCaches &caches, int optimism) {

  assert(!pos.checkers());

  auto [psqt, positional] = network.evaluate(pos, accumulators, caches);

  Value nnue = psqt + positional;

  // Blend optimism and eval with nnue complexity
  int nnueComplexity = std::abs(psqt - positional);
  optimism += optimism * i64(nnueComplexity) / 476;
  nnue -= nnue * i64(nnueComplexity) / 18236;

  int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
  int v = nnue + (nnue * i64(material) + optimism * i64(7675)) / 91000;

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
std::string Eval::trace(Position &pos, const Eval::NNUE::Network &network) {

  if (pos.checkers())
    return "Final evaluation: none (in check)";

  auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
  auto caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(network);

  std::stringstream ss;
  ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
  ss << '\n' << NNUE::trace(pos, network, *caches) << '\n';

  ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2)
     << std::setw(15);

  auto [psqt, positional] = network.evaluate(pos, *accumulators, *caches);
  Value v = psqt + positional;
  ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
  v = pos.side_to_move() == WHITE ? v : -v;
  ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos)
     << " (white side)\n";

  v = evaluate(network, pos, *accumulators, *caches, VALUE_ZERO);
  v = pos.side_to_move() == WHITE ? v : -v;

  ss << "Final evaluation      ";
  ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
  ss << " [with scaled NNUE, ...]\n";

  return ss.str();
}

} // namespace MetalFish
