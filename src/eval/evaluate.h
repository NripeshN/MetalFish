/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include <string>

#include "core/types.h"

namespace MetalFish {

class Position;

namespace Eval {

// The default net name MUST follow the format nn-[SHA256 first 12 digits].nnue
// for the build process (profile-build and fishtest) to work. Do not change the
// name of the macro or the location where this macro is defined, as it is used
// in the Makefile/Fishtest.
#define EvalFileDefaultName "nn-1a298aa575a0.nnue"

namespace NNUE {
class Network;
struct AccumulatorCaches;
class AccumulatorStack;
} // namespace NNUE

std::string trace(Position &pos, const Eval::NNUE::Network &network);

Value evaluate(const NNUE::Network &network, const Position &pos,
               Eval::NNUE::AccumulatorStack &accumulators,
               Eval::NNUE::AccumulatorCaches &caches, int optimism);

// NNUE is CPU-only. These legacy accessors intentionally keep the old API
// surface inert so external callers cannot enable GPU NNUE.
void set_use_apple_silicon_nnue(bool use);
bool use_apple_silicon_nnue();
} // namespace Eval

} // namespace MetalFish

#endif // #ifndef EVALUATE_H_INCLUDED
