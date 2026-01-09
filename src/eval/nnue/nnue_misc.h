/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef NNUE_MISC_H_INCLUDED
#define NNUE_MISC_H_INCLUDED

#include <cstddef>
#include <memory>
#include <string>

#include "core/misc.h"
#include "core/types.h"
#include "nnue_architecture.h"

namespace MetalFish {

class Position;

namespace Eval::NNUE {

// EvalFile uses fixed string types because it's part of the network structure
// which must be trivial.
struct EvalFile {
  // Default net name, will use one of the EvalFileDefaultName* macros defined
  // in evaluate.h
  FixedString<256> defaultName;
  // Selected net name, either via uci option or default
  FixedString<256> current;
  // Net description extracted from the net file
  FixedString<256> netDescription;
};

struct NnueEvalTrace {
  static_assert(LayerStacks == PSQTBuckets);

  Value psqt[LayerStacks];
  Value positional[LayerStacks];
  std::size_t correctBucket;
};

struct Networks;
struct AccumulatorCaches;

std::string trace(Position &pos, const Networks &networks,
                  AccumulatorCaches &caches);

} // namespace Eval::NNUE
} // namespace MetalFish

template <> struct std::hash<MetalFish::Eval::NNUE::EvalFile> {
  std::size_t
  operator()(const MetalFish::Eval::NNUE::EvalFile &evalFile) const noexcept {
    std::size_t h = 0;
    MetalFish::hash_combine(h, evalFile.defaultName);
    MetalFish::hash_combine(h, evalFile.current);
    MetalFish::hash_combine(h, evalFile.netDescription);
    return h;
  }
};

#endif // #ifndef NNUE_MISC_H_INCLUDED
