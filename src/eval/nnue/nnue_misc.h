/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef NNUE_MISC_H_INCLUDED
#define NNUE_MISC_H_INCLUDED

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "../evaluate.h"
#include "core/misc.h"
#include "core/types.h"
#include "nnue_architecture.h"

namespace MetalFish {

class Position;

namespace Eval::NNUE {

// NNUE file metadata uses fixed string types so it stays trivially copyable and
// cheap to move around between the engine and the network loader.
struct EvalFile {
  // Default net name, will use one of the EvalFileDefaultName* macros defined
  // in evaluate.h
  constexpr static std::string_view defaultName = EvalFileDefaultName;
  // Selected net name, either via uci option or default
  std::optional<std::filesystem::path> current;
  // Net description extracted from the net file
  std::string netDescription;
};

struct NnueEvalTrace {
  static_assert(LayerStacks == PSQTBuckets);

  Value psqt[LayerStacks];
  Value positional[LayerStacks];
  usize correctBucket;
};

class Network;
struct AccumulatorCaches;

std::string trace(Position &pos, const Network &network,
                  AccumulatorCaches &caches);

} // namespace Eval::NNUE
} // namespace MetalFish

#endif // #ifndef NNUE_MISC_H_INCLUDED
