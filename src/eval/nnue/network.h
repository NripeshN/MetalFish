/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "core/misc.h"
#include "core/types.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::Eval::NNUE {

class AccumulatorStack;
struct AccumulatorCaches;

using NetworkOutput = std::tuple<Value, Value>;

// The network must be a trivial type, i.e. the memory must be in-line.
// This is required to allow sharing the network via shared memory, as
// there is no way to run destructors.
class Network {
public:
  Network() = default;

  Network(const Network &other) = default;
  Network(Network &&other) = default;

  Network &operator=(const Network &other) = default;
  Network &operator=(Network &&other) = default;

  void load(const std::filesystem::path &rootDirectory,
            std::filesystem::path evalfilePath, EvalFile &evalFile);
  bool save(const EvalFile &evalFile,
            const std::optional<std::filesystem::path> &filename) const;

  usize get_content_hash() const;

  NetworkOutput evaluate(const Position &pos,
                         AccumulatorStack &accumulatorStack,
                         AccumulatorCaches &cache) const;

  void verify(const std::function<void(std::string_view)> &f,
              const EvalFile &evalFile,
              std::filesystem::path evalfilePath) const;

  NnueEvalTrace trace_evaluate(const Position &pos,
                               AccumulatorStack &accumulatorStack,
                               AccumulatorCaches &cache) const;

  void load_external(const std::filesystem::path &,
                     const std::filesystem::path &, EvalFile &);
  void load_internal(EvalFile &);

private:
  void initialize();

  bool save(std::ostream &, const std::string &) const;
  std::optional<std::string> load(std::istream &);

  bool read_header(std::istream &, u32 *, std::string *) const;
  bool write_header(std::ostream &, u32, const std::string &) const;

  bool read_parameters(std::istream &, std::string &);
  bool write_parameters(std::ostream &, const std::string &) const;

  // Input feature converter
  FeatureTransformer featureTransformer;

  // Evaluation function
  NetworkArchitecture network[LayerStacks];

  bool initialized = false;

  // Hash value of evaluation function structure
  static constexpr u32 hash = FeatureTransformer::get_hash_value() ^
                              NetworkArchitecture::get_hash_value();

  friend struct AccumulatorCaches;
};

} // namespace MetalFish::Eval::NNUE

template <> struct std::hash<MetalFish::Eval::NNUE::Network> {
  MetalFish::usize
  operator()(const MetalFish::Eval::NNUE::Network &network) const noexcept {
    return network.get_content_hash();
  }
};

#endif
