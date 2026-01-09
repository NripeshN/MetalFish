/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "core/misc.h"
#include "core/types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::Eval::NNUE {

enum class EmbeddedNNUEType {
    BIG,
    SMALL,
};

using NetworkOutput = std::tuple<Value, Value>;

// The network must be a trivial type, i.e. the memory must be in-line.
// This is required to allow sharing the network via shared memory, as
// there is no way to run destructors.
template<typename Arch, typename Transformer>
class Network {
    static constexpr IndexType FTDimensions = Arch::TransformedFeatureDimensions;

   public:
    Network(EvalFile file, EmbeddedNNUEType type) :
        evalFile(file),
        embeddedType(type) {}

    Network(const Network& other) = default;
    Network(Network&& other)      = default;

    Network& operator=(const Network& other) = default;
    Network& operator=(Network&& other)      = default;

    void load(const std::string& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    std::size_t get_content_hash() const;

    NetworkOutput evaluate(const Position&                         pos,
                           AccumulatorStack&                       accumulatorStack,
                           AccumulatorCaches::Cache<FTDimensions>& cache) const;


    void verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const;
    NnueEvalTrace trace_evaluate(const Position&                         pos,
                                 AccumulatorStack&                       accumulatorStack,
                                 AccumulatorCaches::Cache<FTDimensions>& cache) const;

   private:
    void load_user_net(const std::string&, const std::string&);
    void load_internal();

    void initialize();

    bool                       save(std::ostream&, const std::string&, const std::string&) const;
    std::optional<std::string> load(std::istream&);

    bool read_header(std::istream&, std::uint32_t*, std::string*) const;
    bool write_header(std::ostream&, std::uint32_t, const std::string&) const;

    bool read_parameters(std::istream&, std::string&);
    bool write_parameters(std::ostream&, const std::string&) const;

    // Input feature converter
    Transformer featureTransformer;

    // Evaluation function
    Arch network[LayerStacks];

    EvalFile         evalFile;
    EmbeddedNNUEType embeddedType;

    bool initialized = false;

    // Hash value of evaluation function structure
    static constexpr std::uint32_t hash = Transformer::get_hash_value() ^ Arch::get_hash_value();

    template<IndexType Size>
    friend struct AccumulatorCaches::Cache;
};

// Definitions of the network types
using SmallFeatureTransformer = FeatureTransformer<TransformedFeatureDimensionsSmall>;
using SmallNetworkArchitecture =
  NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>;

using BigFeatureTransformer  = FeatureTransformer<TransformedFeatureDimensionsBig>;
using BigNetworkArchitecture = NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>;

using NetworkBig   = Network<BigNetworkArchitecture, BigFeatureTransformer>;
using NetworkSmall = Network<SmallNetworkArchitecture, SmallFeatureTransformer>;


struct Networks {
    Networks(EvalFile bigFile, EvalFile smallFile) :
        big(bigFile, EmbeddedNNUEType::BIG),
        small(smallFile, EmbeddedNNUEType::SMALL) {}

    NetworkBig   big;
    NetworkSmall small;
};


}  // namespace MetalFish

template<typename ArchT, typename FeatureTransformerT>
struct std::hash<MetalFish::Eval::NNUE::Network<ArchT, FeatureTransformerT>> {
    std::size_t operator()(
      const MetalFish::Eval::NNUE::Network<ArchT, FeatureTransformerT>& network) const noexcept {
        return network.get_content_hash();
    }
};

template<>
struct std::hash<MetalFish::Eval::NNUE::Networks> {
    std::size_t operator()(const MetalFish::Eval::NNUE::Networks& networks) const noexcept {
        std::size_t h = 0;
        MetalFish::hash_combine(h, networks.big);
        MetalFish::hash_combine(h, networks.small);
        return h;
    }
};

#endif
