/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  NNUE Weight Accessor for GPU

  This header provides read-only access to NNUE network weights
  for GPU upload. It uses template metaprogramming to extract
  weights without modifying the core NNUE classes.
*/

#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "eval/nnue/network.h"
#include "eval/nnue/nnue_architecture.h"
#include "eval/nnue/nnue_feature_transformer.h"

namespace MetalFish::GPU {

// Weight accessor that provides read-only access to NNUE weights
// This works by accessing the public members through template specialization

template <typename Network> struct NNUEWeightAccessor;

// Specialization for Big Network
template <> struct NNUEWeightAccessor<Eval::NNUE::NetworkBig> {
  using Network = Eval::NNUE::NetworkBig;
  using FT = Eval::NNUE::BigFeatureTransformer;
  using Arch = Eval::NNUE::BigNetworkArchitecture;

  static constexpr int HiddenDim = Eval::NNUE::TransformedFeatureDimensionsBig;
  static constexpr int FC0Out = Eval::NNUE::L2Big;
  static constexpr int FC1Out = Eval::NNUE::L3Big;
  static constexpr int LayerStacks = Eval::NNUE::LayerStacks;
  static constexpr int PSQTBuckets = Eval::NNUE::PSQTBuckets;
  static constexpr bool HasThreats = true;

  // Feature transformer input dimensions
  static constexpr int FT_InputDims = FT::InputDimensions;
  static constexpr int FT_ThreatDims = FT::ThreatInputDimensions;
};

// Specialization for Small Network
template <> struct NNUEWeightAccessor<Eval::NNUE::NetworkSmall> {
  using Network = Eval::NNUE::NetworkSmall;
  using FT = Eval::NNUE::SmallFeatureTransformer;
  using Arch = Eval::NNUE::SmallNetworkArchitecture;

  static constexpr int HiddenDim =
      Eval::NNUE::TransformedFeatureDimensionsSmall;
  static constexpr int FC0Out = Eval::NNUE::L2Small;
  static constexpr int FC1Out = Eval::NNUE::L3Small;
  static constexpr int LayerStacks = Eval::NNUE::LayerStacks;
  static constexpr int PSQTBuckets = Eval::NNUE::PSQTBuckets;
  static constexpr bool HasThreats = false;

  static constexpr int FT_InputDims = FT::InputDimensions;
  static constexpr int FT_ThreatDims = 0;
};

// Structure to hold extracted weights for GPU upload
struct ExtractedFTWeights {
  const int16_t *biases = nullptr;
  const int16_t *weights = nullptr;
  const int32_t *psqt_weights = nullptr;
  const int8_t *threat_weights = nullptr; // Only for big network
  const int32_t *threat_psqt = nullptr;

  size_t biases_size = 0;
  size_t weights_size = 0;
  size_t psqt_size = 0;
  size_t threat_weights_size = 0;
  size_t threat_psqt_size = 0;
};

struct ExtractedLayerWeights {
  const int8_t *fc0_weights = nullptr;
  const int32_t *fc0_biases = nullptr;
  const int8_t *fc1_weights = nullptr;
  const int32_t *fc1_biases = nullptr;
  const int8_t *fc2_weights = nullptr;
  const int32_t *fc2_biases = nullptr;

  size_t fc0_weights_size = 0;
  size_t fc0_biases_size = 0;
  size_t fc1_weights_size = 0;
  size_t fc1_biases_size = 0;
  size_t fc2_weights_size = 0;
  size_t fc2_biases_size = 0;
};

struct ExtractedNetworkWeights {
  ExtractedFTWeights ft;
  std::array<ExtractedLayerWeights, 8> layers;
  int hidden_dim = 0;
  int fc0_out = 0;
  int fc1_out = 0;
  bool has_threats = false;
  bool valid = false;
};

template <typename Network> class GPUNNUEWeightExtractor {
public:
  using Accessor = NNUEWeightAccessor<Network>;

  static ExtractedNetworkWeights extract(const Network &network) {
    ExtractedNetworkWeights result;
    result.hidden_dim = Accessor::HiddenDim;
    result.fc0_out = Accessor::FC0Out;
    result.fc1_out = Accessor::FC1Out;
    result.has_threats = Accessor::HasThreats;

    // Extract feature transformer weights
    result.ft.biases = network.featureTransformer.biases.data();
    result.ft.biases_size =
        network.featureTransformer.biases.size() * sizeof(int16_t);

    result.ft.weights = network.featureTransformer.weights.data();
    result.ft.weights_size =
        network.featureTransformer.weights.size() * sizeof(int16_t);

    result.ft.psqt_weights = network.featureTransformer.psqtWeights.data();
    result.ft.psqt_size =
        network.featureTransformer.psqtWeights.size() * sizeof(int32_t);

    // Extract threat weights if applicable
    if constexpr (Accessor::HasThreats) {
      if (network.featureTransformer.threatWeights.size() > 0) {
        result.ft.threat_weights =
            network.featureTransformer.threatWeights.data();
        result.ft.threat_weights_size =
            network.featureTransformer.threatWeights.size() * sizeof(int8_t);
      }
      if (network.featureTransformer.threatPsqtWeights.size() > 0) {
        result.ft.threat_psqt =
            network.featureTransformer.threatPsqtWeights.data();
        result.ft.threat_psqt_size =
            network.featureTransformer.threatPsqtWeights.size() *
            sizeof(int32_t);
      }
    }

    // Extract layer weights for each bucket
    for (int i = 0; i < Accessor::LayerStacks; i++) {
      auto &layer = result.layers[i];
      const auto &arch = network.network[i];

      // FC0 layer
      layer.fc0_weights = arch.fc_0.weights;
      layer.fc0_weights_size = sizeof(arch.fc_0.weights);
      layer.fc0_biases = arch.fc_0.biases;
      layer.fc0_biases_size = sizeof(arch.fc_0.biases);

      // FC1 layer
      layer.fc1_weights = arch.fc_1.weights;
      layer.fc1_weights_size = sizeof(arch.fc_1.weights);
      layer.fc1_biases = arch.fc_1.biases;
      layer.fc1_biases_size = sizeof(arch.fc_1.biases);

      // FC2 layer
      layer.fc2_weights = arch.fc_2.weights;
      layer.fc2_weights_size = sizeof(arch.fc_2.weights);
      layer.fc2_biases = arch.fc_2.biases;
      layer.fc2_biases_size = sizeof(arch.fc_2.biases);
    }

    result.valid = true;
    return result;
  }
};

template <typename Network> ExtractedNetworkWeights get_network_info() {
  using Accessor = NNUEWeightAccessor<Network>;

  ExtractedNetworkWeights result;
  result.hidden_dim = Accessor::HiddenDim;
  result.fc0_out = Accessor::FC0Out;
  result.fc1_out = Accessor::FC1Out;
  result.has_threats = Accessor::HasThreats;

  // Calculate sizes
  result.ft.biases_size = Accessor::HiddenDim * sizeof(int16_t);
  result.ft.weights_size =
      Accessor::FT_InputDims * Accessor::HiddenDim * sizeof(int16_t);
  result.ft.psqt_size =
      Accessor::FT_InputDims * Accessor::PSQTBuckets * sizeof(int32_t);

  if (Accessor::HasThreats) {
    result.ft.threat_weights_size =
        Accessor::FT_ThreatDims * Accessor::HiddenDim * sizeof(int8_t);
    result.ft.threat_psqt_size =
        Accessor::FT_ThreatDims * Accessor::PSQTBuckets * sizeof(int32_t);
  }

  // Layer sizes (same for all buckets)
  for (int i = 0; i < Accessor::LayerStacks; i++) {
    auto &layer = result.layers[i];

    // FC0: hidden_dim*2 inputs (both perspectives), FC0Out+1 outputs
    // Weights are stored transposed for sparse input optimization
    layer.fc0_weights_size =
        Accessor::HiddenDim * (Accessor::FC0Out + 1) * sizeof(int8_t);
    layer.fc0_biases_size = (Accessor::FC0Out + 1) * sizeof(int32_t);

    // FC1: FC0Out*2 inputs, FC1Out outputs
    layer.fc1_weights_size =
        Accessor::FC0Out * 2 * Accessor::FC1Out * sizeof(int8_t);
    layer.fc1_biases_size = Accessor::FC1Out * sizeof(int32_t);

    // FC2: FC1Out inputs, 1 output
    layer.fc2_weights_size = Accessor::FC1Out * sizeof(int8_t);
    layer.fc2_biases_size = sizeof(int32_t);
  }

  return result;
}

// Get total memory required for a network
template <typename Network> size_t get_network_memory_requirement() {
  auto info = get_network_info<Network>();

  size_t total = info.ft.biases_size + info.ft.weights_size + info.ft.psqt_size;
  total += info.ft.threat_weights_size + info.ft.threat_psqt_size;

  for (const auto &layer : info.layers) {
    total += layer.fc0_weights_size + layer.fc0_biases_size;
    total += layer.fc1_weights_size + layer.fc1_biases_size;
    total += layer.fc2_weights_size + layer.fc2_biases_size;
  }

  return total;
}

// Print network information (to stderr to avoid interfering with UCI protocol)
inline void print_network_info(const ExtractedNetworkWeights &info,
                               const char *name) {
  std::cerr << name << " Network:\n";
  std::cerr << "  Hidden dim: " << info.hidden_dim << "\n";
  std::cerr << "  FC0 out: " << info.fc0_out << "\n";
  std::cerr << "  FC1 out: " << info.fc1_out << "\n";
  std::cerr << "  Has threats: " << (info.has_threats ? "Yes" : "No") << "\n";
  std::cerr << "  FT weights: " << info.ft.weights_size / 1024 << " KB\n";
  std::cerr << "  FT biases: " << info.ft.biases_size << " bytes\n";
  std::cerr << "  FT PSQT: " << info.ft.psqt_size / 1024 << " KB\n";
  if (info.has_threats) {
    std::cerr << "  Threat weights: " << info.ft.threat_weights_size / 1024
              << " KB\n";
    std::cerr << "  Threat PSQT: " << info.ft.threat_psqt_size / 1024
              << " KB\n";
  }

  size_t layer_total = 0;
  for (const auto &layer : info.layers) {
    layer_total += layer.fc0_weights_size + layer.fc0_biases_size;
    layer_total += layer.fc1_weights_size + layer.fc1_biases_size;
    layer_total += layer.fc2_weights_size + layer.fc2_biases_size;
  }
  std::cerr << "  Layer weights (8 buckets): " << layer_total / 1024 << " KB\n";
}

} // namespace MetalFish::GPU
