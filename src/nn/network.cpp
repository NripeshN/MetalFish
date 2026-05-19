/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network.h"

#ifdef USE_METAL
#include "metal/metal_network.h"
#endif
#ifdef USE_CUDA
#include "cuda/cuda_network.h"
#endif

#include <iostream>
#include <stdexcept>

namespace MetalFish {
namespace NN {

class StubNetwork : public Network {
public:
  StubNetwork(const WeightsFile &weights) : weights_(weights) {}

  NetworkOutput Evaluate(const InputPlanes &input) override {
    NetworkOutput output;
    output.policy.fill(1.0f / kPolicyOutputs);
    output.value = 0.0f;
    output.has_wdl = false;
    output.wdl[0] = output.wdl[1] = output.wdl[2] = 0.0f;
    output.has_moves_left = false;
    return output;
  }

  std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) override {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &input : inputs) {
      outputs.push_back(Evaluate(input));
    }
    return outputs;
  }

  std::string GetNetworkInfo() const override {
    return "Stub network (not functional)";
  }

private:
  WeightsFile weights_;
};

std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const std::string &backend) {
#ifdef USE_METAL
  if (backend == "auto" || backend == "metal") {
    try {
      return std::make_unique<Metal::MetalNetwork>(weights);
    } catch (const std::exception &e) {
      std::cerr << "Metal backend unavailable: " << e.what() << std::endl;
      if (backend == "metal") {
        throw;
      }
    }
  }
#else
  if (backend == "metal") {
    throw std::runtime_error("Metal backend was not compiled into this build");
  }
#endif

#ifdef USE_CUDA
  if (backend == "auto" || backend == "cuda") {
    try {
      return std::make_unique<Cuda::CudaNetwork>(weights);
    } catch (const std::exception &e) {
      std::cerr << "CUDA backend unavailable: " << e.what() << std::endl;
      if (backend == "cuda") {
        throw;
      }
    }
  }
#else
  if (backend == "cuda") {
    throw std::runtime_error("CUDA backend was not compiled into this build");
  }
#endif

  if (backend == "stub") {
    return std::make_unique<StubNetwork>(weights);
  }

  if (backend == "auto") {
    throw std::runtime_error("No functional NN backend available");
  }

  throw std::runtime_error("Unknown NN backend: " + backend);
}

std::unique_ptr<Network> CreateNetwork(const std::string &weights_path,
                                       const std::string &backend) {
  if (backend == "stub") {
    WeightsFile empty_weights;
    return CreateNetwork(empty_weights, backend);
  }

  auto weights_opt = LoadWeights(weights_path);

  if (!weights_opt.has_value()) {
    throw std::runtime_error("Could not load network weights from: " +
                             weights_path);
  }

  return CreateNetwork(weights_opt.value(), backend);
}

} // namespace NN
} // namespace MetalFish
