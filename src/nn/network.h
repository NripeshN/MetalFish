/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "input_planes.h"
#include "network_output.h"
#include "weights_file.h"

namespace MetalFish {
namespace NN {

class Network {
public:
  virtual ~Network() = default;

  virtual NetworkOutput Evaluate(const InputPlanes &input) = 0;

  virtual std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) = 0;

  virtual std::string GetNetworkInfo() const = 0;
  virtual bool HasWDL() const { return false; }
  virtual bool HasMovesLeft() const { return false; }
};

struct BackendConfig {
  std::string backend = "auto";
  std::string coreml_model_path;
  std::string coreml_compute_units = "cpu-ne";
  int cuda_device = -1;
  bool cuda_graph_execution = true;
  int cuda_stable_execution_batch_size = 0;
  bool cuda_deterministic_attention_softmax = true;
  bool cuda_full_buffer_clear = true;
};

std::unique_ptr<Network> CreateNetwork(const std::string &weights_path,
                                       const std::string &backend = "auto");
std::unique_ptr<Network> CreateNetwork(const std::string &weights_path,
                                       const BackendConfig &config);
std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const std::string &backend = "auto");
std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const BackendConfig &config);
std::unique_ptr<Network>
CreateNetwork(const WeightsFile &weights, const std::string &backend,
              const std::string &model_path,
              const std::string &compute_units = "cpu-ne");

} // namespace NN
} // namespace MetalFish
