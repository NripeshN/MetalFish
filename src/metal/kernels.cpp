/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

*/

#include "allocator.h"
#include "metal/device.h"
#include <iostream>
#include <vector>

namespace MetalFish {
namespace Metal {

// GPU kernel management for chess engine operations

class KernelManager {
public:
  static KernelManager &instance() {
    static KernelManager manager;
    return manager;
  }

  bool initialize(const std::string &libPath) {
    try {
      Device &device = get_device();
      library_ = device.get_library("metalfish", libPath);

      if (library_) {
        // Load kernels
        nnueFeatureTransform_ =
            device.get_kernel("nnue_feature_transform", library_);
        nnueFullyConnected_ =
            device.get_kernel("nnue_fully_connected", library_);
        evaluatePosition_ = device.get_kernel("evaluate_position", library_);
        generateMoves_ = device.get_kernel("generate_moves", library_);

        initialized_ = true;
        std::cout << "[MetalFish] GPU kernels initialized successfully\n";
        return true;
      }
    } catch (const std::exception &e) {
      std::cerr << "[MetalFish] Failed to initialize GPU kernels: " << e.what()
                << "\n";
    }

    return false;
  }

  bool is_initialized() const { return initialized_; }

  // NNUE Feature Transform
  void dispatch_feature_transform(Buffer &input, Buffer &output, size_t count) {
    if (!nnueFeatureTransform_)
      return;

    Device &device = get_device();
    CommandEncoder &encoder = device.get_command_encoder();

    encoder.set_compute_pipeline_state(nnueFeatureTransform_);
    encoder.set_buffer(input.ptr, 0);
    encoder.set_buffer(output.ptr, 1);

    MTL::Size gridSize = MTL::Size::Make(count, 1, 1);
    NS::UInteger threadGroupSize =
        nnueFeatureTransform_->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > count)
      threadGroupSize = count;
    MTL::Size groupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    encoder.dispatch_threads(gridSize, groupSize);
  }

  // NNUE Fully Connected Layer
  void dispatch_fully_connected(Buffer &input, Buffer &weights, Buffer &bias,
                                Buffer &output, uint32_t inputSize,
                                uint32_t outputSize) {
    if (!nnueFullyConnected_)
      return;

    Device &device = get_device();
    CommandEncoder &encoder = device.get_command_encoder();

    encoder.set_compute_pipeline_state(nnueFullyConnected_);
    encoder.set_buffer(input.ptr, 0);
    encoder.set_buffer(weights.ptr, 1);
    encoder.set_buffer(bias.ptr, 2);
    encoder.set_buffer(output.ptr, 3);
    encoder.set_bytes(&inputSize, sizeof(inputSize), 4);
    encoder.set_bytes(&outputSize, sizeof(outputSize), 5);

    MTL::Size gridSize = MTL::Size::Make(outputSize, 1, 1);
    NS::UInteger threadGroupSize =
        nnueFullyConnected_->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > outputSize)
      threadGroupSize = outputSize;
    MTL::Size groupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    encoder.dispatch_threads(gridSize, groupSize);
  }

  // Batch position evaluation
  void dispatch_position_eval(Buffer &boardStates, Buffer &scores,
                              size_t count) {
    if (!evaluatePosition_)
      return;

    Device &device = get_device();
    CommandEncoder &encoder = device.get_command_encoder();

    encoder.set_compute_pipeline_state(evaluatePosition_);
    encoder.set_buffer(boardStates.ptr, 0);
    encoder.set_buffer(scores.ptr, 1);

    MTL::Size gridSize = MTL::Size::Make(count, 1, 1);
    NS::UInteger threadGroupSize =
        evaluatePosition_->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > count)
      threadGroupSize = count;
    MTL::Size groupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    encoder.dispatch_threads(gridSize, groupSize);
  }

  // Move generation
  void dispatch_move_gen(Buffer &boardStates, Buffer &moves, Buffer &moveCounts,
                         size_t count) {
    if (!generateMoves_)
      return;

    Device &device = get_device();
    CommandEncoder &encoder = device.get_command_encoder();

    encoder.set_compute_pipeline_state(generateMoves_);
    encoder.set_buffer(boardStates.ptr, 0);
    encoder.set_buffer(moves.ptr, 1);
    encoder.set_buffer(moveCounts.ptr, 2);

    MTL::Size gridSize = MTL::Size::Make(count, 1, 1);
    NS::UInteger threadGroupSize =
        generateMoves_->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > count)
      threadGroupSize = count;
    MTL::Size groupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    encoder.dispatch_threads(gridSize, groupSize);
  }

  void synchronize() {
    Device &device = get_device();
    device.end_encoding();
    device.commit_command_buffer();
  }

private:
  KernelManager() = default;

  MTL::Library *library_ = nullptr;
  MTL::ComputePipelineState *nnueFeatureTransform_ = nullptr;
  MTL::ComputePipelineState *nnueFullyConnected_ = nullptr;
  MTL::ComputePipelineState *evaluatePosition_ = nullptr;
  MTL::ComputePipelineState *generateMoves_ = nullptr;
  bool initialized_ = false;
};

// Initialize GPU kernels
bool init_kernels(const std::string &libPath) {
  return KernelManager::instance().initialize(libPath);
}

bool kernels_initialized() {
  return KernelManager::instance().is_initialized();
}

} // namespace Metal
} // namespace MetalFish
