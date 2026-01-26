/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/



























#pragma once

#include <memory>
#include <vector>

#include "../proto/net.pb.h"
#include "utils/exception.h"

namespace MetalFish::NN {

const int kInputPlanes = 112;

// All input planes are 64 value vectors, every element of which is either
// 0 or some value, unique for the plane. Therefore, input is defined as
// a bitmask showing where to set the value, and the value itself.
struct InputPlane {
  InputPlane() = default;
  void SetAll() { mask = ~0ull; }
  void Fill(float val) {
    SetAll();
    value = val;
  }
  std::uint64_t mask = 0ull;
  float value = 1.0f;
};
using InputPlanes = std::vector<InputPlane>;

// An interface to implement by computing backends.
class NetworkComputation {
 public:
  // Adds a sample to the batch.
  virtual void AddInput(InputPlanes&& input) = 0;
  // Do the computation.
  virtual void ComputeBlocking() = 0;
  // Returns how many times AddInput() was called.
  virtual int GetBatchSize() const = 0;
  // Returns Q value of @sample.
  virtual float GetQVal(int sample) const = 0;
  virtual float GetDVal(int sample) const = 0;
  // Returns P value @move_id of @sample.
  virtual float GetPVal(int sample, int move_id) const = 0;
  virtual float GetMVal(int sample) const = 0;
  virtual ~NetworkComputation() = default;
};

// The plan:
// 1. Search must not look directly into any fields of NetworkFormat anymore.
// 2. Backends populate NetworkCapabilities that show search how to use NN, both
//    for input and output.
// 3. Input part of NetworkCapabilities is just copy of InputFormat for now, and
//    is likely to stay so (because search not knowing how to use NN is not very
//    useful), but it's fine if it will change.
// 4. On the other hand, output part of NetworkCapabilities is set of
//    independent parameters (like WDL, moves left head etc), because search can
//    look what's set and act accordingly. Backends may derive it from
//    output head format fields or other places.

struct NetworkCapabilities {
  pbMetalFish::NN::NetworkFormat::InputFormat input_format;
  pbMetalFish::NN::NetworkFormat::OutputFormat output_format;
  pbMetalFish::NN::NetworkFormat::MovesLeftFormat moves_left;
  // TODO expose information of whether GetDVal() is usable or always zero.

  // Combines capabilities by setting the most restrictive ones. May throw
  // exception.
  void Merge(const NetworkCapabilities& other) {
    if (input_format != other.input_format) {
      throw Exception("Incompatible input formats, " +
                      std::to_string(input_format) + " vs " +
                      std::to_string(other.input_format));
    }
    if (output_format != other.output_format) {
      throw Exception("Incompatible output formats, " +
                      std::to_string(output_format) + " vs " +
                      std::to_string(other.output_format));
    }
    if (!other.has_mlh()) moves_left = pbMetalFish::NN::NetworkFormat::MOVES_LEFT_NONE;
  }

  bool has_mlh() const {
    return moves_left != pbMetalFish::NN::NetworkFormat::MOVES_LEFT_NONE;
  }

  bool has_wdl() const {
    return output_format == pbMetalFish::NN::NetworkFormat::OUTPUT_WDL;
  }
};

class Network {
 public:
  virtual const NetworkCapabilities& GetCapabilities() const = 0;
  virtual std::unique_ptr<NetworkComputation> NewComputation() = 0;
  virtual int GetThreads() const { return 1; }
  virtual void InitThread(int /*id*/) {}
  virtual bool IsCpu() const { return false; }
  virtual int GetMiniBatchSize() const { return 256; }
  virtual int GetPreferredBatchStep() const { return 1; }
  virtual ~Network() = default;
};

}  // namespace MetalFish::NN
