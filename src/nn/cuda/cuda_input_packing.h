/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../input_plane_packing.h"
#include "cuda_smoke_status.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaInputPackingSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
};

constexpr int kCudaInputPlanes = kPackedInputPlaneCount;
constexpr int kCudaSquares = kPackedInputSquareCount;

void PackInputPlanesHostRaw(const float *inputs, int batch_size,
                            std::vector<std::uint64_t> &masks,
                            std::vector<float> &values);
void PackInputPlaneBatchHostRaw(const std::vector<const float *> &inputs,
                                std::vector<std::uint64_t> &masks,
                                std::vector<float> &values);

CudaInputPackingSmokeResult RunInputPackingSmokeRaw(const float *input);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
