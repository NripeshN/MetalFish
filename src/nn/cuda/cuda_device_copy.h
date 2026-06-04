/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cuda_runtime_api.h>

#include <string_view>

namespace MetalFish {
namespace NN {
namespace Cuda {

void CopyDeviceFloatRows(float *dst, int dst_stride, const float *src,
                         int src_stride, int rows, int width,
                         std::string_view name, cudaStream_t stream);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
