/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_buffers.h"

#include <memory>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

class CudaExecutor {
public:
  virtual ~CudaExecutor() = default;

  virtual void Execute(const NetworkTensorPlan &plan,
                       CudaInferenceBuffers &buffers, int batch_size) = 0;
  virtual std::string Name() const = 0;
};

std::unique_ptr<CudaExecutor> CreateMissingCudaExecutor();
std::unique_ptr<CudaExecutor> CreateNullCudaExecutorForSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish
