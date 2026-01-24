/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Advanced Memory Management Header

  Interface for optimized memory management utilities.
*/

#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <cuda_runtime.h>
#include <memory>

namespace MetalFish {
namespace GPU {
namespace CUDA {

// Forward declarations
class UnifiedMemoryManager;
class PinnedMemoryManager;
template <typename T> class DoubleBuffer;
class MemoryPool;
class CacheAlignedAllocator;
class AsyncMemoryOps;
class MemoryStats;

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // CUDA_MEMORY_H
