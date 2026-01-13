/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Constants - Single source of truth for all GPU-related constants
*/

#pragma once

#include <cstdint>

namespace MetalFish::GPU {

// NNUE Network Architecture
constexpr int GPU_FT_DIM_BIG = 1024;
constexpr int GPU_FT_DIM_SMALL = 128;
constexpr int GPU_FC0_OUT = 15;
constexpr int GPU_FC1_OUT = 32;
constexpr int GPU_PSQT_BUCKETS = 8;
constexpr int GPU_LAYER_STACKS = 8;

// Feature Dimensions
constexpr int GPU_HALFKA_DIMS = 45056;
constexpr int GPU_THREAT_DIMS = 1536;

// Batch Processing
constexpr int GPU_MAX_BATCH_SIZE = 256;
constexpr int GPU_MAX_FEATURES = 32;

// Legacy aliases for backward compatibility
constexpr int NNUE_FEATURE_DIM_BIG = GPU_FT_DIM_BIG;
constexpr int NNUE_FEATURE_DIM_SMALL = GPU_FT_DIM_SMALL;
constexpr int MAX_BATCH_SIZE = GPU_MAX_BATCH_SIZE;
constexpr int MAX_FEATURES_PER_POSITION = GPU_MAX_FEATURES;
constexpr int HALFKA_DIMS = GPU_HALFKA_DIMS;
constexpr int PSQT_DIMS = GPU_PSQT_BUCKETS;

} // namespace MetalFish::GPU
