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

// Batch Processing - OPTIMIZED
constexpr int GPU_MAX_BATCH_SIZE = 4096;  // Increased from 256

// Feature limits per position:
// HalfKAv2_hm: max 30 non-king pieces × 1 feature each = 30 per perspective
// Total per position (both perspectives): 60 features
// Add safety margin: 64 features per perspective, 128 total per position
constexpr int GPU_MAX_FEATURES_PER_PERSPECTIVE = 64;
constexpr int GPU_MAX_FEATURES = 128;  // Increased from 32 to handle all positions

// SIMD/Threadgroup optimization constants
constexpr int GPU_THREADGROUP_SIZE = 256;  // Optimal for M-series GPUs
constexpr int GPU_SIMDGROUP_SIZE = 32;     // Apple GPU SIMD width
constexpr int GPU_FEATURE_TRANSFORM_THREADS = 256;
constexpr int GPU_FORWARD_THREADS = 64;

// Memory alignment for optimal GPU access
constexpr int GPU_BUFFER_ALIGNMENT = 256;

// Legacy aliases for backward compatibility
constexpr int NNUE_FEATURE_DIM_BIG = GPU_FT_DIM_BIG;
constexpr int NNUE_FEATURE_DIM_SMALL = GPU_FT_DIM_SMALL;
constexpr int MAX_BATCH_SIZE = GPU_MAX_BATCH_SIZE;
constexpr int MAX_FEATURES_PER_POSITION = GPU_MAX_FEATURES;
constexpr int HALFKA_DIMS = GPU_HALFKA_DIMS;
constexpr int PSQT_DIMS = GPU_PSQT_BUCKETS;

} // namespace MetalFish::GPU
