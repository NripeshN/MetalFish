/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  Policy mapping utilities for neural network inference
*/

#pragma once

#include "../../core/types.h"
#include "policy_map.h"
#include "attention_policy_map.h"

namespace MetalFish {
namespace NN {

// Number of policy outputs for standard chess
constexpr int kPolicyOutputs = 1858;

// Number of policy outputs for attention-based networks
constexpr int kAttentionPolicyOutputs = 4672;

// Transform flags for board symmetry
enum Transform {
    NoTransform = 0,
    FlipTransform = 1,
    MirrorTransform = 2,
    TransposeTransform = 4
};

// Encode a move to neural network policy index
// For conventional policy head (1858 outputs)
int MoveToNNIndex(Move move, int transform = 0);

// Decode neural network policy index to a move
// For conventional policy head (1858 outputs)
Move MoveFromNNIndex(int idx, int transform = 0);

// Encode a move to attention policy index
// For attention-based policy head (4672 outputs)
int MoveToAttentionIndex(Move move, int transform = 0);

// Decode attention policy index to a move
// For attention-based policy head (4672 outputs)
Move MoveFromAttentionIndex(int idx, int transform = 0);

}  // namespace NN
}  // namespace MetalFish
