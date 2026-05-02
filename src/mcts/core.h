/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Core Utilities

  Fast math, moves-left evaluation, collision tracking,
  and score transformation helpers.

  Licensed under GPL-3.0
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

#include "search_params.h"

namespace MetalFish {
namespace MCTS {

// ============================================================================
// Fast Math Utilities
// ============================================================================

namespace FastMath {

inline float FastLog(float x) {
  static constexpr float kScale = 8.2629582881927490e-8f;
  static constexpr int32_t kBias = (127 << 23);
  int32_t i;
  std::memcpy(&i, &x, sizeof(float));
  return static_cast<float>(i - kBias) * kScale;
}

inline float FastSign(float x) {
  return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

inline float FastExp(float x) {
  x = std::max(-88.0f, std::min(88.0f, x));
  int32_t i = static_cast<int32_t>(x * 12102203.0f) + 1065353216;
  float result;
  std::memcpy(&result, &i, sizeof(float));
  return result;
}

inline float FastLogistic(float x) { return 1.0f / (1.0f + FastExp(-x)); }

inline float FastTanh(float x) {
  if (x < -4.97f)
    return -1.0f;
  if (x > 4.97f)
    return 1.0f;
  float x2 = x * x;
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

inline float FastSqrt(float x) {
  if (x <= 0.0f)
    return 0.0f;
  int32_t i;
  std::memcpy(&i, &x, sizeof(float));
  i = 0x1FBD1DF5 + (i >> 1);
  float y;
  std::memcpy(&y, &i, sizeof(float));
  y = 0.5f * (y + x / y);
  return y;
}

} // namespace FastMath

// ============================================================================
// Moves Left Head (MLH) Utility
// ============================================================================

class MovesLeftEvaluator {
public:
  MovesLeftEvaluator() : enabled_(false) {}

  MovesLeftEvaluator(const SearchParams &params, float parent_m = 0.0f,
                     float parent_q = 0.0f)
      : enabled_(params.moves_left_max_effect > 0.0f),
        m_slope_(params.moves_left_slope), m_cap_(params.moves_left_max_effect),
        a_constant_(params.moves_left_constant_factor),
        a_linear_(params.moves_left_scaled_factor),
        a_square_(params.moves_left_quadratic_factor),
        q_threshold_(params.moves_left_threshold), parent_m_(parent_m),
        parent_within_threshold_(std::abs(parent_q) >
                                 params.moves_left_threshold) {}

  void SetParent(float parent_m, float parent_q) {
    parent_m_ = parent_m;
    parent_within_threshold_ = std::abs(parent_q) > q_threshold_;
  }

  float GetMUtility(float child_m, float q) const {
    if (!enabled_ || !parent_within_threshold_)
      return 0.0f;

    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastMath::FastSign(-q);

    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      float q_scaled =
          std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
      m *= a_constant_ + a_linear_ * std::abs(q_scaled) +
           a_square_ * q_scaled * q_scaled;
    } else {
      m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    }

    return m;
  }

  float GetDefaultMUtility() const { return 0.0f; }
  bool IsEnabled() const { return enabled_; }

private:
  bool enabled_;
  float m_slope_ = 0.0f;
  float m_cap_ = 0.0f;
  float a_constant_ = 0.0f;
  float a_linear_ = 0.0f;
  float a_square_ = 0.0f;
  float q_threshold_ = 0.0f;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

// ============================================================================
// WDL Rescaling
// ============================================================================

struct WDLRescaler {
    float ratio;
    float diff;

    bool IsActive() const { return ratio != 1.0f || diff != 0.0f; }

    float Rescale(float q) const {
        if (!IsActive()) return q;
        float clamped = std::clamp(q, -0.9999f, 0.9999f);
        return std::tanh(std::atanh(clamped) * ratio + diff);
    }
};

// ============================================================================
// Score Transformation
// ============================================================================

inline int QToNnueScore(float q) {
  q = std::clamp(q, -0.999f, 0.999f);
  return static_cast<int>(std::atanh(q) * 300.0f);
}

inline float TablebaseWDLToParentWL(int wdl) {
  if (wdl >= 2)
    return -1.0f;
  if (wdl <= -2)
    return 1.0f;
  return 0.0f;
}

inline float TablebaseWDLToDraw(int wdl) {
  return std::abs(wdl) >= 2 ? 0.0f : 1.0f;
}

// ============================================================================
// Collision Handling
// ============================================================================

struct CollisionStats {
  int max_collision_events = 917;
  int max_collision_visits = 80000;

  int scaling_start = 28;
  int scaling_end = 145000;
  float scaling_power = 1.25f;

  int GetMaxCollisionVisits(uint32_t tree_size) const {
    if (tree_size < static_cast<uint32_t>(scaling_start)) {
      return 1;
    }
    if (tree_size >= static_cast<uint32_t>(scaling_end)) {
      return max_collision_visits;
    }

    float progress = static_cast<float>(tree_size - scaling_start) /
                     (scaling_end - scaling_start);
    return 1 + static_cast<int>((max_collision_visits - 1) *
                                std::pow(progress, scaling_power));
  }
};

} // namespace MCTS
} // namespace MetalFish
