#pragma once

#include "../core/position.h"
#include "../search/tt.h"

#include <algorithm>
#include <cmath>

namespace MetalFish {
namespace MCTS {

inline float SharedTTCpToWinProbability(int cp, float cp_scale) {
  const float x = static_cast<float>(cp) / cp_scale;
  return 1.0f / (1.0f + std::exp(-x));
}

class SharedTTReader {
public:
  explicit SharedTTReader(TranspositionTable *tt, float cp_scale = 230.0f)
      : tt_(tt), cp_scale_(cp_scale) {}

  struct TTResult {
    float value;
    float draw;
    int depth;
    bool found;
  };

  TTResult Probe(const Position &pos, int depth_threshold = 8) const {
    TTResult result{0.0f, 0.0f, 0, false};
    if (!tt_)
      return result;

    auto [found, data, writer] = tt_->probe(pos.key());
    if (!found)
      return result;
    if (data.depth < depth_threshold)
      return result;

    int cp = std::clamp(static_cast<int>(data.value), -10000, 10000);

    // UPPER bound (fail-low): true value ≤ stored cp. Using it as a
    // point estimate is only safe when it indicates a bad position.
    // LOWER bound (fail-high): true value ≥ stored cp. Only safe when
    // it indicates a good position. Skip non-EXACT entries where the
    // bound direction makes the point-estimate interpretation unreliable.
    if (data.bound == BOUND_UPPER && cp > 500)
      return result;
    if (data.bound == BOUND_LOWER && cp < -500)
      return result;

    const float win_prob = SharedTTCpToWinProbability(cp, cp_scale_);

    float draw_est = std::max(0.0f, 1.0f - 2.0f * std::abs(win_prob - 0.5f));
    float w = win_prob * (1.0f - draw_est);
    float l = (1.0f - win_prob) * (1.0f - draw_est);

    result.value = w - l;
    result.draw = draw_est;
    result.depth = data.depth;
    result.found = true;
    return result;
  }

private:
  TranspositionTable *tt_ = nullptr;
  float cp_scale_ = 230.0f;
};

} // namespace MCTS
} // namespace MetalFish
