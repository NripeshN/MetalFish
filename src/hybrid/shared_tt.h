#pragma once

#include "../search/tt.h"
#include "../core/position.h"

#include <algorithm>
#include <cmath>

namespace MetalFish {
namespace MCTS {

class SharedTTReader {
public:
    explicit SharedTTReader(TranspositionTable* tt) : tt_(tt) {}

    struct TTResult {
        float value;
        float draw;
        int depth;
        bool found;
    };

    TTResult Probe(const Position& pos, int depth_threshold = 8) const {
        TTResult result{0.0f, 0.0f, 0, false};
        if (!tt_) return result;

        auto [found, data, writer] = tt_->probe(pos.key());
        if (!found) return result;
        if (data.depth < depth_threshold) return result;

        int cp = std::clamp(static_cast<int>(data.value), -10000, 10000);

        // Logistic cp → win probability
        float win_prob = 1.0f / (1.0f + std::pow(10.0f, -cp / 400.0f));

        // Estimate draw probability from score magnitude
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
    TranspositionTable* tt_ = nullptr;
};

} // namespace MCTS
} // namespace MetalFish
