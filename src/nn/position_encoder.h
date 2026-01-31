/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  Position encoder for neural network inference
  Simplified implementation that works with MetalFish Position
*/

#pragma once

#include "../core/position.h"
#include "../core/types.h"
#include "input_planes.h"
#include <deque>

namespace MetalFish {
namespace NN {

// Position history for encoding
// We store FEN strings instead of Position objects since Position is non-copyable
class PositionHistory {
public:
    PositionHistory() = default;
    
    // Add a position to history (stores FEN string)
    void Push(const Position& pos);
    
    // Get number of positions in history
    size_t Size() const { return fens_.size(); }
    
    // Get FEN at index (0 = most recent)
    std::string GetFEN(int index) const;
    
    // Clear history
    void Clear() { fens_.clear(); }
    
private:
    std::deque<std::string> fens_;
    static constexpr size_t kMaxHistory = 8;
};

// Encode a position (with history) into 112 input planes
InputPlanes EncodePosition(const PositionHistory& history, Color side_to_move);

// Simplified encoder for single position (no history)
InputPlanes EncodePosition(const Position& pos);

}  // namespace NN
}  // namespace MetalFish
