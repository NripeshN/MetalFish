/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  Input planes data structure for neural network inference
*/

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace MetalFish {
namespace NN {

// Number of input planes for neural network
constexpr int kInputPlanes = 112;

// History depth (number of previous positions to encode)
constexpr int kMoveHistory = 8;

// Planes per board position
constexpr int kPlanesPerBoard = 13;

// Base index for auxiliary planes
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;  // 104

// Each plane is an 8x8 board represented as 64 floats
struct InputPlane {
    std::array<float, 64> values;
    
    InputPlane() { values.fill(0.0f); }
    
    void Set(int square, float value) {
        if (square >= 0 && square < 64) {
            values[square] = value;
        }
    }
    
    float Get(int square) const {
        if (square >= 0 && square < 64) {
            return values[square];
        }
        return 0.0f;
    }
    
    void Fill(float value) {
        values.fill(value);
    }
};

// Input planes for neural network (112 planes total)
using InputPlanes = std::array<InputPlane, kInputPlanes>;

}  // namespace NN
}  // namespace MetalFish
