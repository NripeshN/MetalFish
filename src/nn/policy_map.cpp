/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "policy_map.h"

#include <algorithm>
#include <unordered_map>
#include <string>

namespace MetalFish {
namespace NN {

namespace {

// Simplified policy mapping
// Real implementation would have 1858-element lookup tables
std::unordered_map<std::string, int> move_to_index;
std::unordered_map<int, Move> index_to_move;

bool tables_initialized = false;

}  // namespace

void InitPolicyTables() {
  if (tables_initialized) return;
  
  // Simplified initialization
  // Real implementation would build full 1858-move mapping
  // based on from_square (64) x to_square (64) x promotion (5)
  // with special handling for underpromotions
  
  tables_initialized = true;
}

int MoveToNNIndex(Move move) {
  InitPolicyTables();
  
  // Extract move components using member functions
  Square from = move.from_sq();
  Square to = move.to_sq();
  MoveType mt = move.type_of();
  
  int from_idx = static_cast<int>(from);
  int to_idx = static_cast<int>(to);
  
  // Simplified mapping - real implementation needs full lookup tables
  // This is a placeholder that returns a valid index in range [0, 1857]
  int base_index = from_idx * 28 + (to_idx % 28);
  
  // Add offset for promotions (simplified)
  if (mt == PROMOTION) {
    PieceType pt = move.promotion_type();
    int promo_offset = (static_cast<int>(pt) - KNIGHT) * 64;
    base_index = 1792 + (promo_offset + from_idx) % 66;  // Keep in range
  }
  
  // Ensure within bounds [0, 1857]
  return std::min(base_index, 1857);
}

Move IndexToNNMove(int index) {
  InitPolicyTables();
  
  // Simplified reverse mapping
  // Real implementation would use proper lookup tables
  
  // For now, return a placeholder
  return Move::none();
}

}  // namespace NN
}  // namespace MetalFish
