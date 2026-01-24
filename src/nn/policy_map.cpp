/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "policy_map.h"

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
  
  // Basic formula: from * 64 + to (simplified)
  // Real formula accounts for underpromotions and special moves
  int base_index = from_idx * 64 + to_idx;
  
  // Add promotion offset if needed
  if (mt == PROMOTION) {
    PieceType pt = move.promotion_type();
    base_index += (static_cast<int>(pt) - KNIGHT) * 64 * 64;
  }
  
  // Ensure within bounds
  return base_index % 1858;
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
