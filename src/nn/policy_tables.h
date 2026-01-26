/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file policy_tables.h
 * @brief MetalFish source file.
 */

  Lc0 Policy Tables - UCI Move ↔ NN Policy Index Mapping
  
  Maps between UCI move notation and neural network policy output indices.
  Lc0 uses 1858 policy outputs for standard chess (excluding underpromotions).
  
  Policy encoding:
  - Queen moves: 56 directions × 64 squares = 3584 indices
  - Knight moves: 8 directions × 64 squares = 512 indices  
  - Underpromotions: 3 types × 64 squares × 2 directions = 384 indices
  
  After deduplication: 1858 unique legal move outputs
  
  Licensed under GPL-3.0
*/

#pragma once

#include "../core/types.h"
#include <array>
#include <cstdint>

namespace MetalFish {

class Position;

namespace NN {

// Lc0 policy output size
constexpr int POLICY_OUTPUTS = 1858;

// Policy index for a move
// Returns -1 if move is illegal or invalid
int move_to_policy_index(Move move, const Position& pos);

// Convert policy index back to move
// Returns Move::none() if index is invalid or doesn't map to a legal move
Move policy_index_to_move(int index, const Position& pos);

// Get all legal moves with their policy indices
// Returns vector of (move, policy_index) pairs
std::vector<std::pair<Move, int>> get_legal_moves_with_indices(const Position& pos);

// Policy mapping tables (initialized at startup)
namespace PolicyTables {

// Initialize policy tables (call once at startup)
void initialize();

// Check if tables are initialized
bool is_initialized();

// Direction encodings for queen-like moves
// 8 directions: N, NE, E, SE, S, SW, W, NW
// Each direction can move 1-7 squares
constexpr int QUEEN_DIRECTIONS = 8;
constexpr int MAX_QUEEN_DISTANCE = 7;

// Knight move encodings
// 8 possible knight moves from any square
constexpr int KNIGHT_DIRECTIONS = 8;

// Underpromotion encodings
// 3 types (N, B, R) × 2 directions (left, right) × 64 squares
constexpr int UNDERPROMOTION_TYPES = 3; // Knight, Bishop, Rook

// Helper: Get direction index for queen-like moves
int get_queen_direction(int from_sq, int to_sq);

// Helper: Get direction index for knight moves
int get_knight_direction(int from_sq, int to_sq);

// Helper: Check if move is an underpromotion
bool is_underpromotion(Move move);

} // namespace PolicyTables

} // namespace NN
} // namespace MetalFish