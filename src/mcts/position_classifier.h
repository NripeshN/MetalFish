/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file position_classifier.h
 * @brief MetalFish source file.
 */

  Position Classifier

  Classifies positions as tactical or strategic to determine the optimal
  search strategy. This is crucial for the hybrid approach - we want to
  use alpha-beta for tactical positions and MCTS for strategic ones.

  Key indicators:
  - Tactical: checks, captures available, hanging pieces, king safety issues
  - Strategic: closed positions, pawn structures, piece maneuvering

  Licensed under GPL-3.0
*/

#pragma once

#include "../core/bitboard.h"
#include "../core/movegen.h"
#include "../core/position.h"

namespace MetalFish {

// Position type classification
enum class PositionType {
  HIGHLY_TACTICAL, // Forcing moves, immediate threats
  TACTICAL,        // Some tactical elements
  BALANCED,        // Mix of tactical and strategic
  STRATEGIC,       // Quiet maneuvering
  HIGHLY_STRATEGIC // Closed position, long-term planning
};

// Detailed position features for analysis
struct PositionFeatures {
  // Tactical indicators
  bool in_check = false;
  int num_checks_available = 0;
  int num_captures = 0;
  int num_promotions = 0;
  int hanging_pieces[2] = {0, 0}; // [WHITE, BLACK]
  int attacked_pieces[2] = {0, 0};
  bool has_mate_threat = false;

  // King safety
  int king_attackers[2] = {0, 0};
  int king_zone_attacks[2] = {0, 0};
  bool kings_castled[2] = {false, false};

  // Strategic indicators
  int pawn_islands[2] = {0, 0};
  int passed_pawns[2] = {0, 0};
  int backward_pawns[2] = {0, 0};
  int isolated_pawns[2] = {0, 0};
  int pawn_chain_length[2] = {0, 0};

  // Piece activity
  int mobility[2] = {0, 0};
  int outposts[2] = {0, 0};
  int piece_coordination[2] = {0, 0};

  // Position structure
  bool is_closed = false;
  bool is_open = false;
  int center_control[2] = {0, 0};
  int space_advantage[2] = {0, 0};

  // Material
  int material[2] = {0, 0};
  int material_imbalance = 0;
  bool is_endgame = false;
  bool is_middlegame = false;

  // Computed scores
  float tactical_score = 0.0f;  // 0-1, higher = more tactical
  float strategic_score = 0.0f; // 0-1, higher = more strategic
  float complexity = 0.0f;      // 0-1, higher = more complex

  PositionType classify() const;
};

// Position classifier
class PositionClassifier {
public:
  PositionClassifier() = default;

  // Analyze position and return features
  PositionFeatures analyze(const Position &pos) const;

  // Quick classification without full analysis
  PositionType quick_classify(const Position &pos) const;

  // Get tactical score (0-1)
  float tactical_score(const Position &pos) const;

  // Get strategic score (0-1)
  float strategic_score(const Position &pos) const;

  // Check specific conditions
  bool is_tactical(const Position &pos) const;
  bool is_strategic(const Position &pos) const;
  bool has_forcing_moves(const Position &pos) const;
  bool is_quiet(const Position &pos) const;

private:
  // Helper methods
  int count_hanging_pieces(const Position &pos, Color c) const;
  int count_attacked_pieces(const Position &pos, Color c) const;
  int count_king_attackers(const Position &pos, Color c) const;
  int count_pawn_islands(const Position &pos, Color c) const;
  int count_passed_pawns(const Position &pos, Color c) const;
  int calculate_mobility(const Position &pos, Color c) const;
  bool is_position_closed(const Position &pos) const;
  bool is_endgame_position(const Position &pos) const;

  // Bitboard helpers
  Bitboard get_king_zone(Square ksq) const;
  Bitboard get_outpost_squares(const Position &pos, Color c) const;
};

// Search strategy recommendation based on position
struct SearchStrategy {
  PositionType position_type;

  // Recommended approach
  float mcts_weight = 0.5f; // 0-1, how much to rely on MCTS
  float ab_weight = 0.5f;   // 0-1, how much to rely on alpha-beta

  // MCTS parameters
  float cpuct = 2.5f;
  int min_mcts_nodes = 100;
  int max_mcts_depth = 50;
  bool use_policy_network = true;

  // Alpha-beta parameters
  int ab_depth = 6;
  bool use_null_move = true;
  bool use_lmr = true;
  int aspiration_window = 25;

  // Hybrid parameters
  int ab_verify_depth = 4;            // Depth for AB verification of MCTS moves
  float ab_override_threshold = 0.3f; // Score diff to override MCTS choice
  bool use_ab_for_tactics = true;

  // Time allocation
  float time_multiplier = 1.0f; // Adjust time based on position
};

// Strategy selector
class StrategySelector {
public:
  StrategySelector() = default;

  // Get recommended strategy for position
  SearchStrategy get_strategy(const Position &pos) const;
  SearchStrategy get_strategy(const PositionFeatures &features) const;

  // Adjust strategy based on game phase
  void adjust_for_time(SearchStrategy &strategy, int time_left_ms,
                       int increment_ms) const;
  void adjust_for_score(SearchStrategy &strategy, int current_score) const;

private:
  PositionClassifier classifier_;
};

// Global classifier instance
PositionClassifier &position_classifier();

} // namespace MetalFish