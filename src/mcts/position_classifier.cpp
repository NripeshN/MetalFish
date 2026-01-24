/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Position Classifier - Implementation

  Licensed under GPL-3.0
*/

#include "position_classifier.h"
#include "../search/movepick.h"
#include <algorithm>
#include <cmath>

namespace MetalFish {

// Piece values for material calculation
constexpr int PieceValues[] = {0, 100, 320, 330, 500, 900, 20000};

// ============================================================================
// PositionFeatures
// ============================================================================

PositionType PositionFeatures::classify() const {
  // Weight the tactical vs strategic indicators
  float tactical = tactical_score;
  float strategic = strategic_score;

  // Adjust based on specific conditions
  if (in_check || has_mate_threat) {
    tactical += 0.3f;
  }

  if (num_captures > 3 || num_promotions > 0) {
    tactical += 0.2f;
  }

  if (is_closed) {
    strategic += 0.2f;
  }

  // Normalize
  float total = tactical + strategic;
  if (total > 0) {
    tactical /= total;
    strategic /= total;
  } else {
    tactical = strategic = 0.5f;
  }

  // Classify
  if (tactical > 0.8f)
    return PositionType::HIGHLY_TACTICAL;
  if (tactical > 0.6f)
    return PositionType::TACTICAL;
  if (strategic > 0.8f)
    return PositionType::HIGHLY_STRATEGIC;
  if (strategic > 0.6f)
    return PositionType::STRATEGIC;
  return PositionType::BALANCED;
}

// ============================================================================
// PositionClassifier
// ============================================================================

PositionFeatures PositionClassifier::analyze(const Position &pos) const {
  PositionFeatures f;

  // Basic checks
  f.in_check = pos.checkers();

  // Count move types
  MoveList<LEGAL> moves(pos);
  for (const auto &m : moves) {
    if (pos.capture(m))
      f.num_captures++;
    if (m.type_of() == PROMOTION)
      f.num_promotions++;
    if (pos.gives_check(m))
      f.num_checks_available++;
  }

  // Hanging and attacked pieces
  for (Color c : {WHITE, BLACK}) {
    f.hanging_pieces[c] = count_hanging_pieces(pos, c);
    f.attacked_pieces[c] = count_attacked_pieces(pos, c);
    f.king_attackers[c] = count_king_attackers(pos, c);
  }

  // Pawn structure
  for (Color c : {WHITE, BLACK}) {
    f.pawn_islands[c] = count_pawn_islands(pos, c);
    f.passed_pawns[c] = count_passed_pawns(pos, c);
  }

  // Mobility
  for (Color c : {WHITE, BLACK}) {
    f.mobility[c] = calculate_mobility(pos, c);
  }

  // Position structure
  f.is_closed = is_position_closed(pos);
  f.is_open = !f.is_closed && popcount(pos.pieces(PAWN)) < 10;
  f.is_endgame = is_endgame_position(pos);
  f.is_middlegame = !f.is_endgame && pos.game_ply() > 10;

  // Material
  for (Color c : {WHITE, BLACK}) {
    f.material[c] = 0;
    for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
      f.material[c] += popcount(pos.pieces(c, pt)) * PieceValues[pt];
    }
  }
  f.material_imbalance = std::abs(f.material[WHITE] - f.material[BLACK]);

  // Compute tactical score
  float tactical = 0.0f;
  if (f.in_check)
    tactical += 0.3f;
  if (f.num_checks_available > 0)
    tactical += 0.1f * std::min(f.num_checks_available, 3);
  if (f.num_captures > 0)
    tactical += 0.05f * std::min(f.num_captures, 5);
  if (f.num_promotions > 0)
    tactical += 0.2f;
  if (f.hanging_pieces[WHITE] + f.hanging_pieces[BLACK] > 0)
    tactical += 0.2f;
  if (f.king_attackers[WHITE] > 2 || f.king_attackers[BLACK] > 2)
    tactical += 0.2f;
  if (f.material_imbalance > 200)
    tactical += 0.1f;
  f.tactical_score = std::min(1.0f, tactical);

  // Compute strategic score
  float strategic = 0.0f;
  if (f.is_closed)
    strategic += 0.3f;
  if (f.pawn_islands[WHITE] > 2 || f.pawn_islands[BLACK] > 2)
    strategic += 0.1f;
  if (f.passed_pawns[WHITE] > 0 || f.passed_pawns[BLACK] > 0)
    strategic += 0.15f;
  if (f.is_endgame)
    strategic += 0.2f;
  if (!f.in_check && f.num_captures < 3)
    strategic += 0.1f;
  int mobility_diff = std::abs(f.mobility[WHITE] - f.mobility[BLACK]);
  if (mobility_diff > 10)
    strategic += 0.1f;
  f.strategic_score = std::min(1.0f, strategic);

  // Complexity
  f.complexity = (f.tactical_score + f.strategic_score) / 2.0f;
  f.complexity += 0.1f * std::min(moves.size() / 40.0f, 1.0f);
  f.complexity = std::min(1.0f, f.complexity);

  return f;
}

PositionType PositionClassifier::quick_classify(const Position &pos) const {
  // Fast classification without full analysis

  // Check for immediate tactical elements
  if (pos.checkers()) {
    return PositionType::HIGHLY_TACTICAL;
  }

  // Count captures quickly
  int captures = 0;
  MoveList<CAPTURES> cap_moves(pos);
  captures = cap_moves.size();

  if (captures > 5) {
    return PositionType::TACTICAL;
  }

  // Check for closed position
  if (is_position_closed(pos)) {
    return PositionType::STRATEGIC;
  }

  // Check piece count for endgame
  if (is_endgame_position(pos)) {
    return PositionType::STRATEGIC;
  }

  // Default to balanced
  if (captures > 2) {
    return PositionType::TACTICAL;
  }

  return PositionType::BALANCED;
}

float PositionClassifier::tactical_score(const Position &pos) const {
  float score = 0.0f;

  if (pos.checkers())
    score += 0.4f;

  MoveList<CAPTURES> caps(pos);
  score += 0.05f * std::min(static_cast<int>(caps.size()), 8);

  // King safety
  for (Color c : {WHITE, BLACK}) {
    int attackers = count_king_attackers(pos, c);
    if (attackers > 2)
      score += 0.2f;
  }

  return std::min(1.0f, score);
}

float PositionClassifier::strategic_score(const Position &pos) const {
  float score = 0.0f;

  if (is_position_closed(pos))
    score += 0.3f;
  if (is_endgame_position(pos))
    score += 0.2f;
  if (!pos.checkers())
    score += 0.1f;

  // Pawn structure complexity
  int islands = count_pawn_islands(pos, WHITE) + count_pawn_islands(pos, BLACK);
  if (islands > 4)
    score += 0.2f;

  return std::min(1.0f, score);
}

bool PositionClassifier::is_tactical(const Position &pos) const {
  return tactical_score(pos) > 0.5f;
}

bool PositionClassifier::is_strategic(const Position &pos) const {
  return strategic_score(pos) > 0.5f;
}

bool PositionClassifier::has_forcing_moves(const Position &pos) const {
  if (pos.checkers())
    return true;

  // Check for checks available
  MoveList<LEGAL> moves(pos);
  for (const auto &m : moves) {
    if (pos.gives_check(m))
      return true;
  }

  return false;
}

bool PositionClassifier::is_quiet(const Position &pos) const {
  if (pos.checkers())
    return false;

  MoveList<CAPTURES> caps(pos);
  return caps.size() < 3;
}

int PositionClassifier::count_hanging_pieces(const Position &pos,
                                             Color c) const {
  int count = 0;
  Color them = ~c;

  Bitboard dominated =
      pos.pieces(c) & ~pos.pieces(c, PAWN) & ~pos.pieces(c, KING);

  while (dominated) {
    Square s = pop_lsb(dominated);
    Bitboard attackers = pos.attackers_to(s) & pos.pieces(them);
    Bitboard defenders = pos.attackers_to(s) & pos.pieces(c);

    if (attackers && !defenders) {
      count++;
    }
  }

  return count;
}

int PositionClassifier::count_attacked_pieces(const Position &pos,
                                              Color c) const {
  int count = 0;
  Color them = ~c;

  Bitboard dominated = pos.pieces(c) & ~pos.pieces(c, KING);

  while (dominated) {
    Square s = pop_lsb(dominated);
    if (pos.attackers_to(s) & pos.pieces(them)) {
      count++;
    }
  }

  return count;
}

int PositionClassifier::count_king_attackers(const Position &pos,
                                             Color c) const {
  Color them = ~c;
  Square ksq = pos.square<KING>(c);
  Bitboard king_zone = get_king_zone(ksq);

  int attackers = 0;

  // Count pieces attacking king zone
  Bitboard their_pieces =
      pos.pieces(them) & ~pos.pieces(them, PAWN) & ~pos.pieces(them, KING);
  Bitboard occupied = pos.pieces();

  while (their_pieces) {
    Square s = pop_lsb(their_pieces);
    PieceType pt = type_of(pos.piece_on(s));
    Bitboard attacks;

    switch (pt) {
    case KNIGHT:
      attacks = attacks_bb<KNIGHT>(s);
      break;
    case BISHOP:
      attacks = attacks_bb<BISHOP>(s, occupied);
      break;
    case ROOK:
      attacks = attacks_bb<ROOK>(s, occupied);
      break;
    case QUEEN:
      attacks = attacks_bb<QUEEN>(s, occupied);
      break;
    default:
      attacks = 0;
    }

    if (attacks & king_zone) {
      attackers++;
    }
  }

  return attackers;
}

int PositionClassifier::count_pawn_islands(const Position &pos, Color c) const {
  Bitboard pawns = pos.pieces(c, PAWN);
  if (!pawns)
    return 0;

  int islands = 0;
  bool in_island = false;

  for (File f = FILE_A; f <= FILE_H; ++f) {
    Bitboard file_pawns = pawns & file_bb(f);
    if (file_pawns) {
      if (!in_island) {
        islands++;
        in_island = true;
      }
    } else {
      in_island = false;
    }
  }

  return islands;
}

int PositionClassifier::count_passed_pawns(const Position &pos, Color c) const {
  int count = 0;
  Bitboard pawns = pos.pieces(c, PAWN);
  Bitboard their_pawns = pos.pieces(~c, PAWN);

  while (pawns) {
    Square s = pop_lsb(pawns);
    File f = file_of(s);
    Rank r = rank_of(s);

    // Check if pawn is passed
    Bitboard front_span;
    if (c == WHITE) {
      front_span = 0ULL;
      for (Rank rr = Rank(r + 1); rr <= RANK_8; ++rr) {
        if (f > FILE_A)
          front_span |= make_square(File(f - 1), rr);
        front_span |= make_square(f, rr);
        if (f < FILE_H)
          front_span |= make_square(File(f + 1), rr);
      }
    } else {
      front_span = 0ULL;
      for (Rank rr = Rank(r - 1); rr >= RANK_1; --rr) {
        if (f > FILE_A)
          front_span |= make_square(File(f - 1), rr);
        front_span |= make_square(f, rr);
        if (f < FILE_H)
          front_span |= make_square(File(f + 1), rr);
      }
    }

    if (!(front_span & their_pawns)) {
      count++;
    }
  }

  return count;
}

int PositionClassifier::calculate_mobility(const Position &pos, Color c) const {
  int mobility = 0;

  Bitboard occupied = pos.pieces();
  Bitboard our_pieces = pos.pieces(c);

  // Knights
  Bitboard knights = pos.pieces(c, KNIGHT);
  while (knights) {
    Square s = pop_lsb(knights);
    mobility += popcount(attacks_bb<KNIGHT>(s) & ~our_pieces);
  }

  // Bishops
  Bitboard bishops = pos.pieces(c, BISHOP);
  while (bishops) {
    Square s = pop_lsb(bishops);
    mobility += popcount(attacks_bb<BISHOP>(s, occupied) & ~our_pieces);
  }

  // Rooks
  Bitboard rooks = pos.pieces(c, ROOK);
  while (rooks) {
    Square s = pop_lsb(rooks);
    mobility += popcount(attacks_bb<ROOK>(s, occupied) & ~our_pieces);
  }

  // Queens
  Bitboard queens = pos.pieces(c, QUEEN);
  while (queens) {
    Square s = pop_lsb(queens);
    mobility += popcount(attacks_bb<QUEEN>(s, occupied) & ~our_pieces);
  }

  return mobility;
}

bool PositionClassifier::is_position_closed(const Position &pos) const {
  // A position is closed if there are many pawns and few open files
  Bitboard all_pawns = pos.pieces(PAWN);
  int pawn_count = popcount(all_pawns);

  if (pawn_count < 10)
    return false;

  // Count open files (files with no pawns)
  int open_files = 0;
  for (File f = FILE_A; f <= FILE_H; ++f) {
    if (!(all_pawns & file_bb(f))) {
      open_files++;
    }
  }

  // Position is closed if few open files and many pawns
  return open_files <= 2 && pawn_count >= 12;
}

bool PositionClassifier::is_endgame_position(const Position &pos) const {
  // Simple endgame detection based on material
  int total_material = 0;
  for (Color c : {WHITE, BLACK}) {
    for (PieceType pt = KNIGHT; pt <= QUEEN; ++pt) {
      total_material += popcount(pos.pieces(c, pt)) * PieceValues[pt];
    }
  }

  // Endgame if total non-pawn material is low
  return total_material < 2600; // Roughly 2 rooks + minor piece
}

Bitboard PositionClassifier::get_king_zone(Square ksq) const {
  // 3x3 area around king
  Bitboard zone = attacks_bb<KING>(ksq) | square_bb(ksq);

  // Extend one rank forward
  File f = file_of(ksq);
  Rank r = rank_of(ksq);

  if (r < RANK_7) {
    zone |= shift<NORTH>(zone & rank_bb(Rank(r + 1)));
  }
  if (r > RANK_2) {
    zone |= shift<SOUTH>(zone & rank_bb(Rank(r - 1)));
  }

  return zone;
}

Bitboard PositionClassifier::get_outpost_squares(const Position &pos,
                                                 Color c) const {
  // Squares that cannot be attacked by enemy pawns
  Color them = ~c;
  Bitboard their_pawns = pos.pieces(them, PAWN);

  Bitboard safe = ~0ULL;

  while (their_pawns) {
    Square s = pop_lsb(their_pawns);
    // Remove squares attacked by pawn advances
    Bitboard pawn_attacks = (them == WHITE)
                                ? pawn_attacks_bb<WHITE>(square_bb(s))
                                : pawn_attacks_bb<BLACK>(square_bb(s));
    safe &= ~pawn_attacks;
  }

  // Only consider central and advanced squares
  Bitboard central =
      (file_bb(FILE_C) | file_bb(FILE_D) | file_bb(FILE_E) | file_bb(FILE_F));
  if (c == WHITE) {
    central &= (rank_bb(RANK_4) | rank_bb(RANK_5) | rank_bb(RANK_6));
  } else {
    central &= (rank_bb(RANK_3) | rank_bb(RANK_4) | rank_bb(RANK_5));
  }

  return safe & central;
}

// ============================================================================
// StrategySelector
// ============================================================================

SearchStrategy StrategySelector::get_strategy(const Position &pos) const {
  PositionFeatures features = classifier_.analyze(pos);
  return get_strategy(features);
}

SearchStrategy StrategySelector::get_strategy(const PositionFeatures &f) const {
  SearchStrategy s;
  s.position_type = f.classify();

  // REBALANCED: Give MCTS more weight across all position types
  // The original weights were too aggressive toward alpha-beta,
  // causing hybrid to underperform pure MCTS
  switch (s.position_type) {
  case PositionType::HIGHLY_TACTICAL:
    // Tactical positions still favor alpha-beta but not as extreme
    s.mcts_weight = 0.35f;          // was 0.2f
    s.ab_weight = 0.65f;            // was 0.8f
    s.cpuct = 1.2f;                 // Lower exploration for tactics
    s.ab_depth = 8;                 // was 10
    s.ab_verify_depth = 5;          // was 6
    s.ab_override_threshold = 0.3f; // was 0.1f - harder to override MCTS
    s.use_ab_for_tactics = true;
    s.time_multiplier = 1.1f; // was 1.2f
    break;

  case PositionType::TACTICAL:
    s.mcts_weight = 0.45f;           // was 0.35f
    s.ab_weight = 0.55f;             // was 0.65f
    s.cpuct = 1.5f;                  // was 2.0f
    s.ab_depth = 7;                  // was 8
    s.ab_verify_depth = 4;           // was 5
    s.ab_override_threshold = 0.35f; // was 0.2f
    s.use_ab_for_tactics = true;
    s.time_multiplier = 1.05f; // was 1.1f
    break;

  case PositionType::BALANCED:
    // Balanced positions should favor MCTS slightly (it's our strength)
    s.mcts_weight = 0.55f; // was 0.5f
    s.ab_weight = 0.45f;   // was 0.5f
    s.cpuct = 1.5f;        // was 2.5f
    s.ab_depth = 6;
    s.ab_verify_depth = 4;
    s.ab_override_threshold = 0.4f; // was 0.3f
    s.use_ab_for_tactics = true;
    s.time_multiplier = 1.0f;
    break;

  case PositionType::STRATEGIC:
    // Strategic positions favor MCTS more
    s.mcts_weight = 0.70f; // was 0.65f
    s.ab_weight = 0.30f;   // was 0.35f
    s.cpuct = 1.8f;        // was 3.0f - still moderate exploration
    s.ab_depth = 5;
    s.ab_verify_depth = 3;
    s.ab_override_threshold = 0.5f; // was 0.4f
    s.use_ab_for_tactics = false;
    s.time_multiplier = 0.95f; // was 0.9f
    break;

  case PositionType::HIGHLY_STRATEGIC:
    // Highly strategic positions strongly favor MCTS
    s.mcts_weight = 0.85f; // was 0.8f
    s.ab_weight = 0.15f;   // was 0.2f
    s.cpuct = 2.0f;        // was 3.5f
    s.ab_depth = 4;
    s.ab_verify_depth = 2;
    s.ab_override_threshold = 0.6f; // was 0.5f
    s.use_ab_for_tactics = false;
    s.time_multiplier = 0.9f; // was 0.8f
    break;
  }

  // Adjust for endgame - MCTS can be good in endgames too
  if (f.is_endgame) {
    s.ab_depth += 1; // was +2
    s.ab_verify_depth += 1;
    // Don't reduce MCTS weight as much in endgames
    s.mcts_weight *= 0.9f; // was 0.8f
    s.ab_weight = 1.0f - s.mcts_weight;
  }

  // Adjust for complexity
  if (f.complexity > 0.7f) {
    s.time_multiplier *= 1.15f; // was 1.2f
    s.min_mcts_nodes = 150;     // was 200
  }

  return s;
}

void StrategySelector::adjust_for_time(SearchStrategy &s, int time_left_ms,
                                       int increment_ms) const {
  // Adjust strategy based on time pressure
  if (time_left_ms < 10000) {
    // Low time - favor faster alpha-beta
    s.mcts_weight *= 0.5f;
    s.ab_weight = 1.0f - s.mcts_weight;
    s.ab_depth = std::max(4, s.ab_depth - 2);
    s.time_multiplier *= 0.5f;
  } else if (time_left_ms < 30000) {
    // Medium time pressure
    s.mcts_weight *= 0.8f;
    s.ab_weight = 1.0f - s.mcts_weight;
    s.time_multiplier *= 0.8f;
  } else if (time_left_ms > 300000) {
    // Plenty of time - can explore more
    s.time_multiplier *= 1.3f;
    s.min_mcts_nodes = 300;
  }

  // Increment affects strategy
  if (increment_ms > 5000) {
    s.time_multiplier *= 1.1f;
  }
}

void StrategySelector::adjust_for_score(SearchStrategy &s,
                                        int current_score) const {
  // Adjust based on current evaluation
  if (std::abs(current_score) > 500) {
    // Winning or losing badly - be more tactical
    s.mcts_weight *= 0.7f;
    s.ab_weight = 1.0f - s.mcts_weight;
    s.use_ab_for_tactics = true;
  } else if (std::abs(current_score) < 50) {
    // Equal position - can explore more
    s.mcts_weight *= 1.1f;
    s.mcts_weight = std::min(0.9f, s.mcts_weight);
    s.ab_weight = 1.0f - s.mcts_weight;
  }
}

// Global instance
PositionClassifier &position_classifier() {
  static PositionClassifier instance;
  return instance;
}

} // namespace MetalFish
