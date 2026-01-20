/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Alpha-Beta Integration for Hybrid Search - Implementation

  Licensed under GPL-3.0
*/

#include "ab_integration.h"
#include "../core/bitboard.h"
#include "../core/movegen.h"
#include "../eval/evaluate.h"
#include <algorithm>
#include <cmath>
#include <mutex>

namespace MetalFish {
namespace MCTS {

// Reduction table for LMR
static int LMRTable[64][64];

// Initialize LMR table
static void init_lmr_table() {
  static bool initialized = false;
  if (initialized)
    return;

  for (int d = 1; d < 64; ++d) {
    for (int m = 1; m < 64; ++m) {
      LMRTable[d][m] = static_cast<int>(0.8 + std::log(d) * std::log(m) * 0.4);
    }
  }
  initialized = true;
}

ABSearcher::ABSearcher() {
  init_lmr_table();
  state_stack_.resize(MAX_PLY + 10);

  // Clear history tables
  std::memset(main_history_, 0, sizeof(main_history_));
  std::memset(capture_history_, 0, sizeof(capture_history_));
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(counter_moves_, 0, sizeof(counter_moves_));

  // Initialize static eval stack to VALUE_NONE
  for (int i = 0; i < MAX_PLY; ++i) {
    static_eval_stack_[i] = VALUE_NONE;
  }
}

ABSearcher::~ABSearcher() = default;

void ABSearcher::initialize(TranspositionTable *tt) { tt_ = tt; }

ABSearchResult ABSearcher::search(const Position &pos, int depth) {
  reset_stats();
  stop_flag_ = false;
  start_time_ = std::chrono::steady_clock::now();

  ABSearchResult result;
  result.depth = depth;

  // Create mutable position copy
  Position search_pos;
  StateInfo st;
  search_pos.set(pos.fen(), false, &st);

  // PV storage
  Move pv[MAX_PLY];
  int pv_length = 0;

  // Search
  Value score = search_internal<true>(search_pos, depth, -VALUE_INFINITE,
                                      VALUE_INFINITE, 0, pv, pv_length);

  result.score = score;
  result.nodes_searched = static_cast<int>(nodes_.load());

  // Extract PV
  for (int i = 0; i < pv_length; ++i) {
    result.pv.push_back(pv[i]);
  }

  if (!result.pv.empty()) {
    result.best_move = result.pv[0];
  }

  // Check for mate
  if (score > VALUE_MATE_IN_MAX_PLY) {
    result.is_mate = true;
    result.mate_in = (VALUE_MATE - score + 1) / 2;
  } else if (score < VALUE_MATED_IN_MAX_PLY) {
    result.is_mate = true;
    result.mate_in = -(VALUE_MATE + score) / 2;
  }

  return result;
}

ABSearchResult ABSearcher::search_aspiration(const Position &pos, int depth,
                                             Value prev_score) {
  if (depth < 4 || prev_score == VALUE_NONE) {
    return search(pos, depth);
  }

  reset_stats();
  stop_flag_ = false;
  start_time_ = std::chrono::steady_clock::now();

  ABSearchResult result;
  result.depth = depth;

  Position search_pos;
  StateInfo st;
  search_pos.set(pos.fen(), false, &st);

  Move pv[MAX_PLY];
  int pv_length = 0;

  Value delta = Value(config_.aspiration_window);
  Value alpha = std::max(prev_score - delta, -VALUE_INFINITE);
  Value beta = std::min(prev_score + delta, VALUE_INFINITE);

  while (true) {
    Value score =
        search_internal<true>(search_pos, depth, alpha, beta, 0, pv, pv_length);

    if (score <= alpha) {
      beta = (alpha + beta) / 2;
      alpha = std::max(score - delta, -VALUE_INFINITE);
      result.bound = BOUND_UPPER;
    } else if (score >= beta) {
      beta = std::min(score + delta, VALUE_INFINITE);
      result.bound = BOUND_LOWER;
    } else {
      result.score = score;
      result.bound = BOUND_EXACT;
      break;
    }

    delta += delta / 3;

    if (should_stop()) {
      result.score = score;
      break;
    }
  }

  result.nodes_searched = static_cast<int>(nodes_.load());

  for (int i = 0; i < pv_length; ++i) {
    result.pv.push_back(pv[i]);
  }

  if (!result.pv.empty()) {
    result.best_move = result.pv[0];
  }

  if (result.score > VALUE_MATE_IN_MAX_PLY) {
    result.is_mate = true;
    result.mate_in = (VALUE_MATE - result.score + 1) / 2;
  } else if (result.score < VALUE_MATED_IN_MAX_PLY) {
    result.is_mate = true;
    result.mate_in = -(VALUE_MATE + result.score) / 2;
  }

  return result;
}

bool ABSearcher::verify_move(const Position &pos, Move move, int depth,
                             Value *out_score) {
  ABSearchResult result = search(pos, depth);

  if (out_score) {
    *out_score = result.score;
  }

  if (result.best_move == move) {
    return true;
  }

  // Check if move is within acceptable range
  if (!result.pv.empty()) {
    Position test_pos;
    StateInfo st1, st2;
    test_pos.set(pos.fen(), false, &st1);

    // Get score of the move to verify
    test_pos.do_move(move, st2);
    Value move_score = -qsearch(test_pos, -VALUE_INFINITE, VALUE_INFINITE, 0);

    // Allow 50cp tolerance
    return move_score >= result.score - 50;
  }

  return false;
}

std::vector<std::pair<Move, float>>
ABSearcher::get_move_scores(const Position &pos) {
  std::vector<std::pair<Move, float>> scores;

  MoveList<LEGAL> moves(pos);
  if (moves.size() == 0)
    return scores;

  // Score each move
  float max_score = -1e9f;

  std::vector<std::pair<Move, float>> raw_scores;

  for (const auto &m : moves) {
    float score = 0.0f;

    // TT move bonus
    if (tt_) {
      auto [found, tt_data, writer] = tt_->probe(pos.key());
      if (found && tt_data.move == m) {
        score += 10000.0f;
      }
    }

    // Capture bonus (MVV-LVA)
    if (pos.capture(m)) {
      // For en passant, the captured pawn is not on the destination square
      PieceType captured =
          m.type_of() == EN_PASSANT ? PAWN : type_of(pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(pos.piece_on(m.from_sq()));
      static const float pv[] = {0, 100, 320, 330, 500, 900, 0};
      score += pv[captured] * 10 - pv[attacker];
    }

    // Promotion bonus
    if (m.type_of() == PROMOTION) {
      score += 8000.0f;
    }

    // History score
    Color us = pos.side_to_move();
    score +=
        static_cast<float>(main_history_[us][m.from_sq()][m.to_sq()]) / 100.0f;

    // Killer bonus
    for (int i = 0; i < 2; ++i) {
      if (killers_[0][i] == m) {
        score += 5000.0f - i * 100.0f;
      }
    }

    raw_scores.emplace_back(m, score);
    max_score = std::max(max_score, score);
  }

  // Normalize to probabilities using softmax
  float sum = 0.0f;
  for (auto &[move, score] : raw_scores) {
    score = std::exp((score - max_score) / 1000.0f);
    sum += score;
  }

  for (auto &[move, score] : raw_scores) {
    scores.emplace_back(move, score / sum);
  }

  return scores;
}

template <bool PvNode>
Value ABSearcher::search_internal(Position &pos, int depth, Value alpha,
                                  Value beta, int ply, Move *pv,
                                  int &pv_length) {
  if (should_stop()) {
    return VALUE_ZERO;
  }

  nodes_++;
  pv_length = 0;

  // Quiescence search at leaf
  if (depth <= 0) {
    return qsearch(pos, alpha, beta, ply);
  }

  // Check for draw
  if (pos.is_draw(ply)) {
    return VALUE_DRAW;
  }

  // Max ply check
  if (ply >= MAX_PLY - 1) {
    return evaluate(pos);
  }

  // TT probe
  Move tt_move = Move::none();
  Value tt_value = VALUE_NONE;
  bool tt_hit = false;

  if (config_.use_tt && tt_) {
    auto [found, tt_data, writer] = tt_->probe(pos.key());
    if (found) {
      tt_hit = true;
      tt_hits_++;
      tt_move = tt_data.move;
      tt_value = tt_data.value;

      // TT cutoff for non-PV nodes
      if (!PvNode && tt_data.depth >= depth) {
        if (tt_data.bound == BOUND_EXACT) {
          tt_cutoffs_++;
          return tt_value;
        }
        if (tt_data.bound == BOUND_LOWER && tt_value >= beta) {
          tt_cutoffs_++;
          return tt_value;
        }
        if (tt_data.bound == BOUND_UPPER && tt_value <= alpha) {
          tt_cutoffs_++;
          return tt_value;
        }
      }
    }
  }

  bool in_check = pos.checkers();
  Value static_eval = VALUE_NONE;

  if (!in_check) {
    static_eval = evaluate(pos);
  } else {
    // When in check, use evaluation from 2 plies ago (same as main search)
    static_eval = (ply >= 2) ? static_eval_stack_[ply - 2] : VALUE_NONE;
  }

  // Store static evaluation for this ply
  static_eval_stack_[ply] = static_eval;

  // Null move pruning (skip - requires TT for do_null_move)

  // Futility pruning
  bool futility_pruning = false;
  if (config_.use_futility && !PvNode && !in_check && depth <= 6 &&
      static_eval + 150 * depth <= alpha) {
    futility_pruning = true;
  }

  // Generate and score moves
  MoveList<LEGAL> moves(pos);

  if (moves.size() == 0) {
    if (in_check) {
      return Value(-VALUE_MATE + ply);
    }
    return VALUE_DRAW;
  }

  // Sort moves by score
  std::vector<std::pair<Move, int>> scored_moves;
  for (const auto &m : moves) {
    int score = 0;
    if (m == tt_move) {
      score = 100000;
    } else if (pos.capture(m)) {
      // For en passant, the captured pawn is not on the destination square
      PieceType captured =
          m.type_of() == EN_PASSANT ? PAWN : type_of(pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(pos.piece_on(m.from_sq()));
      static const int pv[] = {0, 100, 320, 330, 500, 900, 0};
      score = 50000 + pv[captured] * 10 - pv[attacker];
    } else if (m.type_of() == PROMOTION) {
      score = 40000;
    } else {
      Color us = pos.side_to_move();
      score = main_history_[us][m.from_sq()][m.to_sq()];

      if (killers_[ply % MAX_PLY][0] == m) {
        score += 9000;
      } else if (killers_[ply % MAX_PLY][1] == m) {
        score += 8000;
      }
    }
    scored_moves.emplace_back(m, score);
  }

  std::stable_sort(
      scored_moves.begin(), scored_moves.end(),
      [](const auto &a, const auto &b) { return a.second > b.second; });

  Move best_move = Move::none();
  Value best_value = -VALUE_INFINITE;
  int move_count = 0;
  // Improving flag: compare current static eval to eval from 2 plies ago (same
  // side to move) This is the correct semantic - alpha represents search
  // window, not previous evaluation
  bool improving = !in_check && ply >= 2 && static_eval != VALUE_NONE &&
                   static_eval_stack_[ply - 2] != VALUE_NONE &&
                   static_eval > static_eval_stack_[ply - 2];
  Value original_alpha = alpha; // Save for TT bound calculation

  Move child_pv[MAX_PLY];
  int child_pv_len = 0;

  for (const auto &[m, move_score] : scored_moves) {
    if (should_stop()) {
      break;
    }

    move_count++;
    bool is_capture = pos.capture(m);
    bool gives_check = pos.gives_check(m);

    // Futility pruning
    if (futility_pruning && move_count > 1 && !is_capture && !gives_check) {
      continue;
    }

    // Late move reductions
    int reduction = 0;
    if (config_.use_lmr && depth >= 3 && move_count > 1 + 2 * PvNode &&
        !is_capture && !gives_check) {
      reduction = late_move_reduction(depth, move_count, improving);
      if (PvNode) {
        reduction = std::max(0, reduction - 1);
      }
    }

    // Make move
    pos.do_move(m, state_stack_[ply]);

    Value value;

    // PVS
    if (move_count == 1) {
      value = -search_internal<PvNode>(pos, depth - 1, -beta, -alpha, ply + 1,
                                       child_pv, child_pv_len);
    } else {
      // Reduced search
      value = -search_internal<false>(pos, depth - 1 - reduction, -alpha - 1,
                                      -alpha, ply + 1, child_pv, child_pv_len);

      // Re-search if needed
      if (value > alpha && (PvNode || reduction > 0)) {
        value = -search_internal<PvNode>(pos, depth - 1, -beta, -alpha, ply + 1,
                                         child_pv, child_pv_len);
      }
    }

    pos.undo_move(m);

    if (value > best_value) {
      best_value = value;
      best_move = m;

      if (value > alpha) {
        alpha = value;

        // Update PV
        pv[0] = m;
        for (int i = 0; i < child_pv_len; ++i) {
          pv[i + 1] = child_pv[i];
        }
        pv_length = child_pv_len + 1;

        if (value >= beta) {
          // Update killers
          if (!is_capture) {
            killers_[ply % MAX_PLY][1] = killers_[ply % MAX_PLY][0];
            killers_[ply % MAX_PLY][0] = m;
          }

          // Update history with clamping to prevent int16_t overflow
          if (config_.use_history && !is_capture) {
            int bonus = std::min(depth * depth, 400);
            Color us = pos.side_to_move();
            int16_t &entry = main_history_[us][m.from_sq()][m.to_sq()];
            int new_value = static_cast<int>(entry) + bonus;
            entry = static_cast<int16_t>(
                std::clamp(new_value, static_cast<int>(INT16_MIN),
                           static_cast<int>(INT16_MAX)));
          }

          break;
        }
      }
    }
  }

  // Store in TT
  if (config_.use_tt && tt_ && !should_stop()) {
    Bound bound = best_value >= beta            ? BOUND_LOWER
                  : best_value > original_alpha ? BOUND_EXACT
                                                : BOUND_UPPER;

    auto [found, tt_data, writer] = tt_->probe(pos.key());
    writer.write(pos.key(), best_value, PvNode, bound, Depth(depth), best_move,
                 static_eval, tt_->generation());
  }

  return best_value;
}

Value ABSearcher::qsearch(Position &pos, Value alpha, Value beta, int ply) {
  nodes_++;

  if (ply >= MAX_PLY - 1) {
    return evaluate(pos);
  }

  bool in_check = pos.checkers();
  Value stand_pat = VALUE_NONE;

  if (!in_check) {
    stand_pat = evaluate(pos);

    if (stand_pat >= beta) {
      return stand_pat;
    }

    if (stand_pat > alpha) {
      alpha = stand_pat;
    }
  }

  // Generate captures (and checks if in check)
  MoveList<LEGAL> moves(pos);

  Value best_value = in_check ? -VALUE_INFINITE : stand_pat;

  for (const auto &m : moves) {
    bool is_capture = pos.capture(m);
    bool gives_check = pos.gives_check(m);

    // In qsearch, only consider captures and evasions
    if (!in_check && !is_capture && !gives_check) {
      continue;
    }

    // SEE pruning for captures
    if (!in_check && is_capture) {
      if (!pos.see_ge(m, Value(-50))) {
        continue;
      }
    }

    pos.do_move(m, state_stack_[ply]);
    Value value = -qsearch(pos, -beta, -alpha, ply + 1);
    pos.undo_move(m);

    if (value > best_value) {
      best_value = value;
      if (value > alpha) {
        alpha = value;
        if (value >= beta) {
          break;
        }
      }
    }
  }

  // Checkmate/stalemate detection
  if (in_check && best_value == -VALUE_INFINITE) {
    return Value(-VALUE_MATE + ply);
  }

  return best_value;
}

void ABSearcher::score_moves(const Position &pos, MoveList<LEGAL> &moves,
                             Move tt_move) {
  // Scoring is now done inline in search_internal
}

int ABSearcher::late_move_reduction(int depth, int move_count,
                                    bool improving) const {
  int r = LMRTable[std::min(depth, 63)][std::min(move_count, 63)];
  if (!improving) {
    r++;
  }
  return r;
}

Value ABSearcher::evaluate(const Position &pos) {
  // Use simple_eval for the ABSearcher since we don't have access to 
  // the full NNUE infrastructure (networks, accumulators, caches).
  // The main MCTS evaluation uses GPU NNUE for strong evaluation.
  // This ABSearcher is primarily used for tactical verification where
  // simple material evaluation is sufficient for move ordering.
  return Value(Eval::simple_eval(pos));
}

bool ABSearcher::should_stop() const {
  if (stop_flag_) {
    return true;
  }

  if (config_.max_nodes > 0 &&
      nodes_ >= static_cast<uint64_t>(config_.max_nodes)) {
    return true;
  }

  if (config_.max_time_ms > 0) {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (ms >= config_.max_time_ms) {
      return true;
    }
  }

  return false;
}

// Explicit template instantiations
template Value ABSearcher::search_internal<true>(Position &, int, Value, Value,
                                                 int, Move *, int &);
template Value ABSearcher::search_internal<false>(Position &, int, Value, Value,
                                                  int, Move *, int &);

// ============================================================================
// ABPolicyGenerator Implementation
// ============================================================================

ABPolicyGenerator::ABPolicyGenerator() {
  std::memset(mcts_history_, 0, sizeof(mcts_history_));
}

void ABPolicyGenerator::initialize(const ButterflyHistory *main_history,
                                   const CapturePieceToHistory *capture_history,
                                   TranspositionTable *tt) {
  main_history_ = main_history;
  capture_history_ = capture_history;
  tt_ = tt;
}

std::vector<std::pair<Move, float>>
ABPolicyGenerator::generate_policy(const Position &pos) {
  std::vector<std::pair<Move, float>> policy;

  MoveList<LEGAL> moves(pos);
  if (moves.size() == 0)
    return policy;

  std::vector<float> scores(moves.size());
  float max_score = -1e9f;

  // TT move
  Move tt_move = Move::none();
  if (tt_) {
    auto [found, tt_data, writer] = tt_->probe(pos.key());
    if (found) {
      tt_move = tt_data.move;
    }
  }

  Color us = pos.side_to_move();
  int idx = 0;

  for (const auto &m : moves) {
    float score = 0.0f;

    // TT move gets highest priority
    if (m == tt_move) {
      score += 5000.0f;
    }

    // Captures scored by MVV-LVA and SEE
    if (pos.capture(m)) {
      // For en passant, the captured pawn is not on the destination square
      PieceType captured =
          m.type_of() == EN_PASSANT ? PAWN : type_of(pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(pos.piece_on(m.from_sq()));
      static const float piece_values[] = {0, 100, 320, 330, 500, 900, 0};
      score += piece_values[captured] * 6.0f - piece_values[attacker];

      // SEE bonus
      if (pos.see_ge(m, Value(0))) {
        score += 200.0f;
      }
    }

    // Promotions
    if (m.type_of() == PROMOTION) {
      PieceType promo = m.promotion_type();
      if (promo == QUEEN)
        score += 3000.0f;
      else if (promo == KNIGHT)
        score += 500.0f;
    }

    // Checks
    if (pos.gives_check(m)) {
      score += 300.0f;
    }

    // MCTS-specific history
    score +=
        static_cast<float>(mcts_history_[us][m.from_sq()][m.to_sq()]) / 100.0f;

    // Center control bonus
    int to_file = file_of(m.to_sq());
    int to_rank = rank_of(m.to_sq());
    float center_dist = std::abs(to_file - 3.5f) + std::abs(to_rank - 3.5f);
    score += (7.0f - center_dist) * 10.0f;

    // Piece development bonus in opening
    if (pos.count<ALL_PIECES>() > 28) {
      PieceType pt = type_of(pos.piece_on(m.from_sq()));
      if (pt == KNIGHT || pt == BISHOP) {
        int from_rank = rank_of(m.from_sq());
        if ((us == WHITE && from_rank == RANK_1) ||
            (us == BLACK && from_rank == RANK_8)) {
          score += 100.0f;
        }
      }
    }

    scores[idx++] = score;
    max_score = std::max(max_score, score);
  }

  // Softmax normalization with temperature
  float temperature = 1.0f;
  float sum = 0.0f;

  for (float &s : scores) {
    s = std::exp((s - max_score) / (temperature * 500.0f));
    sum += s;
  }

  idx = 0;
  for (const auto &m : moves) {
    policy.emplace_back(m, scores[idx] / sum);
    idx++;
  }

  return policy;
}

void ABPolicyGenerator::update_history(const Position &pos, Move best_move,
                                       const std::vector<Move> &searched_moves,
                                       int depth) {
  Color us = pos.side_to_move();
  int bonus = std::min(depth * depth, 400);

  // Bonus for best move with clamping to prevent int16_t overflow
  {
    int16_t &entry = mcts_history_[us][best_move.from_sq()][best_move.to_sq()];
    int new_value = static_cast<int>(entry) + bonus;
    entry = static_cast<int16_t>(std::clamp(
        new_value, static_cast<int>(INT16_MIN), static_cast<int>(INT16_MAX)));
  }

  // Penalty for other searched moves with clamping
  for (const Move &m : searched_moves) {
    if (m != best_move) {
      int16_t &entry = mcts_history_[us][m.from_sq()][m.to_sq()];
      int new_value = static_cast<int>(entry) - bonus / 2;
      entry = static_cast<int16_t>(std::clamp(
          new_value, static_cast<int>(INT16_MIN), static_cast<int>(INT16_MAX)));
    }
  }
}

// ============================================================================
// TacticalAnalyzer Implementation
// ============================================================================

TacticalAnalyzer::TacticalAnalyzer() = default;

void TacticalAnalyzer::initialize(TranspositionTable *tt) {
  tt_ = tt;
  qsearcher_.initialize(tt);
}

TacticalAnalyzer::TacticalInfo TacticalAnalyzer::analyze(const Position &pos) {
  TacticalInfo info;

  info.in_check = pos.checkers() != 0;

  // Count captures and checks
  MoveList<LEGAL> moves(pos);
  for (const auto &m : moves) {
    if (pos.capture(m)) {
      info.num_captures++;
    }
    if (pos.gives_check(m)) {
      info.num_checks++;
    }
  }

  info.has_forcing_moves = info.num_captures > 3 || info.num_checks > 1;

  // Check for hanging pieces
  Color us = pos.side_to_move();
  Color them = ~us;

  Bitboard dominated = pos.pieces(them);
  while (dominated) {
    Square s = pop_lsb(dominated);
    Bitboard attackers = pos.attackers_to(s, pos.pieces()) & pos.pieces(us);
    if (!attackers) {
      continue;
    }
    // Check if piece is defended
    Bitboard defenders = pos.attackers_to(s, pos.pieces()) & pos.pieces(them);
    if (popcount(attackers) > popcount(defenders)) {
      info.has_hanging_pieces = true;
      break;
    }
  }

  // Check for threats
  Bitboard our_pieces = pos.pieces(us);
  while (our_pieces) {
    Square s = pop_lsb(our_pieces);
    Bitboard attackers = pos.attackers_to(s, pos.pieces()) & pos.pieces(them);
    if (attackers) {
      PieceType pt = type_of(pos.piece_on(s));
      if (pt >= KNIGHT) {
        info.has_threats = true;
        break;
      }
    }
  }

  // Get quiescence score
  ABSearchConfig qconfig;
  qconfig.max_depth = 0;
  qsearcher_.set_config(qconfig);
  ABSearchResult qresult = qsearcher_.search(pos, 0);
  info.quiescence_score = qresult.score;

  // Static eval - use simple_eval since we don't have NNUE infrastructure
  info.tactical_score = Value(Eval::simple_eval(pos));

  return info;
}

bool TacticalAnalyzer::needs_deep_search(const Position &pos) {
  TacticalInfo info = analyze(pos);

  // Deep search needed for tactical positions
  if (info.in_check)
    return true;
  if (info.has_hanging_pieces)
    return true;
  if (info.num_captures >= 4)
    return true;
  if (info.num_checks >= 2)
    return true;

  // Check for significant eval difference
  Value diff = std::abs(info.tactical_score - info.quiescence_score);
  if (diff > 100)
    return true;

  return false;
}

std::vector<std::pair<Move, Value>>
TacticalAnalyzer::get_capture_scores(const Position &pos) {
  std::vector<std::pair<Move, Value>> captures;

  MoveList<LEGAL> moves(pos);
  for (const auto &m : moves) {
    if (pos.capture(m)) {
      Value see_value = pos.see_ge(m, Value(0)) ? Value(100) : Value(-100);
      captures.emplace_back(m, see_value);
    }
  }

  std::sort(captures.begin(), captures.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  return captures;
}

// ============================================================================
// HybridSearchBridge Implementation
// ============================================================================

HybridSearchBridge::HybridSearchBridge()
    : ab_searcher_(std::make_unique<ABSearcher>()),
      policy_generator_(std::make_unique<ABPolicyGenerator>()),
      tactical_analyzer_(std::make_unique<TacticalAnalyzer>()) {}

HybridSearchBridge::~HybridSearchBridge() = default;

void HybridSearchBridge::initialize(TranspositionTable *tt,
                                    GPU::GPUNNUEManager *gpu_manager) {
  tt_ = tt;
  gpu_manager_ = gpu_manager;

  ab_searcher_->initialize(tt);
  policy_generator_->initialize(nullptr, nullptr, tt);
  tactical_analyzer_->initialize(tt);

  initialized_ = true;
}

HybridSearchBridge::VerificationResult
HybridSearchBridge::verify_mcts_move(const Position &pos, Move mcts_move,
                                     int verification_depth,
                                     float override_threshold) {
  VerificationResult result;
  result.mcts_move = mcts_move;

  if (!initialized_) {
    result.verified = true;
    return result;
  }

  auto start = std::chrono::steady_clock::now();

  // Lock to serialize search operations - ABSearcher has non-thread-safe state
  std::lock_guard<std::mutex> search_lock(search_mutex_);

  // Run AB search
  ABSearchConfig config;
  config.max_depth = verification_depth;
  config.use_tt = true;
  ab_searcher_->set_config(config);

  ABSearchResult ab_result = ab_searcher_->search(pos, verification_depth);

  result.ab_move = ab_result.best_move;
  result.ab_score = ab_result.score;
  result.ab_depth = ab_result.depth;
  result.ab_pv = ab_result.pv;

  // Get MCTS move score
  Position test_pos;
  StateInfo st1, st2;
  test_pos.set(pos.fen(), false, &st1);

  if (test_pos.pseudo_legal(mcts_move) && test_pos.legal(mcts_move)) {
    test_pos.do_move(mcts_move, st2);

    // Quick qsearch for MCTS move
    ABSearchConfig qconfig;
    qconfig.max_depth = 0;
    ab_searcher_->set_config(qconfig);
    ABSearchResult mcts_result = ab_searcher_->search(test_pos, 0);
    result.mcts_score = -mcts_result.score;
  } else {
    result.mcts_score = VALUE_NONE;
  }

  // Calculate score difference in pawns
  if (result.mcts_score != VALUE_NONE && result.ab_score != VALUE_NONE) {
    result.score_difference =
        static_cast<float>(result.ab_score - result.mcts_score) / 100.0f;
  }

  // Determine if MCTS move is verified
  if (mcts_move == ab_result.best_move) {
    result.verified = true;
  } else if (result.score_difference < override_threshold) {
    result.verified = true;
  } else {
    result.override_mcts = true;
  }

  auto end = std::chrono::steady_clock::now();
  double time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Update statistics
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.verifications++;
    if (result.override_mcts) {
      stats_.overrides++;
    }
    stats_.ab_nodes += ab_searcher_->nodes_searched();
    stats_.avg_verification_time_ms =
        (stats_.avg_verification_time_ms * (stats_.verifications - 1) +
         time_ms) /
        stats_.verifications;
  }

  return result;
}

std::vector<std::pair<MCTSMove, float>>
HybridSearchBridge::get_enhanced_policy(const Position &pos) {
  std::vector<std::pair<MCTSMove, float>> mcts_policy;

  if (!initialized_) {
    // Return uniform policy
    MoveList<LEGAL> moves(pos);
    float uniform = moves.size() > 0 ? 1.0f / moves.size() : 0.0f;
    for (const auto &m : moves) {
      mcts_policy.emplace_back(MCTSMove::FromStockfish(m), uniform);
    }
    return mcts_policy;
  }

  // Lock to serialize access to policy_generator_ which has mutable
  // mcts_history_
  std::lock_guard<std::mutex> search_lock(search_mutex_);

  auto ab_policy = policy_generator_->generate_policy(pos);

  for (const auto &[move, prob] : ab_policy) {
    mcts_policy.emplace_back(MCTSMove::FromStockfish(move), prob);
  }

  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.policy_generations++;
  }

  return mcts_policy;
}

ABSearchResult HybridSearchBridge::run_ab_search(const Position &pos, int depth,
                                                 int time_ms) {
  if (!initialized_) {
    return ABSearchResult();
  }

  // Lock to serialize search operations - ABSearcher has non-thread-safe state
  std::lock_guard<std::mutex> search_lock(search_mutex_);

  ABSearchConfig config;
  config.max_depth = depth;
  config.max_time_ms = time_ms;
  config.use_tt = true;
  ab_searcher_->set_config(config);

  return ab_searcher_->search(pos, depth);
}

bool HybridSearchBridge::is_tactical_position(const Position &pos) {
  if (!initialized_) {
    return pos.checkers() != 0;
  }
  // Lock to serialize access to tactical_analyzer_ which has internal
  // ABSearcher
  std::lock_guard<std::mutex> search_lock(search_mutex_);
  return tactical_analyzer_->needs_deep_search(pos);
}

Value HybridSearchBridge::get_tactical_score(const Position &pos) {
  if (!initialized_) {
    return Eval::simple_eval(pos);
  }

  // Lock to serialize access to tactical_analyzer_ which has internal
  // ABSearcher
  std::lock_guard<std::mutex> search_lock(search_mutex_);
  auto info = tactical_analyzer_->analyze(pos);
  return info.quiescence_score;
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<HybridSearchBridge> g_hybrid_bridge;
static std::once_flag g_hybrid_bridge_init_flag;

HybridSearchBridge &hybrid_bridge() {
  std::call_once(g_hybrid_bridge_init_flag, []() {
    g_hybrid_bridge = std::make_unique<HybridSearchBridge>();
  });
  return *g_hybrid_bridge;
}

bool initialize_hybrid_bridge(TranspositionTable *tt,
                              GPU::GPUNNUEManager *gpu_manager) {
  hybrid_bridge().initialize(tt, gpu_manager);
  return true;
}

} // namespace MCTS
} // namespace MetalFish
