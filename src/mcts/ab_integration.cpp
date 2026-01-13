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
      PieceType captured = type_of(pos.piece_on(m.to_sq()));
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
    score += static_cast<float>(main_history_[us][m.from_sq()][m.to_sq()]) /
             100.0f;

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
  }

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
      PieceType captured = type_of(pos.piece_on(m.to_sq()));
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

  std::stable_sort(scored_moves.begin(), scored_moves.end(),
                   [](const auto &a, const auto &b) {
                     return a.second > b.second;
                   });

  Move best_move = Move::none();
  Value best_value = -VALUE_INFINITE;
  int move_count = 0;
  bool improving = !in_check && static_eval > alpha;

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

          // Update history
          if (config_.use_history && !is_capture) {
            int bonus = std::min(depth * depth, 400);
            Color us = pos.side_to_move();
            main_history_[us][m.from_sq()][m.to_sq()] += bonus;
          }

          break;
        }
      }
    }
  }

  // Store in TT
  if (config_.use_tt && tt_ && !should_stop()) {
    Bound bound = best_value >= beta   ? BOUND_LOWER
                  : best_value > alpha ? BOUND_EXACT
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
  return Eval::simple_eval(pos);
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

} // namespace MCTS
} // namespace MetalFish
