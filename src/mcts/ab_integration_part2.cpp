/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Alpha-Beta Integration - Part 2 (Policy Generator, Tactical Analyzer, Bridge)

  Licensed under GPL-3.0
*/

#include "ab_integration.h"
#include "../core/bitboard.h"
#include "../core/movegen.h"
#include <algorithm>
#include <cmath>

namespace MetalFish {
namespace MCTS {

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
      PieceType captured = type_of(pos.piece_on(m.to_sq()));
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

    // History scores - skip if using external history (complex type)
    // We use our own simplified mcts_history_ instead

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

  // Bonus for best move
  mcts_history_[us][best_move.from_sq()][best_move.to_sq()] += bonus;

  // Penalty for other searched moves
  for (const Move &m : searched_moves) {
    if (m != best_move) {
      mcts_history_[us][m.from_sq()][m.to_sq()] -= bonus / 2;
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

  // Static eval
  info.tactical_score = Eval::simple_eval(pos);

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
  return tactical_analyzer_->needs_deep_search(pos);
}

Value HybridSearchBridge::get_tactical_score(const Position &pos) {
  if (!initialized_) {
    return Eval::simple_eval(pos);
  }

  auto info = tactical_analyzer_->analyze(pos);
  return info.quiescence_score;
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<HybridSearchBridge> g_hybrid_bridge;

HybridSearchBridge &hybrid_bridge() {
  if (!g_hybrid_bridge) {
    g_hybrid_bridge = std::make_unique<HybridSearchBridge>();
  }
  return *g_hybrid_bridge;
}

bool initialize_hybrid_bridge(TranspositionTable *tt,
                              GPU::GPUNNUEManager *gpu_manager) {
  hybrid_bridge().initialize(tt, gpu_manager);
  return true;
}

} // namespace MCTS
} // namespace MetalFish
