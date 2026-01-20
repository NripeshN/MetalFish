/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Stockfish Search Wrapper Implementation

  Licensed under GPL-3.0
*/

#include "stockfish_search_wrapper.h"
#include "../core/bitboard.h"
#include "../core/memory.h"
#include "../core/misc.h"
#include "../core/movegen.h"
#include "../core/numa.h"
#include "../eval/evaluate.h"
#include "../eval/nnue/network.h"
#include "../search/history.h"
#include "../search/search.h"
#include "../search/thread.h"
#include "../search/timeman.h"
#include "../search/tt.h"
#include "../uci/ucioption.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// StockfishSearchWrapper Implementation
// ============================================================================

StockfishSearchWrapper::StockfishSearchWrapper() = default;

StockfishSearchWrapper::~StockfishSearchWrapper() = default;

bool StockfishSearchWrapper::initialize() {
  if (initialized_) {
    return true;
  }

  try {
    // Create a dedicated transposition table for hybrid search
    // Use a smaller size to avoid memory pressure
    tt_ = std::make_unique<TranspositionTable>();
    // Note: We can't resize without a ThreadPool, so just use default size

    // Create options map with defaults
    options_ = std::make_unique<OptionsMap>();

    // Set up essential options
    options_->add("Threads", Option(1, 1, 1024));
    options_->add("Hash", Option(64, 1, 33554432));
    options_->add("MultiPV", Option(1, 1, 500));
    options_->add("UCI_Chess960", Option(false));
    options_->add("Move Overhead", Option(10, 0, 5000));
    options_->add("nodestime", Option(0, 0, 10000));
    options_->add("Skill Level", Option(20, 0, 20));
    options_->add("UCI_LimitStrength", Option(false));
    options_->add("UCI_Elo", Option(3190, 1320, 3190));
    options_->add("SyzygyPath", Option(""));
    options_->add("SyzygyProbeDepth", Option(1, 1, 100));
    options_->add("Syzygy50MoveRule", Option(true));
    options_->add("SyzygyProbeLimit", Option(7, 0, 7));

    initialized_ = true;
    return true;
  } catch (...) {
    return false;
  }
}

StockfishSearchResult
StockfishSearchWrapper::search(const Position &pos,
                               const StockfishSearchConfig &config) {
  if (!initialized_) {
    return StockfishSearchResult();
  }

  std::lock_guard<std::mutex> lock(search_mutex_);

  Search::LimitsType limits;
  limits.startTime = now();

  if (config.max_depth > 0) {
    limits.depth = config.max_depth;
  }
  if (config.max_time_ms > 0) {
    limits.movetime = config.max_time_ms;
  }
  if (config.max_nodes > 0) {
    limits.nodes = config.max_nodes;
  }

  return run_search_internal(pos.fen(), limits);
}

StockfishSearchResult StockfishSearchWrapper::quick_search(const Position &pos,
                                                           int depth,
                                                           int time_ms) {
  StockfishSearchConfig config;
  config.max_depth = depth;
  config.max_time_ms = time_ms;
  config.num_threads = 1;
  return search(pos, config);
}

bool StockfishSearchWrapper::verify_move(const Position &pos, Move move,
                                         int depth, float threshold_pawns,
                                         Value *out_score) {
  auto result = quick_search(pos, depth);

  if (out_score) {
    *out_score = result.score;
  }

  // Move is verified if it's the best move
  if (result.best_move == move) {
    return true;
  }

  // Or if it's within threshold of the best
  Value move_score = get_move_score(pos, move, depth);
  Value threshold = static_cast<Value>(threshold_pawns * 100);

  return move_score >= result.score - threshold;
}

std::vector<std::pair<Move, float>>
StockfishSearchWrapper::get_move_policy(const Position &pos, int search_depth) {
  std::vector<std::pair<Move, float>> policy;

  MoveList<LEGAL> moves(pos);
  if (moves.size() == 0) {
    return policy;
  }

  // Do a quick search to get the best move and TT info
  auto result = quick_search(pos, search_depth);

  // Score each move
  std::vector<std::pair<Move, float>> scored_moves;
  float max_score = -1e9f;

  for (const auto &m : moves) {
    float score = 0.0f;

    // Best move from search gets highest score
    if (m == result.best_move) {
      score += 10000.0f;
    }

    // Second best (ponder move) gets high score too
    if (m == result.ponder_move) {
      score += 5000.0f;
    }

    // Captures scored by MVV-LVA
    if (pos.capture(m)) {
      PieceType captured =
          m.type_of() == EN_PASSANT ? PAWN : type_of(pos.piece_on(m.to_sq()));
      PieceType attacker = type_of(pos.piece_on(m.from_sq()));
      static const float piece_values[] = {0, 100, 320, 330, 500, 900, 0};
      score += piece_values[captured] * 6.0f - piece_values[attacker];

      // SEE bonus for good captures
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

    // Center control
    int to_file = file_of(m.to_sq());
    int to_rank = rank_of(m.to_sq());
    float center_dist = std::abs(to_file - 3.5f) + std::abs(to_rank - 3.5f);
    score += (7.0f - center_dist) * 10.0f;

    scored_moves.emplace_back(m, score);
    max_score = std::max(max_score, score);
  }

  // Softmax normalization
  float sum = 0.0f;
  for (auto &[move, score] : scored_moves) {
    score = std::exp((score - max_score) / 500.0f);
    sum += score;
  }

  for (auto &[move, score] : scored_moves) {
    policy.emplace_back(move, score / sum);
  }

  return policy;
}

Value StockfishSearchWrapper::get_move_score(const Position &pos, Move move,
                                             int depth) {
  // Make the move and search
  Position search_pos;
  StateInfo st1, st2;
  search_pos.set(pos.fen(), false, &st1);

  if (!search_pos.pseudo_legal(move) || !search_pos.legal(move)) {
    return VALUE_NONE;
  }

  search_pos.do_move(move, st2);
  auto result = quick_search(search_pos, depth - 1);

  // Negate because we're searching from opponent's perspective
  return -result.score;
}

void StockfishSearchWrapper::stop() {
  // No-op for now since we run searches synchronously
}

double StockfishSearchWrapper::avg_search_time_ms() const {
  uint64_t searches = total_searches_.load();
  if (searches == 0)
    return 0.0;
  return total_time_ms_.load() / searches;
}

void StockfishSearchWrapper::reset_stats() {
  total_nodes_ = 0;
  total_searches_ = 0;
  total_time_ms_ = 0;
}

StockfishSearchResult
StockfishSearchWrapper::run_search_internal(const std::string &fen,
                                            const Search::LimitsType &limits) {
  StockfishSearchResult result;

  auto start = std::chrono::steady_clock::now();

  // We need to create a minimal search infrastructure
  // This is a simplified approach that doesn't require the full Engine

  // Create position
  Position pos;
  StateInfo st;
  pos.set(fen, false, &st);

  // Generate legal moves
  MoveList<LEGAL> moves(pos);
  if (moves.size() == 0) {
    // No legal moves - checkmate or stalemate
    result.score = pos.checkers() ? -VALUE_MATE : VALUE_DRAW;
    return result;
  }

  // For now, use a simplified iterative deepening search
  // This is a fallback - the real power comes from using the full Engine

  // Use the simplified ABSearcher but with NNUE evaluation
  // We'll improve this to use the full search later

  Move best_move = *moves.begin();
  Value best_score = -VALUE_INFINITE;
  int target_depth = limits.depth > 0 ? limits.depth : 12;

  // Simple alpha-beta with NNUE eval
  std::function<Value(Position &, int, Value, Value, std::vector<StateInfo> &)>
      search_fn;

  search_fn = [&](Position &p, int depth, Value alpha, Value beta,
                  std::vector<StateInfo> &states) -> Value {
    if (depth <= 0) {
      // Use simple evaluation since we don't have full NNUE infrastructure
      // The main MCTS uses GPU NNUE for strong evaluation
      return Value(Eval::simple_eval(p));
    }

    MoveList<LEGAL> legal_moves(p);
    if (legal_moves.size() == 0) {
      return p.checkers() ? -VALUE_MATE + (target_depth - depth) : VALUE_DRAW;
    }

    Value best = -VALUE_INFINITE;

    for (const auto &m : legal_moves) {
      states.emplace_back();
      p.do_move(m, states.back());
      Value score = -search_fn(p, depth - 1, -beta, -alpha, states);
      p.undo_move(m);
      states.pop_back();

      if (score > best) {
        best = score;
        if (score > alpha) {
          alpha = score;
          if (score >= beta) {
            break;
          }
        }
      }
    }

    return best;
  };

  // Search each move at root
  std::vector<StateInfo> states;
  states.reserve(128);

  for (const auto &m : moves) {
    states.emplace_back();
    pos.do_move(m, states.back());
    Value score = -search_fn(pos, target_depth - 1, -VALUE_INFINITE,
                             -best_score + 1, states);
    pos.undo_move(m);
    states.pop_back();

    if (score > best_score) {
      best_score = score;
      best_move = m;
    }
  }

  result.best_move = best_move;
  result.score = best_score;
  result.depth = target_depth;

  // Check for mate
  if (best_score > VALUE_MATE_IN_MAX_PLY) {
    result.is_mate = true;
    result.mate_in = (VALUE_MATE - best_score + 1) / 2;
  } else if (best_score < VALUE_MATED_IN_MAX_PLY) {
    result.is_mate = true;
    result.mate_in = -(VALUE_MATE + best_score) / 2;
  }

  result.pv.push_back(best_move);

  auto end = std::chrono::steady_clock::now();
  double time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Update stats
  total_searches_++;
  total_time_ms_ = total_time_ms_.load() + time_ms;

  return result;
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<StockfishSearchWrapper> g_stockfish_search;
static std::once_flag g_stockfish_search_init_flag;

StockfishSearchWrapper &stockfish_search() {
  std::call_once(g_stockfish_search_init_flag, []() {
    g_stockfish_search = std::make_unique<StockfishSearchWrapper>();
  });
  return *g_stockfish_search;
}

bool initialize_stockfish_search() {
  return stockfish_search().initialize();
}

} // namespace MCTS
} // namespace MetalFish
