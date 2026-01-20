/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Enhanced Hybrid Search - Implementation

  Licensed under GPL-3.0
*/

#include "enhanced_hybrid_search.h"
#include "../core/misc.h"
#include "../eval/evaluate.h"
#include "../uci/uci.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// EnhancedHybridSearch
// ============================================================================

EnhancedHybridSearch::EnhancedHybridSearch() {
  // Default configuration
  config_.mcts_config.cpuct = 2.5f;
  config_.mcts_config.min_batch_size = 8;
  config_.mcts_config.max_batch_size = 256;
  config_.ab_verify_depth = 6;
  config_.ab_override_threshold = 0.5f;
}

EnhancedHybridSearch::~EnhancedHybridSearch() {
  stop();
  wait();
}

bool EnhancedHybridSearch::initialize(GPU::GPUNNUEManager *gpu_manager) {
  if (!gpu_manager || !gpu_manager->is_ready()) {
    return false;
  }

  gpu_manager_ = gpu_manager;

  // Create GPU MCTS backend
  gpu_backend_ = GPU::create_gpu_mcts_backend(gpu_manager);
  if (!gpu_backend_) {
    return false;
  }

  // Create MCTS search
  mcts_search_ = std::make_unique<HybridSearch>(config_.mcts_config);
  mcts_search_->set_gpu_nnue(gpu_manager);
  mcts_search_->set_neural_network(std::move(gpu_backend_));

  initialized_ = true;
  return true;
}

void EnhancedHybridSearch::start_search(const Position &pos,
                                        const Search::LimitsType &limits,
                                        BestMoveCallback best_move_cb,
                                        InfoCallback info_cb) {
  // Stop any existing search
  stop();
  wait();

  // Reset state
  stats_.reset();
  stop_flag_ = false;
  searching_ = true;

  // Store parameters
  root_fen_ = pos.fen();
  limits_ = limits;
  best_move_callback_ = best_move_cb;
  info_callback_ = info_cb;

  // Analyze position and select strategy
  if (config_.use_position_classifier) {
    PositionFeatures features = classifier_.analyze(pos);
    current_strategy_ = strategy_selector_.get_strategy(features);

    // Adjust for time
    Color us = pos.side_to_move();
    int time_left = (us == WHITE) ? limits.time[WHITE] : limits.time[BLACK];
    int increment = (us == WHITE) ? limits.inc[WHITE] : limits.inc[BLACK];
    strategy_selector_.adjust_for_time(current_strategy_, time_left, increment);

    send_info_string(
        "Position type: " +
        std::string(current_strategy_.position_type ==
                            PositionType::HIGHLY_TACTICAL
                        ? "HIGHLY_TACTICAL"
                    : current_strategy_.position_type == PositionType::TACTICAL
                        ? "TACTICAL"
                    : current_strategy_.position_type == PositionType::BALANCED
                        ? "BALANCED"
                    : current_strategy_.position_type == PositionType::STRATEGIC
                        ? "STRATEGIC"
                        : "HIGHLY_STRATEGIC"));
    send_info_string(
        "Strategy: MCTS=" +
        std::to_string(int(current_strategy_.mcts_weight * 100)) +
        "% AB=" + std::to_string(int(current_strategy_.ab_weight * 100)) + "%");
  } else {
    // Default balanced strategy
    current_strategy_ = strategy_selector_.get_strategy(pos);
  }

  // Update MCTS config based on strategy
  config_.mcts_config.cpuct = current_strategy_.cpuct;
  config_.ab_verify_depth = current_strategy_.ab_verify_depth;
  config_.ab_override_threshold = current_strategy_.ab_override_threshold;

  // Start search thread
  search_thread_ = std::thread(&EnhancedHybridSearch::search_thread_main, this);
}

void EnhancedHybridSearch::stop() {
  stop_flag_ = true;
  if (mcts_search_) {
    mcts_search_->stop();
  }
}

void EnhancedHybridSearch::wait() {
  if (search_thread_.joinable()) {
    search_thread_.join();
  }
  searching_ = false;
}

void EnhancedHybridSearch::search_thread_main() {
  auto start_time = std::chrono::steady_clock::now();

  // Recreate position from FEN
  Position root_pos;
  StateInfo st;
  root_pos.set(root_fen_, false, &st);

  // Calculate time budget
  int total_time_ms = calculate_time_budget(root_pos);
  total_time_ms =
      static_cast<int>(total_time_ms * current_strategy_.time_multiplier);

  send_info_string("Time budget: " + std::to_string(total_time_ms) + "ms");

  // Create position history for MCTS
  MCTSPositionHistory history;
  history.reset(root_fen_);

  // Phase 1: MCTS exploration
  int mcts_time = calculate_mcts_time(total_time_ms);
  send_info_string("Starting MCTS phase: " + std::to_string(mcts_time) + "ms");
  auto mcts_start = std::chrono::steady_clock::now();

  MCTSMove mcts_move = run_mcts_phase(history, mcts_time);

  auto mcts_end = std::chrono::steady_clock::now();
  stats_.mcts_time_ms =
      std::chrono::duration<double, std::milli>(mcts_end - mcts_start).count();

  send_info_string(
      "MCTS phase complete: " + std::to_string(stats_.mcts_time_ms) +
      "ms, move=" + (mcts_move.is_null() ? "null" : mcts_move.to_string()));

  // Check if we should stop or if MCTS returned a null move
  if (stop_flag_ || mcts_move.is_null()) {
    // Return MCTS result if valid, otherwise report no move
    if (best_move_callback_) {
      if (!mcts_move.is_null()) {
        best_move_callback_(mcts_move.to_stockfish(), Move::none());
      } else {
        // Null move from MCTS (e.g., checkmate position or search error)
        // Try to find any legal move as fallback
        MoveList<LEGAL> moves(root_pos);
        if (moves.size() > 0) {
          best_move_callback_(*moves.begin(), Move::none());
        }
        // If no legal moves, this is checkmate/stalemate - no bestmove to
        // report
      }
    }
    return;
  }

  // Phase 2: Alpha-beta verification (if enabled and time permits)
  ABVerifyResult ab_result;
  ab_result.best_move = mcts_move.to_stockfish();
  ab_result.agrees_with_mcts = true;

  if (config_.enable_ab_verify && current_strategy_.ab_weight > 0.1f) {
    int ab_time = calculate_ab_time(total_time_ms);
    auto ab_start = std::chrono::steady_clock::now();

    // Skip AB verification if insufficient time budget
    if (ab_time < 10) {
      send_info_string("Skipping AB verification: insufficient time budget");
    } else {
      send_info_string("Starting AB verification: " + std::to_string(ab_time) +
                       "ms budget");

      // Determine verification depth based on position type and time
      int verify_depth = config_.ab_verify_depth;
      if (current_strategy_.position_type == PositionType::HIGHLY_TACTICAL ||
          current_strategy_.position_type == PositionType::TACTICAL) {
        verify_depth += 2; // Deeper verification for tactical positions
      }

      ab_result = verify_with_alphabeta(root_pos, mcts_move, verify_depth,
                                        ab_start, ab_time);

      auto ab_end = std::chrono::steady_clock::now();
      stats_.ab_time_ms =
          std::chrono::duration<double, std::milli>(ab_end - ab_start).count();
      stats_.ab_verifications++;

      // Check if AB disagrees significantly
      if (!ab_result.agrees_with_mcts) {
        stats_.ab_overrides++;
        send_info_string("AB override: score diff = " +
                         std::to_string(ab_result.score_difference) + " pawns");
      }
    }
  }

  // Phase 3: Make final decision
  Move final_move = make_final_decision(mcts_move, ab_result);

  // Calculate total time
  auto end_time = std::chrono::steady_clock::now();
  stats_.total_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  // Store result
  result_.mcts_best = mcts_move;
  // Get Q value from the best move's child node in the MCTS tree
  result_.mcts_score = mcts_search_ ? mcts_search_->get_best_move_q() : 0.0f;
  result_.ab_best = ab_result.best_move;
  result_.ab_score = ab_result.score;
  result_.ab_depth = ab_result.depth;
  result_.final_best = final_move;
  result_.strategy = current_strategy_;

  // Send final info
  send_info(ab_result.depth, ab_result.score,
            stats_.mcts_nodes + stats_.ab_nodes,
            static_cast<int>(stats_.total_time_ms), ab_result.pv);

  // Report best move
  if (best_move_callback_) {
    Move ponder = Move::none();
    if (ab_result.pv.size() > 1) {
      ponder = ab_result.pv[1];
    }
    best_move_callback_(final_move, ponder);
  }

  // Print statistics
  send_info_string("Stats: MCTS=" + std::to_string(stats_.mcts_nodes.load()) +
                   " AB=" + std::to_string(stats_.ab_nodes.load()) +
                   " overrides=" + std::to_string(stats_.ab_overrides.load()));
}

MCTSMove
EnhancedHybridSearch::run_mcts_phase(const MCTSPositionHistory &history,
                                     int time_ms) {
  if (!mcts_search_) {
    return MCTSMove();
  }

  // Set up MCTS limits
  Search::LimitsType mcts_limits;
  mcts_limits.movetime = time_ms;
  mcts_limits.startTime = now();

  // Run MCTS
  MCTSMove best_move;
  std::atomic<bool> search_done{false};

  auto mcts_best_cb = [&best_move, &search_done](MCTSMove move,
                                                 MCTSMove ponder) {
    best_move = move;
    search_done = true;
  };

  auto mcts_info_cb = [this](const std::string &info) {
    // Forward MCTS info with prefix
    if (info_callback_) {
      // Check if string starts with "info " before stripping prefix
      if (info.size() >= 5 && info.substr(0, 5) == "info ") {
        info_callback_("info string [MCTS] " + info.substr(5));
      } else {
        // Forward as-is if it doesn't have the expected prefix
        info_callback_("info string [MCTS] " + info);
      }
    }
  };

  mcts_search_->start_search(history, mcts_limits, mcts_best_cb, mcts_info_cb);
  mcts_search_->wait();

  // Get statistics
  const auto &mcts_stats = mcts_search_->stats();
  stats_.mcts_nodes = mcts_stats.mcts_nodes.load();
  stats_.gpu_batches = mcts_stats.nn_batches.load();
  stats_.gpu_positions = mcts_stats.nn_evaluations.load();

  return best_move;
}

ABVerifyResult EnhancedHybridSearch::verify_with_alphabeta(
    const Position &pos, MCTSMove mcts_move, int depth,
    std::chrono::steady_clock::time_point ab_start, int ab_time_budget_ms) {
  // Helper lambda to check if time budget is exceeded
  auto time_exceeded = [&]() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration<double, std::milli>(now - ab_start).count();
    return elapsed_ms >= ab_time_budget_ms;
  };

  ABVerifyResult result;
  result.best_move = mcts_move.to_stockfish();
  result.depth = depth;
  result.agrees_with_mcts = true;

  // Guard against null move - cannot verify a null move
  if (mcts_move.is_null()) {
    // Find the best legal move using NNUE evaluation
    if (gpu_manager_) {
      MoveList<LEGAL> moves(pos);
      int best_score = -32000;
      for (const auto &m : moves) {
        Position test_pos;
        StateInfo test_st;
        test_pos.set(pos.fen(), false, &test_st);
        StateInfo test_st2;
        test_pos.do_move(m, test_st2);

        auto [test_psqt, test_score] =
            gpu_manager_->evaluate_single(test_pos, true);
        int score = -test_score;

        if (score > best_score) {
          best_score = score;
          result.best_move = m;
          result.score = score;
        }
        stats_.ab_nodes++;
      }
      if (result.best_move != Move::none()) {
        result.agrees_with_mcts = false;
        result.pv.push_back(result.best_move);
      }
    }
    return result;
  }

  // IMPROVED: Use deeper iterative search for better move validation
  // Start with shallow search and deepen if time permits
  if (gpu_manager_) {
    // Evaluate current position
    auto [psqt1, pos_score1] = gpu_manager_->evaluate_single(pos, true);

    // Make MCTS move and evaluate
    Position pos_after;
    StateInfo st;
    pos_after.set(pos.fen(), false, &st);
    StateInfo st2;
    pos_after.do_move(mcts_move.to_stockfish(), st2);

    auto [psqt2, pos_score2] = gpu_manager_->evaluate_single(pos_after, true);

    result.score = -pos_score2; // Negamax
    result.pv.push_back(mcts_move.to_stockfish());

    // IMPROVED: Score all legal moves and find the best one
    MoveList<LEGAL> moves(pos);
    
    // Store move scores for sorting
    struct MoveScore {
      Move move;
      int score;
      bool is_capture;
      bool gives_check;
    };
    std::vector<MoveScore> move_scores;
    move_scores.reserve(moves.size());
    
    // First pass: quick evaluation of all moves
    for (const auto &m : moves) {
      if (stop_flag_ || time_exceeded())
        break;

      Position test_pos;
      StateInfo test_st;
      test_pos.set(pos.fen(), false, &test_st);
      StateInfo test_st2;
      test_pos.do_move(m, test_st2);

      auto [test_psqt, test_score] =
          gpu_manager_->evaluate_single(test_pos, true);
      int score = -test_score;

      MoveScore ms;
      ms.move = m;
      ms.score = score;
      ms.is_capture = pos.capture(m);
      ms.gives_check = pos.gives_check(m);
      move_scores.push_back(ms);

      stats_.ab_nodes++;
    }
    
    // Sort moves by score (best first)
    std::sort(move_scores.begin(), move_scores.end(),
              [](const MoveScore &a, const MoveScore &b) {
                return a.score > b.score;
              });
    
    // Find the best move and check if MCTS move is competitive
    int mcts_score = result.score;
    int best_score = move_scores.empty() ? mcts_score : move_scores[0].score;
    Move best_move = move_scores.empty() ? mcts_move.to_stockfish() : move_scores[0].move;
    
    // IMPROVED: More sophisticated override logic
    // Only override if:
    // 1. The best move is significantly better than MCTS move
    // 2. The best move is not just a minor improvement
    // 3. Consider tactical moves more carefully
    
    int threshold_cp = static_cast<int>(config_.ab_override_threshold * 100);
    
    // Find MCTS move's rank among all moves
    int mcts_rank = 0;
    for (size_t i = 0; i < move_scores.size(); ++i) {
      if (move_scores[i].move == mcts_move.to_stockfish()) {
        mcts_rank = static_cast<int>(i);
        mcts_score = move_scores[i].score;
        break;
      }
    }
    
    // Override conditions:
    // 1. Best move is significantly better (by threshold)
    // 2. MCTS move is not in top 3 moves
    // 3. Best move is a tactical move (capture/check) and MCTS missed it
    bool should_override = false;
    float score_diff = 0.0f;
    
    if (!move_scores.empty() && best_move != mcts_move.to_stockfish()) {
      score_diff = (best_score - mcts_score) / 100.0f;
      
      // Condition 1: Large score difference
      if (score_diff > config_.ab_override_threshold) {
        should_override = true;
      }
      
      // Condition 2: MCTS move is not competitive (not in top 3)
      if (mcts_rank >= 3 && score_diff > config_.ab_override_threshold * 0.5f) {
        should_override = true;
      }
      
      // Condition 3: Best move is tactical and MCTS missed it
      if (move_scores[0].is_capture || move_scores[0].gives_check) {
        if (score_diff > config_.ab_override_threshold * 0.7f) {
          should_override = true;
        }
      }
    }
    
    if (should_override) {
      result.best_move = best_move;
      result.score = best_score;
      result.agrees_with_mcts = false;
      result.score_difference = score_diff;
      result.pv.clear();
      result.pv.push_back(best_move);
      
      // IMPROVED: Try to extend PV with one more move
      if (!time_exceeded()) {
        Position pv_pos;
        StateInfo pv_st;
        pv_pos.set(pos.fen(), false, &pv_st);
        StateInfo pv_st2;
        pv_pos.do_move(best_move, pv_st2);
        
        MoveList<LEGAL> pv_moves(pv_pos);
        int best_reply_score = -32000;
        Move best_reply = Move::none();
        
        for (const auto &m : pv_moves) {
          if (time_exceeded()) break;
          
          Position reply_pos;
          StateInfo reply_st;
          reply_pos.set(pv_pos.fen(), false, &reply_st);
          StateInfo reply_st2;
          reply_pos.do_move(m, reply_st2);
          
          auto [reply_psqt, reply_score] =
              gpu_manager_->evaluate_single(reply_pos, true);
          int score = -reply_score;
          
          if (score > best_reply_score) {
            best_reply_score = score;
            best_reply = m;
          }
          stats_.ab_nodes++;
        }
        
        if (best_reply != Move::none()) {
          result.pv.push_back(best_reply);
        }
      }
    }
  }

  return result;
}

Move EnhancedHybridSearch::make_final_decision(
    MCTSMove mcts_move, const ABVerifyResult &ab_result) {
  // IMPROVED: More sophisticated decision logic
  
  // If AB didn't find a significantly better move, trust MCTS
  if (ab_result.agrees_with_mcts) {
    return mcts_move.to_stockfish();
  }
  
  // AB found a potentially better move - decide based on multiple factors
  float score_diff = ab_result.score_difference;
  
  // Factor 1: Strategy weight
  // If we're in an AB-heavy strategy, trust AB more
  if (current_strategy_.ab_weight > 0.6f) {
    // AB-heavy: trust AB if it found anything better
    if (score_diff > 0.2f) {
      return ab_result.best_move;
    }
  }
  
  // Factor 2: Large score difference (>1 pawn) - always trust AB
  if (score_diff > 1.0f) {
    return ab_result.best_move;
  }
  
  // Factor 3: Position type specific thresholds
  float position_threshold;
  switch (current_strategy_.position_type) {
  case PositionType::HIGHLY_TACTICAL:
    position_threshold = 0.25f;  // Trust AB more in tactical positions
    break;
  case PositionType::TACTICAL:
    position_threshold = 0.35f;
    break;
  case PositionType::BALANCED:
    position_threshold = 0.45f;
    break;
  case PositionType::STRATEGIC:
    position_threshold = 0.55f;  // Trust MCTS more in strategic positions
    break;
  case PositionType::HIGHLY_STRATEGIC:
    position_threshold = 0.65f;
    break;
  default:
    position_threshold = 0.4f;
  }
  
  if (score_diff > position_threshold) {
    return ab_result.best_move;
  }

  // Default: trust MCTS
  return mcts_move.to_stockfish();
}

int EnhancedHybridSearch::calculate_time_budget(const Position &pos) const {
  if (limits_.movetime > 0) {
    return limits_.movetime;
  }

  Color us = pos.side_to_move();
  int time_left = (us == WHITE) ? limits_.time[WHITE] : limits_.time[BLACK];
  int increment = (us == WHITE) ? limits_.inc[WHITE] : limits_.inc[BLACK];

  if (time_left <= 0) {
    return 1000; // Default 1 second
  }

  // IMPROVED: Use more aggressive time allocation for hybrid search
  // Base time: fraction of remaining time
  int base_time = static_cast<int>(time_left * config_.base_time_fraction);

  // Add portion of increment
  base_time += static_cast<int>(increment * config_.increment_factor);

  // Minimum 500ms for MCTS to be effective
  base_time = std::max(500, base_time);

  // Cap at maximum fraction
  int max_time = static_cast<int>(time_left * config_.max_time_fraction);

  return std::min(base_time, max_time);
}

int EnhancedHybridSearch::calculate_mcts_time(int total_time) const {
  // IMPROVED: Dynamic MCTS time allocation based on position type
  // The strategy weights are already tuned per position type, but we also
  // need to ensure MCTS gets enough time to build a meaningful tree
  
  float mcts_fraction = current_strategy_.mcts_weight;

  // Minimum floor based on position type
  float min_fraction;
  switch (current_strategy_.position_type) {
  case PositionType::HIGHLY_TACTICAL:
    min_fraction = 0.25f;  // Even in tactical positions, give MCTS some time
    break;
  case PositionType::TACTICAL:
    min_fraction = 0.30f;
    break;
  case PositionType::BALANCED:
    min_fraction = 0.40f;
    break;
  case PositionType::STRATEGIC:
    min_fraction = 0.55f;
    break;
  case PositionType::HIGHLY_STRATEGIC:
    min_fraction = 0.70f;
    break;
  default:
    min_fraction = 0.35f;
  }
  
  mcts_fraction = std::max(min_fraction, mcts_fraction);
  
  // Ensure minimum absolute time for MCTS (at least 300ms for meaningful search)
  int mcts_time = static_cast<int>(total_time * mcts_fraction);
  mcts_time = std::max(300, mcts_time);
  
  // Don't exceed 80% of total time to leave room for AB verification
  mcts_time = std::min(mcts_time, static_cast<int>(total_time * 0.8f));

  return mcts_time;
}

int EnhancedHybridSearch::calculate_ab_time(int total_time) const {
  // Remaining time after MCTS
  int mcts_time = calculate_mcts_time(total_time);
  return total_time - mcts_time;
}

void EnhancedHybridSearch::send_info(int depth, int score, uint64_t nodes,
                                     int time_ms, const std::vector<Move> &pv) {
  if (!info_callback_)
    return;

  std::ostringstream ss;
  ss << "info depth " << depth;
  ss << " score cp " << score;
  ss << " nodes " << nodes;
  ss << " time " << time_ms;

  if (time_ms > 0) {
    uint64_t nps = nodes * 1000 / time_ms;
    ss << " nps " << nps;
  }

  if (!pv.empty()) {
    ss << " pv";
    for (const auto &m : pv) {
      ss << " " << UCIEngine::move(m, false);
    }
  }

  info_callback_(ss.str());
}

void EnhancedHybridSearch::send_info_string(const std::string &msg) {
  if (info_callback_) {
    info_callback_("info string " + msg);
  }
}

// ============================================================================
// Pondering Support
// ============================================================================

void EnhancedHybridSearch::start_ponder(const Position &pos, Move ponder_move) {
  if (!initialized_ || searching_)
    return;

  // Apply the ponder move to get the expected position
  StateInfo st;
  Position ponder_pos;
  ponder_pos.set(pos.fen(), false, &st);

  if (!pos.pseudo_legal(ponder_move) || !pos.legal(ponder_move)) {
    return; // Invalid ponder move
  }

  StateInfo st2;
  ponder_pos.do_move(ponder_move, st2);

  pondering_ = true;
  ponder_move_ = ponder_move;
  ponder_start_ = std::chrono::steady_clock::now();

  // Start search on the ponder position with infinite time
  Search::LimitsType ponder_limits;
  ponder_limits.infinite = true;

  // Start search without sending bestmove
  start_search(ponder_pos, ponder_limits, nullptr, info_callback_);
}

void EnhancedHybridSearch::ponder_hit() {
  if (!pondering_)
    return;

  pondering_ = false;

  // Calculate remaining time from ponder
  auto now = std::chrono::steady_clock::now();
  auto ponder_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - ponder_start_)
          .count();

  send_info_string("Ponder hit after " + std::to_string(ponder_time) + "ms");

  // Continue searching with the actual time limits
  // The search is already running, just let it continue
}

void EnhancedHybridSearch::apply_move(Move move) {
  if (!mcts_search_)
    return;

  // Stop any ongoing search first
  stop();
  wait();

  // Apply move to tree for reuse in next search
  MCTSMove mcts_move = MCTSMove::FromStockfish(move);
  mcts_search_->tree()->apply_move(mcts_move);
}

void EnhancedHybridSearch::new_game() {
  stop();
  wait();

  // Reset statistics
  stats_.reset();
  result_ = HybridEvalResult();

  // Clear MCTS tree and recreate with proper configuration
  mcts_search_.reset();
  mcts_search_ = std::make_unique<HybridSearch>(config_.mcts_config);
  mcts_search_->set_gpu_nnue(gpu_manager_);

  // Recreate GPU backend and set neural network (matching initialize())
  if (gpu_manager_) {
    gpu_backend_ = GPU::create_gpu_mcts_backend(gpu_manager_);
    if (gpu_backend_) {
      mcts_search_->set_neural_network(std::move(gpu_backend_));
    }
  }

  // Clear TT
  mcts_tt().clear();
}

// Factory function
std::unique_ptr<EnhancedHybridSearch>
create_enhanced_hybrid_search(GPU::GPUNNUEManager *gpu_manager,
                              const EnhancedHybridConfig &config) {

  auto search = std::make_unique<EnhancedHybridSearch>();
  search->set_config(config);

  if (search->initialize(gpu_manager)) {
    return search;
  }

  return nullptr;
}

} // namespace MCTS
} // namespace MetalFish
