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

  // Simple alpha-beta search for verification
  // In a full implementation, this would use Stockfish's search

  // For now, use NNUE evaluation to verify the move
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

    // Check all legal moves for better options
    MoveList<LEGAL> moves(pos);
    int mcts_score =
        result.score; // Original MCTS score for threshold comparison
    int best_score = result.score;
    Move best_move = mcts_move.to_stockfish();

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

      // Compare against original MCTS score (not sliding best_score) to find
      // moves that are significantly better than MCTS, then track the actual
      // best. Use configurable threshold (convert from pawns to centipawns).
      int threshold_cp = static_cast<int>(config_.ab_override_threshold * 100);
      if (score > mcts_score + threshold_cp && score > best_score) {
        best_score = score;
        best_move = m;
      }

      stats_.ab_nodes++;
    }

    // Check if we found a better move
    if (best_move != mcts_move.to_stockfish()) {
      float score_diff =
          (best_score - result.score) / 100.0f; // Convert to pawns

      if (score_diff > config_.ab_override_threshold) {
        result.best_move = best_move;
        result.score = best_score;
        result.agrees_with_mcts = false;
        result.score_difference = score_diff;
        result.pv.clear();
        result.pv.push_back(best_move);
      }
    }
  }

  return result;
}

Move EnhancedHybridSearch::make_final_decision(
    MCTSMove mcts_move, const ABVerifyResult &ab_result) {
  // If AB found a significantly better move, use it
  if (!ab_result.agrees_with_mcts) {
    // Weight the decision based on strategy
    if (current_strategy_.ab_weight > 0.5f) {
      // AB-heavy strategy: trust AB override
      return ab_result.best_move;
    } else if (ab_result.score_difference > 1.0f) {
      // Large score difference: trust AB even in MCTS-heavy strategy
      return ab_result.best_move;
    }
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

  // Base time: fraction of remaining time
  int base_time = static_cast<int>(time_left * config_.base_time_fraction);

  // Add portion of increment
  base_time += static_cast<int>(increment * config_.increment_factor);

  // Cap at maximum fraction
  int max_time = static_cast<int>(time_left * config_.max_time_fraction);

  return std::min(base_time, max_time);
}

int EnhancedHybridSearch::calculate_mcts_time(int total_time) const {
  // Allocate time based on strategy weights from position classifier
  // The strategy weights are already tuned per position type:
  // - HIGHLY_TACTICAL: 20% MCTS, 80% AB
  // - TACTICAL: 35% MCTS, 65% AB
  // - BALANCED: 50% MCTS, 50% AB
  // - STRATEGIC: 65% MCTS, 35% AB
  // - HIGHLY_STRATEGIC: 80% MCTS, 20% AB
  float mcts_fraction = current_strategy_.mcts_weight;

  // Use a minimal floor (10%) only to prevent zero allocation,
  // but respect the position-based strategy weights
  mcts_fraction = std::max(0.1f, mcts_fraction);

  return static_cast<int>(total_time * mcts_fraction);
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

  // Apply move to tree for reuse in next search

  // Reset for next search
  stop();
  wait();
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
