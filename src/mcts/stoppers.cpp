/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Search Stoppers - Time and node management
  Licensed under GPL-3.0
*/

#include "stoppers.h"

#include <algorithm>
#include <cmath>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// ChainedStopper
// ============================================================================

void ChainedStopper::Add(std::unique_ptr<SearchStopper> s) {
  stoppers_.push_back(std::move(s));
}

bool ChainedStopper::ShouldStop(const SearchStats &stats) {
  for (auto &s : stoppers_) {
    if (s->ShouldStop(stats)) return true;
  }
  return false;
}

// ============================================================================
// TimeLimitStopper
// ============================================================================

TimeLimitStopper::TimeLimitStopper(int64_t limit_ms) : limit_ms_(limit_ms) {}

bool TimeLimitStopper::ShouldStop(const SearchStats &stats) {
  return stats.time_since_movestart_ms >= limit_ms_;
}

// ============================================================================
// NodeLimitStopper
// ============================================================================

NodeLimitStopper::NodeLimitStopper(uint64_t limit) : limit_(limit) {}

bool NodeLimitStopper::ShouldStop(const SearchStats &stats) {
  return stats.total_nodes >= limit_;
}

// ============================================================================
// SmartPruningStopper
// ============================================================================

SmartPruningStopper::SmartPruningStopper(float factor, int64_t time_limit_ms)
    : factor_(factor), time_limit_ms_(time_limit_ms) {}

bool SmartPruningStopper::ShouldStop(const SearchStats &stats) {
  if (stats.edge_n.size() <= 1) return true;

  if (stats.nodes_since_movestart % 32 != 0) return false;

  uint32_t best_n = 0, second_n = 0;
  for (uint32_t n : stats.edge_n) {
    if (n > best_n) {
      second_n = best_n;
      best_n = n;
    } else if (n > second_n) {
      second_n = n;
    }
  }

  int64_t elapsed = std::max(stats.time_since_movestart_ms, int64_t(1));
  double nps = static_cast<double>(stats.nodes_since_movestart) * 1000.0 /
               static_cast<double>(elapsed);

  int64_t remaining_time =
      std::max(int64_t(0), time_limit_ms_ - stats.time_since_movestart_ms);
  double remaining_playouts = remaining_time / 1000.0 * nps;

  if (static_cast<double>(best_n) >
      static_cast<double>(second_n) + remaining_playouts / factor_) {
    return true;
  }

  return false;
}

// ============================================================================
// KLDGainStopper
// ============================================================================

KLDGainStopper::KLDGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}

bool KLDGainStopper::ShouldStop(const SearchStats &stats) {
  const double new_child_nodes = static_cast<double>(stats.total_nodes) - 1.0;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;

  const auto& new_visits = stats.edge_n;
  if (!prev_visits_.empty() && prev_visits_.size() == new_visits.size()) {
    double kldgain = 0.0;
    for (size_t i = 0; i < new_visits.size(); ++i) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = static_cast<double>(new_visits[i]) / new_child_nodes;
      if (prev_visits_[i] > 0 && n_p > 0) {
        kldgain += o_p * std::log(o_p / n_p);
      }
    }
    double per_node = kldgain / (new_child_nodes - prev_child_nodes_);
    if (per_node < static_cast<double>(min_gain_)) {
      return true;
    }
  }

  prev_visits_.clear();
  prev_visits_.reserve(new_visits.size());
  for (uint32_t v : new_visits) {
    prev_visits_.push_back(static_cast<double>(v));
  }
  prev_child_nodes_ = new_child_nodes;
  return false;
}

// ============================================================================
// SigmoidTimeManager
// ============================================================================

std::unique_ptr<SearchStopper>
SigmoidTimeManager::CreateStopper(Color /*us*/, int64_t time_left, int64_t inc,
                                  int movestogo, int ply, uint64_t node_limit,
                                  int64_t movetime, bool infinite,
                                  float smart_pruning_factor) {
  if (movetime > 0) {
    return std::make_unique<TimeLimitStopper>(movetime);
  }

  if (infinite) return nullptr;

  double est_moves = 10.0 + 50.0 / (1.0 + std::exp((ply - 45.0) / 6.0));

  if (movestogo > 0) {
    est_moves = static_cast<double>(movestogo);
  }

  double total_time =
      static_cast<double>(time_left) +
      static_cast<double>(inc) * std::max(0.0, est_moves - 1.0);

  int64_t time_for_move =
      static_cast<int64_t>(total_time / est_moves);

  int64_t max_time = static_cast<int64_t>(time_left * 0.3);
  time_for_move = std::min(time_for_move, max_time);
  time_for_move = std::max(time_for_move, int64_t(500));

  auto chain = std::make_unique<ChainedStopper>();
  chain->Add(std::make_unique<TimeLimitStopper>(time_for_move));
  chain->Add(
      std::make_unique<SmartPruningStopper>(smart_pruning_factor, time_for_move));

  if (node_limit > 0) {
    chain->Add(std::make_unique<NodeLimitStopper>(node_limit));
  }

  return chain;
}

} // namespace MCTS
} // namespace MetalFish
