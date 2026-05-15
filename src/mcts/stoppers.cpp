/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  Licensed under GPL-3.0
*/

#include "stoppers.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace MetalFish {
namespace MCTS {

void StoppersHints::Reset() {
  estimated_remaining_time_ms_.reset();
  estimated_remaining_playouts_.reset();
  estimated_nps_.reset();
}

void StoppersHints::UpdateEstimatedRemainingTimeMs(int64_t v) {
  if (!estimated_remaining_time_ms_ || v < *estimated_remaining_time_ms_)
    estimated_remaining_time_ms_ = v;
}

void StoppersHints::UpdateEstimatedRemainingPlayouts(int64_t v) {
  if (!estimated_remaining_playouts_ || v < *estimated_remaining_playouts_)
    estimated_remaining_playouts_ = v;
}

void StoppersHints::UpdateEstimatedNps(float v) { estimated_nps_ = v; }

int64_t StoppersHints::GetEstimatedRemainingTimeMs() const {
  return estimated_remaining_time_ms_.value_or(
      std::numeric_limits<int64_t>::max());
}

int64_t StoppersHints::GetEstimatedRemainingPlayouts() const {
  return std::max<int64_t>(1, estimated_remaining_playouts_.value_or(
                                  std::numeric_limits<int64_t>::max()));
}

std::optional<float> StoppersHints::GetEstimatedNps() const {
  return estimated_nps_;
}

bool StoppersHints::HasEstimatedRemainingPlayouts() const {
  return estimated_remaining_playouts_.has_value();
}

void ChainedStopper::Add(std::unique_ptr<SearchStopper> s) {
  if (s)
    stoppers_.push_back(std::move(s));
}

bool ChainedStopper::ShouldStop(const SearchStats &stats,
                                StoppersHints *hints) {
  for (auto &s : stoppers_) {
    if (s->ShouldStop(stats, hints))
      return true;
  }
  return false;
}

void ChainedStopper::OnSearchDone(const SearchStats &stats) {
  for (auto &s : stoppers_)
    s->OnSearchDone(stats);
}

TimeLimitStopper::TimeLimitStopper(int64_t limit_ms) : limit_ms_(limit_ms) {}

bool TimeLimitStopper::ShouldStop(const SearchStats &stats,
                                  StoppersHints *hints) {
  if (hints)
    hints->UpdateEstimatedRemainingTimeMs(limit_ms_ -
                                          stats.time_since_movestart_ms);
  return limit_ms_ >= 0 && stats.time_since_movestart_ms >= limit_ms_;
}

NodeLimitStopper::NodeLimitStopper(uint64_t limit) : limit_(limit) {}

bool NodeLimitStopper::ShouldStop(const SearchStats &stats,
                                  StoppersHints *hints) {
  const int64_t remaining =
      limit_ > stats.nodes_since_movestart
          ? static_cast<int64_t>(limit_ - stats.nodes_since_movestart)
          : 0;
  if (hints)
    hints->UpdateEstimatedRemainingPlayouts(remaining);
  return stats.nodes_since_movestart >= limit_;
}

SmartPruningStopper::SmartPruningStopper(float factor, int64_t minimum_batches)
    : factor_(factor), minimum_batches_(minimum_batches) {}

bool SmartPruningStopper::ShouldStop(const SearchStats &stats,
                                     StoppersHints *hints) {
  if (factor_ <= 0.0f || !hints)
    return false;

  std::lock_guard<std::mutex> lock(mutex_);

  if (stats.edge_n.size() == 1)
    return true;

  if (stats.edge_n.size() <= static_cast<size_t>(stats.num_losing_edges) +
                                 (stats.may_resign ? 0u : 1u)) {
    return true;
  }

  if (stats.win_found)
    return true;

  if (stats.nodes_since_movestart > 0 && !first_eval_time_ms_) {
    first_eval_time_ms_ = stats.time_since_movestart_ms;
    return false;
  }

  if (!first_eval_time_ms_ || stats.edge_n.empty())
    return false;

  constexpr int64_t kSmartPruningToleranceMs = 200;
  constexpr int64_t kSmartPruningToleranceNodes = 300;
  if (stats.time_since_movestart_ms <
      *first_eval_time_ms_ + kSmartPruningToleranceMs) {
    return false;
  }

  const int64_t time = std::max<int64_t>(1, stats.time_since_movestart_ms -
                                                *first_eval_time_ms_);
  const auto nodes = stats.nodes_since_movestart + kSmartPruningToleranceNodes;
  const double fallback_nps =
      1000.0 * static_cast<double>(nodes) / static_cast<double>(time) + 1.0;
  const double nps =
      hints->GetEstimatedNps().value_or(static_cast<float>(fallback_nps));

  const double remaining_time_s =
      static_cast<double>(hints->GetEstimatedRemainingTimeMs()) / 1000.0;
  const double time_playouts = remaining_time_s * nps / factor_;
  const double hint_playouts =
      static_cast<double>(hints->GetEstimatedRemainingPlayouts()) / factor_;
  const double remaining_playouts = std::min(time_playouts, hint_playouts);
  if (remaining_playouts <
      static_cast<double>(std::numeric_limits<int64_t>::max()))
    hints->UpdateEstimatedRemainingPlayouts(
        static_cast<int64_t>(std::max(0.0, remaining_playouts)));

  if (stats.batches_since_movestart <
      static_cast<uint64_t>(std::max<int64_t>(0, minimum_batches_))) {
    return false;
  }

  uint32_t best_n = 0, second_n = 0;
  for (uint32_t n : stats.edge_n) {
    if (n > best_n) {
      second_n = best_n;
      best_n = n;
    } else if (n > second_n) {
      second_n = n;
    }
  }

  return remaining_playouts < static_cast<double>(best_n - second_n);
}

KLDGainStopper::KLDGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}

bool KLDGainStopper::ShouldStop(const SearchStats &stats,
                                StoppersHints * /*hints*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  const double new_child_nodes = static_cast<double>(stats.total_nodes) - 1.0;

  if (new_child_nodes < prev_child_nodes_ + average_interval_)
    return false;

  const auto &new_visits = stats.edge_n;
  if (!prev_visits_.empty() && prev_visits_.size() == new_visits.size() &&
      prev_child_nodes_ > 0.0) {
    double kldgain = 0.0;
    for (size_t i = 0; i < new_visits.size(); ++i) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = static_cast<double>(new_visits[i]) / new_child_nodes;
      if (prev_visits_[i] > 0 && n_p > 0.0) {
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

MemoryWatchingStopper::MemoryWatchingStopper(size_t max_bytes)
    : max_bytes_(max_bytes) {}

bool MemoryWatchingStopper::ShouldStop(const SearchStats &stats,
                                       StoppersHints * /*hints*/) {
  size_t estimated = stats.total_nodes * 624;
  return estimated > max_bytes_;
}

std::unique_ptr<SearchStopper>
SigmoidTimeManager::CreateStopper(Color /*us*/, int64_t time_left, int64_t inc,
                                  int movestogo, int ply, uint64_t node_limit,
                                  int64_t movetime, bool infinite,
                                  float smart_pruning_factor) {
  if (movetime > 0) {
    return std::make_unique<TimeLimitStopper>(movetime);
  }

  if (infinite)
    return nullptr;

  double est_moves = 10.0 + 50.0 / (1.0 + std::exp((ply - 45.0) / 6.0));

  if (movestogo > 0) {
    est_moves = static_cast<double>(movestogo);
  }

  double total_time = static_cast<double>(time_left) +
                      static_cast<double>(inc) * std::max(0.0, est_moves - 1.0);

  int64_t time_for_move = static_cast<int64_t>(total_time / est_moves);

  int64_t max_time = static_cast<int64_t>(time_left * 0.3);
  time_for_move = std::min(time_for_move, max_time);
  time_for_move = std::max(time_for_move, int64_t(500));

  auto chain = std::make_unique<ChainedStopper>();
  chain->Add(std::make_unique<TimeLimitStopper>(time_for_move));
  chain->Add(std::make_unique<SmartPruningStopper>(smart_pruning_factor, 0));

  if (node_limit > 0) {
    chain->Add(std::make_unique<NodeLimitStopper>(node_limit));
  }

  return chain;
}

} // namespace MCTS
} // namespace MetalFish
