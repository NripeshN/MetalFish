/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Search Stoppers - Time and node management
  Licensed under GPL-3.0
*/

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "../core/types.h"

namespace MetalFish {
namespace MCTS {

struct SearchStats {
  enum class TimeUsageHint : uint8_t {
    Normal = 0,
    ImmediateMove = 1,
    NeedMoreTime = 2
  };

  uint64_t total_nodes = 0;
  uint64_t nodes_since_movestart = 0;
  uint64_t batches_since_movestart = 0;
  int64_t time_since_movestart_ms = 0;
  int64_t time_since_first_batch_ms = 0;
  std::vector<uint32_t> edge_n;
  int num_losing_edges = 0;
  bool win_found = false;
  bool may_resign = true;
  TimeUsageHint time_usage_hint = TimeUsageHint::Normal;
};

class StoppersHints {
public:
  void Reset();
  void UpdateEstimatedRemainingTimeMs(int64_t v);
  void UpdateEstimatedRemainingPlayouts(int64_t v);
  void UpdateEstimatedNps(float v);

  int64_t GetEstimatedRemainingTimeMs() const;
  int64_t GetEstimatedRemainingPlayouts() const;
  std::optional<float> GetEstimatedNps() const;
  bool HasEstimatedRemainingPlayouts() const;

private:
  std::optional<int64_t> estimated_remaining_time_ms_;
  std::optional<int64_t> estimated_remaining_playouts_;
  std::optional<float> estimated_nps_;
};

class SearchStopper {
public:
  virtual ~SearchStopper() = default;
  virtual bool ShouldStop(const SearchStats &stats, StoppersHints *hints) = 0;
  virtual void OnSearchDone(const SearchStats &) {}
};

class ChainedStopper : public SearchStopper {
public:
  void Add(std::unique_ptr<SearchStopper> s);
  bool ShouldStop(const SearchStats &stats, StoppersHints *hints) override;
  void OnSearchDone(const SearchStats &stats) override;

private:
  std::vector<std::unique_ptr<SearchStopper>> stoppers_;
};

class TimeLimitStopper : public SearchStopper {
public:
  explicit TimeLimitStopper(int64_t limit_ms);
  bool ShouldStop(const SearchStats &stats, StoppersHints *hints) override;

private:
  int64_t limit_ms_;
};

class NodeLimitStopper : public SearchStopper {
public:
  explicit NodeLimitStopper(uint64_t limit);
  bool ShouldStop(const SearchStats &stats, StoppersHints *hints) override;

private:
  uint64_t limit_;
};

class SmartPruningStopper : public SearchStopper {
public:
  SmartPruningStopper(float factor, int64_t minimum_batches);
  bool ShouldStop(const SearchStats &stats, StoppersHints *hints) override;

private:
  float factor_;
  int64_t minimum_batches_;
  std::optional<int64_t> first_eval_time_ms_;
  mutable std::mutex mutex_;
};

class KLDGainStopper : public SearchStopper {
public:
  KLDGainStopper(float min_gain, int average_interval);
  bool ShouldStop(const SearchStats &stats, StoppersHints *hints) override;

private:
  float min_gain_;
  int average_interval_;
  std::vector<double> prev_visits_;
  double prev_child_nodes_ = 0.0;
  mutable std::mutex mutex_;
};

class MemoryWatchingStopper : public SearchStopper {
public:
  explicit MemoryWatchingStopper(size_t max_bytes);
  bool ShouldStop(const SearchStats &stats, StoppersHints *hints) override;

private:
  size_t max_bytes_;
};

class SigmoidTimeManager {
public:
  std::unique_ptr<SearchStopper> CreateStopper(Color us, int64_t time_left,
                                               int64_t inc, int movestogo,
                                               int ply, uint64_t node_limit,
                                               int64_t movetime, bool infinite,
                                               float smart_pruning_factor);
};

} // namespace MCTS
} // namespace MetalFish
