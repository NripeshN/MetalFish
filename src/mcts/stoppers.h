/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Search Stoppers - Time and node management
  Licensed under GPL-3.0
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../core/types.h"

namespace MetalFish {
namespace MCTS {

struct SearchStats {
  uint64_t total_nodes = 0;
  uint64_t nodes_since_movestart = 0;
  int64_t time_since_movestart_ms = 0;
  std::vector<uint32_t> edge_n;
};

class SearchStopper {
public:
  virtual ~SearchStopper() = default;
  virtual bool ShouldStop(const SearchStats &stats) = 0;
};

class ChainedStopper : public SearchStopper {
public:
  void Add(std::unique_ptr<SearchStopper> s);
  bool ShouldStop(const SearchStats &stats) override;

private:
  std::vector<std::unique_ptr<SearchStopper>> stoppers_;
};

class TimeLimitStopper : public SearchStopper {
public:
  explicit TimeLimitStopper(int64_t limit_ms);
  bool ShouldStop(const SearchStats &stats) override;

private:
  int64_t limit_ms_;
};

class NodeLimitStopper : public SearchStopper {
public:
  explicit NodeLimitStopper(uint64_t limit);
  bool ShouldStop(const SearchStats &stats) override;

private:
  uint64_t limit_;
};

class SmartPruningStopper : public SearchStopper {
public:
  explicit SmartPruningStopper(float factor, int64_t time_limit_ms);
  bool ShouldStop(const SearchStats &stats) override;

private:
  float factor_;
  int64_t time_limit_ms_;
};

class KLDGainStopper : public SearchStopper {
public:
  KLDGainStopper(float min_gain, int average_interval);
  bool ShouldStop(const SearchStats &stats) override;

private:
  float min_gain_;
  int average_interval_;
  std::vector<double> prev_visits_;
  double prev_child_nodes_ = 0.0;
};

class MemoryWatchingStopper : public SearchStopper {
public:
  explicit MemoryWatchingStopper(size_t max_bytes);
  bool ShouldStop(const SearchStats &stats) override;

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
