/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef ENGINE_H_INCLUDED
#define ENGINE_H_INCLUDED

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "core/misc.h"
#include "core/numa.h"
#include "core/position.h"
#include "core/types.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_misc.h"
#include "search/history.h"
#include "search/search.h"
#include "search/thread.h"
#include "search/tt.h"
#include "syzygy/tbprobe.h" // for MetalFish::Depth
#include "uci/ucioption.h"

namespace MetalFish {

constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;
extern int MaxThreads;

class Engine {
public:
  using InfoShort = Search::InfoShort;
  using InfoFull = Search::InfoFull;
  using InfoIter = Search::InfoIteration;

  Engine(std::optional<std::filesystem::path> path = std::nullopt);

  // Cannot be movable due to components holding backreferences to fields
  Engine(const Engine &) = delete;
  Engine(Engine &&) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine &operator=(Engine &&) = delete;

  ~Engine() { wait_for_search_finished(); }

  std::variant<u64, PositionSetError> perft(const std::string &fen, Depth depth,
                                            bool isChess960);

  // non blocking call to start searching
  void go(Search::LimitsType &);
  // non blocking call to stop searching
  void stop();

  // blocking call to wait for search to finish
  void wait_for_search_finished();
  // set a new position, moves are in UCI format
  std::optional<PositionSetError>
  set_position(const std::string &fen, const std::vector<std::string> &moves);

  // modifiers

  bool set_numa_config_from_option(const std::string &o);
  void resize_threads();
  void set_tt_size(usize mb);
  void set_ponderhit(bool);
  void search_clear();

  void set_on_update_no_moves(std::function<void(const InfoShort &)> &&);
  void set_on_update_full(std::function<void(const InfoFull &)> &&);
  void set_on_iter(std::function<void(const InfoIter &)> &&);
  void
  set_on_bestmove(std::function<void(std::string_view, std::string_view)> &&);
  void set_on_start(std::function<void()> &&);
  void set_on_verify_network(std::function<void(std::string_view)> &&);

  std::function<void(std::string_view, std::string_view)> get_on_bestmove();
  std::function<void(const InfoFull &)> get_on_update_full();

  Thread *threads_get_best();
  uint64_t threads_nodes_searched();

  // network related

  void verify_network() const;
  std::unique_ptr<Eval::NNUE::Network> get_default_network();
  void load_network(const std::filesystem::path &file);
  void save_network(const std::optional<std::filesystem::path> &file);

  // utility functions

  void trace_eval() const;

  const OptionsMap &get_options() const;
  OptionsMap &get_options();

  int get_hashfull(int maxAge = 0) const;

  std::string fen() const;
  std::optional<PositionSetError> flip();
  std::string visualize() const;
  std::vector<std::pair<usize, usize>>
  get_bound_thread_count_by_numa_node() const;
  std::string get_numa_config_as_string() const;
  std::string numa_config_information_as_string() const;
  std::string thread_allocation_information_as_string() const;
  std::string thread_binding_information_as_string() const;

  struct QuickSearchResult {
    Move best_move = Move::none();
    Move ponder_move = Move::none();
    Value score = VALUE_NONE;
    int depth = 0;
    uint64_t nodes = 0;
    std::vector<Move> pv;
  };

  struct RootMoveSnapshot {
    Move move = Move::none();
    Value score = VALUE_NONE;
    Value previous_score = VALUE_NONE;
    Value average_score = VALUE_NONE;
    bool score_lowerbound = false;
    bool score_upperbound = false;
    uint64_t effort = 0;
    int sel_depth = 0;
    std::vector<Move> pv;
  };

  std::vector<RootMoveSnapshot> root_move_snapshot(size_t max_moves = 0) const;

  QuickSearchResult search_sync(const std::string &fen, int depth,
                                int time_ms = 0);

  QuickSearchResult search_silent(const std::string &fen, int depth,
                                  int time_ms = 0);

  // Iterative deepening with per-iteration callback. Preserves all state
  // (TT, aspiration windows, killers, history) across iterations.
  using IterationCallback =
      std::function<void(const QuickSearchResult &result)>;
  void search_with_callbacks(const std::string &fen, int time_ms,
                             IterationCallback on_iteration,
                             std::atomic<bool> &stop_flag);

  TranspositionTable &get_tt() { return tt; }
  const TranspositionTable &get_tt() const { return tt; }

  ThreadPool &get_threads() { return threads; }

private:
  const std::filesystem::path binaryDirectory;

  NumaReplicationContext numaContext;

  Position pos;
  StateListPtr states;

  OptionsMap options;
  ThreadPool threads;
  TranspositionTable tt;
  Eval::NNUE::EvalFile networkFile;
  LazyNumaReplicatedSystemWide<Eval::NNUE::Network> network;

  Search::SearchManager::UpdateContext updateContext;
  std::function<void(std::string_view)> onVerifyNetwork;
  std::map<NumaIndex, SharedHistories> sharedHists;

  // Hybrid search calls go() once per move; verify the network only when the
  // selected file changes instead of on every search start.
  mutable bool networkVerified = false;
  mutable std::string verifiedEvalFile;
};

} // namespace MetalFish

#endif // #ifndef ENGINE_H_INCLUDED
