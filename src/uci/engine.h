/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file engine.h
 * @brief MetalFish source file.
 */

  Licensed under GPL-3.0
*/

#ifndef ENGINE_H_INCLUDED
#define ENGINE_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/numa.h"
#include "core/position.h"
#include "eval/nnue/network.h"
#include "search/history.h"
#include "search/search.h"
#include "search/thread.h"
#include "search/tt.h"
#include "syzygy/tbprobe.h" // for MetalFish::Depth
#include "uci/ucioption.h"

namespace MetalFish {

class Engine {
public:
  using InfoShort = Search::InfoShort;
  using InfoFull = Search::InfoFull;
  using InfoIter = Search::InfoIteration;

  Engine(std::optional<std::string> path = std::nullopt);

  // Cannot be movable due to components holding backreferences to fields
  Engine(const Engine &) = delete;
  Engine(Engine &&) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine &operator=(Engine &&) = delete;

  ~Engine() { wait_for_search_finished(); }

  std::uint64_t perft(const std::string &fen, Depth depth, bool isChess960);

  // non blocking call to start searching
  void go(Search::LimitsType &);
  // non blocking call to stop searching
  void stop();

  // blocking call to wait for search to finish
  void wait_for_search_finished();
  // set a new position, moves are in UCI format
  void set_position(const std::string &fen,
                    const std::vector<std::string> &moves);

  // modifiers

  void set_numa_config_from_option(const std::string &o);
  void resize_threads();
  void set_tt_size(size_t mb);
  void set_ponderhit(bool);
  void search_clear();

  void set_on_update_no_moves(std::function<void(const InfoShort &)> &&);
  void set_on_update_full(std::function<void(const InfoFull &)> &&);
  void set_on_iter(std::function<void(const InfoIter &)> &&);
  void
  set_on_bestmove(std::function<void(std::string_view, std::string_view)> &&);
  void set_on_verify_networks(std::function<void(std::string_view)> &&);

  // network related

  void verify_networks() const;
  void load_networks();
  void load_big_network(const std::string &file);
  void load_small_network(const std::string &file);
  void save_network(
      const std::pair<std::optional<std::string>, std::string> files[2]);

  // utility functions

  void trace_eval() const;

  const OptionsMap &get_options() const;
  OptionsMap &get_options();

  int get_hashfull(int maxAge = 0) const;

  std::string fen() const;
  void flip();
  std::string visualize() const;
  std::vector<std::pair<size_t, size_t>>
  get_bound_thread_count_by_numa_node() const;
  std::string get_numa_config_as_string() const;
  std::string numa_config_information_as_string() const;
  std::string thread_allocation_information_as_string() const;
  std::string thread_binding_information_as_string() const;

  // ============================================================================
  // Hybrid Search Integration
  // ============================================================================
  // These methods allow the hybrid MCTS search to use the full AB search
  // implementation for move verification.

  // Run a synchronous search on the given FEN and return the best move
  // This uses the full Stockfish search with all optimizations
  struct QuickSearchResult {
    Move best_move = Move::none();
    Move ponder_move = Move::none();
    Value score = VALUE_NONE;
    int depth = 0;
    uint64_t nodes = 0;
    std::vector<Move> pv;
  };

  QuickSearchResult search_sync(const std::string &fen, int depth,
                                int time_ms = 0);

  // Silent search - runs AB search WITHOUT triggering bestmove callback
  // Used by hybrid search where the coordinator handles bestmove output
  QuickSearchResult search_silent(const std::string &fen, int depth,
                                  int time_ms = 0);

  // Get access to the transposition table for sharing with hybrid search
  TranspositionTable &get_tt() { return tt; }
  const TranspositionTable &get_tt() const { return tt; }

  // Get access to thread pool for advanced integration
  ThreadPool &get_threads() { return threads; }

private:
  const std::string binaryDirectory;

  NumaReplicationContext numaContext;

  Position pos;
  StateListPtr states;

  OptionsMap options;
  ThreadPool threads;
  TranspositionTable tt;
  LazyNumaReplicatedSystemWide<Eval::NNUE::Networks> networks;

  Search::SearchManager::UpdateContext updateContext;
  std::function<void(std::string_view)> onVerifyNetworks;
  std::map<NumaIndex, SharedHistories> sharedHists;
};

} // namespace MetalFish

#endif // #ifndef ENGINE_H_INCLUDED