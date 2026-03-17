/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "uci/engine.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "core/misc.h"
#include "core/numa.h"
#include "core/perft.h"
#include "core/position.h"
#include "core/shm.h"
#include "core/types.h"
#include "eval/evaluate.h"
#include "eval/gpu_backend.h"
#include "eval/gpu_integration.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_common.h"
#include "eval/nnue/nnue_misc.h"
#include "search/search.h"
#include "syzygy/tbprobe.h"
#include "uci/uci.h"
#include "uci/ucioption.h"

namespace MetalFish {

namespace NN = Eval::NNUE;

constexpr auto StartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;
int MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

Engine::Engine(std::optional<std::string> path)
    : binaryDirectory(path ? CommandLine::get_binary_directory(*path) : ""),
      numaContext(NumaConfig::from_system()),
      states(new std::deque<StateInfo>(1)), threads(),
      networks(numaContext,
               // Heap-allocate because sizeof(NN::Networks) is large
               std::make_unique<NN::Networks>(
                   NN::EvalFile{EvalFileDefaultNameBig, "None", ""},
                   NN::EvalFile{EvalFileDefaultNameSmall, "None", ""})) {

  pos.set(StartFEN, false, &states->back());

  options.add( //
      "Debug Log File", Option("", [](const Option &o) {
        start_logger(o);
        return std::nullopt;
      }));

  options.add( //
      "NumaPolicy", Option("auto", [this](const Option &o) {
        set_numa_config_from_option(o);
        return numa_config_information_as_string() + "\n" +
               thread_allocation_information_as_string();
      }));

  options.add( //
      "Threads", Option(1, 1, MaxThreads, [this](const Option &) {
        resize_threads();
        return thread_allocation_information_as_string();
      }));

  options.add( //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option &o) {
        set_tt_size(o);
        return std::nullopt;
      }));

  options.add( //
      "Clear Hash", Option([this](const Option &) {
        search_clear();
        return std::nullopt;
      }));

  options.add( //
      "Ponder", Option(false));

  options.add( //
      "MultiPV", Option(1, 1, MAX_MOVES));

  options.add("Skill Level", Option(20, 0, 20));

  options.add("Move Overhead", Option(10, 0, 5000));

  options.add("nodestime", Option(0, 0, 10000));

  options.add("UCI_Chess960", Option(false));

  options.add("UCI_LimitStrength", Option(false));

  options.add("UCI_Elo", Option(MetalFish::Search::Skill::LowestElo,
                                MetalFish::Search::Skill::LowestElo,
                                MetalFish::Search::Skill::HighestElo));

  options.add("UCI_ShowWDL", Option(false));

  options.add( //
      "SyzygyPath", Option("", [](const Option &o) {
        Tablebases::init(o);
        return std::nullopt;
      }));

  options.add("SyzygyProbeDepth", Option(1, 1, 100));

  options.add("Syzygy50MoveRule", Option(true));

  options.add("SyzygyProbeLimit", Option(7, 0, 7));

  options.add( //
      "EvalFile", Option(EvalFileDefaultNameBig, [this](const Option &o) {
        load_big_network(o);
        return std::nullopt;
      }));

  options.add( //
      "EvalFileSmall",
      Option(EvalFileDefaultNameSmall, [this](const Option &o) {
        load_small_network(o);
        return std::nullopt;
      }));

  // GPU acceleration options
  // NOTE: GPU is used ONLY for transformer network inference (MCTS/Hybrid).
  // AB search always uses CPU NNUE. There is no "GPU NNUE" mode.
  // Default to false -- Metal is initialized on demand when MCTS/Hybrid starts.
  options.add("UseGPU", Option(false));

  // Transformer network weights for MCTS/Hybrid (.pb or .pb.gz file)
  options.add("NNWeights", Option("", [](const Option &) {
                // Path is read when MCTS/Hybrid search starts
                return std::nullopt;
              }));

  // Hybrid search mode - use parallel MCTS+AB instead of pure AB
  options.add("UseHybridSearch", Option(false));

  // Pure MCTS mode - use GPU-accelerated MCTS instead of AB
  options.add("UseMCTS", Option(false));

  // Optional parity preset and exposed MCTS tuning controls
  options.add("MCTSParityPreset", Option(false));
  options.add("MCTSCPuct", Option("1.745"));
  options.add("MCTSCPuctAtRoot", Option("1.745"));
  options.add("MCTSCPuctBase", Option("38739"));
  options.add("MCTSCPuctFactor", Option("3.894"));
  options.add("MCTSCPuctBaseAtRoot", Option("38739"));
  options.add("MCTSCPuctFactorAtRoot", Option("3.894"));
  options.add("MCTSFpuAbsolute", Option(false));
  options.add("MCTSFpuAbsoluteAtRoot", Option(false));
  options.add("MCTSFpuValue", Option("0.33"));
  options.add("MCTSFpuValueAtRoot", Option("0.33"));
  options.add("MCTSFpuReduction", Option("0.33"));
  options.add("MCTSFpuReductionAtRoot", Option("0.33"));
  options.add("MCTSPolicySoftmaxTemp", Option("1.359"));
  options.add("MCTSVirtualLoss", Option(1, 1, 128));
  options.add("MCTSMinibatchSize", Option(256, 1, 4096));
  options.add("MCTSMaxOutOfOrderFactor", Option("2.4"));
  options.add("MCTSAddDirichletNoise", Option(false));
  options.add("MCTSNoiseEpsilon", Option("0.0"));
  options.add("MCTSNoiseAlpha", Option("0.3"));

  load_networks();
  resize_threads();
}

std::uint64_t Engine::perft(const std::string &fen, Depth depth,
                            bool isChess960) {
  verify_networks();

  return Benchmark::perft(fen, depth, isChess960);
}

void Engine::go(Search::LimitsType &limits) {
  assert(limits.perft == 0);
  verify_networks();

  threads.start_thinking(options, pos, states, limits);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
  wait_for_search_finished();

  tt.clear(threads);
  threads.clear();

  // @TODO wont work with multiple instances
  Tablebases::init(options["SyzygyPath"]); // Free mapped files
}

void Engine::set_on_update_no_moves(
    std::function<void(const Engine::InfoShort &)> &&f) {
  updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(
    std::function<void(const Engine::InfoFull &)> &&f) {
  updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter &)> &&f) {
  updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(
    std::function<void(std::string_view, std::string_view)> &&f) {
  updateContext.onBestmove = std::move(f);
}

void Engine::set_on_verify_networks(std::function<void(std::string_view)> &&f) {
  onVerifyNetworks = std::move(f);
}

std::function<void(std::string_view, std::string_view)>
Engine::get_on_bestmove() {
  return updateContext.onBestmove;
}

std::function<void(const Engine::InfoFull &)> Engine::get_on_update_full() {
  return updateContext.onUpdateFull;
}

Thread *Engine::threads_get_best() { return threads.get_best_thread(); }

uint64_t Engine::threads_nodes_searched() { return threads.nodes_searched(); }

void Engine::wait_for_search_finished() {
  threads.main_thread()->wait_for_search_finished();
}

void Engine::set_position(const std::string &fen,
                          const std::vector<std::string> &moves) {
  // Drop the old state and create a new one
  states = StateListPtr(new std::deque<StateInfo>(1));
  pos.set(fen, options["UCI_Chess960"], &states->back());

  for (const auto &move : moves) {
    auto m = UCIEngine::to_move(pos, move);

    if (m == Move::none())
      break;

    states->emplace_back();
    pos.do_move(m, states->back());
  }
}

// modifiers

void Engine::set_numa_config_from_option(const std::string &o) {
  if (o == "auto" || o == "system") {
    numaContext.set_numa_config(NumaConfig::from_system());
  } else if (o == "hardware") {
    // Don't respect affinity set in the system.
    numaContext.set_numa_config(NumaConfig::from_system(false));
  } else if (o == "none") {
    numaContext.set_numa_config(NumaConfig{});
  } else {
    numaContext.set_numa_config(NumaConfig::from_string(o));
  }

  // Force reallocation of threads in case affinities need to change.
  resize_threads();
  threads.ensure_network_replicated();
}

void Engine::resize_threads() {
  threads.wait_for_search_finished();
  threads.set(numaContext.get_numa_config(),
              {options, threads, tt, sharedHists, networks}, updateContext);

  // Reallocate the hash with the new threadpool size
  set_tt_size(options["Hash"]);
  threads.ensure_network_replicated();
}

void Engine::set_tt_size(size_t mb) {
  wait_for_search_finished();
  tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_networks() const {
  networks->big.verify(options["EvalFile"], onVerifyNetworks);
  networks->small.verify(options["EvalFileSmall"], onVerifyNetworks);

  auto statuses = networks.get_status_and_errors();
  for (size_t i = 0; i < statuses.size(); ++i) {
    const auto [status, error] = statuses[i];
    std::string message = "Network replica " + std::to_string(i + 1) + ": ";
    if (status == SystemWideSharedConstantAllocationStatus::NoAllocation) {
      message += "No allocation.";
    } else if (status ==
               SystemWideSharedConstantAllocationStatus::LocalMemory) {
      message += "Local memory.";
    } else if (status ==
               SystemWideSharedConstantAllocationStatus::SharedMemory) {
      message += "Shared memory.";
    } else {
      message += "Unknown status.";
    }

    if (error.has_value()) {
      message += " " + *error;
    }

    onVerifyNetworks(message);
  }
}

void Engine::load_networks() {
  networks.modify_and_replicate([this](NN::Networks &networks_) {
    networks_.big.load(binaryDirectory, options["EvalFile"]);
    networks_.small.load(binaryDirectory, options["EvalFileSmall"]);
  });
  threads.clear();
  threads.ensure_network_replicated();

  // NOTE: No GPU NNUE initialization here.
  // AB search uses CPU NNUE only.
  // Transformer network (for MCTS/Hybrid) is loaded on demand when
  // those search modes are activated, using the NNWeights UCI option.
}

void Engine::load_big_network(const std::string &file) {
  networks.modify_and_replicate([this, &file](NN::Networks &networks_) {
    networks_.big.load(binaryDirectory, file);
  });
  threads.clear();
  threads.ensure_network_replicated();
}

void Engine::load_small_network(const std::string &file) {
  networks.modify_and_replicate([this, &file](NN::Networks &networks_) {
    networks_.small.load(binaryDirectory, file);
  });
  threads.clear();
  threads.ensure_network_replicated();
}

void Engine::save_network(
    const std::pair<std::optional<std::string>, std::string> files[2]) {
  networks.modify_and_replicate([&files](NN::Networks &networks_) {
    networks_.big.save(files[0].first);
    networks_.small.save(files[1].first);
  });
}

// utility functions

void Engine::trace_eval() const {
  StateListPtr trace_states(new std::deque<StateInfo>(1));
  Position p;
  p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());

  verify_networks();

  sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
}

const OptionsMap &Engine::get_options() const { return options; }
OptionsMap &Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

void Engine::flip() { pos.flip(); }

std::string Engine::visualize() const {
  std::stringstream ss;
  ss << pos;
  return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<size_t, size_t>>
Engine::get_bound_thread_count_by_numa_node() const {
  auto counts = threads.get_bound_thread_count_by_numa_node();
  const NumaConfig &cfg = numaContext.get_numa_config();
  std::vector<std::pair<size_t, size_t>> ratios;
  NumaIndex n = 0;
  for (; n < counts.size(); ++n)
    ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
  if (!counts.empty())
    for (; n < cfg.num_numa_nodes(); ++n)
      ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
  return ratios;
}

std::string Engine::get_numa_config_as_string() const {
  return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
  auto cfgStr = get_numa_config_as_string();
  return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
  auto boundThreadsByNode = get_bound_thread_count_by_numa_node();
  std::stringstream ss;
  if (boundThreadsByNode.empty())
    return ss.str();

  bool isFirst = true;

  for (auto &&[current, total] : boundThreadsByNode) {
    if (!isFirst)
      ss << ":";
    ss << current << "/" << total;
    isFirst = false;
  }

  return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
  std::stringstream ss;

  size_t threadsSize = threads.size();
  ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

  auto boundThreadsByNodeStr = thread_binding_information_as_string();
  if (boundThreadsByNodeStr.empty())
    return ss.str();

  ss << " with NUMA node thread binding: ";
  ss << boundThreadsByNodeStr;

  return ss.str();
}

// ============================================================================
// Hybrid Search Integration
// ============================================================================

Engine::QuickSearchResult Engine::search_sync(const std::string &fen, int depth,
                                              int time_ms) {
  QuickSearchResult result;

  // Set up the position
  set_position(fen, {});

  // Set up search limits
  Search::LimitsType limits;
  limits.startTime = now();
  if (depth > 0) {
    limits.depth = depth;
  }
  if (time_ms > 0) {
    limits.movetime = time_ms;
  }

  // Run the search
  go(limits);
  wait_for_search_finished();

  // Get the best thread's result (Engine is now a friend of Worker)
  Thread *best_thread = threads.get_best_thread();
  if (best_thread && !best_thread->worker->rootMoves.empty()) {
    const auto &root_move = best_thread->worker->rootMoves[0];
    result.best_move = root_move.pv[0];
    result.score = root_move.score;
    result.depth = best_thread->worker->completedDepth;
    result.nodes = threads.nodes_searched();
    result.pv = root_move.pv;

    if (root_move.pv.size() > 1) {
      result.ponder_move = root_move.pv[1];
    }
  }

  return result;
}

// Silent search - runs AB search WITHOUT triggering bestmove callback
// This is used by hybrid search where the coordinator handles bestmove output
Engine::QuickSearchResult Engine::search_silent(const std::string &fen,
                                                int depth, int time_ms) {
  QuickSearchResult result;

  // Set up the position
  set_position(fen, {});

  // Set up search limits
  Search::LimitsType limits;
  limits.startTime = now();
  if (depth > 0) {
    limits.depth = depth;
  }
  if (time_ms > 0) {
    limits.movetime = time_ms;
  }

  // IMPORTANT: We cannot safely modify the callback while search is running
  // because it's called from the search thread. Instead, we use a flag-based
  // approach where the callback checks if it should be silent.
  //
  // For now, we use a simpler approach: temporarily set a no-op callback
  // BEFORE starting the search, and restore it AFTER the search completes.
  // This is safe because wait_for_search_finished() ensures the search thread
  // has completed before we restore.

  // Save the current bestmove callback
  auto saved_callback = updateContext.onBestmove;

  // Set a no-op callback before starting search
  updateContext.onBestmove = [](std::string_view, std::string_view) {
    // Silent - do nothing
  };

  // Run the search - the no-op callback is in place
  go(limits);
  wait_for_search_finished(); // This ensures search thread has finished

  // Now safe to restore the callback since search thread is done
  updateContext.onBestmove = saved_callback;

  // Get the best thread's result
  Thread *best_thread = threads.get_best_thread();
  if (best_thread && !best_thread->worker->rootMoves.empty()) {
    const auto &root_move = best_thread->worker->rootMoves[0];
    result.best_move = root_move.pv[0];
    result.score = root_move.score;
    result.depth = best_thread->worker->completedDepth;
    result.nodes = threads.nodes_searched();
    result.pv = root_move.pv;

    if (root_move.pv.size() > 1) {
      result.ponder_move = root_move.pv[1];
    }
  }

  return result;
}

void Engine::search_with_callbacks(const std::string &fen, int time_ms,
                                   IterationCallback on_iteration,
                                   std::atomic<bool> &stop_flag) {
  // Set up the position
  set_position(fen, {});

  // Set up search limits
  Search::LimitsType limits;
  limits.startTime = now();
  if (time_ms > 0)
    limits.movetime = time_ms;

  // Save original callbacks
  auto saved_bestmove = updateContext.onBestmove;
  auto saved_update = updateContext.onUpdateFull;

  // Suppress bestmove output (hybrid coordinator handles this)
  updateContext.onBestmove = [](std::string_view, std::string_view) {};

  // Hook into the per-iteration update to call our callback.
  // This fires after each depth of iterative deepening completes,
  // giving us the current best move + PV with full search state preserved.
  updateContext.onUpdateFull = [this, &on_iteration,
                                &saved_update](const Search::InfoFull &info) {
    // Build QuickSearchResult from the search state
    Thread *best = threads.get_best_thread();
    if (best && !best->worker->rootMoves.empty()) {
      QuickSearchResult result;
      const auto &rm = best->worker->rootMoves[0];
      result.best_move = rm.pv[0];
      result.score = rm.score;
      result.depth = best->worker->completedDepth;
      result.nodes = threads.nodes_searched();
      result.pv = rm.pv;
      if (rm.pv.size() > 1)
        result.ponder_move = rm.pv[1];

      on_iteration(result);
    }
    // Chain to original for UCI info output
    if (saved_update)
      saved_update(info);
  };

  // Run the search -- this is a single iterative deepening run that
  // preserves all state (TT, aspiration windows, killers, history)
  // across depth iterations. Much more efficient than calling
  // search_silent() in a loop.
  go(limits);

  // Poll for external stop signal from hybrid coordinator
  // while the search is running. The search checks threads.stop
  // internally, so setting it will cause the search to wind down.
  while (!threads.stop.load(std::memory_order_acquire)) {
    if (stop_flag.load(std::memory_order_acquire)) {
      threads.stop = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  wait_for_search_finished();

  // Restore original callbacks
  updateContext.onBestmove = saved_bestmove;
  updateContext.onUpdateFull = saved_update;
}
} // namespace MetalFish
