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

  // Default to false -- Metal is initialized on demand when MCTS/Hybrid starts.
  options.add("UseGPU", Option(false));

  options.add("NNWeights",
              Option("", [](const Option &) { return std::nullopt; }));
  options.add("NNBackend", Option("auto"));
  options.add("NNBackendRequireAccelerator", Option(false));
  options.add("NNCoreMLModelPath", Option(""));
  options.add("NNCoreMLComputeUnits", Option("cpu-ne"));
  options.add("NNCudaDevice", Option(-1, -1, 255));
  options.add("NNCudaGraphExecution", Option(true));
  options.add("NNCudaStableExecutionBatchSize", Option(0, 0, 256));
  options.add("NNCudaDeterministicAttentionSoftmax", Option(true));
  options.add("NNCudaFullBufferClear", Option(true));

  options.add("UseHybridSearch", Option(false));

  options.add("UseMCTS", Option(false));

  // 0 = auto (derived from Threads; coordinator is outside the worker budget).
  options.add("HybridMCTSThreads", Option(0, 0, MaxThreads));
  options.add("HybridABThreads", Option(0, 0, MaxThreads));
  options.add("HybridAutoABThreadsCap", Option(0, 0, MaxThreads));
  options.add("HybridABPolicyWeight", Option("0.0"));
  options.add("HybridMCTSMinimumKLDGainPerNode", Option("0.0"));
  options.add("HybridABRootRejectMCTS", Option(true));
  options.add("HybridMCTSRootReject", Option(false));
  options.add("HybridMCTSUseSharedTT", Option(false));
  options.add("HybridMCTSABRootHints", Option(true));
  options.add("HybridMCTSABRootHintDelayMs", Option(0, 0, 1000));
  options.add("HybridMCTSABRootHintCount", Option(8, 1, 16));
  options.add("HybridABCandidateVerifyMs", Option(240, 0, 1000));
  options.add("HybridABCandidateVerifyCount", Option(5, 1, 10));
  options.add("HybridRootPawnLeverTieBreak", Option(true));
  options.add("HybridANERootProbe", Option(false));
  options.add("HybridANERootHints", Option(false));
  options.add("HybridANEConfirmMCTSOverride", Option(false));
  options.add("HybridANEOnlyPawnEndgames", Option(false));
  options.add("HybridANEWeights", Option(""));
  options.add("HybridANEModelPath", Option(""));
  options.add("HybridANEComputeUnits", Option("cpu-ne"));
  options.add("HybridANERootHintCount", Option(6, 1, 32));
  options.add("HybridANERootHintWaitMs", Option(0, 0, 1000));
  options.add("HybridANEMinBudgetMs", Option(0, 0, 30000));
  options.add("HybridTrace", Option(false));
  options.add("TransformerLowTimeFallbackMs", Option(3000, 0, 30000));
  options.add("TransformerMinMoveBudgetMs", Option(400, 0, 5000));

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
  options.add("MCTSPolicyTemperature", Option("1.359"));
  options.add("MCTSRootPolicySoftmaxTemp", Option("1.6"));
  options.add("MCTSHighPolicyRootLever", Option(true));
  options.add("MCTSLowPolicyRootLever", Option(true));
  options.add("MCTSRootTacticalCaptureProbe", Option(true));
  options.add("MCTSMovesLeftMaxEffect", Option("0.0345"));
  options.add("MCTSMovesLeftThreshold", Option("0.8"));
  options.add("MCTSMovesLeftSlope", Option("0.0027"));
  options.add("MCTSMovesLeftConstantFactor", Option("0.0"));
  options.add("MCTSMovesLeftScaledFactor", Option("1.6521"));
  options.add("MCTSMovesLeftQuadraticFactor", Option("-0.6521"));
  options.add("MCTSTemperature", Option("0.0"));
  options.add("MCTSTempValueCutoff", Option("100.0"));
  options.add("MCTSSmartPruningFactor", Option("1.33"));
  options.add("PureMCTSSmartPruningFactor", Option("0.5"));
  options.add("PureMCTSCPuctAtRoot", Option("2.4"));
  options.add("PureMCTSFpuReductionAtRoot", Option("0.55"));
  options.add("MCTSSmartPruningMinimumBatches", Option(0, 0, 10000));
  options.add("MCTSMinimumKLDGainPerNode", Option("0.00005"));
  options.add("MCTSKLDGainAverageInterval", Option(100, 1, 10000000));
  options.add("MCTSTimeManager", Option("smooth"));
  options.add("MCTSCacheHistoryLength", Option(0, 0, 7));
  options.add("MCTSNNCacheSize", Option(2000000, 1, 100000000));
  options.add("MCTSSolidTreeThreshold", Option(100, 0, 2000000000));
  options.add("MCTSMaxPrefetch", Option(32, 0, 1024));
  options.add("MCTSMaxCollisionEvents", Option(917, 1, 65536));
  options.add("MCTSMaxCollisionVisits", Option(80000, 1, 100000000));
  options.add("MCTSMaxCollisionVisitsScalingStart", Option(28, 1, 100000));
  options.add("MCTSMaxCollisionVisitsScalingEnd", Option(145000, 0, 100000000));
  options.add("MCTSMaxCollisionVisitsScalingPower", Option("1.25"));
  options.add("MCTSVirtualLoss", Option(1, 1, 128));
  // 0 = auto; on Apple Silicon smaller batches are better at low thread counts.
  options.add("MCTSMinibatchSize", Option(0, 0, 4096));
  options.add("MCTSCudaAutoMinibatchSize", Option(0, 0, 256));
  options.add("MCTSMaxThreads", Option(0, 0, MaxThreads));
  options.add("MCTSParallelSearch", Option(false));
  options.add("MCTSMaxOutOfOrderFactor", Option("4.0"));
  options.add("MCTSMaxOutOfOrderEvalsFactor", Option("4.0"));
  options.add("MCTSAddDirichletNoise", Option(false));
  options.add("MCTSNoiseEpsilon", Option("0.0"));
  options.add("MCTSNoiseAlpha", Option("0.3"));

  load_networks();
  resize_threads();
}

std::uint64_t Engine::perft(const std::string &fen, Depth depth,
                            bool isChess960) {
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

  // Thread-count changes do not require rebuilding the TT. Preserving it is
  // important for hybrid search, which temporarily resizes the AB worker pool
  // around every move. Hash option changes and Clear Hash still reset it.
  if (!tt.is_allocated())
    set_tt_size(options["Hash"]);
  threads.ensure_network_replicated();
  networksVerified = false;
}

void Engine::set_tt_size(size_t mb) {
  wait_for_search_finished();
  tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

void Engine::verify_networks() const {
  const std::string bigEvalFile = std::string(options["EvalFile"]);
  const std::string smallEvalFile = std::string(options["EvalFileSmall"]);

  if (networksVerified && verifiedBigEvalFile == bigEvalFile &&
      verifiedSmallEvalFile == smallEvalFile)
    return;

  networks->big.verify(bigEvalFile, onVerifyNetworks);
  networks->small.verify(smallEvalFile, onVerifyNetworks);

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

  verifiedBigEvalFile = bigEvalFile;
  verifiedSmallEvalFile = smallEvalFile;
  networksVerified = true;
}

void Engine::load_networks() {
  networksVerified = false;
  networks.modify_and_replicate([this](NN::Networks &networks_) {
    networks_.big.load(binaryDirectory, options["EvalFile"]);
    networks_.small.load(binaryDirectory, options["EvalFileSmall"]);
  });
  threads.clear();
  threads.ensure_network_replicated();
}

void Engine::load_big_network(const std::string &file) {
  networksVerified = false;
  networks.modify_and_replicate([this, &file](NN::Networks &networks_) {
    networks_.big.load(binaryDirectory, file);
  });
  threads.clear();
  threads.ensure_network_replicated();
}

void Engine::load_small_network(const std::string &file) {
  networksVerified = false;
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

std::vector<Engine::RootMoveSnapshot>
Engine::root_move_snapshot(size_t max_moves) const {
  std::vector<RootMoveSnapshot> snapshot;
  if (threads.empty())
    return snapshot;

  bool can_vote_best_thread = true;
  for (auto it = threads.cbegin(); it != threads.cend(); ++it) {
    const Thread *thread = it->get();
    if (!thread || !thread->worker || thread->worker->rootMoves.empty() ||
        thread->worker->rootMoves[0].pv.empty()) {
      can_vote_best_thread = false;
      break;
    }
  }

  Thread *best_thread = nullptr;
  if (can_vote_best_thread)
    best_thread = threads.get_best_thread();
  else {
    for (auto it = threads.cbegin(); it != threads.cend(); ++it) {
      Thread *thread = it->get();
      if (thread && thread->worker && !thread->worker->rootMoves.empty()) {
        best_thread = thread;
        break;
      }
    }
  }

  if (!best_thread || best_thread->worker->rootMoves.empty())
    return snapshot;

  const auto &root_moves = best_thread->worker->rootMoves;
  const size_t count = max_moves > 0 ? std::min(max_moves, root_moves.size())
                                     : root_moves.size();
  snapshot.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto &rm = root_moves[i];
    RootMoveSnapshot item;
    item.move = rm.pv.empty() ? Move::none() : rm.pv[0];
    item.score = rm.score;
    item.previous_score = rm.previousScore;
    item.average_score = rm.averageScore;
    item.score_lowerbound = rm.scoreLowerbound;
    item.score_upperbound = rm.scoreUpperbound;
    item.effort = rm.effort;
    item.sel_depth = rm.selDepth;
    item.pv = rm.pv;
    snapshot.push_back(std::move(item));
  }
  return snapshot;
}

Engine::QuickSearchResult Engine::search_sync(const std::string &fen, int depth,
                                              int time_ms) {
  QuickSearchResult result;

  set_position(fen, {});

  Search::LimitsType limits;
  limits.startTime = now();
  if (depth > 0) {
    limits.depth = depth;
  }
  if (time_ms > 0) {
    limits.movetime = time_ms;
  }

  go(limits);
  wait_for_search_finished();

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

// Runs AB search without triggering the bestmove callback.
Engine::QuickSearchResult Engine::search_silent(const std::string &fen,
                                                int depth, int time_ms) {
  QuickSearchResult result;

  set_position(fen, {});

  Search::LimitsType limits;
  limits.startTime = now();
  if (depth > 0) {
    limits.depth = depth;
  }
  if (time_ms > 0) {
    limits.movetime = time_ms;
  }

  auto saved_callback = updateContext.onBestmove;

  updateContext.onBestmove = [](std::string_view, std::string_view) {};

  go(limits);
  wait_for_search_finished();

  updateContext.onBestmove = saved_callback;

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
  set_position(fen, {});

  Search::LimitsType limits;
  limits.startTime = now();
  if (time_ms > 0)
    limits.movetime = time_ms;

  auto saved_bestmove = updateContext.onBestmove;
  auto saved_update = updateContext.onUpdateFull;

  updateContext.onBestmove = [](std::string_view, std::string_view) {};

  updateContext.onUpdateFull = [this, &on_iteration,
                                &saved_update](const Search::InfoFull &info) {
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
    if (saved_update)
      saved_update(info);
  };

  go(limits);

  while (!threads.stop.load(std::memory_order_acquire)) {
    if (stop_flag.load(std::memory_order_acquire)) {
      threads.stop = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  wait_for_search_finished();

  updateContext.onBestmove = saved_bestmove;
  updateContext.onUpdateFull = saved_update;
}
} // namespace MetalFish
