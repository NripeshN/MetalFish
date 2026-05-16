/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "search/thread.h"

#include <algorithm>
#if defined(__APPLE__)
#include <pthread.h>
#endif
#include <cassert>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/bitboard.h"
#include "core/memory.h"
#include "core/movegen.h"
#include "core/types.h"
#include "search/history.h"
#include "search/search.h"
#include "search/timeman.h"
#include "syzygy/tbprobe.h"
#include "uci/uci.h"
#include "uci/ucioption.h"

namespace MetalFish {

Thread::Thread(Search::SharedState &sharedState,
               std::unique_ptr<Search::ISearchManager> sm, size_t n,
               size_t numaN, size_t totalNumaCount,
               OptionalThreadToNumaNodeBinder binder)
    : idx(n), idxInNuma(numaN), totalNuma(totalNumaCount),
      nthreads(sharedState.options["Threads"]),
      stdThread(&Thread::idle_loop, this) {

  wait_for_search_finished();

  run_custom_job([this, &binder, &sharedState, &sm, n]() {
    this->numaAccessToken = binder();
    this->worker = make_unique_large_page<Search::Worker>(
        sharedState, std::move(sm), n, idxInNuma, totalNuma,
        this->numaAccessToken);
  });

  wait_for_search_finished();
}

Thread::~Thread() {

  assert(!searching);

  exit = true;
  start_searching();
  stdThread.join();
}

void Thread::start_searching() {
  assert(worker != nullptr);
  run_custom_job([this]() { worker->start_searching(); });
}

void Thread::clear_worker() {
  assert(worker != nullptr);
  run_custom_job([this]() { worker->clear(); });
}

void Thread::wait_for_search_finished() {

  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [&] { return !searching; });
}

void Thread::run_custom_job(std::function<void()> f) {
  {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [&] { return !searching; });
    jobFunc = std::move(f);
    searching = true;
  }
  cv.notify_one();
}

void Thread::ensure_network_replicated() {
  worker->ensure_network_replicated();
}

void Thread::idle_loop() {
#if defined(__APPLE__)
  // Request P-core scheduling via QoS class.
  // USER_INTERACTIVE signals to macOS that this is a latency-sensitive thread
  // and should prefer performance cores over efficiency cores.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  while (true) {
    std::unique_lock<std::mutex> lk(mutex);
    searching = false;
    cv.notify_one();
    cv.wait(lk, [&] { return searching; });

    if (exit)
      return;

    std::function<void()> job = std::move(jobFunc);
    jobFunc = nullptr;

    lk.unlock();

    if (job)
      job();
  }
}

Search::SearchManager *ThreadPool::main_manager() {
  return main_thread()->worker->main_manager();
}

uint64_t ThreadPool::nodes_searched() const {
  return accumulate(&Search::Worker::nodes);
}
uint64_t ThreadPool::tb_hits() const {
  return accumulate(&Search::Worker::tbHits);
}

static size_t next_power_of_two(uint64_t count) {
  return count > 1 ? (2ULL << msb(count - 1)) : 1;
}

void ThreadPool::set(
    const NumaConfig &numaConfig, Search::SharedState sharedState,
    const Search::SearchManager::UpdateContext &updateContext) {

  if (threads.size() > 0) {
    main_thread()->wait_for_search_finished();

    threads.clear();

    boundThreadToNumaNode.clear();
  }

  const size_t requested = sharedState.options["Threads"];

  if (requested > 0) {
    // Binding threads may be problematic when there's multiple NUMA nodes and
    // multiple MetalFish instances running. In particular, if each instance
    // runs a single thread then they would all be mapped to the first NUMA
    // node. This is undesirable, and so the default behaviour (i.e. when the
    // user does not change the NumaConfig UCI setting) is to not bind the
    // threads to processors unless we know for sure that we span NUMA nodes and
    // replication is required.
    const std::string numaPolicy(sharedState.options["NumaPolicy"]);
    const bool doBindThreads = [&]() {
      if (numaPolicy == "none")
        return false;

      if (numaPolicy == "auto")
        return numaConfig.suggests_binding_threads(requested);

      // numaPolicy == "system", or explicitly set by the user
      return true;
    }();

    std::map<NumaIndex, size_t> counts;
    boundThreadToNumaNode =
        doBindThreads
            ? numaConfig.distribute_threads_among_numa_nodes(requested)
            : std::vector<NumaIndex>{};

    if (boundThreadToNumaNode.empty())
      counts[0] = requested;
    else {
      for (size_t i = 0; i < boundThreadToNumaNode.size(); ++i)
        counts[boundThreadToNumaNode[i]]++;
    }

    sharedState.sharedHistories.clear();
    for (auto pair : counts) {
      NumaIndex numaIndex = pair.first;
      uint64_t count = pair.second;
      auto f = [&]() {
        sharedState.sharedHistories.try_emplace(numaIndex,
                                                next_power_of_two(count));
      };
      if (doBindThreads)
        numaConfig.execute_on_numa_node(numaIndex, f);
      else
        f();
    }

    auto threadsPerNode = counts;
    counts.clear();

    while (threads.size() < requested) {
      const size_t threadId = threads.size();
      const NumaIndex numaId =
          doBindThreads ? boundThreadToNumaNode[threadId] : 0;
      auto create_thread = [&]() {
        auto manager =
            threadId == 0
                ? std::unique_ptr<Search::ISearchManager>(
                      std::make_unique<Search::SearchManager>(updateContext))
                : std::make_unique<Search::NullSearchManager>();

        auto binder = doBindThreads
                          ? OptionalThreadToNumaNodeBinder(numaConfig, numaId)
                          : OptionalThreadToNumaNodeBinder(numaId);

        threads.emplace_back(std::make_unique<Thread>(
            sharedState, std::move(manager), threadId, counts[numaId]++,
            threadsPerNode[numaId], binder));
      };

      if (doBindThreads)
        numaConfig.execute_on_numa_node(numaId, create_thread);
      else
        create_thread();
    }

    clear();

    main_thread()->wait_for_search_finished();
  }
}

void ThreadPool::clear() {
  if (threads.size() == 0)
    return;

  for (auto &&th : threads)
    th->clear_worker();

  for (auto &&th : threads)
    th->wait_for_search_finished();

  main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
  main_manager()->previousTimeReduction = 0.85;

  main_manager()->callsCnt = 0;
  main_manager()->bestPreviousScore = VALUE_INFINITE;
  main_manager()->originalTimeAdjust = -1;
  main_manager()->tm.clear();
}

void ThreadPool::run_on_thread(size_t threadId, std::function<void()> f) {
  assert(threads.size() > threadId);
  threads[threadId]->run_custom_job(std::move(f));
}

void ThreadPool::wait_on_thread(size_t threadId) {
  assert(threads.size() > threadId);
  threads[threadId]->wait_for_search_finished();
}

size_t ThreadPool::num_threads() const { return threads.size(); }

void ThreadPool::start_thinking(const OptionsMap &options, Position &pos,
                                StateListPtr &states,
                                Search::LimitsType limits) {

  main_thread()->wait_for_search_finished();

  main_manager()->stopOnPonderhit = stop = abortedSearch = false;
  main_manager()->ponder = limits.ponderMode;

  increaseDepth = true;

  Search::RootMoves rootMoves;
  const auto legalmoves = MoveList<LEGAL>(pos);

  for (const auto &uciMove : limits.searchmoves) {
    auto move = UCIEngine::to_move(pos, uciMove);

    if (std::find(legalmoves.begin(), legalmoves.end(), move) !=
            legalmoves.end() &&
        std::find(rootMoves.begin(), rootMoves.end(), move) == rootMoves.end())
      rootMoves.emplace_back(move);
  }

  if (rootMoves.empty())
    for (const auto &m : legalmoves)
      rootMoves.emplace_back(m);

  Tablebases::Config tbConfig =
      Tablebases::rank_root_moves(options, pos, rootMoves);

  assert(states.get() || setupStates.get());

  if (states.get())
    setupStates = std::move(states);

  const std::string rootFen = pos.fen();
  const bool chess960 = pos.is_chess960();

  for (auto &&th : threads) {
    Thread *thread = th.get();
    thread->run_custom_job([&, thread]() {
      auto &worker = *thread->worker;
      worker.limits = limits;
      worker.nodes = worker.tbHits = worker.bestMoveChanges = 0;
      worker.nmpMinPly = 0;
      worker.rootDepth = worker.completedDepth = 0;
      worker.rootMoves = rootMoves;
      worker.rootPos.set(rootFen, chess960, &worker.rootState);
      worker.rootState = setupStates->back();
      worker.tbConfig = tbConfig;
    });
  }

  for (auto &&th : threads)
    th->wait_for_search_finished();

  main_thread()->start_searching();
}

Thread *ThreadPool::get_best_thread() const {

  Thread *bestThread = threads.front().get();
  Value minScore = VALUE_NONE;

  std::unordered_map<Move, int64_t, Move::MoveHash> votes(
      2 * std::min(size(), bestThread->worker->rootMoves.size()));

  for (auto &&th : threads)
    minScore = std::min(minScore, th->worker->rootMoves[0].score);

  auto thread_voting_value = [minScore](Thread *th) {
    return (th->worker->rootMoves[0].score - minScore + 14) *
           int(th->worker->completedDepth);
  };

  for (auto &&th : threads)
    votes[th->worker->rootMoves[0].pv[0]] += thread_voting_value(th.get());

  for (auto &&th : threads) {
    const auto bestThreadScore = bestThread->worker->rootMoves[0].score;
    const auto newThreadScore = th->worker->rootMoves[0].score;

    const auto &bestThreadPV = bestThread->worker->rootMoves[0].pv;
    const auto &newThreadPV = th->worker->rootMoves[0].pv;

    const auto bestThreadMoveVote = votes[bestThreadPV[0]];
    const auto newThreadMoveVote = votes[newThreadPV[0]];

    const bool bestThreadInProvenWin = is_win(bestThreadScore);
    const bool newThreadInProvenWin = is_win(newThreadScore);

    const bool bestThreadInProvenLoss =
        bestThreadScore != -VALUE_INFINITE && is_loss(bestThreadScore);
    const bool newThreadInProvenLoss =
        newThreadScore != -VALUE_INFINITE && is_loss(newThreadScore);

    // We make sure not to pick a thread with truncated principal variation
    const bool betterVotingValue =
        thread_voting_value(th.get()) * int(newThreadPV.size() > 2) >
        thread_voting_value(bestThread) * int(bestThreadPV.size() > 2);

    if (bestThreadInProvenWin) {
      if (newThreadScore > bestThreadScore)
        bestThread = th.get();
    } else if (bestThreadInProvenLoss) {
      if (newThreadInProvenLoss && newThreadScore < bestThreadScore)
        bestThread = th.get();
    } else if (newThreadInProvenWin || newThreadInProvenLoss ||
               (!is_loss(newThreadScore) &&
                (newThreadMoveVote > bestThreadMoveVote ||
                 (newThreadMoveVote == bestThreadMoveVote &&
                  betterVotingValue))))
      bestThread = th.get();
  }

  return bestThread;
}

void ThreadPool::start_searching() {
  for (auto &&th : threads)
    if (th != threads.front())
      th->start_searching();
}

void ThreadPool::wait_for_search_finished() const {
  for (auto &&th : threads)
    if (th != threads.front())
      th->wait_for_search_finished();
}

std::vector<size_t> ThreadPool::get_bound_thread_count_by_numa_node() const {
  std::vector<size_t> counts;

  if (!boundThreadToNumaNode.empty()) {
    NumaIndex highestNumaNode = 0;
    for (NumaIndex n : boundThreadToNumaNode)
      if (n > highestNumaNode)
        highestNumaNode = n;

    counts.resize(highestNumaNode + 1, 0);

    for (NumaIndex n : boundThreadToNumaNode)
      counts[n] += 1;
  }

  return counts;
}

void ThreadPool::ensure_network_replicated() {
  for (auto &&th : threads)
    th->ensure_network_replicated();
}

} // namespace MetalFish
