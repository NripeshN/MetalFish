/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "core/memory.h"
#include "core/numa.h"
#include "core/position.h"
#include "search/search.h"
#include "thread_win32_osx.h"

namespace MetalFish {

class OptionsMap;
using Value = int;

// Sometimes we don't want to actually bind the threads, but the recipient still
// needs to think it runs on *some* NUMA node, such that it can access
// structures that rely on NUMA node knowledge. This class encapsulates this
// optional process such that the recipient does not need to know whether the
// binding happened or not.
class OptionalThreadToNumaNodeBinder {
public:
  OptionalThreadToNumaNodeBinder(NumaIndex n)
      : numaConfig(nullptr), numaId(n) {}

  OptionalThreadToNumaNodeBinder(const NumaConfig &cfg, NumaIndex n)
      : numaConfig(&cfg), numaId(n) {}

  NumaReplicatedAccessToken operator()() const {
    if (numaConfig != nullptr)
      return numaConfig->bind_current_thread_to_numa_node(numaId);
    else
      return NumaReplicatedAccessToken(numaId);
  }

private:
  const NumaConfig *numaConfig;
  NumaIndex numaId;
};

// Abstraction of a thread. It contains a pointer to the worker and a native
// thread. After construction, the native thread is started with idle_loop()
// waiting for a signal to start searching.
// When the signal is received, the thread starts searching and when
// the search is finished, it goes back to idle_loop() waiting for a new signal.
class Thread {
public:
  Thread(Search::SharedState &, std::unique_ptr<Search::ISearchManager>, size_t,
         size_t, size_t, OptionalThreadToNumaNodeBinder);
  virtual ~Thread();

  void idle_loop();
  void start_searching();
  void clear_worker();
  void run_custom_job(std::function<void()> f);

  void ensure_network_replicated();

  // Thread has been slightly altered to allow running custom jobs, so
  // this name is no longer correct. However, this class (and ThreadPool)
  // require further work to make them properly generic while maintaining
  // appropriate specificity regarding search, from the point of view of an
  // outside user, so renaming of this function is left for whenever that
  // happens.
  void wait_for_search_finished();
  size_t id() const { return idx; }

  LargePagePtr<Search::Worker> worker;
  std::function<void()> jobFunc;

private:
  std::mutex mutex;
  std::condition_variable cv;
  size_t idx, idxInNuma, totalNuma, nthreads;
  bool exit = false, searching = true; // Set before starting std::thread
  NativeThread stdThread;
  NumaReplicatedAccessToken numaAccessToken;
};

// ThreadPool struct handles all the threads-related stuff like init, starting,
// parking and, most importantly, launching a thread. All the access to threads
// is done through this class.
class ThreadPool {
public:
  ThreadPool() {}

  ~ThreadPool() {
    // destroy any existing thread(s)
    if (threads.size() > 0) {
      main_thread()->wait_for_search_finished();

      threads.clear();
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;

  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  void start_thinking(const OptionsMap &, Position &, StateListPtr &,
                      Search::LimitsType);
  void run_on_thread(size_t threadId, std::function<void()> f);
  void wait_on_thread(size_t threadId);
  size_t num_threads() const;
  void clear();
  void set(const NumaConfig &numaConfig, Search::SharedState,
           const Search::SearchManager::UpdateContext &);

  Search::SearchManager *main_manager();
  Thread *main_thread() const { return threads.front().get(); }
  uint64_t nodes_searched() const;
  uint64_t tb_hits() const;
  Thread *get_best_thread() const;
  void start_searching();
  void wait_for_search_finished() const;

  std::vector<size_t> get_bound_thread_count_by_numa_node() const;

  void ensure_network_replicated();

  std::atomic_bool stop, abortedSearch, increaseDepth;

  auto cbegin() const noexcept { return threads.cbegin(); }
  auto begin() noexcept { return threads.begin(); }
  auto end() noexcept { return threads.end(); }
  auto cend() const noexcept { return threads.cend(); }
  auto size() const noexcept { return threads.size(); }
  auto empty() const noexcept { return threads.empty(); }

private:
  StateListPtr setupStates;
  std::vector<std::unique_ptr<Thread>> threads;
  std::vector<NumaIndex> boundThreadToNumaNode;

  uint64_t accumulate(std::atomic<uint64_t> Search::Worker::*member) const {

    uint64_t sum = 0;
    for (auto &&th : threads)
      sum += (th->worker.get()->*member).load(std::memory_order_relaxed);
    return sum;
  }
};

} // namespace MetalFish

#endif // #ifndef THREAD_H_INCLUDED
