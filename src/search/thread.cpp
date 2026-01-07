/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/search.h"
#include "search/tt.h"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace MetalFish {

namespace Search {

// Thread pool for search
// Currently single-threaded, but infrastructure ready for Lazy SMP

class ThreadPool {
public:
  ThreadPool() = default;
  ~ThreadPool() { shutdown(); }

  void set_size(size_t count) {
    // For now, always use at least 1 worker
    count = std::max(size_t(1), count);

    // Stop any ongoing search first
    stop_searching();

    workers_.clear();

    // Create workers
    for (size_t i = 0; i < count; ++i) {
      workers_.push_back(std::make_unique<Worker>(i));
    }

    threadCount_ = count;
  }

  size_t size() const { return workers_.size(); }

  Worker *main_worker() {
    return workers_.empty() ? nullptr : workers_[0].get();
  }

  Worker *worker(size_t idx) {
    return idx < workers_.size() ? workers_[idx].get() : nullptr;
  }

  void stop_searching() {
    Signals_stop = true;
    for (auto &w : workers_) {
      w->stopRequested = true;
    }
  }

  void wait_for_search_finished() {
    // Wait for main thread to finish
    if (!workers_.empty()) {
      workers_[0]->wait_for_search_finished();
    }
  }

  void shutdown() {
    stop_searching();
    wait_for_search_finished();
    workers_.clear();
  }

  void clear() {
    for (auto &w : workers_) {
      w->clear();
    }
  }

  uint64_t nodes_searched() const {
    uint64_t total = 0;
    for (const auto &w : workers_) {
      total += w->nodes.load();
    }
    return total;
  }

  // Get best thread (for single-threaded, just return main)
  Worker *best_thread() {
    return main_worker();
  }

private:
  std::vector<std::unique_ptr<Worker>> workers_;
  size_t threadCount_ = 1;
};

// Global thread pool
static ThreadPool Threads;

void set_thread_count(size_t count) {
  count = std::max(size_t(1), std::min(count, size_t(512)));
  Threads.set_size(count);
}

size_t thread_count() { return Threads.size(); }

Worker *main_thread() { return Threads.main_worker(); }

Worker *best_thread() { return Threads.best_thread(); }

uint64_t total_nodes() { return Threads.nodes_searched(); }

void start_search(Position &pos, const LimitsType &limits,
                  StateListPtr &states) {
  // For now, use single-threaded search
  Worker *main = Threads.main_worker();
  if (main) {
    Signals_stop = false;
    main->stopRequested = false;
    main->start_searching(pos, limits, states);
  }
}

void stop_search() { Threads.stop_searching(); }

void wait_for_search() { Threads.wait_for_search_finished(); }

void clear_threads() { Threads.clear(); }

} // namespace Search

} // namespace MetalFish
