/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

*/

#include "search/search.h"
#include "search/tt.h"
#include "core/movegen.h"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace MetalFish {

namespace Search {

// =============================================================================
// Lazy SMP Thread Pool Implementation
// =============================================================================
// Each thread has its own Worker with its own Position copy.
// All threads share the TranspositionTable for communication.
// Threads search at slightly different depths to maximize diversity.

class ThreadPool {
public:
  ThreadPool() = default;
  ~ThreadPool() { shutdown(); }

  void set_size(size_t count) {
    count = std::max(size_t(1), count);

    // Stop any ongoing search first
    stop_searching();
    wait_for_search_finished();

    // Clear existing threads
    for (auto &t : threads_) {
      if (t.joinable())
        t.join();
    }
    threads_.clear();
    workers_.clear();

    // Create workers (each with their own index for depth variation)
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

  // Start all helper threads (not main thread)
  void start_helper_threads(Position &pos, const LimitsType &limits,
                            [[maybe_unused]] StateListPtr &states) {
    if (workers_.size() <= 1)
      return;

    // Wait for any previous threads to finish
    for (auto &t : threads_) {
      if (t.joinable())
        t.join();
    }
    threads_.clear();

    // Start helper threads (index 1 onwards)
    for (size_t i = 1; i < workers_.size(); ++i) {
      Worker *w = workers_[i].get();
      
      // Each helper thread gets a copy of the position
      threads_.emplace_back([this, w, &pos, &limits]() {
        // Copy position for this thread
        StateListPtr localStates = std::make_unique<std::deque<StateInfo>>(1);
        Position localPos;
        localPos.set(pos.fen(), pos.is_chess960(), &localStates->back());
        
        // Copy root moves from main thread
        w->rootMoves = workers_[0]->rootMoves;
        
        // Start searching with local position copy
        w->start_searching(localPos, limits, localStates);
      });
    }
  }

  void stop_searching() {
    Signals_stop = true;
    for (auto &w : workers_) {
      w->stopRequested = true;
    }
  }

  void wait_for_search_finished() {
    // Wait for all helper threads
    for (auto &t : threads_) {
      if (t.joinable())
        t.join();
    }
    threads_.clear();

    // Wait for main thread
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

  // Get best thread based on completed depth and score
  Worker *best_thread() {
    if (workers_.empty())
      return nullptr;
    if (workers_.size() == 1)
      return workers_[0].get();

    Worker *best = workers_[0].get();
    int bestScore = best->rootMoves.empty() ? -VALUE_INFINITE : best->rootMoves[0].score;
    Depth bestDepth = best->completedDepth;

    for (size_t i = 1; i < workers_.size(); ++i) {
      Worker *w = workers_[i].get();
      if (w->rootMoves.empty())
        continue;

      int score = w->rootMoves[0].score;
      Depth depth = w->completedDepth;

      // Prefer deeper search, or equal depth with better score
      if (depth > bestDepth || (depth == bestDepth && score > bestScore)) {
        best = w;
        bestScore = score;
        bestDepth = depth;
      }
    }

    return best;
  }

private:
  std::vector<std::unique_ptr<Worker>> workers_;
  std::vector<std::thread> threads_;
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
  Worker *main = Threads.main_worker();
  if (!main)
    return;

  Signals_stop = false;

  // Reset all workers
  for (size_t i = 0; i < Threads.size(); ++i) {
    Worker *w = Threads.worker(i);
    if (w) {
      w->stopRequested = false;
      w->nodes = 0;
    }
  }

  // Generate root moves on main thread first
  main->rootMoves.clear();
  for (const auto &m : MoveList<LEGAL>(pos)) {
    if (limits.searchmoves.empty() ||
        std::find(limits.searchmoves.begin(), limits.searchmoves.end(), m) !=
            limits.searchmoves.end())
      main->rootMoves.emplace_back(m);
  }

  if (main->rootMoves.empty()) {
    // No legal moves - checkmate or stalemate
    main->rootMoves.emplace_back(Move::none());
    return;
  }

  // Start helper threads with position copies (Lazy SMP)
  Threads.start_helper_threads(pos, limits, states);

  // Main thread searches directly
  main->start_searching(pos, limits, states);
}

void stop_search() { Threads.stop_searching(); }

void wait_for_search() { Threads.wait_for_search_finished(); }

void clear_threads() { Threads.clear(); }

} // namespace Search

} // namespace MetalFish
