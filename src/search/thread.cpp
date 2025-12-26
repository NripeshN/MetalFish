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
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace MetalFish {

namespace Search {

// Thread pool for parallel search (simplified implementation)
// A full implementation would have a proper thread pool with work stealing

class ThreadPool {
public:
    ThreadPool() = default;
    ~ThreadPool() { stop(); }
    
    void set_size(size_t count) {
        stop();
        
        workers.clear();
        for (size_t i = 0; i < count; ++i) {
            workers.push_back(std::make_unique<Worker>(i));
        }
    }
    
    size_t size() const { return workers.size(); }
    
    Worker* main_worker() {
        return workers.empty() ? nullptr : workers[0].get();
    }
    
    void start_searching(Position& pos, const LimitsType& limits, StateListPtr& states) {
        // For now, only use main thread
        if (!workers.empty()) {
            workers[0]->start_searching(pos, limits, states);
        }
    }
    
    void stop() {
        Signals_stop = true;
        for (auto& w : workers) {
            w->stopRequested = true;
            w->wait_for_search_finished();
        }
    }
    
    void clear() {
        for (auto& w : workers) {
            w->clear();
        }
    }
    
    uint64_t nodes_searched() const {
        uint64_t total = 0;
        for (const auto& w : workers) {
            total += w->nodes.load();
        }
        return total;
    }

private:
    std::vector<std::unique_ptr<Worker>> workers;
};

// Global thread pool
static ThreadPool Threads;

void set_thread_count(size_t count) {
    Threads.set_size(count);
}

size_t thread_count() {
    return Threads.size();
}

} // namespace Search

} // namespace MetalFish
