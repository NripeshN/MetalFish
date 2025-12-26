/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Main search implementation - Hybrid CPU-GPU alpha-beta search with
  batched GPU evaluation.
*/

#include "core/types.h"
#include <atomic>
#include <vector>

namespace MetalFish {
namespace Search {

// Search stack for each ply
struct Stack {
    Move* pv;
    int ply;
    Move currentMove;
    Move excludedMove;
    Value staticEval;
    bool inCheck;
    bool ttPv;
    int moveCount;
};

// Root move structure
struct RootMove {
    Move pv[MAX_PLY + 1];
    int pvLength;
    Value score;
    Value previousScore;
    int selDepth;
    
    explicit RootMove(Move m) : score(-VALUE_INFINITE), previousScore(-VALUE_INFINITE), 
                                 selDepth(0), pvLength(1) {
        pv[0] = m;
    }
    
    bool operator<(const RootMove& other) const {
        return score > other.score; // Higher score first
    }
};

// Search limits
struct Limits {
    int depth = 0;
    uint64_t nodes = 0;
    int movetime = 0;
    int time[COLOR_NB] = {0, 0};
    int inc[COLOR_NB] = {0, 0};
    int movestogo = 0;
    bool infinite = false;
    bool ponder = false;
};

// Search state
class SearchState {
public:
    std::vector<RootMove> rootMoves;
    Limits limits;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> nodes{0};
    
    int rootDepth = 0;
    int completedDepth = 0;
    
    void clear() {
        rootMoves.clear();
        stop = false;
        nodes = 0;
        rootDepth = 0;
        completedDepth = 0;
    }
};

// Global search state
static SearchState searchState;

/**
 * Alpha-beta search with GPU-batched evaluation
 * 
 * The key optimization is collecting leaf nodes during search
 * and evaluating them in batches on the GPU.
 */
template<bool PvNode>
Value search(Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {
    // Base case: quiescence search at depth 0
    if (depth <= 0) {
        return alpha; // TODO: qsearch
    }
    
    // Check for stop signal
    if (searchState.stop) {
        return VALUE_ZERO;
    }
    
    searchState.nodes++;
    
    // TODO: Implement full search
    // - Transposition table lookup
    // - Move generation
    // - Move ordering
    // - Null move pruning
    // - Late move reductions
    // - Recursive search
    // - GPU batch evaluation at leaf nodes
    
    return VALUE_ZERO;
}

/**
 * Iterative deepening main loop
 */
void iterative_deepening() {
    Stack stack[MAX_PLY + 10];
    Stack* ss = stack + 7; // Allow negative indices
    
    for (int d = 1; d <= searchState.limits.depth; ++d) {
        searchState.rootDepth = d;
        
        Value alpha = -VALUE_INFINITE;
        Value beta = VALUE_INFINITE;
        
        // Search at current depth
        Value score = search<true>(ss, alpha, beta, d, false);
        
        if (searchState.stop) break;
        
        searchState.completedDepth = d;
        
        // TODO: Print UCI info
        // TODO: Time management
    }
}

/**
 * Start the search
 */
void start(const Limits& limits) {
    searchState.limits = limits;
    searchState.stop = false;
    
    // Set default depth if not specified
    if (searchState.limits.depth == 0) {
        searchState.limits.depth = limits.infinite ? MAX_PLY : 20;
    }
    
    iterative_deepening();
}

/**
 * Stop the search
 */
void stop() {
    searchState.stop = true;
}

/**
 * Get search statistics
 */
uint64_t get_nodes() {
    return searchState.nodes;
}

} // namespace Search
} // namespace MetalFish

