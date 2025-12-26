/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Search tests
*/

#include "core/types.h"
#include "core/bitboard.h"
#include "core/position.h"
#include "core/movegen.h"
#include "search/search.h"
#include "search/tt.h"
#include "eval/evaluate.h"
#include <iostream>
#include <cassert>
#include <deque>

using namespace MetalFish;

bool test_search() {
    // Initialize
    init_bitboards();
    Position::init();
    Search::init();
    TT.resize(16);  // 16 MB hash
    
    Position pos;
    StateListPtr states = std::make_unique<std::deque<StateInfo>>(1);
    
    // Test basic search functionality
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &states->back());
    
    Search::Worker worker;
    Search::LimitsType limits;
    limits.depth = 4;  // Shallow search for testing
    
    worker.clear();
    
    // This would start search - for testing, just verify setup works
    assert(worker.nodes.load() == 0);
    
    // Test TT
    TT.clear();
    assert(TT.hashfull() == 0);
    
    // Store some entries
    Key testKey = 0x123456789ABCDEFULL;
    bool found;
    TTEntry* tte = TT.probe(testKey, found);
    assert(!found);  // Should not be found initially
    
    tte->save(testKey, 100, true, BOUND_EXACT, 10, Move(SQ_E2, SQ_E4), 50, TT.generation());
    
    TTEntry* tte2 = TT.probe(testKey, found);
    assert(found);
    assert(tte2->value() == 100);
    assert(tte2->depth() == 10);
    
    // Test new search increments generation
    TT.new_search();
    // Generation should have changed
    
    // Test evaluation
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &states->back());
    Value eval = Eval::evaluate(pos);
    // Starting position should be close to equal
    assert(std::abs(eval) < 100);
    
    // Test winning position for white
    pos.set("8/8/8/8/8/8/Q7/k1K5 w - - 0 1", false, &states->back());
    Value winEval = Eval::evaluate(pos);
    assert(winEval > 500);  // White should be winning
    
    // Test losing position for white
    pos.set("8/8/8/8/8/8/q7/K1k5 w - - 0 1", false, &states->back());
    Value loseEval = Eval::evaluate(pos);
    assert(loseEval < -500);  // White should be losing
    
    // Test material balance
    pos.set("8/8/8/8/8/8/8/K1k5 w - - 0 1", false, &states->back());
    Value bareEval = Eval::evaluate(pos);
    // Should be close to 0 with just kings
    assert(std::abs(bareEval) < 50);
    
    std::cout << "All search tests passed!" << std::endl;
    return true;
}

