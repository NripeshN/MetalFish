#include "core/perft.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include "uci/uci.h"
#include <iostream>
#include <vector>

namespace MetalFish {

uint64_t perft(Position& pos, int depth, bool root) {
    if (depth == 0)
        return 1;
    
    uint64_t nodes = 0;
    
    MoveList<LEGAL> moves(pos);
    
    // At depth 1, we can just count the moves (leaf nodes)
    if (depth == 1 && !root)
        return moves.size();
    
    for (const Move* it = moves.begin(); it != moves.end(); ++it) {
        Move m = *it;
        StateInfo st;
        
        pos.do_move(m, st);
        uint64_t cnt = perft(pos, depth - 1, false);
        pos.undo_move(m);
        
        nodes += cnt;
        
        if (root)
            std::cout << UCI::move_to_uci(m, false) << ": " << cnt << std::endl;
    }
    
    return nodes;
}

void divide(Position& pos, int depth) {
    perft(pos, depth, true);
}

} // namespace MetalFish
