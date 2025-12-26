/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Position tests
*/

#include "core/types.h"
#include "core/bitboard.h"
#include "core/position.h"
#include "core/zobrist.h"
#include <iostream>
#include <cassert>

using namespace MetalFish;

bool test_position() {
    // Initialize
    init_bitboards();
    Position::init();
    
    Position pos;
    StateInfo st;
    
    // Test starting position
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st);
    
    assert(pos.side_to_move() == WHITE);
    assert(pos.piece_on(SQ_E1) == W_KING);
    assert(pos.piece_on(SQ_E8) == B_KING);
    assert(pos.piece_on(SQ_D1) == W_QUEEN);
    assert(pos.piece_on(SQ_D8) == B_QUEEN);
    assert(pos.piece_on(SQ_E4) == NO_PIECE);
    
    // Test piece counts
    assert(pos.count<PAWN>(WHITE) == 8);
    assert(pos.count<PAWN>(BLACK) == 8);
    assert(pos.count<KING>(WHITE) == 1);
    assert(pos.count<KING>(BLACK) == 1);
    
    // Test FEN output
    std::string fen = pos.fen();
    assert(fen.find("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") == 0);
    
    // Test castling rights
    assert(pos.can_castle(WHITE_OO));
    assert(pos.can_castle(WHITE_OOO));
    assert(pos.can_castle(BLACK_OO));
    assert(pos.can_castle(BLACK_OOO));
    
    // Test position with en passant
    pos.set("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3", false, &st);
    assert(pos.ep_square() == SQ_E6);
    
    // Test position after e4
    StateInfo st2;
    Move e2e4(SQ_E2, SQ_E4);
    
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st);
    pos.do_move(e2e4, st2);
    
    assert(pos.side_to_move() == BLACK);
    assert(pos.piece_on(SQ_E4) == W_PAWN);
    assert(pos.piece_on(SQ_E2) == NO_PIECE);
    assert(pos.ep_square() == SQ_E3);
    
    // Test undo move
    pos.undo_move(e2e4);
    assert(pos.side_to_move() == WHITE);
    assert(pos.piece_on(SQ_E2) == W_PAWN);
    assert(pos.piece_on(SQ_E4) == NO_PIECE);
    
    // Test complex position
    pos.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", false, &st);
    assert(pos.piece_on(SQ_E5) == W_KNIGHT);
    assert(pos.piece_on(SQ_B4) == B_PAWN);
    
    // Test checkers
    pos.set("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3", false, &st);
    assert(pos.checkers() != 0);  // Black king is in check
    
    // Test key uniqueness (different positions should have different keys)
    StateInfo st3, st4;
    Position pos2;
    
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st3);
    pos2.set("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", false, &st4);
    
    assert(pos.key() != pos2.key());
    
    // Test position validation
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st);
    assert(pos.pos_is_ok());
    
    std::cout << "All position tests passed!" << std::endl;
    return true;
}


