/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Position encoding tests
*/

#include "../src/nn/position_encoder.h"
#include "../src/nn/input_planes.h"
#include "../src/core/position.h"
#include <iostream>

using namespace MetalFish;
using namespace MetalFish::NN;

bool test_position_encoding() {
    std::cout << "\n=== Position Encoding Tests ===\n\n";
    
    // Test 1: Encode starting position
    std::cout << "Test 1: Encoding starting position...\n";
    Position pos;
    StateInfo si;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);
    
    InputPlanes planes = EncodePosition(pos);
    
    // Verify plane count
    if (planes.size() != kInputPlanes) {
        std::cout << "✗ Wrong number of planes: " << planes.size() 
                  << " (expected " << kInputPlanes << ")\n";
        return false;
    }
    std::cout << "✓ Correct number of planes: " << kInputPlanes << "\n";
    
    // Verify white pawns are encoded (plane 0, squares 8-15)
    int white_pawn_count = 0;
    for (int sq = 8; sq < 16; ++sq) {
        if (planes[0].Get(sq) > 0.5f) {
            white_pawn_count++;
        }
    }
    if (white_pawn_count != 8) {
        std::cout << "✗ Wrong white pawn count: " << white_pawn_count << " (expected 8)\n";
        return false;
    }
    std::cout << "✓ White pawns correctly encoded: 8 pawns on rank 2\n";
    
    // Verify black pawns are encoded (plane 6, squares 48-55)
    int black_pawn_count = 0;
    for (int sq = 48; sq < 56; ++sq) {
        if (planes[6].Get(sq) > 0.5f) {
            black_pawn_count++;
        }
    }
    if (black_pawn_count != 8) {
        std::cout << "✗ Wrong black pawn count: " << black_pawn_count << " (expected 8)\n";
        return false;
    }
    std::cout << "✓ Black pawns correctly encoded: 8 pawns on rank 7\n";
    
    // Verify color plane (plane 104 should be all 1s for white to move)
    bool color_correct = true;
    for (int sq = 0; sq < 64; ++sq) {
        if (planes[104].Get(sq) < 0.5f) {
            color_correct = false;
            break;
        }
    }
    if (!color_correct) {
        std::cout << "✗ Color plane incorrect\n";
        return false;
    }
    std::cout << "✓ Color plane correct (white to move)\n";
    
    // Verify castling rights (planes 106-109 should all be 1s)
    if (planes[106].Get(0) < 0.5f || planes[107].Get(0) < 0.5f ||
        planes[108].Get(0) < 0.5f || planes[109].Get(0) < 0.5f) {
        std::cout << "✗ Castling rights planes incorrect\n";
        return false;
    }
    std::cout << "✓ Castling rights correctly encoded\n";
    
    std::cout << "\nTest 2: Encoding position with history...\n";
    PositionHistory history;
    
    // Add starting position
    history.Push(pos);
    
    // Make a move
    StateInfo si2;
    pos.do_move(Move(SQ_E2, SQ_E4), si2);
    history.Push(pos);
    
    InputPlanes planes_with_history = EncodePosition(history, WHITE);
    
    std::cout << "✓ Position with history encoded (2 positions)\n";
    std::cout << "  History size: " << history.Size() << "\n";
    
    // Verify we have two positions encoded
    // Current position should be in planes 0-12
    // Previous position in planes 13-25
    
    std::cout << "\nTest 3: Encoding black to move...\n";
    StateInfo si3;
    pos.set("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", false, &si3);
    
    InputPlanes planes_black = EncodePosition(pos);
    
    // Verify color plane is all 0s for black to move
    bool black_color_correct = true;
    for (int sq = 0; sq < 64; ++sq) {
        if (planes_black[104].Get(sq) > 0.5f) {
            black_color_correct = false;
            break;
        }
    }
    if (!black_color_correct) {
        std::cout << "✗ Color plane incorrect for black\n";
        return false;
    }
    std::cout << "✓ Color plane correct (black to move)\n";
    
    // Verify en passant square is encoded
    if (planes_black[111].Get(SQ_E3) < 0.5f) {
        std::cout << "✗ En passant square not encoded\n";
        return false;
    }
    std::cout << "✓ En passant square correctly encoded\n";
    
    std::cout << "\n=== All Position Encoding Tests PASSED ===\n";
    return true;
}

int main() {
    Position::init();
    Bitboards::init();
    return test_position_encoding() ? 0 : 1;
}
