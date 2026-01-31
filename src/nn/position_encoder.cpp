/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  Position encoder implementation
*/

#include "position_encoder.h"
#include "../core/bitboard.h"

namespace MetalFish {
namespace NN {

namespace {

// Encode a single board position into planes (13 planes per position)
// Planes 0-5: Our pieces (P, N, B, R, Q, K)
// Planes 6-11: Opponent pieces (p, n, b, r, q, k)
// Plane 12: Repetitions (for now, always 0)
void EncodeBoardPosition(InputPlanes& planes, int plane_offset, 
                         const Position& pos, Color us) {
    // Encode our pieces (planes 0-5)
    for (PieceType pt = PAWN; pt <= KING; ++pt) {
        Bitboard bb = pos.pieces(us, pt);
        int plane_idx = plane_offset + (pt - PAWN);
        
        while (bb) {
            Square sq = pop_lsb(bb);
            planes[plane_idx].Set(sq, 1.0f);
        }
    }
    
    // Encode opponent pieces (planes 6-11)
    Color them = ~us;
    for (PieceType pt = PAWN; pt <= KING; ++pt) {
        Bitboard bb = pos.pieces(them, pt);
        int plane_idx = plane_offset + 6 + (pt - PAWN);
        
        while (bb) {
            Square sq = pop_lsb(bb);
            planes[plane_idx].Set(sq, 1.0f);
        }
    }
    
    // Plane 12: Repetitions (simplified - always 0 for now)
    // TODO: Implement proper repetition detection
}

// Encode auxiliary planes (8 planes: 104-111)
void EncodeAuxiliaryPlanes(InputPlanes& planes, const Position& pos, Color us) {
    // Plane 104: Color (all 1s if white to move, all 0s if black)
    if (us == WHITE) {
        planes[104].Fill(1.0f);
    }
    
    // Plane 105: Total move count (normalized)
    int total_moves = pos.game_ply();
    float move_count_normalized = std::min(total_moves / 100.0f, 1.0f);
    planes[105].Fill(move_count_normalized);
    
    // Planes 106-109: Castling rights
    // Plane 106: Our king-side castling
    if (pos.can_castle(us == WHITE ? WHITE_OO : BLACK_OO)) {
        planes[106].Fill(1.0f);
    }
    
    // Plane 107: Our queen-side castling
    if (pos.can_castle(us == WHITE ? WHITE_OOO : BLACK_OOO)) {
        planes[107].Fill(1.0f);
    }
    
    // Plane 108: Opponent king-side castling
    Color them = ~us;
    if (pos.can_castle(them == WHITE ? WHITE_OO : BLACK_OO)) {
        planes[108].Fill(1.0f);
    }
    
    // Plane 109: Opponent queen-side castling
    if (pos.can_castle(them == WHITE ? WHITE_OOO : BLACK_OOO)) {
        planes[109].Fill(1.0f);
    }
    
    // Plane 110: No-progress count (50-move rule counter, normalized)
    int no_progress = pos.rule50_count();
    float no_progress_normalized = std::min(no_progress / 100.0f, 1.0f);
    planes[110].Fill(no_progress_normalized);
    
    // Plane 111: Reserved / En passant (simplified)
    if (pos.ep_square() != SQ_NONE) {
        planes[111].Set(pos.ep_square(), 1.0f);
    }
}

}  // namespace

// PositionHistory implementation
void PositionHistory::Push(const Position& pos) {
    if (fens_.size() >= kMaxHistory) {
        fens_.pop_front();
    }
    fens_.push_back(pos.fen());
}

std::string PositionHistory::GetFEN(int index) const {
    if (index < 0 || index >= static_cast<int>(fens_.size())) {
        return "";
    }
    return fens_[fens_.size() - 1 - index];
}

// Encode position with history
InputPlanes EncodePosition(const PositionHistory& history, Color side_to_move) {
    InputPlanes planes;
    
    // Encode up to 8 historical positions (planes 0-103)
    int positions_to_encode = std::min(static_cast<int>(history.Size()), kMoveHistory);
    
    StateInfo si;
    for (int i = 0; i < positions_to_encode; ++i) {
        std::string fen = history.GetFEN(i);
        if (!fen.empty()) {
            Position pos;
            pos.set(fen, false, &si);
            
            int plane_offset = i * kPlanesPerBoard;
            EncodeBoardPosition(planes, plane_offset, pos, side_to_move);
        }
    }
    
    // Encode auxiliary planes (planes 104-111)
    if (positions_to_encode > 0) {
        std::string current_fen = history.GetFEN(0);
        if (!current_fen.empty()) {
            Position pos;
            pos.set(current_fen, false, &si);
            EncodeAuxiliaryPlanes(planes, pos, side_to_move);
        }
    }
    
    return planes;
}

// Simplified encoder for single position
InputPlanes EncodePosition(const Position& pos) {
    InputPlanes planes;
    
    Color us = pos.side_to_move();
    
    // Encode current position in planes 0-12
    EncodeBoardPosition(planes, 0, pos, us);
    
    // Remaining 91 planes (13-103) stay zero (no history)
    
    // Encode auxiliary planes (planes 104-111)
    EncodeAuxiliaryPlanes(planes, pos, us);
    
    return planes;
}

}  // namespace NN
}  // namespace MetalFish
