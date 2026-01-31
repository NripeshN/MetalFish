/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  Policy mapping utilities implementation
*/

#include "policy_utils.h"
#include <cassert>

namespace MetalFish {
namespace NN {

namespace {

// Helper to get file and rank from square (use different names to avoid conflict)
inline int get_file(Square sq) { return sq & 7; }
inline int get_rank(Square sq) { return sq >> 3; }

// Helper to create square from file and rank
inline Square make_square(int file, int rank) {
    return Square((rank << 3) + file);
}

// Apply transform to square
Square TransformSquare(Square sq, int transform) {
    int file = get_file(sq);
    int rank = get_rank(sq);
    
    if (transform & FlipTransform) {
        file = 7 - file;
    }
    if (transform & MirrorTransform) {
        rank = 7 - rank;
    }
    if (transform & TransposeTransform) {
        std::swap(file, rank);
    }
    
    return make_square(file, rank);
}

}  // namespace

// Simplified move encoding for conventional policy head
// This is a basic implementation that needs to match lc0's encoding
int MoveToNNIndex(Move move, int transform) {
    if (!move.is_ok()) return -1;
    
    Square from = move.from_sq();
    Square to = move.to_sq();
    
    // Apply transform
    from = TransformSquare(from, transform);
    to = TransformSquare(to, transform);
    
    int from_file = get_file(from);
    int from_rank = get_rank(from);
    int to_file = get_file(to);
    int to_rank = get_rank(to);
    
    int from_idx = from_rank * 8 + from_file;
    
    // Calculate direction
    int delta_file = to_file - from_file;
    int delta_rank = to_rank - from_rank;
    
    // Determine move type (simplified - needs full knight move handling, etc.)
    int move_type = 0;
    
    // For now, use basic queen moves (horizontal, vertical, diagonal)
    // This is simplified and will need proper implementation
    if (delta_rank == 0 && delta_file != 0) {
        // Horizontal
        move_type = (delta_file > 0) ? 0 : 28; // East/West
    } else if (delta_file == 0 && delta_rank != 0) {
        // Vertical
        move_type = (delta_rank > 0) ? 14 : 42; // North/South
    } else if (abs(delta_file) == abs(delta_rank)) {
        // Diagonal
        if (delta_file > 0 && delta_rank > 0) move_type = 7;   // NE
        else if (delta_file > 0 && delta_rank < 0) move_type = 21; // SE
        else if (delta_file < 0 && delta_rank > 0) move_type = 35; // NW
        else move_type = 49; // SW
    } else {
        // Knight moves or promotions - TODO
        return -1; // Not implemented yet
    }
    
    // Look up in policy map
    int map_idx = move_type * 64 + from_idx;
    if (map_idx >= 0 && map_idx < 73 * 64) {
        return kConvPolicyMap[map_idx];
    }
    
    return -1;
}

// Simplified move decoding
Move MoveFromNNIndex(int idx, int transform) {
    if (idx < 0 || idx >= kPolicyOutputs) {
        return Move::none();
    }
    
    // Find the policy map entry that matches this index
    // This is inefficient but works for now
    for (int i = 0; i < 73 * 64; i++) {
        if (kConvPolicyMap[i] == idx) {
            int move_type = i / 64;
            int from_idx = i % 64;
            
            int from_file = from_idx % 8;
            int from_rank = from_idx / 8;
            
            Square from = make_square(from_file, from_rank);
            
            // Decode move type to get destination
            // This is simplified - needs full implementation
            // TODO: Implement full decoding logic
            
            return Move::none(); // Placeholder
        }
    }
    
    return Move::none();
}

// Attention policy encoding (simplified)
int MoveToAttentionIndex(Move move, int transform) {
    if (!move.is_ok()) return -1;
    
    Square from = move.from_sq();
    Square to = move.to_sq();
    
    // Apply transform
    from = TransformSquare(from, transform);
    to = TransformSquare(to, transform);
    
    // For attention policy, it's simpler: from_square * 64 + to_square + promotions
    int base_idx = static_cast<int>(from) * 64 + static_cast<int>(to);
    
    // Add promotion offset if needed (TODO)
    
    // Look up in attention policy map
    if (base_idx >= 0 && base_idx < 64 * 64 + 8 * 24) {
        return kAttnPolicyMap[base_idx];
    }
    
    return -1;
}

// Attention policy decoding (simplified)
Move MoveFromAttentionIndex(int idx, int transform) {
    if (idx < 0 || idx >= kAttentionPolicyOutputs) {
        return Move::none();
    }
    
    // Find the attention policy map entry
    for (int i = 0; i < 64 * 64 + 8 * 24; i++) {
        if (kAttnPolicyMap[i] == idx) {
            if (i < 64 * 64) {
                int from = i / 64;
                int to = i % 64;
                
                Square from_sq = TransformSquare(Square(from), transform);
                Square to_sq = TransformSquare(Square(to), transform);
                
                return Move(from_sq, to_sq);
            }
            // TODO: Handle promotions
        }
    }
    
    return Move::none();
}

}  // namespace NN
}  // namespace MetalFish
