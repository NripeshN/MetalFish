/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Move generation kernels for GPU-accelerated chess.
  
  Move generation on GPU is challenging due to variable output size.
  We use a two-pass approach:
  1. Count moves for each position (to allocate output)
  2. Generate actual moves

  For batched processing during search, we generate moves for multiple
  positions in parallel, leveraging the massive parallelism of the GPU.
*/

#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Bitboard Utilities
///////////////////////////////////////////////////////////////////////////////

// Direction shifts for sliding pieces
constant int DIRECTIONS[8] = {8, -8, 1, -1, 9, -9, 7, -7}; // N, S, E, W, NE, SW, NW, SE

// Pre-computed attack tables would be loaded from buffer
// For now, compute on the fly

/**
 * Get knight attack bitboard for a square
 */
inline uint64_t knight_attacks(int sq) {
    uint64_t bb = 1ULL << sq;
    uint64_t attacks = 0;
    
    // Knight moves: +/- 17, 15, 10, 6 with file bounds checking
    int rank = sq / 8;
    int file = sq % 8;
    
    // +17 = up 2, right 1
    if (rank < 6 && file < 7) attacks |= bb << 17;
    // +15 = up 2, left 1
    if (rank < 6 && file > 0) attacks |= bb << 15;
    // +10 = up 1, right 2
    if (rank < 7 && file < 6) attacks |= bb << 10;
    // +6 = up 1, left 2
    if (rank < 7 && file > 1) attacks |= bb << 6;
    // -17 = down 2, left 1
    if (rank > 1 && file > 0) attacks |= bb >> 17;
    // -15 = down 2, right 1
    if (rank > 1 && file < 7) attacks |= bb >> 15;
    // -10 = down 1, left 2
    if (rank > 0 && file > 1) attacks |= bb >> 10;
    // -6 = down 1, right 2
    if (rank > 0 && file < 6) attacks |= bb >> 6;
    
    return attacks;
}

/**
 * Get king attack bitboard for a square
 */
inline uint64_t king_attacks(int sq) {
    uint64_t bb = 1ULL << sq;
    uint64_t attacks = 0;
    
    int rank = sq / 8;
    int file = sq % 8;
    
    // All 8 directions
    if (rank < 7) attacks |= bb << 8;  // N
    if (rank > 0) attacks |= bb >> 8;  // S
    if (file < 7) attacks |= bb << 1;  // E
    if (file > 0) attacks |= bb >> 1;  // W
    if (rank < 7 && file < 7) attacks |= bb << 9;  // NE
    if (rank < 7 && file > 0) attacks |= bb << 7;  // NW
    if (rank > 0 && file < 7) attacks |= bb >> 7;  // SE
    if (rank > 0 && file > 0) attacks |= bb >> 9;  // SW
    
    return attacks;
}

/**
 * Get white pawn attacks
 */
inline uint64_t pawn_attacks_white(uint64_t pawns) {
    return ((pawns << 7) & 0x7F7F7F7F7F7F7F7FULL) |  // NW, mask off H file
           ((pawns << 9) & 0xFEFEFEFEFEFEFEFEULL);   // NE, mask off A file
}

/**
 * Get black pawn attacks
 */
inline uint64_t pawn_attacks_black(uint64_t pawns) {
    return ((pawns >> 9) & 0x7F7F7F7F7F7F7F7FULL) |  // SW, mask off H file
           ((pawns >> 7) & 0xFEFEFEFEFEFEFEFEULL);   // SE, mask off A file
}

///////////////////////////////////////////////////////////////////////////////
// Move Count Kernel (First Pass)
///////////////////////////////////////////////////////////////////////////////

/**
 * Count legal moves for each position in batch
 * This is used to allocate output buffers
 */
kernel void count_moves(
    device const uint64_t* bitboards [[buffer(0)]],  // [batch x 12] - piece bitboards
    device const uint64_t* occupancy [[buffer(1)]],  // [batch x 3] - white, black, all
    device const uint8_t* state [[buffer(2)]],       // [batch] - side to move, castling, ep
    device uint32_t* move_counts [[buffer(3)]],      // [batch] - output move counts
    constant int& batch_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    // Load position data
    const device uint64_t* bb = bitboards + tid * 12;
    const device uint64_t* occ = occupancy + tid * 3;
    uint8_t side_to_move = state[tid * 4];
    
    uint64_t white_occ = occ[0];
    uint64_t black_occ = occ[1];
    uint64_t all_occ = occ[2];
    uint64_t friendly = (side_to_move == 0) ? white_occ : black_occ;
    uint64_t enemy = (side_to_move == 0) ? black_occ : white_occ;
    
    uint32_t count = 0;
    
    // Piece indices: 0-5 = white (P,N,B,R,Q,K), 6-11 = black
    int offset = (side_to_move == 0) ? 0 : 6;
    
    // Pawns
    uint64_t pawns = bb[offset + 0];
    if (side_to_move == 0) {
        // White pawns
        uint64_t single_push = (pawns << 8) & ~all_occ;
        count += popcount(single_push);
        
        uint64_t double_push = ((single_push & 0x0000000000FF0000ULL) << 8) & ~all_occ;
        count += popcount(double_push);
        
        uint64_t attacks = pawn_attacks_white(pawns) & enemy;
        count += popcount(attacks);
    } else {
        // Black pawns
        uint64_t single_push = (pawns >> 8) & ~all_occ;
        count += popcount(single_push);
        
        uint64_t double_push = ((single_push & 0x0000FF0000000000ULL) >> 8) & ~all_occ;
        count += popcount(double_push);
        
        uint64_t attacks = pawn_attacks_black(pawns) & enemy;
        count += popcount(attacks);
    }
    
    // Knights
    uint64_t knights = bb[offset + 1];
    while (knights) {
        int sq = ctz(knights);
        knights &= knights - 1;
        uint64_t attacks = knight_attacks(sq) & ~friendly;
        count += popcount(attacks);
    }
    
    // King
    uint64_t king = bb[offset + 5];
    if (king) {
        int sq = ctz(king);
        uint64_t attacks = king_attacks(sq) & ~friendly;
        count += popcount(attacks);
    }
    
    // TODO: Add sliding piece moves (bishops, rooks, queens) using magic bitboards
    // For now, estimate
    count += popcount(bb[offset + 2]) * 7;  // Bishops ~7 moves each
    count += popcount(bb[offset + 3]) * 8;  // Rooks ~8 moves each
    count += popcount(bb[offset + 4]) * 14; // Queens ~14 moves each
    
    move_counts[tid] = count;
}

///////////////////////////////////////////////////////////////////////////////
// Move Generation Kernel (Second Pass)
///////////////////////////////////////////////////////////////////////////////

/**
 * Pack a move into 16 bits
 * Format: [2 bits type][2 bits promo][6 bits from][6 bits to]
 */
inline uint16_t pack_move(int from, int to, int type = 0, int promo = 0) {
    return (type << 14) | (promo << 12) | (from << 6) | to;
}

/**
 * Generate moves for positions in batch
 */
kernel void generate_moves(
    device const uint64_t* bitboards [[buffer(0)]],
    device const uint64_t* occupancy [[buffer(1)]],
    device const uint8_t* state [[buffer(2)]],
    device const uint32_t* move_offsets [[buffer(3)]],  // Cumulative sum of move_counts
    device uint16_t* moves_out [[buffer(4)]],           // Output move list
    constant int& batch_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    const device uint64_t* bb = bitboards + tid * 12;
    const device uint64_t* occ = occupancy + tid * 3;
    uint8_t side_to_move = state[tid * 4];
    
    uint32_t out_idx = (tid > 0) ? move_offsets[tid - 1] : 0;
    device uint16_t* out = moves_out + out_idx;
    
    uint64_t white_occ = occ[0];
    uint64_t black_occ = occ[1];
    uint64_t all_occ = occ[2];
    uint64_t friendly = (side_to_move == 0) ? white_occ : black_occ;
    uint64_t enemy = (side_to_move == 0) ? black_occ : white_occ;
    
    int offset = (side_to_move == 0) ? 0 : 6;
    uint32_t move_idx = 0;
    
    // Generate pawn moves
    uint64_t pawns = bb[offset + 0];
    
    if (side_to_move == 0) {
        // White pawn pushes
        uint64_t single = (pawns << 8) & ~all_occ;
        while (single) {
            int to = ctz(single);
            single &= single - 1;
            int from = to - 8;
            
            if (to >= 56) {
                // Promotion
                out[move_idx++] = pack_move(from, to, 1, 0); // Knight
                out[move_idx++] = pack_move(from, to, 1, 1); // Bishop
                out[move_idx++] = pack_move(from, to, 1, 2); // Rook
                out[move_idx++] = pack_move(from, to, 1, 3); // Queen
            } else {
                out[move_idx++] = pack_move(from, to);
            }
        }
        
        // Double pushes
        uint64_t double_push = ((((pawns << 8) & ~all_occ) & 0x0000000000FF0000ULL) << 8) & ~all_occ;
        while (double_push) {
            int to = ctz(double_push);
            double_push &= double_push - 1;
            out[move_idx++] = pack_move(to - 16, to);
        }
        
        // Captures
        uint64_t left_cap = ((pawns << 7) & 0x7F7F7F7F7F7F7F7FULL) & enemy;
        while (left_cap) {
            int to = ctz(left_cap);
            left_cap &= left_cap - 1;
            int from = to - 7;
            if (to >= 56) {
                out[move_idx++] = pack_move(from, to, 1, 0);
                out[move_idx++] = pack_move(from, to, 1, 1);
                out[move_idx++] = pack_move(from, to, 1, 2);
                out[move_idx++] = pack_move(from, to, 1, 3);
            } else {
                out[move_idx++] = pack_move(from, to);
            }
        }
        
        uint64_t right_cap = ((pawns << 9) & 0xFEFEFEFEFEFEFEFEULL) & enemy;
        while (right_cap) {
            int to = ctz(right_cap);
            right_cap &= right_cap - 1;
            int from = to - 9;
            if (to >= 56) {
                out[move_idx++] = pack_move(from, to, 1, 0);
                out[move_idx++] = pack_move(from, to, 1, 1);
                out[move_idx++] = pack_move(from, to, 1, 2);
                out[move_idx++] = pack_move(from, to, 1, 3);
            } else {
                out[move_idx++] = pack_move(from, to);
            }
        }
    } else {
        // Black pawns - similar but reversed
        uint64_t single = (pawns >> 8) & ~all_occ;
        while (single) {
            int to = ctz(single);
            single &= single - 1;
            int from = to + 8;
            
            if (to < 8) {
                out[move_idx++] = pack_move(from, to, 1, 0);
                out[move_idx++] = pack_move(from, to, 1, 1);
                out[move_idx++] = pack_move(from, to, 1, 2);
                out[move_idx++] = pack_move(from, to, 1, 3);
            } else {
                out[move_idx++] = pack_move(from, to);
            }
        }
    }
    
    // Generate knight moves
    uint64_t knights = bb[offset + 1];
    while (knights) {
        int from = ctz(knights);
        knights &= knights - 1;
        uint64_t attacks = knight_attacks(from) & ~friendly;
        while (attacks) {
            int to = ctz(attacks);
            attacks &= attacks - 1;
            out[move_idx++] = pack_move(from, to);
        }
    }
    
    // Generate king moves
    uint64_t king = bb[offset + 5];
    if (king) {
        int from = ctz(king);
        uint64_t attacks = king_attacks(from) & ~friendly;
        while (attacks) {
            int to = ctz(attacks);
            attacks &= attacks - 1;
            out[move_idx++] = pack_move(from, to);
        }
    }
    
    // TODO: Generate sliding piece moves with magic bitboards
}

///////////////////////////////////////////////////////////////////////////////
// Parallel Move Legality Check
///////////////////////////////////////////////////////////////////////////////

/**
 * Check legality of moves in parallel
 * Filters out moves that leave the king in check
 */
kernel void check_legality(
    device const uint16_t* moves [[buffer(0)]],
    device const uint32_t* move_counts [[buffer(1)]],
    device const uint64_t* bitboards [[buffer(2)]],
    device const uint64_t* occupancy [[buffer(3)]],
    device const uint8_t* state [[buffer(4)]],
    device uint8_t* legal_mask [[buffer(5)]],  // 1 if legal, 0 if illegal
    constant int& batch_size [[buffer(6)]],
    constant int& max_moves [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])  // (move_idx, position_idx)
{
    uint pos_idx = tid.y;
    uint move_idx = tid.x;
    
    if (pos_idx >= (uint)batch_size) return;
    
    uint32_t num_moves = move_counts[pos_idx];
    if (move_idx >= num_moves) return;
    
    uint32_t move_offset = (pos_idx > 0) ? move_counts[pos_idx - 1] : 0;
    uint16_t move = moves[move_offset + move_idx];
    
    // TODO: Implement full legality check
    // For now, assume all moves are legal
    legal_mask[move_offset + move_idx] = 1;
}

