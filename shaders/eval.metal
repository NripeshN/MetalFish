/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Position evaluation kernels for batched GPU processing.
*/

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Material and Piece Square Table Evaluation
///////////////////////////////////////////////////////////////////////////////

// Piece values (centipawns)
constant int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 0}; // None, Pawn, Knight, Bishop, Rook, Queen, King

/**
 * Batched material count evaluation
 * Each thread processes one position from a batch
 */
kernel void material_eval(
    device const uint8_t* boards [[buffer(0)]],    // [batch_size x 64] - piece per square
    device int32_t* output [[buffer(1)]],           // [batch_size] - material scores
    constant int& batch_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    const device uint8_t* board = boards + tid * 64;
    int32_t white_material = 0;
    int32_t black_material = 0;
    
    for (int sq = 0; sq < 64; sq++) {
        uint8_t piece = board[sq];
        if (piece == 0) continue;
        
        int piece_type = piece & 0x07;  // Lower 3 bits = piece type
        int color = (piece >> 3) & 0x01; // Bit 3 = color (0=white, 1=black)
        int value = PIECE_VALUES[piece_type];
        
        if (color == 0) {
            white_material += value;
        } else {
            black_material += value;
        }
    }
    
    output[tid] = white_material - black_material;
}

/**
 * Piece Square Table evaluation kernel
 * Applies positional bonuses based on piece placement
 */
kernel void psqt_eval(
    device const uint8_t* boards [[buffer(0)]],      // [batch_size x 64]
    device const int16_t* psqt [[buffer(1)]],        // [16 x 64] - PSQT for each piece type
    device int32_t* output [[buffer(2)]],            // [batch_size]
    constant int& batch_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    const device uint8_t* board = boards + tid * 64;
    int32_t score = 0;
    
    for (int sq = 0; sq < 64; sq++) {
        uint8_t piece = board[sq];
        if (piece == 0) continue;
        
        int color = (piece >> 3) & 0x01;
        
        // For black pieces, flip the square vertically
        int psqt_sq = (color == 0) ? sq : (sq ^ 56);
        
        // Get PSQT value
        int16_t psqt_value = psqt[piece * 64 + psqt_sq];
        
        // Add for white, subtract for black
        if (color == 0) {
            score += psqt_value;
        } else {
            score -= psqt_value;
        }
    }
    
    output[tid] = score;
}

///////////////////////////////////////////////////////////////////////////////
// Mobility Evaluation
///////////////////////////////////////////////////////////////////////////////

/**
 * Knight mobility evaluation
 * Count legal knight moves for mobility bonus
 */
kernel void knight_mobility(
    device const uint64_t* occupancy [[buffer(0)]],    // [batch_size x 2] - white/black occupancy
    device const uint64_t* knight_bb [[buffer(1)]],    // [batch_size x 2] - knight bitboards
    device const uint64_t* knight_attacks [[buffer(2)]], // [64] - precomputed knight attack tables
    device int32_t* output [[buffer(3)]],              // [batch_size]
    constant int& batch_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    uint64_t white_occ = occupancy[tid * 2];
    uint64_t black_occ = occupancy[tid * 2 + 1];
    uint64_t white_knights = knight_bb[tid * 2];
    uint64_t black_knights = knight_bb[tid * 2 + 1];
    
    int32_t white_mobility = 0;
    int32_t black_mobility = 0;
    
    // Count white knight moves
    while (white_knights) {
        int sq = ctz(white_knights); // Count trailing zeros
        white_knights &= white_knights - 1; // Clear lowest bit
        uint64_t attacks = knight_attacks[sq] & ~white_occ;
        white_mobility += popcount(attacks);
    }
    
    // Count black knight moves
    while (black_knights) {
        int sq = ctz(black_knights);
        black_knights &= black_knights - 1;
        uint64_t attacks = knight_attacks[sq] & ~black_occ;
        black_mobility += popcount(attacks);
    }
    
    // Mobility bonus: 4 centipawns per square
    output[tid] = (white_mobility - black_mobility) * 4;
}

///////////////////////////////////////////////////////////////////////////////
// King Safety Evaluation
///////////////////////////////////////////////////////////////////////////////

/**
 * King safety evaluation
 * Penalize exposed kings based on pawn shield and attacker proximity
 */
kernel void king_safety(
    device const uint64_t* king_zones [[buffer(0)]],    // [batch_size x 2] - zones around kings
    device const uint64_t* attacker_bb [[buffer(1)]],   // [batch_size x 2] - attacker bitboards
    device const uint64_t* pawn_shield [[buffer(2)]],   // [batch_size x 2] - pawns in front of king
    device int32_t* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    uint64_t white_zone = king_zones[tid * 2];
    uint64_t black_zone = king_zones[tid * 2 + 1];
    uint64_t white_attackers = attacker_bb[tid * 2];
    uint64_t black_attackers = attacker_bb[tid * 2 + 1];
    uint64_t white_shield = pawn_shield[tid * 2];
    uint64_t black_shield = pawn_shield[tid * 2 + 1];
    
    // Count attackers in king zone
    int white_danger = popcount(black_attackers & white_zone);
    int black_danger = popcount(white_attackers & black_zone);
    
    // Pawn shield bonus (3 pawns = full shield)
    int white_shield_count = min(3, popcount(white_shield));
    int black_shield_count = min(3, popcount(black_shield));
    
    // Safety score: penalty for attackers, bonus for shield
    int32_t white_safety = white_shield_count * 20 - white_danger * 40;
    int32_t black_safety = black_shield_count * 20 - black_danger * 40;
    
    output[tid] = white_safety - black_safety;
}

///////////////////////////////////////////////////////////////////////////////
// Combined Static Evaluation
///////////////////////////////////////////////////////////////////////////////

/**
 * Combined evaluation kernel
 * Sums all evaluation components with appropriate weights
 */
kernel void combine_eval(
    device const int32_t* material [[buffer(0)]],
    device const int32_t* psqt [[buffer(1)]],
    device const int32_t* mobility [[buffer(2)]],
    device const int32_t* king_safety [[buffer(3)]],
    device const int32_t* nnue [[buffer(4)]],  // Neural network evaluation
    device int32_t* output [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant bool& use_nnue [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)batch_size) return;
    
    if (use_nnue) {
        // When NNUE is available, use it as primary evaluation
        // But still blend with material for endgame scaling
        int32_t nnue_score = nnue[tid];
        int32_t mat_score = material[tid];
        
        // Scale NNUE by material (more weight in complex positions)
        int32_t abs_mat = abs(mat_score);
        int32_t scale = min(1024, 128 + abs_mat / 4);
        
        output[tid] = (nnue_score * scale) / 1024;
    } else {
        // Classical evaluation: weighted sum
        output[tid] = material[tid] + 
                      psqt[tid] + 
                      mobility[tid] + 
                      king_safety[tid];
    }
}

///////////////////////////////////////////////////////////////////////////////
// Batch Position Processing for Search
///////////////////////////////////////////////////////////////////////////////

/**
 * Process multiple positions in parallel for alpha-beta search
 * This is used for the "batch evaluation" optimization where we
 * collect leaf nodes and evaluate them together on the GPU
 */
kernel void batch_static_eval(
    device const uint8_t* positions [[buffer(0)]],     // Packed position data
    device const int16_t* psqt_table [[buffer(1)]],    // PSQT weights
    device int32_t* scores [[buffer(2)]],              // Output scores
    constant int& batch_size [[buffer(3)]],
    constant int& position_size [[buffer(4)]],          // Bytes per position
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (tid >= (uint)batch_size) return;
    
    const device uint8_t* pos = positions + tid * position_size;
    
    // Extract board from position
    // Position format: [64 bytes board][8 bytes extra state]
    const device uint8_t* board = pos;
    
    int32_t score = 0;
    
    // Material + PSQT in one pass
    for (int sq = 0; sq < 64; sq++) {
        uint8_t piece = board[sq];
        if (piece == 0) continue;
        
        int piece_type = piece & 0x07;
        int color = (piece >> 3) & 0x01;
        
        // Material
        int mat_value = PIECE_VALUES[piece_type];
        
        // PSQT (flip square for black)
        int psqt_sq = (color == 0) ? sq : (sq ^ 56);
        int16_t psqt_value = psqt_table[piece * 64 + psqt_sq];
        
        int piece_score = mat_value + psqt_value;
        
        if (color == 0) {
            score += piece_score;
        } else {
            score -= piece_score;
        }
    }
    
    // TODO: Add tempo, mobility, etc.
    
    scores[tid] = score;
}

