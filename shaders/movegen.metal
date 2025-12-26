/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU-accelerated move generation and scoring kernels for Metal.
  
  Key optimizations for Apple Silicon M-series:
  - Use threadgroup memory for intermediate results (32KB limit)
  - Leverage SIMD groups (32 threads) for efficient parallel operations
  - Minimize memory barriers where possible
  - Use unified memory for zero-copy transfers
*/

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Constants
constant int MAX_MOVES = 256;
constant int SIMD_SIZE = 32;

// Piece types
constant int PAWN = 1;
constant int KNIGHT = 2;
constant int BISHOP = 3;
constant int ROOK = 4;
constant int QUEEN = 5;
constant int KING = 6;

// Move encoding
typedef uint32_t Move;
typedef uint64_t Bitboard;
typedef int32_t Value;

// Move scores for sorting
constant int MVV_LVA[7][7] = {
    { 0,  0,  0,  0,  0,  0,  0 },  // No piece
    { 0, 15, 14, 13, 12, 11, 10 },  // Pawn attacks
    { 0, 25, 24, 23, 22, 21, 20 },  // Knight attacks  
    { 0, 35, 34, 33, 32, 31, 30 },  // Bishop attacks
    { 0, 45, 44, 43, 42, 41, 40 },  // Rook attacks
    { 0, 55, 54, 53, 52, 51, 50 },  // Queen attacks
    { 0,  5,  4,  3,  2,  1,  0 }   // King attacks (rarely captures)
};

// Piece values for MVV-LVA
constant int PIECE_VALUES[7] = { 0, 100, 320, 330, 500, 900, 10000 };

// Helper: Count trailing zeros
inline int ctz64(uint64_t x) {
    return x ? __builtin_ctzll(x) : 64;
}

// Helper: Pop least significant bit
inline int pop_lsb(thread uint64_t& x) {
    int idx = ctz64(x);
    x &= x - 1;
    return idx;
}

// Helper: Population count
inline int popcount64(uint64_t x) {
    return popcount(x);
}

/**
 * Move scoring kernel - scores all moves in parallel
 * 
 * This kernel computes move scores for all moves in a batch,
 * enabling parallel sorting for move ordering.
 */
kernel void score_moves_batch(
    device const Move* moves [[buffer(0)]],            // All moves [batch x MAX_MOVES]
    device const int32_t* move_counts [[buffer(1)]],   // Move count per position
    device const int8_t* piece_on [[buffer(2)]],       // Piece on each square [batch x 64]
    device const int16_t* history [[buffer(3)]],       // History heuristic [from x to]
    device const Move* killers [[buffer(4)]],          // Killer moves [batch x 2]
    device int32_t* scores [[buffer(5)]],              // Output scores
    constant int& batch_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    int pos_idx = tid.y;
    int move_idx = tid.x;
    
    if (pos_idx >= batch_size) return;
    
    int num_moves = move_counts[pos_idx];
    if (move_idx >= num_moves) return;
    
    Move m = moves[pos_idx * MAX_MOVES + move_idx];
    
    // Decode move
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    int promo = (m >> 12) & 0x7;
    int flags = (m >> 14) & 0x3;
    
    int score = 0;
    
    // Get piece types
    int piece_base = pos_idx * 64;
    int8_t moving_piece = piece_on[piece_base + from];
    int8_t captured = piece_on[piece_base + to];
    
    int attacker = abs(moving_piece) & 7;  // Piece type 1-6
    int victim = abs(captured) & 7;
    
    // Capture scoring (MVV-LVA)
    if (captured != 0) {
        score = PIECE_VALUES[victim] - attacker + 10000;  // High priority for captures
    }
    // Promotion scoring
    else if (promo > 0) {
        score = PIECE_VALUES[promo] + 5000;
    }
    // Killer moves
    else {
        Move killer1 = killers[pos_idx * 2];
        Move killer2 = killers[pos_idx * 2 + 1];
        
        if (m == killer1) {
            score = 4000;
        } else if (m == killer2) {
            score = 3900;
        } else {
            // History heuristic
            score = history[from * 64 + to];
        }
    }
    
    scores[pos_idx * MAX_MOVES + move_idx] = score;
}

/**
 * Static Exchange Evaluation (SEE) kernel
 * 
 * Evaluates the result of a capture sequence on a square.
 * Runs in parallel for all captures in a batch.
 */
kernel void see_batch(
    device const Move* captures [[buffer(0)]],         // Capture moves to evaluate
    device const Bitboard* occupancy [[buffer(1)]],    // Board occupancy [batch x 2 x 7]
    device const int32_t* piece_squares [[buffer(2)]], // Piece locations
    device Value* results [[buffer(3)]],               // SEE results
    constant int& num_captures [[buffer(4)]],
    constant int& batch_size [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    int pos_idx = tid.y;
    int cap_idx = tid.x;
    
    if (pos_idx >= batch_size || cap_idx >= num_captures) return;
    
    Move m = captures[pos_idx * num_captures + cap_idx];
    
    // Decode move
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    
    // Simplified SEE - actual implementation would be more complex
    // For now, just return MVV-LVA approximation
    int attacker = 0;  // Would extract from position
    int victim = 0;    // Would extract from position
    
    Value result = PIECE_VALUES[victim] - PIECE_VALUES[attacker] / 10;
    results[pos_idx * num_captures + cap_idx] = result;
}

/**
 * Parallel move sorting using bitonic sort
 * 
 * Sorts moves by score for optimal move ordering.
 * Works on powers of 2 for simplicity (pads with low scores).
 */
kernel void bitonic_sort_moves(
    device Move* moves [[buffer(0)]],
    device int32_t* scores [[buffer(1)]],
    constant int& num_moves [[buffer(2)]],
    constant int& stage [[buffer(3)]],
    constant int& step [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    int partner = tid ^ step;
    
    if (partner > tid && (int)tid < num_moves && partner < num_moves) {
        bool ascending = ((tid & stage) == 0);
        
        int32_t my_score = scores[tid];
        int32_t partner_score = scores[partner];
        
        bool should_swap = ascending ? (my_score < partner_score) : (my_score > partner_score);
        
        if (should_swap) {
            // Swap scores
            scores[tid] = partner_score;
            scores[partner] = my_score;
            
            // Swap moves
            Move my_move = moves[tid];
            Move partner_move = moves[partner];
            moves[tid] = partner_move;
            moves[partner] = my_move;
        }
    }
}

/**
 * Knight move generation kernel
 * 
 * Generates all knight moves in parallel for batch of positions.
 */
constant int8_t KNIGHT_OFFSETS[8] = { -17, -15, -10, -6, 6, 10, 15, 17 };

kernel void generate_knight_moves(
    device const Bitboard* knights [[buffer(0)]],      // Knight bitboards [batch]
    device const Bitboard* own_pieces [[buffer(1)]],   // Own piece occupancy
    device Move* moves [[buffer(2)]],                  // Output moves
    device atomic_int* move_counts [[buffer(3)]],      // Atomic move counts
    constant int& batch_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    int pos_idx = tid;
    if (pos_idx >= batch_size) return;
    
    Bitboard knights_bb = knights[pos_idx];
    Bitboard blockers = own_pieces[pos_idx];
    
    int move_count = 0;
    
    while (knights_bb) {
        int from = pop_lsb(knights_bb);
        int from_rank = from / 8;
        int from_file = from % 8;
        
        for (int i = 0; i < 8; i++) {
            int to = from + KNIGHT_OFFSETS[i];
            
            if (to < 0 || to >= 64) continue;
            
            int to_rank = to / 8;
            int to_file = to % 8;
            
            // Check for wrap-around
            int rank_diff = abs(to_rank - from_rank);
            int file_diff = abs(to_file - from_file);
            
            if (!((rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)))
                continue;
            
            // Check if target is not blocked by own piece
            if ((blockers >> to) & 1) continue;
            
            // Encode move
            Move m = from | (to << 6);
            
            int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1, memory_order_relaxed);
            if (idx < MAX_MOVES) {
                moves[pos_idx * MAX_MOVES + idx] = m;
            }
        }
    }
}

/**
 * Batch position evaluation using material count
 * 
 * Fast material evaluation for quiescence search.
 */
kernel void material_eval_batch(
    device const int8_t* pieces [[buffer(0)]],         // Piece array [batch x 64]
    device Value* results [[buffer(1)]],               // Evaluation results
    constant int& batch_size [[buffer(2)]],
    uint pos_idx [[thread_position_in_grid]],
    uint local_idx [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if ((int)pos_idx >= batch_size) return;
    
    // Shared memory for reduction
    threadgroup Value partial_sums[256];
    
    int base = pos_idx * 64;
    Value my_sum = 0;
    
    // Each thread handles some squares
    for (int sq = local_idx; sq < 64; sq += tg_size) {
        int8_t piece = pieces[base + sq];
        if (piece == 0) continue;
        
        int piece_type = abs(piece) & 7;
        int sign = (piece > 0) ? 1 : -1;
        my_sum += sign * PIECE_VALUES[piece_type];
    }
    
    partial_sums[local_idx] = my_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (local_idx == 0) {
        results[pos_idx] = partial_sums[0];
    }
}

/**
 * Attack detection kernel
 * 
 * Checks if any square in a set is attacked by enemy pieces.
 * Used for king safety and check detection.
 */
kernel void is_attacked_batch(
    device const Bitboard* target_squares [[buffer(0)]],  // Squares to check
    device const Bitboard* enemy_pieces [[buffer(1)]],    // Enemy piece bitboards [batch x 6]
    device const Bitboard* occupancy [[buffer(2)]],       // Total occupancy
    device bool* results [[buffer(3)]],                   // Attack results
    constant int& batch_size [[buffer(4)]],
    uint pos_idx [[thread_position_in_grid]])
{
    if ((int)pos_idx >= batch_size) return;
    
    Bitboard targets = target_squares[pos_idx];
    Bitboard occ = occupancy[pos_idx];
    
    // Get enemy pieces (6 piece types)
    device const Bitboard* enemy = enemy_pieces + pos_idx * 6;
    
    bool attacked = false;
    
    // Check each target square
    while (targets && !attacked) {
        int sq = pop_lsb(targets);
        
        // Pawn attacks (would need side-to-move info)
        // Knight attacks
        // Bishop/Queen diagonal attacks
        // Rook/Queen orthogonal attacks
        // King attacks
        
        // Simplified: just check if enemy queen/rook are on same rank/file
        Bitboard rook_queen = enemy[3] | enemy[4];  // Rooks + Queens
        
        int rank = sq / 8;
        int file = sq % 8;
        
        Bitboard rank_mask = 0xFFULL << (rank * 8);
        Bitboard file_mask = 0x0101010101010101ULL << file;
        
        if ((rook_queen & (rank_mask | file_mask)) != 0) {
            attacked = true;
        }
    }
    
    results[pos_idx] = attacked;
}
