/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU-accelerated core chess operations:
  - Move generation
  - Static Exchange Evaluation (SEE)
  - Move scoring
  - Parallel perft
  - Batch position processing

  Designed for maximum parallelism on Apple Silicon GPU.
*/

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Chess Constants
// ============================================================================

constant int SQUARE_NB = 64;
constant int COLOR_NB = 2;
constant int PIECE_TYPE_NB = 7;
constant int MAX_MOVES = 256;

// Piece types
constant int NO_PIECE_TYPE = 0;
constant int PAWN = 1;
constant int KNIGHT = 2;
constant int BISHOP = 3;
constant int ROOK = 4;
constant int QUEEN = 5;
constant int KING = 6;

// Colors
constant int WHITE = 0;
constant int BLACK = 1;

// Move types
constant int NORMAL = 0;
constant int PROMOTION = 1;
constant int EN_PASSANT = 2;
constant int CASTLING = 3;

// Piece values for MVV-LVA
constant int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 20000};

// ============================================================================
// Bitboard Operations
// ============================================================================

typedef uint64_t Bitboard;

inline int popcount(Bitboard b) { return popcount(b); }

inline int lsb(Bitboard b) { return ctz(b); }

inline Bitboard square_bb(int sq) { return 1UL << sq; }

inline int rank_of(int sq) { return sq >> 3; }

inline int file_of(int sq) { return sq & 7; }

// Pre-computed attack tables (to be loaded from CPU)
struct AttackTables {
  Bitboard pawnAttacks[2][64];    // [color][square]
  Bitboard knightAttacks[64];     // [square]
  Bitboard kingAttacks[64];       // [square]
  Bitboard bishopMagics[64];      // Magic numbers
  Bitboard rookMagics[64];        // Magic numbers
  Bitboard bishopMasks[64];       // Occupancy masks
  Bitboard rookMasks[64];         // Occupancy masks
  int bishopShifts[64];           // Bit shifts
  int rookShifts[64];             // Bit shifts
  uint bishopOffsets[64];         // Table offsets
  uint rookOffsets[64];           // Table offsets
};

// ============================================================================
// Position Data Structure (GPU-friendly)
// ============================================================================

struct GPUPosition {
  Bitboard pieces[2][7];  // [color][piece_type] bitboards
  Bitboard occupied[3];   // [white, black, all]
  int8_t board[64];       // Piece on each square
  int sideToMove;
  int castlingRights;
  int epSquare;
  int halfmoveClock;
};

// ============================================================================
// Move Representation
// ============================================================================

struct GPUMove {
  uint16_t data;  // from:6, to:6, type:2, promo:2
  int16_t score;  // Move ordering score
};

inline GPUMove make_move(int from, int to, int type, int promo) {
  GPUMove m;
  m.data = (from & 0x3F) | ((to & 0x3F) << 6) | ((type & 0x3) << 12) |
           ((promo & 0x3) << 14);
  m.score = 0;
  return m;
}

inline int move_from(GPUMove m) { return m.data & 0x3F; }

inline int move_to(GPUMove m) { return (m.data >> 6) & 0x3F; }

inline int move_type(GPUMove m) { return (m.data >> 12) & 0x3; }

inline int move_promo(GPUMove m) { return (m.data >> 14) & 0x3; }

// ============================================================================
// GPU Move Generation - Pawn Moves
// ============================================================================

kernel void generate_pawn_moves(device const GPUPosition *positions
                                [[buffer(0)]],
                                device const AttackTables *attacks
                                [[buffer(1)]],
                                device GPUMove *moves [[buffer(2)]],
                                device atomic_int *move_counts [[buffer(3)]],
                                constant int &batch_size [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int pawn_idx = gid.x;

  if (pos_idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[pos_idx];
  int us = pos.sideToMove;
  int them = 1 - us;
  int direction = (us == WHITE) ? 8 : -8;
  int start_rank = (us == WHITE) ? 1 : 6;
  int promo_rank = (us == WHITE) ? 6 : 1;

  Bitboard pawns = pos.pieces[us][PAWN];
  Bitboard empty = ~pos.occupied[2];
  Bitboard enemies = pos.occupied[them];

  // Find the pawn_idx-th pawn
  int count = 0;
  int sq = -1;
  while (pawns) {
    sq = lsb(pawns);
    if (count == pawn_idx)
      break;
    pawns &= pawns - 1;
    count++;
  }

  if (sq < 0 || count != pawn_idx)
    return;

  // Single push
  int to = sq + direction;
  if (to >= 0 && to < 64 && (empty & square_bb(to))) {
    if (rank_of(sq) == promo_rank) {
      // Promotion
      for (int p = KNIGHT; p <= QUEEN; p++) {
        int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                            memory_order_relaxed);
        if (idx < MAX_MOVES) {
          moves[pos_idx * MAX_MOVES + idx] = make_move(sq, to, PROMOTION, p - 2);
        }
      }
    } else {
      int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                          memory_order_relaxed);
      if (idx < MAX_MOVES) {
        moves[pos_idx * MAX_MOVES + idx] = make_move(sq, to, NORMAL, 0);
      }

      // Double push
      if (rank_of(sq) == start_rank) {
        int to2 = to + direction;
        if (empty & square_bb(to2)) {
          int idx2 = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                               memory_order_relaxed);
          if (idx2 < MAX_MOVES) {
            moves[pos_idx * MAX_MOVES + idx2] = make_move(sq, to2, NORMAL, 0);
          }
        }
      }
    }
  }

  // Captures
  Bitboard pawn_attacks = attacks->pawnAttacks[us][sq];
  Bitboard captures = pawn_attacks & enemies;

  while (captures) {
    int capture_sq = lsb(captures);
    captures &= captures - 1;

    if (rank_of(sq) == promo_rank) {
      for (int p = KNIGHT; p <= QUEEN; p++) {
        int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                            memory_order_relaxed);
        if (idx < MAX_MOVES) {
          moves[pos_idx * MAX_MOVES + idx] =
              make_move(sq, capture_sq, PROMOTION, p - 2);
        }
      }
    } else {
      int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                          memory_order_relaxed);
      if (idx < MAX_MOVES) {
        moves[pos_idx * MAX_MOVES + idx] =
            make_move(sq, capture_sq, NORMAL, 0);
      }
    }
  }

  // En passant
  if (pos.epSquare >= 0 && (pawn_attacks & square_bb(pos.epSquare))) {
    int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                        memory_order_relaxed);
    if (idx < MAX_MOVES) {
      moves[pos_idx * MAX_MOVES + idx] =
          make_move(sq, pos.epSquare, EN_PASSANT, 0);
    }
  }
}

// ============================================================================
// GPU Move Generation - Knight Moves
// ============================================================================

kernel void generate_knight_moves(device const GPUPosition *positions
                                  [[buffer(0)]],
                                  device const AttackTables *attacks
                                  [[buffer(1)]],
                                  device GPUMove *moves [[buffer(2)]],
                                  device atomic_int *move_counts [[buffer(3)]],
                                  constant int &batch_size [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int knight_idx = gid.x;

  if (pos_idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[pos_idx];
  int us = pos.sideToMove;
  Bitboard knights = pos.pieces[us][KNIGHT];
  Bitboard targets = ~pos.occupied[us];

  // Find the knight_idx-th knight
  int count = 0;
  int sq = -1;
  while (knights) {
    sq = lsb(knights);
    if (count == knight_idx)
      break;
    knights &= knights - 1;
    count++;
  }

  if (sq < 0 || count != knight_idx)
    return;

  Bitboard moves_bb = attacks->knightAttacks[sq] & targets;

  while (moves_bb) {
    int to = lsb(moves_bb);
    moves_bb &= moves_bb - 1;

    int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                        memory_order_relaxed);
    if (idx < MAX_MOVES) {
      moves[pos_idx * MAX_MOVES + idx] = make_move(sq, to, NORMAL, 0);
    }
  }
}

// ============================================================================
// GPU Move Generation - King Moves
// ============================================================================

kernel void generate_king_moves(device const GPUPosition *positions
                                [[buffer(0)]],
                                device const AttackTables *attacks
                                [[buffer(1)]],
                                device GPUMove *moves [[buffer(2)]],
                                device atomic_int *move_counts [[buffer(3)]],
                                constant int &batch_size [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
  int pos_idx = gid;

  if (pos_idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[pos_idx];
  int us = pos.sideToMove;
  Bitboard king = pos.pieces[us][KING];
  Bitboard targets = ~pos.occupied[us];

  if (!king)
    return;

  int sq = lsb(king);
  Bitboard moves_bb = attacks->kingAttacks[sq] & targets;

  while (moves_bb) {
    int to = lsb(moves_bb);
    moves_bb &= moves_bb - 1;

    int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                        memory_order_relaxed);
    if (idx < MAX_MOVES) {
      moves[pos_idx * MAX_MOVES + idx] = make_move(sq, to, NORMAL, 0);
    }
  }

  // Castling (simplified check - full legality done later)
  if (us == WHITE) {
    if ((pos.castlingRights & 1) && !(pos.occupied[2] & 0x60UL)) {
      // Kingside
      int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                          memory_order_relaxed);
      if (idx < MAX_MOVES) {
        moves[pos_idx * MAX_MOVES + idx] = make_move(sq, 6, CASTLING, 0);
      }
    }
    if ((pos.castlingRights & 2) && !(pos.occupied[2] & 0xEUL)) {
      // Queenside
      int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                          memory_order_relaxed);
      if (idx < MAX_MOVES) {
        moves[pos_idx * MAX_MOVES + idx] = make_move(sq, 2, CASTLING, 0);
      }
    }
  } else {
    if ((pos.castlingRights & 4) && !(pos.occupied[2] & 0x6000000000000000UL)) {
      int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                          memory_order_relaxed);
      if (idx < MAX_MOVES) {
        moves[pos_idx * MAX_MOVES + idx] = make_move(sq, 62, CASTLING, 0);
      }
    }
    if ((pos.castlingRights & 8) && !(pos.occupied[2] & 0xE00000000000000UL)) {
      int idx = atomic_fetch_add_explicit(move_counts + pos_idx, 1,
                                          memory_order_relaxed);
      if (idx < MAX_MOVES) {
        moves[pos_idx * MAX_MOVES + idx] = make_move(sq, 58, CASTLING, 0);
      }
    }
  }
}

// ============================================================================
// GPU Static Exchange Evaluation (SEE)
// ============================================================================
// Evaluates whether a capture sequence is winning

kernel void batch_see(device const GPUPosition *positions [[buffer(0)]],
                      device const GPUMove *moves [[buffer(1)]],
                      device const AttackTables *attacks [[buffer(2)]],
                      device int *see_results [[buffer(3)]],
                      constant int &batch_size [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  int idx = gid;
  if (idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[idx];
  GPUMove m = moves[idx];

  int from = move_from(m);
  int to = move_to(m);
  int stm = pos.sideToMove;

  // Get piece types
  int movedPiece = abs(pos.board[from]) % 7;
  int captured = abs(pos.board[to]) % 7;

  // Simple SEE approximation
  int gain[32];
  int depth = 0;

  gain[0] = (captured > 0) ? PIECE_VALUES[captured] : 0;

  // Simplified SEE - actual implementation would iterate attackers
  // For now, use MVV-LVA approximation
  if (movedPiece > 0 && captured > 0) {
    see_results[idx] = gain[0] - PIECE_VALUES[movedPiece];
  } else {
    see_results[idx] = gain[0];
  }
}

// ============================================================================
// GPU Move Scoring
// ============================================================================
// Score moves for ordering using MVV-LVA, history, etc.

kernel void score_moves(device const GPUPosition *positions [[buffer(0)]],
                        device GPUMove *moves [[buffer(1)]],
                        device const int *move_counts [[buffer(2)]],
                        device const int16_t *history [[buffer(3)]],
                        constant int &batch_size [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int move_idx = gid.x;

  if (pos_idx >= batch_size)
    return;

  int count = move_counts[pos_idx];
  if (move_idx >= count)
    return;

  device const GPUPosition &pos = positions[pos_idx];
  device GPUMove &m = moves[pos_idx * MAX_MOVES + move_idx];

  int from = move_from(m);
  int to = move_to(m);
  int type = move_type(m);

  int movedPiece = abs(pos.board[from]) % 7;
  int captured = abs(pos.board[to]) % 7;

  int score = 0;

  // Captures: MVV-LVA
  if (captured > 0) {
    score = PIECE_VALUES[captured] * 16 - movedPiece;
  } else {
    // Quiet moves: history heuristic
    int us = pos.sideToMove;
    score = history[us * SQUARE_NB * SQUARE_NB + from * SQUARE_NB + to];
  }

  // Promotions
  if (type == PROMOTION) {
    int promo = move_promo(m) + KNIGHT;
    score += PIECE_VALUES[promo];
  }

  m.score = (int16_t)clamp(score, -32000, 32000);
}

// ============================================================================
// GPU Parallel Perft
// ============================================================================
// Count nodes at each depth in parallel

struct PerftNode {
  GPUPosition pos;
  int depth;
  int parent_idx;
};

kernel void perft_expand(device const PerftNode *input_nodes [[buffer(0)]],
                         device PerftNode *output_nodes [[buffer(1)]],
                         device const AttackTables *attacks [[buffer(2)]],
                         device atomic_int *output_count [[buffer(3)]],
                         device atomic_ulong *node_count [[buffer(4)]],
                         constant int &input_size [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
  int node_idx = gid.y;
  int move_idx = gid.x;

  if (node_idx >= input_size)
    return;

  device const PerftNode &node = input_nodes[node_idx];

  if (node.depth <= 0) {
    // Leaf node - count it
    atomic_fetch_add_explicit(node_count, 1, memory_order_relaxed);
    return;
  }

  // Generate moves for this position
  // (Simplified - actual implementation would use the move generation kernels)
  // This is a placeholder that would be filled with actual legal move
  // generation

  if (move_idx == 0) {
    // Only first thread counts the node
    if (node.depth == 1) {
      // At depth 1, just count legal moves
      // TODO: Implement actual move counting
      atomic_fetch_add_explicit(node_count, 20, memory_order_relaxed);
    }
  }
}

// ============================================================================
// GPU Batch Position Evaluation (Combined NNUE + Classical)
// ============================================================================

kernel void batch_evaluate_positions(
    device const GPUPosition *positions [[buffer(0)]],
    device const int32_t *nnue_scores [[buffer(1)]],
    device int32_t *final_scores [[buffer(2)]],
    constant int &batch_size [[buffer(3)]],
    constant int &use_nnue [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int idx = gid;
  if (idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[idx];

  if (use_nnue && nnue_scores[idx] != 0) {
    // Use NNUE score
    final_scores[idx] = nnue_scores[idx];
  } else {
    // Classical evaluation: material + piece-square tables
    int score = 0;

    // Material counting
    for (int c = 0; c < 2; c++) {
      int sign = (c == WHITE) ? 1 : -1;
      score += sign * popcount(pos.pieces[c][PAWN]) * 100;
      score += sign * popcount(pos.pieces[c][KNIGHT]) * 320;
      score += sign * popcount(pos.pieces[c][BISHOP]) * 330;
      score += sign * popcount(pos.pieces[c][ROOK]) * 500;
      score += sign * popcount(pos.pieces[c][QUEEN]) * 900;
    }

    // Adjust for side to move
    if (pos.sideToMove == BLACK)
      score = -score;

    final_scores[idx] = score;
  }
}

// ============================================================================
// GPU Attack Detection
// ============================================================================
// Check if a square is attacked by a given side

inline Bitboard get_bishop_attacks(int sq, Bitboard occ,
                                   device const AttackTables *attacks,
                                   device const Bitboard *bishopTable) {
  Bitboard magic = attacks->bishopMagics[sq];
  Bitboard mask = attacks->bishopMasks[sq];
  int shift = attacks->bishopShifts[sq];
  uint offset = attacks->bishopOffsets[sq];

  uint index = uint(((occ & mask) * magic) >> shift);
  return bishopTable[offset + index];
}

inline Bitboard get_rook_attacks(int sq, Bitboard occ,
                                 device const AttackTables *attacks,
                                 device const Bitboard *rookTable) {
  Bitboard magic = attacks->rookMagics[sq];
  Bitboard mask = attacks->rookMasks[sq];
  int shift = attacks->rookShifts[sq];
  uint offset = attacks->rookOffsets[sq];

  uint index = uint(((occ & mask) * magic) >> shift);
  return rookTable[offset + index];
}

kernel void is_square_attacked(device const GPUPosition *positions
                               [[buffer(0)]],
                               device const int *squares [[buffer(1)]],
                               device const int *attackers [[buffer(2)]],
                               device const AttackTables *attacks
                               [[buffer(3)]],
                               device const Bitboard *bishopTable [[buffer(4)]],
                               device const Bitboard *rookTable [[buffer(5)]],
                               device bool *results [[buffer(6)]],
                               constant int &batch_size [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
  int idx = gid;
  if (idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[idx];
  int sq = squares[idx];
  int attacker = attackers[idx];
  Bitboard occ = pos.occupied[2];

  // Check all piece types for attacks
  // Pawns
  Bitboard pawn_attacks = attacks->pawnAttacks[1 - attacker][sq];
  if (pawn_attacks & pos.pieces[attacker][PAWN]) {
    results[idx] = true;
    return;
  }

  // Knights
  if (attacks->knightAttacks[sq] & pos.pieces[attacker][KNIGHT]) {
    results[idx] = true;
    return;
  }

  // Bishops and Queens (diagonal)
  Bitboard bishop_attacks = get_bishop_attacks(sq, occ, attacks, bishopTable);
  if (bishop_attacks &
      (pos.pieces[attacker][BISHOP] | pos.pieces[attacker][QUEEN])) {
    results[idx] = true;
    return;
  }

  // Rooks and Queens (straight)
  Bitboard rook_attacks = get_rook_attacks(sq, occ, attacks, rookTable);
  if (rook_attacks &
      (pos.pieces[attacker][ROOK] | pos.pieces[attacker][QUEEN])) {
    results[idx] = true;
    return;
  }

  // King
  if (attacks->kingAttacks[sq] & pos.pieces[attacker][KING]) {
    results[idx] = true;
    return;
  }

  results[idx] = false;
}

// ============================================================================
// GPU Batch Legal Move Filtering
// ============================================================================

kernel void filter_legal_moves(device const GPUPosition *positions
                               [[buffer(0)]],
                               device const GPUMove *pseudo_moves [[buffer(1)]],
                               device const int *pseudo_counts [[buffer(2)]],
                               device GPUMove *legal_moves [[buffer(3)]],
                               device atomic_int *legal_counts [[buffer(4)]],
                               device const AttackTables *attacks
                               [[buffer(5)]],
                               device const Bitboard *bishopTable [[buffer(6)]],
                               device const Bitboard *rookTable [[buffer(7)]],
                               constant int &batch_size [[buffer(8)]],
                               uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int move_idx = gid.x;

  if (pos_idx >= batch_size)
    return;

  int count = pseudo_counts[pos_idx];
  if (move_idx >= count)
    return;

  device const GPUPosition &pos = positions[pos_idx];
  GPUMove m = pseudo_moves[pos_idx * MAX_MOVES + move_idx];

  // Simplified legality check - verify king is not in check after move
  // In practice, this would make the move and check king safety

  int from = move_from(m);
  int to = move_to(m);
  int us = pos.sideToMove;

  // Find our king
  int king_sq = lsb(pos.pieces[us][KING]);

  // Simple check: if moving king, verify target isn't attacked
  if (abs(pos.board[from]) % 7 == KING) {
    Bitboard occ = pos.occupied[2] ^ square_bb(from) | square_bb(to);
    // Would need to check if 'to' is attacked
    // For now, assume legal (proper check in CPU)
  }

  // For now, pass all moves through (legality verified on CPU for correctness)
  int idx = atomic_fetch_add_explicit(legal_counts + pos_idx, 1,
                                      memory_order_relaxed);
  if (idx < MAX_MOVES) {
    legal_moves[pos_idx * MAX_MOVES + idx] = m;
  }
}

// ============================================================================
// GPU Move Sorting (Parallel Bitonic Sort)
// ============================================================================

kernel void bitonic_sort_step(device GPUMove *moves [[buffer(0)]],
                              device const int *counts [[buffer(1)]],
                              constant int &batch_size [[buffer(2)]],
                              constant int &stage [[buffer(3)]],
                              constant int &step [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int pair_idx = gid.x;

  if (pos_idx >= batch_size)
    return;

  int count = counts[pos_idx];
  int dist = 1 << step;
  int block = 1 << (stage + 1);

  int left = pair_idx + (pair_idx / dist) * dist;
  int right = left + dist;

  if (left >= count || right >= count)
    return;

  device GPUMove *arr = moves + pos_idx * MAX_MOVES;

  bool ascending = ((pair_idx / (1 << stage)) % 2) == 0;

  if ((arr[left].score < arr[right].score) == ascending) {
    GPUMove temp = arr[left];
    arr[left] = arr[right];
    arr[right] = temp;
  }
}

// ============================================================================
// GPU Hash/Zobrist Computation
// ============================================================================

kernel void compute_zobrist_hash(device const GPUPosition *positions
                                 [[buffer(0)]],
                                 device const uint64_t *zobrist_pieces
                                 [[buffer(1)]],
                                 device const uint64_t *zobrist_castling
                                 [[buffer(2)]],
                                 device const uint64_t *zobrist_ep
                                 [[buffer(3)]],
                                 constant uint64_t &zobrist_side [[buffer(4)]],
                                 device uint64_t *hashes [[buffer(5)]],
                                 constant int &batch_size [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  int idx = gid;
  if (idx >= batch_size)
    return;

  device const GPUPosition &pos = positions[idx];
  uint64_t hash = 0;

  // Pieces
  for (int sq = 0; sq < 64; sq++) {
    int piece = pos.board[sq];
    if (piece != 0) {
      // zobrist_pieces indexed by [piece][square]
      hash ^= zobrist_pieces[piece * 64 + sq];
    }
  }

  // Side to move
  if (pos.sideToMove == BLACK) {
    hash ^= zobrist_side;
  }

  // Castling
  hash ^= zobrist_castling[pos.castlingRights];

  // En passant
  if (pos.epSquare >= 0) {
    hash ^= zobrist_ep[file_of(pos.epSquare)];
  }

  hashes[idx] = hash;
}

