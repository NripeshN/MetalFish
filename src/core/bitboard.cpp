/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "core/bitboard.h"
#include <algorithm>
#include <cstring>

namespace MetalFish {

// Global attack tables
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];

// Magic bitboard tables
Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

// Attack tables storage
Bitboard RookTable[0x19000];  // ~102K entries
Bitboard BishopTable[0x1480]; // ~5.3K entries

namespace {

// Pre-computed magic numbers for rook and bishop moves
// These are well-known magics that work for all squares
constexpr Bitboard RookMagicNumbers[SQUARE_NB] = {
    0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL,
    0x0080040800100080ULL, 0x0080020400080080ULL, 0x0080010200040080ULL,
    0x0080008001000200ULL, 0x0080002040800100ULL, 0x0000800020400080ULL,
    0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
    0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL,
    0x0000800040800100ULL, 0x0000208000400080ULL, 0x0000404000201000ULL,
    0x0000808010002000ULL, 0x0000808008001000ULL, 0x0000808004000800ULL,
    0x0000808002000400ULL, 0x0000010100020004ULL, 0x0000020000408104ULL,
    0x0000208080004000ULL, 0x0000200040005000ULL, 0x0000100080200080ULL,
    0x0000080080100080ULL, 0x0000040080080080ULL, 0x0000020080040080ULL,
    0x0000010080800200ULL, 0x0000800080004100ULL, 0x0000204000800080ULL,
    0x0000200040401000ULL, 0x0000100080802000ULL, 0x0000080080801000ULL,
    0x0000040080800800ULL, 0x0000020080800400ULL, 0x0000020001010004ULL,
    0x0000800040800100ULL, 0x0000204000808000ULL, 0x0000200040008080ULL,
    0x0000100020008080ULL, 0x0000080010008080ULL, 0x0000040008008080ULL,
    0x0000020004008080ULL, 0x0000010002008080ULL, 0x0000004081020004ULL,
    0x0000204000800080ULL, 0x0000200040008080ULL, 0x0000100020008080ULL,
    0x0000080010008080ULL, 0x0000040008008080ULL, 0x0000020004008080ULL,
    0x0000800100020080ULL, 0x0000800041000080ULL, 0x00FFFCDDFCED714AULL,
    0x007FFCDDFCED714AULL, 0x003FFFCDFFD88096ULL, 0x0000040810002101ULL,
    0x0001000204080011ULL, 0x0001000204000801ULL, 0x0001000082000401ULL,
    0x0001FFFAABFAD1A2ULL};

constexpr Bitboard BishopMagicNumbers[SQUARE_NB] = {
    0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL,
    0x0004040080000000ULL, 0x0001104000000000ULL, 0x0000821040000000ULL,
    0x0000410410400000ULL, 0x0000104104104000ULL, 0x0000040404040400ULL,
    0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
    0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104104000ULL,
    0x0000002082082000ULL, 0x0004000808080800ULL, 0x0002000404040400ULL,
    0x0001000202020200ULL, 0x0000800802004000ULL, 0x0000800400A00000ULL,
    0x0000200100884000ULL, 0x0000400082082000ULL, 0x0000200041041000ULL,
    0x0002080010101000ULL, 0x0001040008080800ULL, 0x0000208004010400ULL,
    0x0000404004010200ULL, 0x0000840000802000ULL, 0x0000404002011000ULL,
    0x0000808001041000ULL, 0x0000404000820800ULL, 0x0001041000202000ULL,
    0x0000820800101000ULL, 0x0000104400080800ULL, 0x0000020080080080ULL,
    0x0000404040040100ULL, 0x0000808100020100ULL, 0x0001010100020800ULL,
    0x0000808080010400ULL, 0x0000820820004000ULL, 0x0000410410002000ULL,
    0x0000082088001000ULL, 0x0000002011000800ULL, 0x0000080100400400ULL,
    0x0001010101000200ULL, 0x0002020202000400ULL, 0x0001010101000200ULL,
    0x0000410410400000ULL, 0x0000208208200000ULL, 0x0000002084100000ULL,
    0x0000000020880000ULL, 0x0000001002020000ULL, 0x0000040408020000ULL,
    0x0004040404040000ULL, 0x0002020202020000ULL, 0x0000104104104000ULL,
    0x0000002082082000ULL, 0x0000000020841000ULL, 0x0000000000208800ULL,
    0x0000000010020200ULL, 0x0000000404080200ULL, 0x0000040404040400ULL,
    0x0002020202020200ULL};

// Shift counts for magic bitboards
constexpr int RookShifts[SQUARE_NB] = {
    52, 53, 53, 53, 53, 53, 53, 52, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 52, 53, 53, 53, 53, 53, 53, 52};

constexpr int BishopShifts[SQUARE_NB] = {
    58, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59, 59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 58};

// Safe step for attack generation
Bitboard safe_destination(Square s, int step) {
  Square to = Square(s + step);
  return is_ok(to) && std::abs(file_of(s) - file_of(to)) <= 2 ? square_bb(to)
                                                              : 0;
}

// Generate sliding attacks
Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
  Bitboard attacks = 0;
  Direction rookDirs[4] = {NORTH, SOUTH, EAST, WEST};
  Direction bishopDirs[4] = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
  Direction *dirs = (pt == ROOK) ? rookDirs : bishopDirs;

  for (int i = 0; i < 4; ++i) {
    Square s = sq;
    while (safe_destination(s, dirs[i])) {
      s += dirs[i];
      attacks |= square_bb(s);
      if (occupied & square_bb(s))
        break;
    }
  }
  return attacks;
}

// Initialize magic bitboards for a piece type
Bitboard *init_magics(PieceType pt, Bitboard table[], Magic magics[]) {
  const Bitboard *magicNumbers =
      (pt == ROOK) ? RookMagicNumbers : BishopMagicNumbers;
  const int *shifts = (pt == ROOK) ? RookShifts : BishopShifts;

  Bitboard occupancy[4096], reference[4096];
  Bitboard *attacks = table;

  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    // Compute the mask for this square
    Bitboard edges = ((Rank1BB | Rank8BB) & ~rank_bb(rank_of(s))) |
                     ((FileABB | FileHBB) & ~file_bb(file_of(s)));

    Magic &m = magics[s];
    m.mask = sliding_attack(pt, s, 0) & ~edges;
    m.magic = magicNumbers[s];
    m.shift = shifts[s];
    m.attacks = attacks;

    // Enumerate all subsets of the mask
    Bitboard b = 0;
    int size = 0;
    do {
      occupancy[size] = b;
      reference[size] = sliding_attack(pt, s, b);
      size++;
      b = (b - m.mask) & m.mask;
    } while (b);

    // Fill the attack table
    for (int i = 0; i < size; ++i) {
      unsigned idx = m.index(occupancy[i]);
      m.attacks[idx] = reference[i];
    }

    attacks += 1ULL << (64 - m.shift);
  }

  return attacks;
}

} // anonymous namespace

void init_bitboards() {
  // Initialize pawn attacks
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    PawnAttacks[WHITE][s] = pawn_attacks_bb<WHITE>(square_bb(s));
    PawnAttacks[BLACK][s] = pawn_attacks_bb<BLACK>(square_bb(s));
  }

  // Initialize knight attacks
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Bitboard b = 0;
    for (int step : {-17, -15, -10, -6, 6, 10, 15, 17})
      b |= safe_destination(s, step);
    KnightAttacks[s] = b;
  }

  // Initialize king attacks
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Bitboard b = 0;
    for (int step : {-9, -8, -7, -1, 1, 7, 8, 9})
      b |= safe_destination(s, step);
    KingAttacks[s] = b;
  }

  // Initialize pseudo attacks (sliding pieces on empty board)
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    PseudoAttacks[KNIGHT][s] = KnightAttacks[s];
    PseudoAttacks[KING][s] = KingAttacks[s];
    PseudoAttacks[BISHOP][s] = sliding_attack(BISHOP, s, 0);
    PseudoAttacks[ROOK][s] = sliding_attack(ROOK, s, 0);
    PseudoAttacks[QUEEN][s] = PseudoAttacks[BISHOP][s] | PseudoAttacks[ROOK][s];
  }

  // Initialize magic bitboards
  init_magics(ROOK, RookTable, RookMagics);
  init_magics(BISHOP, BishopTable, BishopMagics);

  // Initialize between and line bitboards
  for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
    for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
      if (s1 == s2) {
        BetweenBB[s1][s2] = 0;
        LineBB[s1][s2] = 0;
        continue;
      }

      // Check if squares are on same line
      if (PseudoAttacks[BISHOP][s1] & square_bb(s2)) {
        LineBB[s1][s2] =
            (attacks_bb(BISHOP, s1, 0) & attacks_bb(BISHOP, s2, 0)) |
            square_bb(s1) | square_bb(s2);
        BetweenBB[s1][s2] = attacks_bb(BISHOP, s1, square_bb(s2)) &
                            attacks_bb(BISHOP, s2, square_bb(s1));
      } else if (PseudoAttacks[ROOK][s1] & square_bb(s2)) {
        LineBB[s1][s2] = (attacks_bb(ROOK, s1, 0) & attacks_bb(ROOK, s2, 0)) |
                         square_bb(s1) | square_bb(s2);
        BetweenBB[s1][s2] = attacks_bb(ROOK, s1, square_bb(s2)) &
                            attacks_bb(ROOK, s2, square_bb(s1));
      } else {
        LineBB[s1][s2] = 0;
        BetweenBB[s1][s2] = 0;
      }
    }
  }
}

std::string pretty(Bitboard b) {
  std::string s = "+---+---+---+---+---+---+---+---+\n";

  for (Rank r = RANK_8; r >= RANK_1; --r) {
    for (File f = FILE_A; f <= FILE_H; ++f) {
      s += b & square_bb(make_square(f, r)) ? "| X " : "|   ";
    }
    s += "| " + std::to_string(1 + r) + "\n+---+---+---+---+---+---+---+---+\n";
  }
  s += "  a   b   c   d   e   f   g   h\n";

  return s;
}

} // namespace MetalFish
