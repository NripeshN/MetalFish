/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "types.h"
#include <string>

namespace MetalFish {

// Bitboard constants
constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFFULL;
constexpr Bitboard Rank2BB = Rank1BB << 8;
constexpr Bitboard Rank3BB = Rank1BB << 16;
constexpr Bitboard Rank4BB = Rank1BB << 24;
constexpr Bitboard Rank5BB = Rank1BB << 32;
constexpr Bitboard Rank6BB = Rank1BB << 40;
constexpr Bitboard Rank7BB = Rank1BB << 48;
constexpr Bitboard Rank8BB = Rank1BB << 56;

constexpr Bitboard DarkSquares = 0xAA55AA55AA55AA55ULL;
constexpr Bitboard LightSquares = ~DarkSquares;

// Bitboard for a single square
constexpr Bitboard square_bb(Square s) { return 1ULL << s; }

// Rank and file bitboards
constexpr Bitboard rank_bb(Rank r) { return Rank1BB << (8 * r); }
constexpr Bitboard rank_bb(Square s) { return rank_bb(rank_of(s)); }
constexpr Bitboard file_bb(File f) { return FileABB << f; }
constexpr Bitboard file_bb(Square s) { return file_bb(file_of(s)); }
constexpr bool more_than_one(Bitboard b) { return b & (b - 1); }

// Pre-computed attack tables
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];

// Magic bitboard structures for sliding pieces
struct Magic {
  Bitboard mask;
  Bitboard magic;
  Bitboard *attacks;
  unsigned shift;

  // Compute the attack index
  unsigned index(Bitboard occupied) const {
#if defined(USE_PEXT)
    return unsigned(pext(occupied, mask));
#else
    return unsigned(((occupied & mask) * magic) >> shift);
#endif
  }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

// Initialize bitboard tables
void init_bitboards();

// Bitboard utility functions
inline int popcount(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(b);
#else
  // Fallback popcount
  b = b - ((b >> 1) & 0x5555555555555555ULL);
  b = (b & 0x3333333333333333ULL) + ((b >> 2) & 0x3333333333333333ULL);
  b = (b + (b >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return (b * 0x0101010101010101ULL) >> 56;
#endif
}

inline Square lsb(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
  return Square(__builtin_ctzll(b));
#else
  // Fallback
  static const int index64[64] = {
      0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61,
      54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4,  62,
      46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
      25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};
  return Square(index64[((b ^ (b - 1)) * 0x03f79d71b4cb0a89ULL) >> 58]);
#endif
}

inline Square msb(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
  return Square(63 - __builtin_clzll(b));
#else
  // Fallback using de Bruijn
  b |= b >> 1;
  b |= b >> 2;
  b |= b >> 4;
  b |= b >> 8;
  b |= b >> 16;
  b |= b >> 32;
  return lsb(b);
#endif
}

inline Square pop_lsb(Bitboard &b) {
  Square s = lsb(b);
  b &= b - 1;
  return s;
}

// Get attacks for sliding pieces
inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {
  switch (pt) {
  case BISHOP:
    return BishopMagics[s].attacks[BishopMagics[s].index(occupied)];
  case ROOK:
    return RookMagics[s].attacks[RookMagics[s].index(occupied)];
  case QUEEN:
    return attacks_bb(BISHOP, s, occupied) | attacks_bb(ROOK, s, occupied);
  default:
    return PseudoAttacks[pt][s];
  }
}

// Shift a bitboard in a direction
template <Direction D> constexpr Bitboard shift(Bitboard b) {
  return D == NORTH           ? b << 8
         : D == SOUTH         ? b >> 8
         : D == NORTH + NORTH ? b << 16
         : D == SOUTH + SOUTH ? b >> 16
         : D == EAST          ? (b & ~FileHBB) << 1
         : D == WEST          ? (b & ~FileABB) >> 1
         : D == NORTH_EAST    ? (b & ~FileHBB) << 9
         : D == NORTH_WEST    ? (b & ~FileABB) << 7
         : D == SOUTH_EAST    ? (b & ~FileHBB) >> 7
         : D == SOUTH_WEST    ? (b & ~FileABB) >> 9
                              : 0;
}

// Pawn attacks
template <Color C> constexpr Bitboard pawn_attacks_bb(Bitboard b) {
  return C == WHITE ? shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b)
                    : shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
}

inline Bitboard pawn_attacks_bb(Color c, Square s) { return PawnAttacks[c][s]; }

// Pretty print bitboard
std::string pretty(Bitboard b);

} // namespace MetalFish
