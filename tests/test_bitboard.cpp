/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Bitboard tests
*/

#include "core/bitboard.h"
#include "core/types.h"
#include <cassert>
#include <iostream>

using namespace MetalFish;

bool test_bitboard() {
  // Initialize bitboards
  init_bitboards();

  // Test square_bb
  assert(square_bb(SQ_A1) == 1ULL);
  assert(square_bb(SQ_H8) == (1ULL << 63));
  assert(square_bb(SQ_E4) == (1ULL << 28));

  // Test file and rank bitboards
  assert((FileABB & square_bb(SQ_A1)) != 0);
  assert((FileABB & square_bb(SQ_B1)) == 0);
  assert((Rank1BB & square_bb(SQ_A1)) != 0);
  assert((Rank1BB & square_bb(SQ_A2)) == 0);

  // Test popcount
  assert(popcount(0) == 0);
  assert(popcount(1) == 1);
  assert(popcount(0xFFFFFFFFFFFFFFFFULL) == 64);
  assert(popcount(Rank1BB) == 8);

  // Test lsb and msb
  assert(lsb(1ULL) == SQ_A1);
  assert(lsb(1ULL << 63) == SQ_H8);
  assert(msb(1ULL) == SQ_A1);
  assert(msb(1ULL << 63) == SQ_H8);

  // Test pop_lsb
  Bitboard b = Rank1BB;
  assert(pop_lsb(b) == SQ_A1);
  assert(pop_lsb(b) == SQ_B1);

  // Test shift
  assert(shift<NORTH>(square_bb(SQ_E4)) == square_bb(SQ_E5));
  assert(shift<SOUTH>(square_bb(SQ_E4)) == square_bb(SQ_E3));
  assert(shift<EAST>(square_bb(SQ_E4)) == square_bb(SQ_F4));
  assert(shift<WEST>(square_bb(SQ_E4)) == square_bb(SQ_D4));

  // Test knight attacks
  Bitboard knightOnE4 = KnightAttacks[SQ_E4];
  assert(knightOnE4 & square_bb(SQ_D6));
  assert(knightOnE4 & square_bb(SQ_F6));
  assert(knightOnE4 & square_bb(SQ_G5));
  assert(knightOnE4 & square_bb(SQ_G3));

  // Test king attacks
  Bitboard kingOnE4 = KingAttacks[SQ_E4];
  assert(kingOnE4 & square_bb(SQ_E5));
  assert(kingOnE4 & square_bb(SQ_F5));
  assert(kingOnE4 & square_bb(SQ_D4));

  // Test pawn attacks
  assert(PawnAttacks[WHITE][SQ_E4] & square_bb(SQ_D5));
  assert(PawnAttacks[WHITE][SQ_E4] & square_bb(SQ_F5));
  assert(PawnAttacks[BLACK][SQ_E4] & square_bb(SQ_D3));
  assert(PawnAttacks[BLACK][SQ_E4] & square_bb(SQ_F3));

  // Test rook attacks (with occupancy)
  Bitboard occupied = square_bb(SQ_E6);
  Bitboard rookAttacks = attacks_bb(ROOK, SQ_E4, occupied);
  assert(rookAttacks & square_bb(SQ_E5));
  assert(rookAttacks & square_bb(SQ_E6));    // Can capture
  assert(!(rookAttacks & square_bb(SQ_E7))); // Blocked

  // Test bishop attacks
  Bitboard bishopAttacks = attacks_bb(BISHOP, SQ_E4, 0);
  assert(bishopAttacks & square_bb(SQ_D5));
  assert(bishopAttacks & square_bb(SQ_F5));
  assert(bishopAttacks & square_bb(SQ_H7));
  assert(bishopAttacks & square_bb(SQ_A8));

  // Test queen attacks (combination of rook and bishop)
  Bitboard queenAttacks = attacks_bb(QUEEN, SQ_E4, 0);
  assert((queenAttacks & rookAttacks) != 0); // Some overlap expected

  // Test between and line
  assert(BetweenBB[SQ_A1][SQ_A8] != 0);
  assert(LineBB[SQ_A1][SQ_A8] != 0);
  assert(BetweenBB[SQ_A1][SQ_H8] != 0); // Diagonal

  std::cout << "All bitboard tests passed!" << std::endl;
  return true;
}
