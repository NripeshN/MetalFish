/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Move generation tests
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include <cassert>
#include <iostream>

using namespace MetalFish;

bool test_movegen() {
  // Initialize (might already be done, but safe to call again)
  init_bitboards();
  Position::init();

  Position pos;
  StateInfo st;

  std::cout << "Testing starting position... ";

  // Test starting position - should have 20 legal moves
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  // First test NON_EVASIONS since we're not in check
  Move moves[MAX_MOVES];
  Move *end = generate<NON_EVASIONS>(pos, moves);
  size_t nonEvasionCount = end - moves;
  std::cout << "non-evasions: " << nonEvasionCount << "... ";

  // Then test LEGAL
  MoveList<LEGAL> legalMoves(pos);
  std::cout << "legal: " << legalMoves.size() << "... ";

  if (legalMoves.size() != 20) {
    std::cerr << "Expected 20 legal moves, got " << legalMoves.size()
              << std::endl;
    return false;
  }

  std::cout << "OK\n";

  // Test a simple position without check
  std::cout << "Testing simple position... ";
  pos.set("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", false,
          &st);
  MoveList<LEGAL> moves2(pos);
  std::cout << "legal: " << moves2.size() << "... ";

  if (moves2.size() == 0) {
    std::cerr << "Expected some legal moves" << std::endl;
    return false;
  }
  std::cout << "OK\n";

  std::cout << "All move generation tests passed!" << std::endl;
  return true;
}
