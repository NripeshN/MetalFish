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
  Bitboards::init();
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

  // Test Kiwipete position (complex with many moves)
  std::cout << "Testing Kiwipete position... ";
  pos.set(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      false, &st);
  MoveList<LEGAL> kiwipeteMoves(pos);
  std::cout << "legal: " << kiwipeteMoves.size() << "... ";

  if (kiwipeteMoves.size() != 48) {
    std::cerr << "Expected 48 legal moves, got " << kiwipeteMoves.size()
              << std::endl;
    return false;
  }
  std::cout << "OK\n";

  // Test position in check
  std::cout << "Testing check position... ";
  pos.set("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3",
          false, &st);
  MoveList<EVASIONS> evasions(pos);
  std::cout << "evasions: " << evasions.size() << "... ";
  if (evasions.size() == 0) {
    std::cerr << "Expected some evasion moves in check" << std::endl;
    return false;
  }
  std::cout << "OK\n";

  // Test promotion position
  std::cout << "Testing promotion position... ";
  pos.set("8/P7/8/8/8/8/8/4K2k w - - 0 1", false, &st);
  MoveList<LEGAL> promoMoves(pos);
  std::cout << "legal: " << promoMoves.size() << "... ";
  // Should have 4 promotion moves (Q, R, B, N) plus king moves
  bool hasPromotion = false;
  for (const auto &m : promoMoves) {
    if (m.type_of() == PROMOTION) {
      hasPromotion = true;
      break;
    }
  }
  if (!hasPromotion) {
    std::cerr << "Expected promotion moves" << std::endl;
    return false;
  }
  std::cout << "OK\n";

  // Test en passant
  std::cout << "Testing en passant... ";
  pos.set("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3", false,
          &st);
  MoveList<LEGAL> epMoves(pos);
  bool hasEP = false;
  for (const auto &m : epMoves) {
    if (m.type_of() == EN_PASSANT) {
      hasEP = true;
      break;
    }
  }
  if (!hasEP) {
    std::cerr << "Expected en passant move" << std::endl;
    return false;
  }
  std::cout << "OK\n";

  // Test castling
  std::cout << "Testing castling... ";
  pos.set("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", false, &st);
  MoveList<LEGAL> castleMoves(pos);
  int castleCount = 0;
  for (const auto &m : castleMoves) {
    if (m.type_of() == CASTLING) {
      castleCount++;
    }
  }
  if (castleCount != 2) {
    std::cerr << "Expected 2 castling moves, got " << castleCount << std::endl;
    return false;
  }
  std::cout << "OK\n";

  std::cout << "All move generation tests passed!" << std::endl;
  return true;
}
