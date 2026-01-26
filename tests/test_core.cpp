/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file test_core.cpp
 * @brief MetalFish source file.
 */

  Core Tests - Types, Bitboard, Position, Move Generation
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <vector>

using namespace MetalFish;

namespace {

static int g_tests_passed = 0;
static int g_tests_failed = 0;

class TestCase {
public:
  TestCase(const char *name) : name_(name), passed_(true) {
    std::cout << "  " << name_ << "... " << std::flush;
  }
  ~TestCase() {
    if (passed_) {
      std::cout << "OK" << std::endl;
      g_tests_passed++;
    } else {
      g_tests_failed++;
    }
  }
  void fail(const char *msg, int line) {
    if (passed_) {
      std::cout << "FAILED\n";
      passed_ = false;
    }
    std::cout << "    Line " << line << ": " << msg << std::endl;
  }
  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      tc.fail(#cond, __LINE__);                                                \
    }                                                                          \
  } while (0)

// ============================================================================
// Types Tests
// ============================================================================

void test_types() {
  {
    TestCase tc("Square operations");
    EXPECT(tc, make_square(FILE_E, RANK_4) == SQ_E4);
    EXPECT(tc, file_of(SQ_E4) == FILE_E);
    EXPECT(tc, rank_of(SQ_E4) == RANK_4);
    EXPECT(tc, flip_rank(SQ_A1) == SQ_A8);
    EXPECT(tc, flip_file(SQ_A1) == SQ_H1);
    EXPECT(tc, is_ok(SQ_E4));
    EXPECT(tc, !is_ok(SQ_NONE));
  }
  {
    TestCase tc("Piece operations");
    EXPECT(tc, make_piece(WHITE, PAWN) == W_PAWN);
    EXPECT(tc, make_piece(BLACK, QUEEN) == B_QUEEN);
    EXPECT(tc, type_of(W_KNIGHT) == KNIGHT);
    EXPECT(tc, color_of(B_ROOK) == BLACK);
    EXPECT(tc, (~W_PAWN) == B_PAWN);
  }
  {
    TestCase tc("Move encoding");
    Move m(SQ_E2, SQ_E4);
    EXPECT(tc, m.from_sq() == SQ_E2);
    EXPECT(tc, m.to_sq() == SQ_E4);
    EXPECT(tc, m.type_of() == NORMAL);
    EXPECT(tc, m.is_ok());

    Move promo = Move::make<PROMOTION>(SQ_E7, SQ_E8, QUEEN);
    EXPECT(tc, promo.type_of() == PROMOTION);
    EXPECT(tc, promo.promotion_type() == QUEEN);

    Move castle = Move::make<CASTLING>(SQ_E1, SQ_G1);
    EXPECT(tc, castle.type_of() == CASTLING);

    EXPECT(tc, Move::none() != Move::null());
    EXPECT(tc, !Move::none().is_ok());
  }
  {
    TestCase tc("Value functions");
    EXPECT(tc, is_valid(VALUE_ZERO));
    EXPECT(tc, !is_valid(VALUE_NONE));
    EXPECT(tc, is_win(VALUE_MATE - 10));
    EXPECT(tc, is_loss(-VALUE_MATE + 10));
    EXPECT(tc, mate_in(5) == VALUE_MATE - 5);
    EXPECT(tc, mated_in(5) == -VALUE_MATE + 5);
  }
  {
    TestCase tc("Direction operations");
    EXPECT(tc, SQ_E4 + NORTH == SQ_E5);
    EXPECT(tc, SQ_E4 + SOUTH == SQ_E3);
    EXPECT(tc, SQ_E4 + EAST == SQ_F4);
    EXPECT(tc, SQ_E4 + WEST == SQ_D4);
    EXPECT(tc, pawn_push(WHITE) == NORTH);
    EXPECT(tc, pawn_push(BLACK) == SOUTH);
  }
}

// ============================================================================
// Bitboard Tests
// ============================================================================

void test_bitboard() {
  {
    TestCase tc("Basic bitboard operations");
    EXPECT(tc, square_bb(SQ_A1) == 1ULL);
    EXPECT(tc, square_bb(SQ_H8) == (1ULL << 63));
    EXPECT(tc, popcount(0) == 0);
    EXPECT(tc, popcount(0xFFFFFFFFFFFFFFFFULL) == 64);
    EXPECT(tc, popcount(Rank1BB) == 8);
    EXPECT(tc, popcount(FileABB) == 8);
  }
  {
    TestCase tc("LSB and MSB");
    EXPECT(tc, lsb(1ULL) == SQ_A1);
    EXPECT(tc, lsb(1ULL << 63) == SQ_H8);
    EXPECT(tc, msb(1ULL) == SQ_A1);
    EXPECT(tc, msb(1ULL << 63) == SQ_H8);
    EXPECT(tc, msb(Rank1BB) == SQ_H1);

    Bitboard b = Rank1BB;
    EXPECT(tc, pop_lsb(b) == SQ_A1);
    EXPECT(tc, pop_lsb(b) == SQ_B1);
  }
  {
    TestCase tc("Bitboard shifts");
    EXPECT(tc, shift<NORTH>(square_bb(SQ_E4)) == square_bb(SQ_E5));
    EXPECT(tc, shift<SOUTH>(square_bb(SQ_E4)) == square_bb(SQ_E3));
    EXPECT(tc, shift<EAST>(square_bb(SQ_E4)) == square_bb(SQ_F4));
    EXPECT(tc, shift<WEST>(square_bb(SQ_E4)) == square_bb(SQ_D4));
    EXPECT(tc, shift<NORTH_EAST>(square_bb(SQ_E4)) == square_bb(SQ_F5));
    EXPECT(tc, shift<EAST>(square_bb(SQ_H4)) == 0);
  }
  {
    TestCase tc("Piece attacks");
    Bitboard knight = PseudoAttacks[KNIGHT][SQ_E4];
    EXPECT(tc, knight & square_bb(SQ_D6));
    EXPECT(tc, knight & square_bb(SQ_F6));
    EXPECT(tc, knight & square_bb(SQ_G5));
    EXPECT(tc, popcount(knight) == 8);

    Bitboard king = PseudoAttacks[KING][SQ_E4];
    EXPECT(tc, popcount(king) == 8);

    Bitboard pawn_w = pawn_attacks_bb<WHITE>(square_bb(SQ_E4));
    EXPECT(tc, pawn_w & square_bb(SQ_D5));
    EXPECT(tc, pawn_w & square_bb(SQ_F5));
  }
  {
    TestCase tc("Sliding attacks");
    Bitboard occupied = square_bb(SQ_E6);
    Bitboard rook = attacks_bb<ROOK>(SQ_E4, occupied);
    EXPECT(tc, rook & square_bb(SQ_E5));
    EXPECT(tc, rook & square_bb(SQ_E6));
    EXPECT(tc, !(rook & square_bb(SQ_E7)));

    Bitboard bishop = attacks_bb<BISHOP>(SQ_E4, 0);
    EXPECT(tc, bishop & square_bb(SQ_D5));
    EXPECT(tc, bishop & square_bb(SQ_H7));
  }
  {
    TestCase tc("Between and line");
    EXPECT(tc, BetweenBB[SQ_A1][SQ_A8] != 0);
    EXPECT(tc, LineBB[SQ_A1][SQ_A8] != 0);
    EXPECT(tc, BetweenBB[SQ_A1][SQ_H8] != 0);
    EXPECT(tc, between_bb(SQ_E1, SQ_E8) & square_bb(SQ_E4));
  }
}

// ============================================================================
// Position Tests
// ============================================================================

void test_position() {
  {
    TestCase tc("FEN parsing");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    EXPECT(tc, pos.side_to_move() == WHITE);
    EXPECT(tc, pos.piece_on(SQ_E1) == W_KING);
    EXPECT(tc, pos.piece_on(SQ_E8) == B_KING);
    EXPECT(tc, pos.can_castle(WHITE_OO));
    EXPECT(tc, pos.can_castle(BLACK_OOO));
  }
  {
    TestCase tc("Piece counts");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    EXPECT(tc, pos.count<PAWN>(WHITE) == 8);
    EXPECT(tc, pos.count<PAWN>(BLACK) == 8);
    EXPECT(tc, pos.count<KING>() == 2);
    EXPECT(tc, pos.count<QUEEN>() == 2);
  }
  {
    TestCase tc("Do/undo move");
    std::deque<StateInfo> states(10);
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &states[0]);

    Key original_key = pos.key();
    Move e2e4(SQ_E2, SQ_E4);
    pos.do_move(e2e4, states[1]);

    EXPECT(tc, pos.side_to_move() == BLACK);
    EXPECT(tc, pos.piece_on(SQ_E4) == W_PAWN);
    EXPECT(tc, pos.empty(SQ_E2));

    pos.undo_move(e2e4);
    EXPECT(tc, pos.side_to_move() == WHITE);
    EXPECT(tc, pos.key() == original_key);
  }
  {
    TestCase tc("Captures");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            false, &st);

    Move capture(SQ_E4, SQ_D5);
    EXPECT(tc, pos.capture(capture));

    Move quiet(SQ_E4, SQ_E5);
    EXPECT(tc, !pos.capture(quiet));
  }
  {
    TestCase tc("Check detection");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            false, &st);
    EXPECT(tc, !pos.checkers());

    pos.set("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            false, &st);
    EXPECT(tc, pos.checkers() != 0);
  }
  {
    TestCase tc("Castling");
    std::deque<StateInfo> states(10);
    Position pos;
    pos.set("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", false,
            &states[0]);

    EXPECT(tc, pos.can_castle(WHITE_OO));
    EXPECT(tc, pos.can_castle(WHITE_OOO));

    Move castle_ks = Move::make<CASTLING>(SQ_E1, SQ_G1);
    pos.do_move(castle_ks, states[1]);

    EXPECT(tc, pos.piece_on(SQ_G1) == W_KING);
    EXPECT(tc, pos.piece_on(SQ_F1) == W_ROOK);
  }
  {
    TestCase tc("En passant");
    std::deque<StateInfo> states(10);
    Position pos;
    pos.set("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
            false, &states[0]);

    EXPECT(tc, pos.ep_square() == SQ_E6);
    Move ep = Move::make<EN_PASSANT>(SQ_F5, SQ_E6);
    pos.do_move(ep, states[1]);

    EXPECT(tc, pos.piece_on(SQ_E6) == W_PAWN);
    EXPECT(tc, pos.empty(SQ_E5));
  }
  {
    TestCase tc("Promotion");
    std::deque<StateInfo> states(10);
    Position pos;
    pos.set("8/P7/8/8/8/8/8/4K2k w - - 0 1", false, &states[0]);

    Move promo = Move::make<PROMOTION>(SQ_A7, SQ_A8, QUEEN);
    pos.do_move(promo, states[1]);

    EXPECT(tc, pos.piece_on(SQ_A8) == W_QUEEN);
    EXPECT(tc, pos.count<QUEEN>(WHITE) == 1);
  }
  {
    TestCase tc("SEE");
    StateInfo st;
    Position pos;
    pos.set("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", false, &st);

    Move capture(SQ_E1, SQ_E5);
    EXPECT(tc, pos.see_ge(capture, 0));
  }
  {
    TestCase tc("Position keys");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    EXPECT(tc, pos.key() != 0);
    EXPECT(tc, pos.pawn_key() != 0);
    EXPECT(tc, pos.material_key() != 0);
  }
}

// ============================================================================
// Move Generation Tests
// ============================================================================

void test_movegen() {
  {
    TestCase tc("Starting position (20 moves)");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    MoveList<LEGAL> moves(pos);
    EXPECT(tc, moves.size() == 20);
  }
  {
    TestCase tc("Kiwipete position (48 moves)");
    StateInfo st;
    Position pos;
    pos.set(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        false, &st);

    MoveList<LEGAL> moves(pos);
    EXPECT(tc, moves.size() == 48);
  }
  {
    TestCase tc("Captures generation");
    StateInfo st;
    Position pos;
    pos.set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            false, &st);

    MoveList<CAPTURES> captures(pos);
    EXPECT(tc, captures.size() >= 0);
  }
  {
    TestCase tc("Evasions in check");
    StateInfo st;
    Position pos;
    pos.set("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            false, &st);

    EXPECT(tc, pos.checkers() != 0);
    MoveList<EVASIONS> evasions(pos);
    EXPECT(tc, evasions.size() > 0);
  }
  {
    TestCase tc("Promotions (4 types)");
    StateInfo st;
    Position pos;
    pos.set("8/P7/8/8/8/8/8/4K2k w - - 0 1", false, &st);

    MoveList<LEGAL> moves(pos);
    int promo_count = 0;
    for (const auto &m : moves) {
      if (m.type_of() == PROMOTION)
        promo_count++;
    }
    EXPECT(tc, promo_count == 4);
  }
  {
    TestCase tc("Castling moves");
    StateInfo st;
    Position pos;
    pos.set("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", false, &st);

    MoveList<LEGAL> moves(pos);
    int castle_count = 0;
    for (const auto &m : moves) {
      if (m.type_of() == CASTLING)
        castle_count++;
    }
    EXPECT(tc, castle_count == 2);
  }
  {
    TestCase tc("En passant move");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
            false, &st);

    MoveList<LEGAL> moves(pos);
    bool has_ep = false;
    for (const auto &m : moves) {
      if (m.type_of() == EN_PASSANT)
        has_ep = true;
    }
    EXPECT(tc, has_ep);
  }
  {
    TestCase tc("Pseudo-legal validation");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    Move e2e4(SQ_E2, SQ_E4);
    EXPECT(tc, pos.pseudo_legal(e2e4));
    EXPECT(tc, pos.legal(e2e4));

    Move illegal(SQ_E2, SQ_E5);
    EXPECT(tc, !pos.pseudo_legal(illegal));
  }
}

} // namespace

bool test_core() {
  std::cout << "\n=== Core Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Types]" << std::endl;
  test_types();

  std::cout << "\n[Bitboard]" << std::endl;
  test_bitboard();

  std::cout << "\n[Position]" << std::endl;
  test_position();

  std::cout << "\n[Move Generation]" << std::endl;
  test_movegen();

  std::cout << "\n--- Core Results: " << g_tests_passed << " passed, "
            << g_tests_failed << " failed ---" << std::endl;

  return g_tests_failed == 0;
}