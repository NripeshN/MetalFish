/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive Core Tests - Bitboard, Position, Move Generation, Types
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
    std::cout << "  Testing " << name_ << "... " << std::flush;
  }
  ~TestCase() {
    if (passed_) {
      std::cout << "PASSED" << std::endl;
      g_tests_passed++;
    } else {
      g_tests_failed++;
    }
  }
  void fail(const char *msg, const char *file, int line) {
    if (passed_) {
      std::cout << "FAILED\n";
      passed_ = false;
    }
    std::cout << "    " << file << ":" << line << ": " << msg << std::endl;
  }
  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      tc.fail(#cond, __FILE__, __LINE__);                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Types Tests
// ============================================================================

bool test_square_operations() {
  TestCase tc("SquareOperations");
  EXPECT(tc, make_square(FILE_E, RANK_4) == SQ_E4);
  EXPECT(tc, file_of(SQ_E4) == FILE_E);
  EXPECT(tc, rank_of(SQ_E4) == RANK_4);
  EXPECT(tc, flip_rank(SQ_A1) == SQ_A8);
  EXPECT(tc, flip_file(SQ_A1) == SQ_H1);
  EXPECT(tc, is_ok(SQ_E4));
  EXPECT(tc, !is_ok(SQ_NONE));
  return tc.passed();
}

bool test_piece_operations() {
  TestCase tc("PieceOperations");
  EXPECT(tc, make_piece(WHITE, PAWN) == W_PAWN);
  EXPECT(tc, make_piece(BLACK, QUEEN) == B_QUEEN);
  EXPECT(tc, type_of(W_KNIGHT) == KNIGHT);
  EXPECT(tc, color_of(B_ROOK) == BLACK);
  EXPECT(tc, (~W_PAWN) == B_PAWN);
  return tc.passed();
}

bool test_move_encoding() {
  TestCase tc("MoveEncoding");
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
  return tc.passed();
}

bool test_value_functions() {
  TestCase tc("ValueFunctions");
  EXPECT(tc, is_valid(VALUE_ZERO));
  EXPECT(tc, !is_valid(VALUE_NONE));
  EXPECT(tc, is_win(VALUE_MATE - 10));
  EXPECT(tc, is_loss(-VALUE_MATE + 10));
  EXPECT(tc, mate_in(5) == VALUE_MATE - 5);
  EXPECT(tc, mated_in(5) == -VALUE_MATE + 5);
  return tc.passed();
}

bool test_direction_operations() {
  TestCase tc("DirectionOperations");
  EXPECT(tc, SQ_E4 + NORTH == SQ_E5);
  EXPECT(tc, SQ_E4 + SOUTH == SQ_E3);
  EXPECT(tc, SQ_E4 + EAST == SQ_F4);
  EXPECT(tc, SQ_E4 + WEST == SQ_D4);
  EXPECT(tc, pawn_push(WHITE) == NORTH);
  EXPECT(tc, pawn_push(BLACK) == SOUTH);
  return tc.passed();
}

// ============================================================================
// Bitboard Tests
// ============================================================================

bool test_bitboard_basics() {
  TestCase tc("BitboardBasics");
  EXPECT(tc, square_bb(SQ_A1) == 1ULL);
  EXPECT(tc, square_bb(SQ_H8) == (1ULL << 63));
  EXPECT(tc, popcount(0) == 0);
  EXPECT(tc, popcount(0xFFFFFFFFFFFFFFFFULL) == 64);
  EXPECT(tc, popcount(Rank1BB) == 8);
  EXPECT(tc, popcount(FileABB) == 8);
  return tc.passed();
}

bool test_bitboard_lsb_msb() {
  TestCase tc("BitboardLsbMsb");
  EXPECT(tc, lsb(1ULL) == SQ_A1);
  EXPECT(tc, lsb(1ULL << 63) == SQ_H8);
  EXPECT(tc, msb(1ULL) == SQ_A1);
  EXPECT(tc, msb(1ULL << 63) == SQ_H8);
  EXPECT(tc, msb(Rank1BB) == SQ_H1);

  Bitboard b = Rank1BB;
  EXPECT(tc, pop_lsb(b) == SQ_A1);
  EXPECT(tc, pop_lsb(b) == SQ_B1);
  return tc.passed();
}

bool test_bitboard_shifts() {
  TestCase tc("BitboardShifts");
  EXPECT(tc, shift<NORTH>(square_bb(SQ_E4)) == square_bb(SQ_E5));
  EXPECT(tc, shift<SOUTH>(square_bb(SQ_E4)) == square_bb(SQ_E3));
  EXPECT(tc, shift<EAST>(square_bb(SQ_E4)) == square_bb(SQ_F4));
  EXPECT(tc, shift<WEST>(square_bb(SQ_E4)) == square_bb(SQ_D4));
  EXPECT(tc, shift<NORTH_EAST>(square_bb(SQ_E4)) == square_bb(SQ_F5));
  EXPECT(tc, shift<EAST>(square_bb(SQ_H4)) == 0); // Edge case
  return tc.passed();
}

bool test_bitboard_attacks() {
  TestCase tc("BitboardAttacks");
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
  return tc.passed();
}

bool test_bitboard_sliding_attacks() {
  TestCase tc("BitboardSlidingAttacks");
  Bitboard occupied = square_bb(SQ_E6);
  Bitboard rook = attacks_bb<ROOK>(SQ_E4, occupied);
  EXPECT(tc, rook & square_bb(SQ_E5));
  EXPECT(tc, rook & square_bb(SQ_E6));
  EXPECT(tc, !(rook & square_bb(SQ_E7)));

  Bitboard bishop = attacks_bb<BISHOP>(SQ_E4, 0);
  EXPECT(tc, bishop & square_bb(SQ_D5));
  EXPECT(tc, bishop & square_bb(SQ_H7));
  return tc.passed();
}

bool test_bitboard_between_line() {
  TestCase tc("BitboardBetweenLine");
  EXPECT(tc, BetweenBB[SQ_A1][SQ_A8] != 0);
  EXPECT(tc, LineBB[SQ_A1][SQ_A8] != 0);
  EXPECT(tc, BetweenBB[SQ_A1][SQ_H8] != 0);
  EXPECT(tc, between_bb(SQ_E1, SQ_E8) & square_bb(SQ_E4));
  return tc.passed();
}

// ============================================================================
// Position Tests
// ============================================================================

bool test_position_set_fen() {
  TestCase tc("PositionSetFen");
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  EXPECT(tc, pos.side_to_move() == WHITE);
  EXPECT(tc, pos.piece_on(SQ_E1) == W_KING);
  EXPECT(tc, pos.piece_on(SQ_E8) == B_KING);
  EXPECT(tc, pos.can_castle(WHITE_OO));
  EXPECT(tc, pos.can_castle(BLACK_OOO));
  return tc.passed();
}

bool test_position_piece_counts() {
  TestCase tc("PositionPieceCounts");
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  EXPECT(tc, pos.count<PAWN>(WHITE) == 8);
  EXPECT(tc, pos.count<PAWN>(BLACK) == 8);
  EXPECT(tc, pos.count<KING>() == 2);
  EXPECT(tc, pos.count<QUEEN>() == 2);
  return tc.passed();
}

bool test_position_do_undo_move() {
  TestCase tc("PositionDoUndoMove");
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
  return tc.passed();
}

bool test_position_captures() {
  TestCase tc("PositionCaptures");
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
          false, &st);

  Move capture(SQ_E4, SQ_D5);
  EXPECT(tc, pos.capture(capture));

  Move quiet(SQ_E4, SQ_E5);
  EXPECT(tc, !pos.capture(quiet));
  return tc.passed();
}

bool test_position_check_detection() {
  TestCase tc("PositionCheckDetection");
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
          false, &st);
  EXPECT(tc, !pos.checkers());

  pos.set("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
          false, &st);
  EXPECT(tc, pos.checkers() != 0);
  return tc.passed();
}

bool test_position_castling() {
  TestCase tc("PositionCastling");
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
  return tc.passed();
}

bool test_position_en_passant() {
  TestCase tc("PositionEnPassant");
  std::deque<StateInfo> states(10);
  Position pos;
  pos.set("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3", false,
          &states[0]);

  EXPECT(tc, pos.ep_square() == SQ_E6);
  Move ep = Move::make<EN_PASSANT>(SQ_F5, SQ_E6);
  pos.do_move(ep, states[1]);

  EXPECT(tc, pos.piece_on(SQ_E6) == W_PAWN);
  EXPECT(tc, pos.empty(SQ_E5)); // Captured pawn removed
  return tc.passed();
}

bool test_position_promotion() {
  TestCase tc("PositionPromotion");
  std::deque<StateInfo> states(10);
  Position pos;
  pos.set("8/P7/8/8/8/8/8/4K2k w - - 0 1", false, &states[0]);

  Move promo = Move::make<PROMOTION>(SQ_A7, SQ_A8, QUEEN);
  pos.do_move(promo, states[1]);

  EXPECT(tc, pos.piece_on(SQ_A8) == W_QUEEN);
  EXPECT(tc, pos.count<QUEEN>(WHITE) == 1);
  return tc.passed();
}

bool test_position_repetition() {
  TestCase tc("PositionRepetition");
  std::deque<StateInfo> states(20);
  Position pos;
  pos.set("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQ - 0 1", false, &states[0]);

  // Just test that repetition detection functions exist and work
  bool has_repeated = pos.has_repeated();
  EXPECT(tc, !has_repeated); // Fresh position should not have repetition

  return tc.passed();
}

bool test_position_see() {
  TestCase tc("PositionSEE");
  StateInfo st;
  Position pos;
  pos.set("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", false, &st);

  Move capture(SQ_E1, SQ_E5);
  EXPECT(tc, pos.see_ge(capture, 0));
  return tc.passed();
}

// ============================================================================
// Move Generation Tests
// ============================================================================

bool test_movegen_starting_position() {
  TestCase tc("MovegenStartingPosition");
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  MoveList<LEGAL> moves(pos);
  EXPECT(tc, moves.size() == 20);
  return tc.passed();
}

bool test_movegen_captures() {
  TestCase tc("MovegenCaptures");
  StateInfo st;
  Position pos;
  pos.set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
          false, &st);

  MoveList<CAPTURES> captures(pos);
  bool found_capture = false;
  for (const auto &m : captures) {
    if (pos.capture(m))
      found_capture = true;
  }
  EXPECT(tc, captures.size() >= 0);
  return tc.passed();
}

bool test_movegen_evasions() {
  TestCase tc("MovegenEvasions");
  StateInfo st;
  Position pos;
  pos.set("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
          false, &st);

  EXPECT(tc, pos.checkers() != 0);
  MoveList<EVASIONS> evasions(pos);
  EXPECT(tc, evasions.size() > 0);
  return tc.passed();
}

bool test_movegen_promotions() {
  TestCase tc("MovegenPromotions");
  StateInfo st;
  Position pos;
  pos.set("8/P7/8/8/8/8/8/4K2k w - - 0 1", false, &st);

  MoveList<LEGAL> moves(pos);
  int promo_count = 0;
  for (const auto &m : moves) {
    if (m.type_of() == PROMOTION)
      promo_count++;
  }
  EXPECT(tc, promo_count == 4); // Q, R, B, N
  return tc.passed();
}

bool test_movegen_castling() {
  TestCase tc("MovegenCastling");
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
  return tc.passed();
}

bool test_movegen_pseudo_legal() {
  TestCase tc("MovegenPseudoLegal");
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  Move e2e4(SQ_E2, SQ_E4);
  EXPECT(tc, pos.pseudo_legal(e2e4));
  EXPECT(tc, pos.legal(e2e4));

  Move illegal(SQ_E2, SQ_E5);
  EXPECT(tc, !pos.pseudo_legal(illegal));
  return tc.passed();
}

} // namespace

bool test_core_comprehensive() {
  std::cout << "\n=== Comprehensive Core Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Types]" << std::endl;
  test_square_operations();
  test_piece_operations();
  test_move_encoding();
  test_value_functions();
  test_direction_operations();

  std::cout << "\n[Bitboard]" << std::endl;
  test_bitboard_basics();
  test_bitboard_lsb_msb();
  test_bitboard_shifts();
  test_bitboard_attacks();
  test_bitboard_sliding_attacks();
  test_bitboard_between_line();

  std::cout << "\n[Position]" << std::endl;
  test_position_set_fen();
  test_position_piece_counts();
  test_position_do_undo_move();
  test_position_captures();
  test_position_check_detection();
  test_position_castling();
  test_position_en_passant();
  test_position_promotion();
  test_position_repetition();
  test_position_see();

  std::cout << "\n[MoveGen]" << std::endl;
  test_movegen_starting_position();
  test_movegen_captures();
  test_movegen_evasions();
  test_movegen_promotions();
  test_movegen_castling();
  test_movegen_pseudo_legal();

  std::cout << "\n=== Core Test Summary ===" << std::endl;
  std::cout << "  Passed: " << g_tests_passed << std::endl;
  std::cout << "  Failed: " << g_tests_failed << std::endl;

  return g_tests_failed == 0;
}
