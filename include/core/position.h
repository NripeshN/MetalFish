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

#include "bitboard.h"
#include "types.h"
#include <array>
#include <cassert>
#include <deque>
#include <memory>
#include <string>

namespace MetalFish {

class TranspositionTable;

// StateInfo stores information needed to restore a Position to its previous
// state
struct StateInfo {
  // Copied when making a move
  Key materialKey;
  Key pawnKey;
  Value nonPawnMaterial[COLOR_NB];
  int castlingRights;
  int rule50;
  int pliesFromNull;
  Square epSquare;

  // Not copied when making a move
  Key key;
  Bitboard checkersBB;
  StateInfo *previous;
  Bitboard blockersForKing[COLOR_NB];
  Bitboard pinners[COLOR_NB];
  Bitboard checkSquares[PIECE_TYPE_NB];
  Piece capturedPiece;
  int repetition;
};

using StateListPtr = std::unique_ptr<std::deque<StateInfo>>;

// Position class represents a chess position
class Position {
public:
  static void init();

  Position() = default;
  Position(const Position &) = delete;
  Position &operator=(const Position &) = delete;

  // FEN string input/output
  Position &set(const std::string &fenStr, bool isChess960, StateInfo *si);
  std::string fen() const;

  // Position representation
  Bitboard pieces() const { return byTypeBB[ALL_PIECES]; }

  template <typename... PieceTypes> Bitboard pieces(PieceTypes... pts) const {
    return (byTypeBB[pts] | ...);
  }

  Bitboard pieces(Color c) const { return byColorBB[c]; }

  template <typename... PieceTypes>
  Bitboard pieces(Color c, PieceTypes... pts) const {
    return pieces(c) & pieces(pts...);
  }

  Piece piece_on(Square s) const { return board[s]; }
  Square ep_square() const { return st->epSquare; }
  bool empty(Square s) const { return piece_on(s) == NO_PIECE; }

  template <PieceType Pt> int count(Color c) const {
    return pieceCount[make_piece(c, Pt)];
  }

  template <PieceType Pt> int count() const {
    return count<Pt>(WHITE) + count<Pt>(BLACK);
  }

  template <PieceType Pt> Square square(Color c) const {
    assert(pieces(c, Pt));
    return lsb(pieces(c, Pt));
  }

  // Castling
  int castling_rights() const { return st->castlingRights; }
  bool can_castle(CastlingRights cr) const { return st->castlingRights & cr; }
  bool castling_impeded(CastlingRights cr) const {
    return pieces() & castlingPath[cr];
  }
  Square castling_rook_square(CastlingRights cr) const {
    return castlingRookSquare[cr];
  }

  // Checking
  Bitboard checkers() const { return st->checkersBB; }
  Bitboard blockers_for_king(Color c) const { return st->blockersForKing[c]; }
  Bitboard check_squares(PieceType pt) const { return st->checkSquares[pt]; }
  Bitboard pinners(Color c) const { return st->pinners[c]; }

  // Attacks
  Bitboard attackers_to(Square s) const;
  Bitboard attackers_to(Square s, Bitboard occupied) const;
  void update_slider_blockers(Color c) const;

  template <PieceType Pt> Bitboard attacks_by(Color c) const;

  // Properties of moves
  bool legal(Move m) const;
  bool pseudo_legal(Move m) const;
  bool gives_check(Move m) const;

  bool capture(Move m) const {
    return (!empty(m.to_sq()) && m.type_of() != CASTLING) ||
           m.type_of() == EN_PASSANT;
  }
  Piece moved_piece(Move m) const { return piece_on(m.from_sq()); }
  Piece captured_piece() const { return st->capturedPiece; }

  // Doing and undoing moves
  void do_move(Move m, StateInfo &newSt);
  void do_move(Move m, StateInfo &newSt, bool givesCheck);
  void undo_move(Move m);
  void do_null_move(StateInfo &newSt);
  void undo_null_move();

  // Static Exchange Evaluation
  bool see_ge(Move m, int threshold = 0) const;

  // Hash keys
  Key key() const { return st->key; }
  Key material_key() const { return st->materialKey; }
  Key pawn_key() const { return st->pawnKey; }

  // Other properties
  Color side_to_move() const { return sideToMove; }
  int game_ply() const { return gamePly; }
  bool is_chess960() const { return chess960; }
  bool is_draw(int ply) const;
  bool has_game_cycle(int ply) const;
  int rule50_count() const { return st->rule50; }
  Value non_pawn_material(Color c) const { return st->nonPawnMaterial[c]; }
  Value non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
  }

  StateInfo *state() const { return st; }

  // For debugging
  bool pos_is_ok() const;

  // Piece manipulation
  void put_piece(Piece pc, Square s);
  void remove_piece(Square s);
  void move_piece(Square from, Square to);

private:
  // Castling helper
  void do_castling_helper(bool doCastling, Color us, Square from, Square &to,
                          Square &rfrom, Square &rto);
  void set_castling_right(Color c, Square rfrom);
  void set_state() const;
  void set_check_info() const;

  // Data members
  Piece board[SQUARE_NB];
  Bitboard byTypeBB[PIECE_TYPE_NB];
  Bitboard byColorBB[COLOR_NB];
  int pieceCount[PIECE_NB];
  int castlingRightsMask[SQUARE_NB];
  Square castlingRookSquare[CASTLING_RIGHT_NB];
  Bitboard castlingPath[CASTLING_RIGHT_NB];
  StateInfo *st;
  int gamePly;
  Color sideToMove;
  bool chess960;
};

std::ostream &operator<<(std::ostream &os, const Position &pos);

} // namespace MetalFish
