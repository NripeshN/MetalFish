/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Implementation of the position adapter layer for MCTS.

  Licensed under GPL-3.0
*/

#include "position_adapter.h"
#include "../uci/uci.h"
#include <algorithm>

using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// ============================================================================
// MCTSMove Implementation
// ============================================================================

std::string MCTSMove::to_string() const {
  if (is_null())
    return "(none)";
  return UCIEngine::move(move_, false);
}

// Starting position FEN
constexpr auto StartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ============================================================================
// MCTSPosition Implementation
// ============================================================================

MCTSPosition::MCTSPosition() { pos_.set(StartFEN, false, &st_); }

MCTSPosition::MCTSPosition(const MCTSPosition &other) {
  // Direct position copy via FEN (Position has non-trivial state pointers)
  // This is safer than memcpy since Position has internal pointers to
  // StateInfo.
  pos_.set(other.pos_.fen(), false, &st_);
  move_stack_ = other.move_stack_;
}

MCTSPosition &MCTSPosition::operator=(const MCTSPosition &other) {
  if (this != &other) {
    state_stack_.clear();
    pos_.set(other.pos_.fen(), false, &st_);
    move_stack_ = other.move_stack_;
  }
  return *this;
}

void MCTSPosition::set_from_fen(const std::string &fen) {
  state_stack_.clear();
  move_stack_.clear();
  pos_.set(fen, false, &st_);
}

std::string MCTSPosition::fen() const { return pos_.fen(); }

void MCTSPosition::do_move(MCTSMove move) { do_move(move.to_internal()); }

void MCTSPosition::do_move(Move move) {
  state_stack_.push_back(StateInfo());
  pos_.do_move(move, state_stack_.back());
  move_stack_.push_back(move);
}

MCTSMoveList MCTSPosition::generate_legal_moves() const {
  MCTSMoveList result;
  MoveList<LEGAL> moves(pos_);

  for (const auto &m : moves) {
    result.push_back(MCTSMove::FromInternal(m));
  }

  return result;
}

bool MCTSPosition::is_check() const { return pos_.checkers(); }

bool MCTSPosition::is_checkmate() const {
  if (!is_check())
    return false;
  MoveList<LEGAL> moves(pos_);
  return moves.size() == 0;
}

bool MCTSPosition::is_stalemate() const {
  if (is_check())
    return false;
  MoveList<LEGAL> moves(pos_);
  return moves.size() == 0;
}

bool MCTSPosition::is_draw() const {
  // Check for 50-move rule
  if (pos_.rule50_count() >= 100)
    return true;

  // Check for insufficient material
  if (pos_.is_draw(0))
    return true;

  return false;
}

bool MCTSPosition::is_terminal() const {
  MoveList<LEGAL> moves(pos_);
  if (moves.size() == 0)
    return true; // Checkmate or stalemate
  if (is_draw())
    return true;
  return false;
}

int MCTSPosition::repetition_count() const {
  // Simplified - use built-in repetition detection
  return pos_.is_draw(0) ? 2 : 0;
}

bool MCTSPosition::can_castle_kingside(Color c) const {
  return pos_.can_castle(c == WHITE ? WHITE_OO : BLACK_OO);
}

bool MCTSPosition::can_castle_queenside(Color c) const {
  return pos_.can_castle(c == WHITE ? WHITE_OOO : BLACK_OOO);
}

// ============================================================================
// MCTSEncoder Implementation
// ============================================================================

std::vector<float> MCTSEncoder::encode_position(const MCTSPosition &pos) {
  // Uses 112 input planes of 8x8 = 7168 values
  // We'll use a simplified encoding compatible with our GPU NNUE
  std::vector<float> planes(kTotalPlanes * 64, 0.0f);

  const Position &p = pos.internal_position();
  Color us = p.side_to_move();
  Color them = ~us;

  // Encode piece positions (planes 0-11: our pieces, their pieces)
  auto encode_pieces = [&](int base_plane, Color c) {
    for (PieceType pt = PAWN; pt <= KING; ++pt) {
      Bitboard bb = p.pieces(c, pt);
      int plane_idx = base_plane + static_cast<int>(pt) - 1;

      while (bb) {
        Square sq = pop_lsb(bb);
        // Flip square if black to move (standard convention)
        if (us == BLACK)
          sq = flip_rank(sq);
        planes[plane_idx * 64 + sq] = 1.0f;
      }
    }
  };

  encode_pieces(0, us);   // Our pieces: planes 0-5
  encode_pieces(6, them); // Their pieces: planes 6-11

  // Repetition planes (plane 12)
  if (pos.repetition_count() >= 1) {
    std::fill_n(planes.data() + 12 * 64, 64, 1.0f);
  }

  // Castling rights (planes 104-107, we'll use 13-16)
  if (pos.can_castle_kingside(us))
    std::fill_n(planes.data() + 13 * 64, 64, 1.0f);
  if (pos.can_castle_queenside(us))
    std::fill_n(planes.data() + 14 * 64, 64, 1.0f);
  if (pos.can_castle_kingside(them))
    std::fill_n(planes.data() + 15 * 64, 64, 1.0f);
  if (pos.can_castle_queenside(them))
    std::fill_n(planes.data() + 16 * 64, 64, 1.0f);

  // En passant (plane 17)
  Square ep = pos.en_passant_square();
  if (ep != SQ_NONE) {
    Square encoded_ep = (us == BLACK) ? flip_rank(ep) : ep;
    planes[17 * 64 + encoded_ep] = 1.0f;
  }

  // 50-move rule counter (plane 18) - normalized
  float rule50 = static_cast<float>(pos.rule50_count()) / 100.0f;
  std::fill_n(planes.data() + 18 * 64, 64, rule50);

  return planes;
}

} // namespace MCTS
} // namespace MetalFish
