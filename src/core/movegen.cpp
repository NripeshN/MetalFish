/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "core/movegen.h"
#include "core/bitboard.h"
#include "core/position.h"

namespace MetalFish {

namespace {

// Generate moves for pawns
template <Color Us, GenType Type>
Move *generate_pawn_moves(const Position &pos, Move *moveList,
                          Bitboard target) {
  constexpr Color Them = ~Us;
  constexpr Bitboard TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
  constexpr Bitboard TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
  constexpr Direction Up = pawn_push(Us);
  constexpr Direction UpRight = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
  constexpr Direction UpLeft = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

  Bitboard emptySquares = ~pos.pieces();
  Bitboard enemies =
      (Type == EVASIONS ? pos.checkers() : pos.pieces(Them)) & target;

  Bitboard pawnsOn7 = pos.pieces(Us, PAWN) & TRank7BB;
  Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

  // Single and double pawn pushes, no promotions
  if constexpr (Type != CAPTURES) {
    Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
    Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

    if constexpr (Type == EVASIONS) {
      b1 &= target;
      b2 &= target;
    }

    while (b1) {
      Square to = pop_lsb(b1);
      *moveList++ = Move(to - Up, to);
    }

    while (b2) {
      Square to = pop_lsb(b2);
      *moveList++ = Move(to - Up - Up, to);
    }
  }

  // Promotions and underpromotions
  if (pawnsOn7) {
    Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
    Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
    Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

    if constexpr (Type == EVASIONS)
      b3 &= target;

    while (b1) {
      Square to = pop_lsb(b1);
      Square from = to - UpRight;
      *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);
      *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
      *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
      *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
    }

    while (b2) {
      Square to = pop_lsb(b2);
      Square from = to - UpLeft;
      *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);
      *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
      *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
      *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
    }

    while (b3) {
      Square to = pop_lsb(b3);
      Square from = to - Up;
      *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);
      *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
      *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
      *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
    }
  }

  // Standard captures
  if constexpr (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS) {
    Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
    Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

    while (b1) {
      Square to = pop_lsb(b1);
      *moveList++ = Move(to - UpRight, to);
    }

    while (b2) {
      Square to = pop_lsb(b2);
      *moveList++ = Move(to - UpLeft, to);
    }

    // En passant captures
    if (pos.ep_square() != SQ_NONE) {
      // For evasions: en passant is allowed only if capturing the pawn resolves
      // the check. The captured pawn is at ep_square - Up (e.g., if ep is c6,
      // pawn is on c5)
      Square capsq = pos.ep_square() - Up;

      if constexpr (Type == EVASIONS) {
        // If the pawn that gave check is the one we can capture via en passant,
        // then en passant is a valid evasion (it removes the attacker)
        if (!(target & square_bb(capsq)))
          goto skip_ep;
      }

      {
        Bitboard b = pawnsNotOn7 & pawn_attacks_bb(Them, pos.ep_square());

        while (b) {
          Square from = pop_lsb(b);
          *moveList++ = Move::make<EN_PASSANT>(from, pos.ep_square());
        }
      }
    skip_ep:;
    }
  }

  return moveList;
}

// Generate moves for pieces (non-pawns)
template <PieceType Pt, bool Checks>
Move *generate_piece_moves(const Position &pos, Move *moveList, Color us,
                           Bitboard target) {
  static_assert(Pt != KING && Pt != PAWN,
                "Use specialized generators for king and pawn");

  Bitboard bb = pos.pieces(us, Pt);

  while (bb) {
    Square from = pop_lsb(bb);
    Bitboard b = attacks_bb(Pt, from, pos.pieces()) & target;

    if constexpr (Checks) {
      b &= pos.check_squares(Pt);
    }

    while (b) {
      *moveList++ = Move(from, pop_lsb(b));
    }
  }

  return moveList;
}

// Generate all moves for a position (internal implementation)
template <Color Us, GenType Type>
Move *generate_all(const Position &pos, Move *moveList) {
  static_assert(Type != LEGAL, "Use generate<LEGAL> instead");

  constexpr bool Checks = Type == QUIETS;

  Bitboard target;
  if constexpr (Type == CAPTURES) {
    target = pos.pieces(~Us);
  } else if constexpr (Type == QUIETS) {
    target = ~pos.pieces();
  } else if constexpr (Type == EVASIONS) {
    Bitboard checkers = pos.checkers();
    if (!checkers)
      return moveList; // No evasions needed if not in check

    // If double check, only king moves
    if (more_than_one(checkers)) {
      Square ksq = pos.square<KING>(Us);
      Bitboard b = attacks_bb(KING, ksq, 0) & ~pos.pieces(Us);
      while (b)
        *moveList++ = Move(ksq, pop_lsb(b));
      return moveList;
    }

    target = BetweenBB[pos.square<KING>(Us)][lsb(checkers)] | checkers;
  } else {
    target = ~pos.pieces(Us);
  }

  moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
  moveList = generate_piece_moves<KNIGHT, Checks>(pos, moveList, Us, target);
  moveList = generate_piece_moves<BISHOP, Checks>(pos, moveList, Us, target);
  moveList = generate_piece_moves<ROOK, Checks>(pos, moveList, Us, target);
  moveList = generate_piece_moves<QUEEN, Checks>(pos, moveList, Us, target);

  if constexpr (Type != EVASIONS) {
    Square ksq = pos.square<KING>(Us);
    Bitboard b = attacks_bb(KING, ksq, 0) & target;

    while (b)
      *moveList++ = Move(ksq, pop_lsb(b));

    // Castling
    if constexpr (Type != CAPTURES) {
      if (pos.can_castle(Us & KING_SIDE) &&
          !pos.castling_impeded(CastlingRights(Us & KING_SIDE))) {
        *moveList++ = Move::make<CASTLING>(
            ksq, pos.castling_rook_square(CastlingRights(Us & KING_SIDE)));
      }
      if (pos.can_castle(Us & QUEEN_SIDE) &&
          !pos.castling_impeded(CastlingRights(Us & QUEEN_SIDE))) {
        *moveList++ = Move::make<CASTLING>(
            ksq, pos.castling_rook_square(CastlingRights(Us & QUEEN_SIDE)));
      }
    }
  } else {
    // When in check, only generate king moves
    Square ksq = pos.square<KING>(Us);
    Bitboard b = attacks_bb(KING, ksq, 0) & ~pos.pieces(Us);

    while (b)
      *moveList++ = Move(ksq, pop_lsb(b));
  }

  return moveList;
}

} // anonymous namespace

// Template specializations for move generation

template <> Move *generate<CAPTURES>(const Position &pos, Move *moveList) {
  return pos.side_to_move() == WHITE
             ? generate_all<WHITE, CAPTURES>(pos, moveList)
             : generate_all<BLACK, CAPTURES>(pos, moveList);
}

template <> Move *generate<QUIETS>(const Position &pos, Move *moveList) {
  return pos.side_to_move() == WHITE
             ? generate_all<WHITE, QUIETS>(pos, moveList)
             : generate_all<BLACK, QUIETS>(pos, moveList);
}

template <> Move *generate<EVASIONS>(const Position &pos, Move *moveList) {
  return pos.side_to_move() == WHITE
             ? generate_all<WHITE, EVASIONS>(pos, moveList)
             : generate_all<BLACK, EVASIONS>(pos, moveList);
}

template <> Move *generate<NON_EVASIONS>(const Position &pos, Move *moveList) {
  return pos.side_to_move() == WHITE
             ? generate_all<WHITE, NON_EVASIONS>(pos, moveList)
             : generate_all<BLACK, NON_EVASIONS>(pos, moveList);
}

template <> Move *generate<LEGAL>(const Position &pos, Move *moveList) {
  Move *cur = moveList;

  Move *end = pos.checkers() ? generate<EVASIONS>(pos, moveList)
                             : generate<NON_EVASIONS>(pos, moveList);

  // Filter out illegal moves in place
  while (cur != end) {
    if (!pos.legal(*cur))
      *cur = *--end; // Replace with last move and shrink the list
    else
      ++cur; // Move is legal, advance to next
  }

  return cur; // Return pointer to end of legal moves
}

} // namespace MetalFish
