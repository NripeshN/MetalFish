/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  
  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "core/position.h"
#include "core/zobrist.h"
#include <iostream>
#include <sstream>
#include <cstring>

namespace MetalFish {

namespace {

// Piece characters for FEN
constexpr char PieceToChar[] = " PNBRQK  pnbrqk";

} // anonymous namespace

// Initialize position-related data structures
void Position::init() {
    Zobrist::init();
}

// Set position from FEN string
Position& Position::set(const std::string& fenStr, bool isChess960, StateInfo* si) {
    /*
       A FEN string defines a particular position using only the ASCII character set.

       A FEN string contains six fields separated by a space. The fields are:

       1) Piece placement (from rank 8 to rank 1)
       2) Active color
       3) Castling availability
       4) En passant target square
       5) Halfmove clock
       6) Fullmove number
    */

    std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));

    st = si;
    chess960 = isChess960;

    std::istringstream ss(fenStr);
    ss >> std::noskipws;

    unsigned char token;
    size_t idx;
    Square sq = SQ_A8;

    // 1. Piece placement
    while ((ss >> token) && !isspace(token)) {
        if (isdigit(token))
            sq += Direction(token - '0');
        else if (token == '/')
            sq += 2 * SOUTH;
        else if ((idx = std::string(PieceToChar).find(token)) != std::string::npos) {
            put_piece(Piece(idx), sq);
            ++sq;
        }
    }

    // 2. Active color
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    // 3. Castling availability
    while ((ss >> token) && !isspace(token)) {
        Square rsq;
        Color c = islower(token) ? BLACK : WHITE;
        Rank rank = (c == WHITE) ? RANK_1 : RANK_8;

        token = char(toupper(token));

        if (token == 'K') {
            for (rsq = make_square(FILE_H, rank); type_of(piece_on(rsq)) != ROOK; --rsq) {}
        } else if (token == 'Q') {
            for (rsq = make_square(FILE_A, rank); type_of(piece_on(rsq)) != ROOK; ++rsq) {}
        } else if (token >= 'A' && token <= 'H') {
            rsq = make_square(File(token - 'A'), rank);
        } else {
            continue;
        }

        set_castling_right(c, rsq);
    }

    // 4. En passant square
    unsigned char col, row;
    if (((ss >> col) && (col >= 'a' && col <= 'h'))
        && ((ss >> row) && (row == (sideToMove == WHITE ? '6' : '3')))) {
        st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));
    } else {
        st->epSquare = SQ_NONE;
    }

    // 5-6. Halfmove clock and fullmove number
    ss >> std::skipws >> st->rule50 >> gamePly;

    // Convert to game ply (0-indexed)
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    set_state();

    return *this;
}

// Output position as FEN string
std::string Position::fen() const {
    std::ostringstream ss;

    for (Rank r = RANK_8; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_H; ++f) {
            int emptyCnt = 0;
            for (; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_1)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    if (can_castle(WHITE_OO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');
    if (can_castle(WHITE_OOO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');
    if (can_castle(BLACK_OO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');
    if (can_castle(BLACK_OOO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

    if (!can_castle(ANY_CASTLING))
        ss << '-';

    ss << (st->epSquare == SQ_NONE ? " -" 
           : " " + std::string{char('a' + file_of(st->epSquare)), char('1' + rank_of(st->epSquare))});

    ss << ' ' << st->rule50 << ' ' << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

// Set up castling rights
void Position::set_castling_right(Color c, Square rfrom) {
    Square kfrom = square<KING>(c);
    CastlingRights cr = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, rfrom > kfrom ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, rfrom > kfrom ? SQ_F1 : SQ_D1);

    castlingPath[cr] = (BetweenBB[rfrom][rto] | BetweenBB[kfrom][kto])
                     & ~(square_bb(kfrom) | square_bb(rfrom));
}

// Compute position state (keys, checkers, etc.)
void Position::set_state() const {
    st->key = 0;
    st->materialKey = 0;
    st->pawnKey = Zobrist::noPawns;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

    set_check_info();

    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        if (Piece pc = piece_on(s); pc != NO_PIECE) {
            st->key ^= Zobrist::psq[pc][s];

            if (type_of(pc) == PAWN)
                st->pawnKey ^= Zobrist::psq[pc][s];
            else if (type_of(pc) != KING)
                st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];

            st->materialKey ^= Zobrist::psq[pc][pieceCount[pc]];
        }
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];
}

// Set check-related info
void Position::set_check_info() const {
    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);

    Square ksq = square<KING>(~sideToMove);

    st->checkSquares[PAWN]   = pawn_attacks_bb(~sideToMove, ksq);
    st->checkSquares[KNIGHT] = attacks_bb(KNIGHT, ksq, 0);
    st->checkSquares[BISHOP] = attacks_bb(BISHOP, ksq, pieces());
    st->checkSquares[ROOK]   = attacks_bb(ROOK, ksq, pieces());
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;
}

// Update blockers for a king
void Position::update_slider_blockers(Color c) const {
    Square ksq = square<KING>(c);

    st->blockersForKing[c] = 0;
    st->pinners[~c] = 0;

    Bitboard snipers = ((attacks_bb(ROOK, ksq, 0) & pieces(QUEEN, ROOK))
                      | (attacks_bb(BISHOP, ksq, 0) & pieces(QUEEN, BISHOP))) & pieces(~c);
    Bitboard occupied = pieces();

    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        Bitboard b = BetweenBB[ksq][sniperSq] & occupied;

        if (b && !more_than_one(b)) {
            st->blockersForKing[c] |= b;
            if (b & pieces(c))
                st->pinners[~c] |= square_bb(sniperSq);
        }
    }
}

// Check if bitboard has more than one bit set
inline bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

// Get attackers to a square
Bitboard Position::attackers_to(Square s) const {
    return attackers_to(s, pieces());
}

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    return (pawn_attacks_bb(BLACK, s) & pieces(WHITE, PAWN))
         | (pawn_attacks_bb(WHITE, s) & pieces(BLACK, PAWN))
         | (attacks_bb(KNIGHT, s, 0) & pieces(KNIGHT))
         | (attacks_bb(ROOK, s, occupied) & pieces(ROOK, QUEEN))
         | (attacks_bb(BISHOP, s, occupied) & pieces(BISHOP, QUEEN))
         | (attacks_bb(KING, s, 0) & pieces(KING));
}

// Check if move is legal
bool Position::legal(Move m) const {
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();

    // En passant captures are tricky
    if (m.type_of() == EN_PASSANT) {
        Square ksq = square<KING>(us);
        Square capsq = to - pawn_push(us);
        Bitboard occupied = (pieces() ^ square_bb(from) ^ square_bb(capsq)) | square_bb(to);

        return !(attacks_bb(ROOK, ksq, occupied) & pieces(~us, QUEEN, ROOK))
            && !(attacks_bb(BISHOP, ksq, occupied) & pieces(~us, QUEEN, BISHOP));
    }

    // Castling moves
    if (m.type_of() == CASTLING) {
        // Check that path is not attacked
        to = relative_square(us, to > from ? SQ_G1 : SQ_C1);
        Direction step = (to > from ? EAST : WEST);

        for (Square s = from; s != to; s += step)
            if (attackers_to(s) & pieces(~us))
                return false;

        return !chess960 || !(attacks_bb(ROOK, to, pieces() ^ square_bb(to)) & pieces(~us, ROOK));
    }

    // If moving the king, check destination is not attacked
    if (type_of(piece_on(from)) == KING)
        return !(attackers_to(to, pieces() ^ square_bb(from)) & pieces(~us));

    // For other moves, make sure we're not leaving king in check
    return !(blockers_for_king(us) & square_bb(from))
        || (LineBB[square<KING>(us)][from] & square_bb(to));
}

// Check if a move gives check
bool Position::gives_check(Move m) const {
    Square from = m.from_sq();
    Square to = m.to_sq();
    PieceType pt = type_of(piece_on(from));

    // Direct check
    if (check_squares(pt) & square_bb(to))
        return true;

    // Discovered check
    if (blockers_for_king(~sideToMove) & square_bb(from))
        if (!LineBB[square<KING>(~sideToMove)][from] || !(LineBB[square<KING>(~sideToMove)][from] & square_bb(to)))
            return true;

    switch (m.type_of()) {
    case NORMAL:
        return false;

    case PROMOTION:
        return attacks_bb(m.promotion_type(), to, pieces() ^ square_bb(from)) & square_bb(square<KING>(~sideToMove));

    case EN_PASSANT: {
        Square capsq = make_square(file_of(to), rank_of(from));
        Bitboard b = (pieces() ^ square_bb(from) ^ square_bb(capsq)) | square_bb(to);
        return (attacks_bb(ROOK, square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, ROOK))
             | (attacks_bb(BISHOP, square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, BISHOP));
    }

    case CASTLING: {
        Square ksq = square<KING>(~sideToMove);
        Square rto = relative_square(sideToMove, to > from ? SQ_F1 : SQ_D1);
        return check_squares(ROOK) & square_bb(rto);
    }

    default:
        return false;
    }
}

// Static exchange evaluation
bool Position::see_ge(Move m, int threshold) const {
    if (m.type_of() != NORMAL)
        return VALUE_ZERO >= threshold;

    Square from = m.from_sq();
    Square to = m.to_sq();

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    Bitboard occupied = pieces() ^ square_bb(from) ^ square_bb(to);
    Color stm = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int res = 1;

    while (true) {
        stm = ~stm;
        attackers &= occupied;
        stmAttackers = attackers & pieces(stm);
        if (!stmAttackers)
            break;

        if (pinners(~stm) & occupied) {
            stmAttackers &= ~blockers_for_king(stm);
            if (!stmAttackers)
                break;
        }

        res ^= 1;

        if ((bb = stmAttackers & pieces(PAWN))) {
            swap = PawnValue - swap;
            if (swap < res) break;
            occupied ^= lsb(bb);
            attackers |= attacks_bb(BISHOP, to, occupied) & pieces(BISHOP, QUEEN);
        }
        else if ((bb = stmAttackers & pieces(KNIGHT))) {
            swap = KnightValue - swap;
            if (swap < res) break;
            occupied ^= lsb(bb);
        }
        else if ((bb = stmAttackers & pieces(BISHOP))) {
            swap = BishopValue - swap;
            if (swap < res) break;
            occupied ^= lsb(bb);
            attackers |= attacks_bb(BISHOP, to, occupied) & pieces(BISHOP, QUEEN);
        }
        else if ((bb = stmAttackers & pieces(ROOK))) {
            swap = RookValue - swap;
            if (swap < res) break;
            occupied ^= lsb(bb);
            attackers |= attacks_bb(ROOK, to, occupied) & pieces(ROOK, QUEEN);
        }
        else if ((bb = stmAttackers & pieces(QUEEN))) {
            swap = QueenValue - swap;
            if (swap < res) break;
            occupied ^= lsb(bb);
            attackers |= (attacks_bb(BISHOP, to, occupied) & pieces(BISHOP, QUEEN))
                       | (attacks_bb(ROOK, to, occupied) & pieces(ROOK, QUEEN));
        }
        else {
            // King
            if (attackers & ~pieces(stm))
                res ^= 1;
            break;
        }
    }

    return bool(res);
}

// Do a move
void Position::do_move(Move m, StateInfo& newSt) {
    do_move(m, newSt, gives_check(m));
}

void Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {
    Key k = st->key ^ Zobrist::side;

    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st = &newSt;

    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;

    Color us = sideToMove;
    Color them = ~us;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = piece_on(from);
    Piece captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);

    if (m.type_of() == CASTLING) {
        Square rfrom, rto;
        do_castling<true>(us, from, to, rfrom, rto);
        k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        captured = NO_PIECE;
    }

    if (captured) {
        Square capsq = to;

        if (m.type_of() == EN_PASSANT) {
            capsq -= pawn_push(us);
        }

        remove_piece(capsq);

        if (type_of(captured) != PAWN)
            st->nonPawnMaterial[them] -= PieceValue[captured];

        k ^= Zobrist::psq[captured][capsq];
        st->materialKey ^= Zobrist::psq[captured][pieceCount[captured]];

        if (type_of(captured) == PAWN)
            st->pawnKey ^= Zobrist::psq[captured][capsq];

        st->rule50 = 0;
    }

    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    if (st->epSquare != SQ_NONE) {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to])) {
        k ^= Zobrist::castling[st->castlingRights];
        st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
        k ^= Zobrist::castling[st->castlingRights];
    }

    if (m.type_of() != CASTLING)
        move_piece(from, to);

    if (type_of(pc) == PAWN) {
        if ((int(to) ^ int(from)) == 16) {
            st->epSquare = to - pawn_push(us);
            k ^= Zobrist::enpassant[file_of(st->epSquare)];
        }
        else if (m.type_of() == PROMOTION) {
            Piece promotion = make_piece(us, m.promotion_type());

            remove_piece(to);
            put_piece(promotion, to);

            k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[promotion][to];
            st->pawnKey ^= Zobrist::psq[pc][to];
            st->materialKey ^= Zobrist::psq[promotion][pieceCount[promotion] - 1]
                             ^ Zobrist::psq[pc][pieceCount[pc]];
            st->nonPawnMaterial[us] += PieceValue[promotion];
        }

        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
        st->rule50 = 0;
    }

    st->capturedPiece = captured;
    st->key = k;
    st->checkersBB = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : 0;
    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;
    int end = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4) {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2) {
            stp = stp->previous->previous;
            if (stp->key == st->key) {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }
}

// Undo a move
void Position::undo_move(Move m) {
    sideToMove = ~sideToMove;

    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = piece_on(to);

    if (m.type_of() == PROMOTION) {
        remove_piece(to);
        pc = make_piece(us, PAWN);
        put_piece(pc, to);
    }

    if (m.type_of() == CASTLING) {
        Square rfrom, rto;
        do_castling<false>(us, from, to, rfrom, rto);
    }
    else {
        move_piece(to, from);

        if (st->capturedPiece) {
            Square capsq = to;

            if (m.type_of() == EN_PASSANT) {
                capsq -= pawn_push(us);
            }

            put_piece(st->capturedPiece, capsq);
        }
    }

    st = st->previous;
    --gamePly;
}

// Null move
void Position::do_null_move(StateInfo& newSt) {
    std::memcpy(&newSt, st, sizeof(StateInfo));
    newSt.previous = st;
    st = &newSt;

    st->key ^= Zobrist::side;

    if (st->epSquare != SQ_NONE) {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    ++st->rule50;
    st->pliesFromNull = 0;

    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;
}

void Position::undo_null_move() {
    st = st->previous;
    sideToMove = ~sideToMove;
}

// Piece manipulation
void Position::put_piece(Piece pc, Square s) {
    board[s] = pc;
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= square_bb(s);
    byColorBB[color_of(pc)] |= square_bb(s);
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]++;
}

void Position::remove_piece(Square s) {
    Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= square_bb(s);
    byTypeBB[type_of(pc)] ^= square_bb(s);
    byColorBB[color_of(pc)] ^= square_bb(s);
    board[s] = NO_PIECE;
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]--;
}

void Position::move_piece(Square from, Square to) {
    Piece pc = board[from];
    Bitboard fromTo = square_bb(from) | square_bb(to);
    byTypeBB[ALL_PIECES] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to] = pc;
}

// Castling helper
template<bool Do>
void Position::do_castling(Color us, Square from, Square& to, Square& rfrom, Square& rto) {
    bool kingSide = to > from;
    rfrom = to;
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    if (Do) {
        remove_piece(from);
        remove_piece(rfrom);
        put_piece(make_piece(us, KING), to);
        put_piece(make_piece(us, ROOK), rto);
    }
    else {
        remove_piece(to);
        remove_piece(rto);
        put_piece(make_piece(us, KING), from);
        put_piece(make_piece(us, ROOK), rfrom);
    }
}

// Repetition detection
bool Position::is_draw(int ply) const {
    if (st->rule50 > 99)
        return true;

    return st->repetition && st->repetition < ply;
}

bool Position::has_game_cycle(int ply) const {
    int end = std::min(st->rule50, st->pliesFromNull);
    if (end < 3)
        return false;

    Key originalKey = st->key;
    StateInfo* stp = st->previous;

    for (int i = 3; i <= end; i += 2) {
        stp = stp->previous->previous;
        if (stp->key == originalKey)
            return true;
    }

    return false;
}

// Position validation
bool Position::pos_is_ok() const {
    // Check piece counts
    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1)
        return false;

    // Check bitboard consistency
    if ((pieces(WHITE) & pieces(BLACK)) != 0)
        return false;

    if ((pieces(WHITE) | pieces(BLACK)) != pieces())
        return false;

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                return false;

    return true;
}

// Pretty print
std::ostream& operator<<(std::ostream& os, const Position& pos) {
    os << "\n +---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_H; ++f) {
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];
        }
        os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
    }

    os << "   a   b   c   d   e   f   g   h\n\n";
    os << "Fen: " << pos.fen() << "\n";
    os << "Key: " << std::hex << pos.key() << std::dec << "\n";

    return os;
}

// Explicit template instantiations
template void Position::do_castling<true>(Color, Square, Square&, Square&, Square&);
template void Position::do_castling<false>(Color, Square, Square&, Square&, Square&);

} // namespace MetalFish
