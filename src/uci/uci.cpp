/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  UCI protocol implementation stub.
  Main UCI handling is in main.cpp for now.
*/

#include "core/types.h"
#include <string>

namespace MetalFish {
namespace UCI {

// UCI protocol helpers

std::string square_to_string(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string move_to_string(Move m) {
    if (m == Move::none()) return "(none)";
    if (m == Move::null()) return "0000";
    
    std::string str = square_to_string(m.from_sq()) + square_to_string(m.to_sq());
    
    if (m.type_of() == PROMOTION) {
        str += " nbrq"[m.promotion_type() - KNIGHT];
    }
    
    return str;
}

Square string_to_square(const std::string& s) {
    if (s.length() < 2) return SQ_NONE;
    return make_square(File(s[0] - 'a'), Rank(s[1] - '1'));
}

} // namespace UCI
} // namespace MetalFish

