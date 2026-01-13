/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Implementation of the Stockfish-to-MCTS adapter layer.

  Licensed under GPL-3.0
*/

#include "stockfish_adapter.h"
#include "../uci/uci.h"
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// ============================================================================
// MCTSMove Implementation
// ============================================================================

std::string MCTSMove::to_string() const {
  if (is_null()) return "(none)";
  return UCIEngine::move(move_, false);
}

// Starting position FEN
constexpr auto StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ============================================================================
// MCTSPosition Implementation
// ============================================================================

MCTSPosition::MCTSPosition() {
  pos_.set(StartFEN, false, &st_);
}

MCTSPosition::MCTSPosition(const MCTSPosition& other) {
  // Use FEN to copy - simpler and safer
  std::string current_fen = other.pos_.fen();
  pos_.set(current_fen, false, &st_);
  
  // Copy move stack for undo support
  move_stack_ = other.move_stack_;
}

MCTSPosition& MCTSPosition::operator=(const MCTSPosition& other) {
  if (this != &other) {
    // Use FEN to copy - simpler and safer
    std::string current_fen = other.pos_.fen();
    state_stack_.clear();
    pos_.set(current_fen, false, &st_);
    
    // Copy move stack for undo support
    move_stack_ = other.move_stack_;
  }
  return *this;
}

void MCTSPosition::set_from_fen(const std::string& fen) {
  state_stack_.clear();
  move_stack_.clear();
  pos_.set(fen, false, &st_);
}

std::string MCTSPosition::fen() const {
  return pos_.fen();
}

void MCTSPosition::do_move(MCTSMove move) {
  do_move(move.to_stockfish());
}

void MCTSPosition::do_move(Move move) {
  state_stack_.push_back(StateInfo());
  pos_.do_move(move, state_stack_.back());
  move_stack_.push_back(move);
}

void MCTSPosition::undo_move() {
  if (!move_stack_.empty()) {
    pos_.undo_move(move_stack_.back());
    move_stack_.pop_back();
    state_stack_.pop_back();
  }
}

MCTSMoveList MCTSPosition::generate_legal_moves() const {
  MCTSMoveList result;
  MoveList<LEGAL> moves(pos_);
  
  for (const auto& m : moves) {
    result.push_back(MCTSMove::FromStockfish(m));
  }
  
  return result;
}

bool MCTSPosition::is_check() const {
  return pos_.checkers();
}

bool MCTSPosition::is_checkmate() const {
  if (!is_check()) return false;
  MoveList<LEGAL> moves(pos_);
  return moves.size() == 0;
}

bool MCTSPosition::is_stalemate() const {
  if (is_check()) return false;
  MoveList<LEGAL> moves(pos_);
  return moves.size() == 0;
}

bool MCTSPosition::is_draw() const {
  // Check for 50-move rule
  if (pos_.rule50_count() >= 100) return true;
  
  // Check for insufficient material
  if (pos_.is_draw(0)) return true;
  
  return false;
}

bool MCTSPosition::is_terminal() const {
  MoveList<LEGAL> moves(pos_);
  if (moves.size() == 0) return true;  // Checkmate or stalemate
  if (is_draw()) return true;
  return false;
}

GameResult MCTSPosition::get_game_result() const {
  if (!is_terminal()) return GameResult::UNDECIDED;
  
  if (is_checkmate()) {
    return is_black_to_move() ? GameResult::WHITE_WON : GameResult::BLACK_WON;
  }
  
  return GameResult::DRAW;
}

int MCTSPosition::repetition_count() const {
  // Simplified - use Stockfish's built-in repetition detection
  return pos_.is_draw(0) ? 2 : 0;
}

bool MCTSPosition::can_castle_kingside(Color c) const {
  return pos_.can_castle(c == WHITE ? WHITE_OO : BLACK_OO);
}

bool MCTSPosition::can_castle_queenside(Color c) const {
  return pos_.can_castle(c == WHITE ? WHITE_OOO : BLACK_OOO);
}

// ============================================================================
// MCTSPositionHistory Implementation
// ============================================================================

MCTSPositionHistory::MCTSPositionHistory() {
  reset();
}

void MCTSPositionHistory::reset() {
  positions_.clear();
  moves_.clear();
  positions_.emplace_back();
}

void MCTSPositionHistory::reset(const std::string& fen) {
  positions_.clear();
  moves_.clear();
  positions_.emplace_back();
  positions_.back().set_from_fen(fen);
}

void MCTSPositionHistory::reset(const std::string& fen, const std::vector<std::string>& moves) {
  reset(fen);
  
  for (const auto& move_str : moves) {
    // Parse move from UCI string
    Move m = UCIEngine::to_move(positions_.back().stockfish_position(), move_str);
    if (m != Move::none()) {
      do_move(m);
    }
  }
}

void MCTSPositionHistory::do_move(MCTSMove move) {
  do_move(move.to_stockfish());
}

void MCTSPositionHistory::do_move(Move move) {
  positions_.push_back(positions_.back());
  positions_.back().do_move(move);
  moves_.push_back(MCTSMove::FromStockfish(move));
}

int MCTSPositionHistory::compute_repetitions() const {
  if (positions_.empty()) return 0;
  
  uint64_t current_hash = current().hash();
  int count = 0;
  
  for (size_t i = 0; i + 1 < positions_.size(); ++i) {
    if (positions_[i].hash() == current_hash) {
      ++count;
    }
  }
  
  return count;
}

MCTSMove MCTSPositionHistory::last_move() const {
  if (moves_.empty()) return MCTSMove();
  return moves_.back();
}

// ============================================================================
// MCTSEncoder Implementation
// ============================================================================

std::vector<float> MCTSEncoder::encode_position(const MCTSPosition& pos) {
  // Lc0 uses 112 input planes of 8x8 = 7168 values
  // We'll use a simplified encoding compatible with our GPU NNUE
  std::vector<float> planes(kTotalPlanes * 64, 0.0f);
  
  const Position& p = pos.stockfish_position();
  Color us = p.side_to_move();
  Color them = ~us;
  
  // Encode piece positions (planes 0-11: our pieces, their pieces)
  auto encode_pieces = [&](int base_plane, Color c) {
    for (PieceType pt = PAWN; pt <= KING; ++pt) {
      Bitboard bb = p.pieces(c, pt);
      int plane_idx = base_plane + static_cast<int>(pt) - 1;
      
      while (bb) {
        Square sq = pop_lsb(bb);
        // Flip square if black to move (Lc0 convention)
        if (us == BLACK) sq = flip_rank(sq);
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
  
  // Castling rights (planes 104-107 in Lc0, we'll use 13-16)
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

std::vector<float> MCTSEncoder::encode_history(const MCTSPositionHistory& history, int history_length) {
  // For now, just encode the current position
  // A full implementation would encode the last N positions
  return encode_position(history.current());
}

std::vector<std::pair<MCTSMove, float>> MCTSEncoder::decode_policy(
    const MCTSPosition& pos,
    const float* policy_output,
    int policy_size) {
  
  std::vector<std::pair<MCTSMove, float>> result;
  MCTSMoveList legal_moves = pos.generate_legal_moves();
  
  // Normalize policy over legal moves
  float total = 0.0f;
  std::vector<float> probs;
  probs.reserve(legal_moves.size());
  
  for (const auto& move : legal_moves) {
    int idx = move_to_policy_index(pos, move);
    float prob = (idx >= 0 && idx < policy_size) ? policy_output[idx] : 0.0f;
    prob = std::max(prob, 0.0f);  // Ensure non-negative
    probs.push_back(prob);
    total += prob;
  }
  
  // Normalize
  if (total > 0.0f) {
    for (size_t i = 0; i < legal_moves.size(); ++i) {
      result.emplace_back(legal_moves[i], probs[i] / total);
    }
  } else {
    // Uniform distribution if no valid policy
    float uniform = 1.0f / legal_moves.size();
    for (const auto& move : legal_moves) {
      result.emplace_back(move, uniform);
    }
  }
  
  // Sort by probability descending
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  
  return result;
}

int MCTSEncoder::move_to_policy_index(const MCTSPosition& pos, MCTSMove move) {
  // Lc0 uses a 1858-element policy vector
  // Encoding: from_square * 73 + move_type
  // Move types: 56 queen moves + 8 knight moves + 9 underpromotions
  
  Move m = move.to_stockfish();
  Square from = m.from_sq();
  Square to = m.to_sq();
  
  // Flip if black to move
  if (pos.is_black_to_move()) {
    from = flip_rank(from);
    to = flip_rank(to);
  }
  
  int from_idx = static_cast<int>(from);
  
  // Calculate direction and distance
  int df = file_of(to) - file_of(from);
  int dr = rank_of(to) - rank_of(from);
  
  int move_type = -1;
  
  // Knight moves (8 directions)
  if ((std::abs(df) == 2 && std::abs(dr) == 1) || (std::abs(df) == 1 && std::abs(dr) == 2)) {
    static const int knight_dirs[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
    for (int i = 0; i < 8; ++i) {
      if (df == knight_dirs[i][0] && dr == knight_dirs[i][1]) {
        move_type = 56 + i;
        break;
      }
    }
  }
  // Queen-like moves (56 directions: 7 distances * 8 directions)
  else if (df == 0 || dr == 0 || std::abs(df) == std::abs(dr)) {
    int distance = std::max(std::abs(df), std::abs(dr));
    int dir = -1;
    
    // 8 directions: N, NE, E, SE, S, SW, W, NW
    if (df == 0 && dr > 0) dir = 0;       // N
    else if (df > 0 && dr > 0 && df == dr) dir = 1;  // NE
    else if (df > 0 && dr == 0) dir = 2;  // E
    else if (df > 0 && dr < 0 && df == -dr) dir = 3; // SE
    else if (df == 0 && dr < 0) dir = 4;  // S
    else if (df < 0 && dr < 0 && df == dr) dir = 5;  // SW
    else if (df < 0 && dr == 0) dir = 6;  // W
    else if (df < 0 && dr > 0 && -df == dr) dir = 7; // NW
    
    if (dir >= 0 && distance >= 1 && distance <= 7) {
      move_type = dir * 7 + (distance - 1);
    }
  }
  
  // Handle underpromotions (knight, bishop, rook)
  if (move.is_promotion()) {
    PieceType promo = m.promotion_type();
    if (promo != QUEEN) {
      // Underpromotion: 64-72 (3 types * 3 directions)
      int promo_idx = (promo == KNIGHT) ? 0 : (promo == BISHOP) ? 1 : 2;
      int dir_idx = (df == 0) ? 1 : (df > 0) ? 2 : 0;  // Left, straight, right
      move_type = 64 + promo_idx * 3 + dir_idx;
    }
  }
  
  if (move_type < 0) return -1;
  
  return from_idx * 73 + move_type;
}

MCTSMove MCTSEncoder::policy_index_to_move(const MCTSPosition& pos, int index) {
  // This is the inverse of move_to_policy_index
  // For now, return null move - proper implementation would decode the index
  return MCTSMove();
}

}  // namespace MCTS
}  // namespace MetalFish
