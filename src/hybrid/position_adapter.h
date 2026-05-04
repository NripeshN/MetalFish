/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Position adapter layer for the MCTS search infrastructure.

  Licensed under GPL-3.0
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../core/bitboard.h"
#include "../core/movegen.h"
#include "../core/position.h"
#include "../core/types.h"

using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

class MCTSMove {
public:
  MCTSMove() : move_(Move::none()) {}
  MCTSMove(Move m) : move_(m) {}

  static MCTSMove FromInternal(Move m) { return MCTSMove(m); }
  Move to_internal() const { return move_; }

  Square from() const { return move_.from_sq(); }
  Square to() const { return move_.to_sq(); }

  bool is_promotion() const { return move_.type_of() == PROMOTION; }
  bool is_castling() const { return move_.type_of() == CASTLING; }
  bool is_en_passant() const { return move_.type_of() == EN_PASSANT; }
  bool is_null() const { return move_ == Move::none(); }

  PieceType get_promotion_type() const { return move_.promotion_type(); }

  std::string to_string() const;

  bool operator==(const MCTSMove &other) const { return move_ == other.move_; }
  bool operator!=(const MCTSMove &other) const { return move_ != other.move_; }

  uint16_t raw_data() const { return static_cast<uint16_t>(move_.raw()); }

private:
  Move move_;
};

using MCTSMoveList = std::vector<MCTSMove>;

class MCTSPosition {
public:
  MCTSPosition();
  MCTSPosition(const MCTSPosition &other);
  MCTSPosition &operator=(const MCTSPosition &other);

  void set_from_fen(const std::string &fen);
  std::string fen() const;

  Color side_to_move() const { return pos_.side_to_move(); }
  bool is_black_to_move() const { return side_to_move() == BLACK; }

  void do_move(MCTSMove move);
  void do_move(Move move);

  MCTSMoveList generate_legal_moves() const;

  bool is_check() const;
  bool is_checkmate() const;
  bool is_stalemate() const;
  bool is_draw() const;
  bool is_terminal() const;

  int repetition_count() const;
  int ply_count() const { return pos_.game_ply(); }
  int rule50_count() const { return pos_.rule50_count(); }
  uint64_t hash() const { return pos_.key(); }

  const Position &internal_position() const { return pos_; }
  Position &internal_position() { return pos_; }

  Bitboard pieces(Color c) const { return pos_.pieces(c); }
  Bitboard pieces(Color c, PieceType pt) const { return pos_.pieces(c, pt); }
  Square king_square(Color c) const { return pos_.square<KING>(c); }

  bool can_castle_kingside(Color c) const;
  bool can_castle_queenside(Color c) const;

  Square en_passant_square() const { return pos_.ep_square(); }

private:
  Position pos_;
  StateInfo st_;
  std::vector<StateInfo> state_stack_;
  std::vector<Move> move_stack_;
};

// Neural network input encoding
class MCTSEncoder {
public:
  static std::vector<float> encode_position(const MCTSPosition &pos);

private:
  static constexpr int kPawnPlane = 0;
  static constexpr int kKnightPlane = 1;
  static constexpr int kBishopPlane = 2;
  static constexpr int kRookPlane = 3;
  static constexpr int kQueenPlane = 4;
  static constexpr int kKingPlane = 5;

  static constexpr int kPlanesPerPosition = 13;
  static constexpr int kTotalPlanes = 112;
};

struct MCTSEvaluation {
  float wdl[3];
  float q;
  float m;
  std::vector<std::pair<MCTSMove, float>> policy;
};

class MCTSNeuralNetwork {
public:
  virtual ~MCTSNeuralNetwork() = default;

  virtual MCTSEvaluation evaluate(const MCTSPosition &pos) = 0;
  virtual std::vector<MCTSEvaluation>
  evaluate_batch(const std::vector<const MCTSPosition *> &positions) = 0;
  virtual int optimal_batch_size() const = 0;
};

} // namespace MCTS
} // namespace MetalFish
