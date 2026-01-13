/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  This file provides an adapter layer between Stockfish's chess representation
  and Lc0's MCTS search infrastructure. This allows us to use Lc0's battle-tested
  MCTS implementation with Stockfish's efficient position representation.

  Licensed under GPL-3.0
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Include Stockfish headers
#include "../core/bitboard.h"
#include "../core/position.h"
#include "../core/movegen.h"
#include "../core/types.h"

// Use MetalFish namespace for chess types
using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// Forward declarations
class MCTSNode;
class MCTSTree;

// Game result enum compatible with Lc0
enum class GameResult : uint8_t {
  UNDECIDED = 0,
  WHITE_WON = 1,
  DRAW = 2,
  BLACK_WON = 3
};

// Move representation that wraps Stockfish's Move
class MCTSMove {
public:
  MCTSMove() : move_(Move::none()) {}
  MCTSMove(Move m) : move_(m) {}
  
  // Create from Stockfish move
  static MCTSMove FromStockfish(Move m) { return MCTSMove(m); }
  
  // Convert to Stockfish move
  Move to_stockfish() const { return move_; }
  
  // Get from/to squares
  Square from() const { return move_.from_sq(); }
  Square to() const { return move_.to_sq(); }
  
  // Check move type
  bool is_promotion() const { return move_.type_of() == PROMOTION; }
  bool is_castling() const { return move_.type_of() == CASTLING; }
  bool is_en_passant() const { return move_.type_of() == EN_PASSANT; }
  bool is_null() const { return move_ == Move::none(); }
  
  // Get promotion piece type
  PieceType get_promotion_type() const { return move_.promotion_type(); }
  
  // String representation
  std::string to_string() const;
  
  // Comparison
  bool operator==(const MCTSMove& other) const { return move_ == other.move_; }
  bool operator!=(const MCTSMove& other) const { return move_ != other.move_; }
  
  // Raw data access
  uint16_t raw_data() const { return static_cast<uint16_t>(move_.raw()); }

private:
  Move move_;
};

// Move list compatible with Lc0's interface
using MCTSMoveList = std::vector<MCTSMove>;

// Position wrapper that provides Lc0-compatible interface using Stockfish internals
class MCTSPosition {
public:
  MCTSPosition();
  MCTSPosition(const MCTSPosition& other);
  MCTSPosition& operator=(const MCTSPosition& other);
  
  // Set from FEN
  void set_from_fen(const std::string& fen);
  
  // Get current position as FEN
  std::string fen() const;
  
  // Get side to move
  Color side_to_move() const { return pos_.side_to_move(); }
  bool is_black_to_move() const { return side_to_move() == BLACK; }
  
  // Apply a move
  void do_move(MCTSMove move);
  void do_move(Move move);
  
  // Undo last move
  void undo_move();
  
  // Generate legal moves
  MCTSMoveList generate_legal_moves() const;
  
  // Check game state
  bool is_check() const;
  bool is_checkmate() const;
  bool is_stalemate() const;
  bool is_draw() const;
  bool is_terminal() const;
  GameResult get_game_result() const;
  
  // Get repetition count
  int repetition_count() const;
  
  // Get ply count
  int ply_count() const { return pos_.game_ply(); }
  
  // Get rule50 count
  int rule50_count() const { return pos_.rule50_count(); }
  
  // Hash
  uint64_t hash() const { return pos_.key(); }
  
  // Access to underlying Stockfish position
  const Position& stockfish_position() const { return pos_; }
  Position& stockfish_position() { return pos_; }
  
  // Piece access (for neural network encoding)
  Bitboard pieces(Color c) const { return pos_.pieces(c); }
  Bitboard pieces(Color c, PieceType pt) const { return pos_.pieces(c, pt); }
  Square king_square(Color c) const { return pos_.square<KING>(c); }
  
  // Castling rights
  bool can_castle_kingside(Color c) const;
  bool can_castle_queenside(Color c) const;
  
  // En passant square
  Square en_passant_square() const { return pos_.ep_square(); }

private:
  Position pos_;
  StateInfo st_;
  std::vector<StateInfo> state_stack_;
  std::vector<Move> move_stack_;
};

// Position history for repetition detection and game tracking
class MCTSPositionHistory {
public:
  MCTSPositionHistory();
  
  // Reset to starting position
  void reset();
  
  // Reset to position from FEN
  void reset(const std::string& fen);
  
  // Reset with moves
  void reset(const std::string& fen, const std::vector<std::string>& moves);
  
  // Apply move
  void do_move(MCTSMove move);
  void do_move(Move move);
  
  // Get current position
  const MCTSPosition& current() const { return positions_.back(); }
  MCTSPosition& current() { return positions_.back(); }
  
  // Get position at index
  const MCTSPosition& at(size_t idx) const { return positions_[idx]; }
  
  // Get number of positions
  size_t size() const { return positions_.size(); }
  
  // Check for repetition
  int compute_repetitions() const;
  
  // Get last move
  MCTSMove last_move() const;
  
  // Check if position is terminal
  bool is_terminal() const { return current().is_terminal(); }
  
  // Get game result
  GameResult get_game_result() const { return current().get_game_result(); }

private:
  std::vector<MCTSPosition> positions_;
  std::vector<MCTSMove> moves_;
};

// Neural network input encoding (converts Stockfish position to NN input planes)
class MCTSEncoder {
public:
  // Encode position for neural network
  // Returns input planes in Lc0's format (112 planes of 8x8)
  static std::vector<float> encode_position(const MCTSPosition& pos);
  
  // Encode position history (for transformer-style networks)
  static std::vector<float> encode_history(const MCTSPositionHistory& history, int history_length = 8);
  
  // Decode policy output to moves
  static std::vector<std::pair<MCTSMove, float>> decode_policy(
      const MCTSPosition& pos,
      const float* policy_output,
      int policy_size);
  
  // Policy index for a move
  static int move_to_policy_index(const MCTSPosition& pos, MCTSMove move);
  
  // Move from policy index
  static MCTSMove policy_index_to_move(const MCTSPosition& pos, int index);

private:
  // Piece plane indices
  static constexpr int kPawnPlane = 0;
  static constexpr int kKnightPlane = 1;
  static constexpr int kBishopPlane = 2;
  static constexpr int kRookPlane = 3;
  static constexpr int kQueenPlane = 4;
  static constexpr int kKingPlane = 5;
  
  // Number of planes per position
  static constexpr int kPlanesPerPosition = 13;
  
  // Total planes for 8 history positions
  static constexpr int kTotalPlanes = 112;
};

// Evaluation result from neural network
struct MCTSEvaluation {
  float wdl[3];     // Win/Draw/Loss probabilities
  float q;          // Q value (expected outcome)
  float m;          // Moves left estimate
  std::vector<std::pair<MCTSMove, float>> policy;  // Move probabilities
};

// Interface for neural network backend
class MCTSNeuralNetwork {
public:
  virtual ~MCTSNeuralNetwork() = default;
  
  // Evaluate a single position
  virtual MCTSEvaluation evaluate(const MCTSPosition& pos) = 0;
  
  // Evaluate a batch of positions
  virtual std::vector<MCTSEvaluation> evaluate_batch(
      const std::vector<const MCTSPosition*>& positions) = 0;
  
  // Get optimal batch size
  virtual int optimal_batch_size() const = 0;
};

}  // namespace MCTS
}  // namespace MetalFish
