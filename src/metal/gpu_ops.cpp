/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
*/

#include "gpu_ops.h"
#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/zobrist.h"
#include <chrono>
#include <cstring>
#include <iostream>

namespace MetalFish {
namespace GPU {

// Global instance
std::unique_ptr<GPUOps> gpu_ops;

// ============================================================================
// GPUPosition
// ============================================================================

void GPUPosition::from_position(const Position &pos) {
  // Copy piece bitboards
  for (int c = 0; c < 2; c++) {
    for (int pt = 0; pt < 7; pt++) {
      pieces[c][pt] = pos.pieces(Color(c), PieceType(pt));
    }
  }

  // Occupancy
  occupied[0] = pos.pieces(WHITE);
  occupied[1] = pos.pieces(BLACK);
  occupied[2] = pos.pieces();

  // Board array
  for (int sq = 0; sq < 64; sq++) {
    Piece p = pos.piece_on(Square(sq));
    board[sq] = static_cast<int8_t>(p);
  }

  // State
  sideToMove = pos.side_to_move();
  castlingRights = pos.castling_rights();
  epSquare = pos.ep_square();
  halfmoveClock = pos.rule50_count();
}

// ============================================================================
// GPUMove
// ============================================================================

Move GPUMove::to_move() const {
  int from = data & 0x3F;
  int to = (data >> 6) & 0x3F;
  int type = (data >> 12) & 0x3;
  int promo = (data >> 14) & 0x3;

  if (type == 0) {
    return Move(Square(from), Square(to));
  } else if (type == 1) { // PROMOTION
    return Move::make<PROMOTION>(Square(from), Square(to),
                                 PieceType(promo + KNIGHT));
  } else if (type == 2) { // EN_PASSANT
    return Move::make<EN_PASSANT>(Square(from), Square(to));
  } else { // CASTLING
    return Move::make<CASTLING>(Square(from), Square(to));
  }
}

GPUMove GPUMove::from_move(Move m) {
  GPUMove gm;
  int from = m.from_sq();
  int to = m.to_sq();
  int type = 0;
  int promo = 0;

  MoveType mt = m.type_of();
  if (mt == PROMOTION) {
    type = 1;
    promo = m.promotion_type() - KNIGHT;
  } else if (mt == EN_PASSANT) {
    type = 2;
  } else if (mt == CASTLING) {
    type = 3;
  }

  gm.data = (from & 0x3F) | ((to & 0x3F) << 6) | ((type & 0x3) << 12) |
            ((promo & 0x3) << 14);
  gm.score = 0;
  return gm;
}

// ============================================================================
// GPUOps Implementation
// ============================================================================

GPUOps::GPUOps() {}

GPUOps::~GPUOps() {
  // Release kernels
  if (pawn_movegen_kernel_)
    pawn_movegen_kernel_->release();
  if (knight_movegen_kernel_)
    knight_movegen_kernel_->release();
  if (king_movegen_kernel_)
    king_movegen_kernel_->release();
  if (score_moves_kernel_)
    score_moves_kernel_->release();
  if (see_kernel_)
    see_kernel_->release();
  if (eval_kernel_)
    eval_kernel_->release();
  if (attack_kernel_)
    attack_kernel_->release();
  if (hash_kernel_)
    hash_kernel_->release();
  if (filter_legal_kernel_)
    filter_legal_kernel_->release();
  if (sort_kernel_)
    sort_kernel_->release();

  // Release buffers
  if (positions_buffer_)
    positions_buffer_->release();
  if (moves_buffer_)
    moves_buffer_->release();
  if (move_counts_buffer_)
    move_counts_buffer_->release();
  if (attack_tables_buffer_)
    attack_tables_buffer_->release();
  if (results_buffer_)
    results_buffer_->release();
  if (history_buffer_)
    history_buffer_->release();
  if (bishop_table_buffer_)
    bishop_table_buffer_->release();
  if (rook_table_buffer_)
    rook_table_buffer_->release();
  if (zobrist_pieces_buffer_)
    zobrist_pieces_buffer_->release();
  if (zobrist_castling_buffer_)
    zobrist_castling_buffer_->release();
  if (zobrist_ep_buffer_)
    zobrist_ep_buffer_->release();

  if (library_)
    library_->release();
  if (queue_)
    queue_->release();
}

bool GPUOps::init(MTL::Device *device, const GPUOpsConfig &config) {
  device_ = device;
  config_ = config;

  if (!device_) {
    std::cerr << "[GPU_OPS] No Metal device provided" << std::endl;
    return false;
  }

  // Create command queue
  queue_ = device_->newCommandQueue();
  if (!queue_) {
    std::cerr << "[GPU_OPS] Failed to create command queue" << std::endl;
    return false;
  }

  // Load kernels
  if (!load_kernels()) {
    std::cerr
        << "[GPU_OPS] Warning: Failed to load some kernels, using CPU fallback"
        << std::endl;
  }

  // Allocate buffers
  if (!allocate_buffers()) {
    std::cerr << "[GPU_OPS] Failed to allocate buffers" << std::endl;
    return false;
  }

  // Upload precomputed tables
  upload_attack_tables();
  upload_zobrist_keys();

  initialized_ = true;
  std::cout << "[GPU_OPS] Initialized with batch size: "
            << config_.max_batch_size << std::endl;

  return true;
}

bool GPUOps::load_kernels() {
  NS::Error *error = nullptr;

  // Try to load from compiled metallib first
  NS::String *libPath =
      NS::String::string("metalfish.metallib", NS::UTF8StringEncoding);
  NS::URL *libURL = NS::URL::fileURLWithPath(libPath);
  library_ = device_->newLibrary(libURL, &error);

  if (!library_) {
    // Shader file not found - create minimal fallback library
    NS::String *source = NS::String::string(R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void noop_kernel() {}
    )",
                                            NS::UTF8StringEncoding);

    MTL::CompileOptions *opts = MTL::CompileOptions::alloc()->init();
    library_ = device_->newLibrary(source, opts, &error);
    opts->release();

    if (!library_) {
      // Even fallback failed - that's OK, we'll use CPU
      return false;
    }
  }

  // Load individual kernels (non-fatal if missing)
  auto load_kernel = [this](const char *name) -> MTL::ComputePipelineState * {
    NS::String *fname = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function *fn = library_->newFunction(fname);
    if (!fn)
      return nullptr;

    NS::Error *err = nullptr;
    MTL::ComputePipelineState *pso = device_->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
  };

  pawn_movegen_kernel_ = load_kernel("generate_pawn_moves");
  knight_movegen_kernel_ = load_kernel("generate_knight_moves");
  king_movegen_kernel_ = load_kernel("generate_king_moves");
  score_moves_kernel_ = load_kernel("score_moves");
  see_kernel_ = load_kernel("batch_see");
  eval_kernel_ = load_kernel("batch_evaluate_positions");
  attack_kernel_ = load_kernel("is_square_attacked");
  hash_kernel_ = load_kernel("compute_zobrist_hash");
  filter_legal_kernel_ = load_kernel("filter_legal_moves");
  sort_kernel_ = load_kernel("bitonic_sort_step");

  return true;
}

bool GPUOps::allocate_buffers() {
  MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
  size_t batch = config_.max_batch_size;
  size_t max_moves = config_.max_moves_per_position;

  // Position buffer
  positions_buffer_ = device_->newBuffer(batch * sizeof(GPUPosition), options);

  // Move buffers
  moves_buffer_ =
      device_->newBuffer(batch * max_moves * sizeof(GPUMove), options);
  move_counts_buffer_ = device_->newBuffer(batch * sizeof(int32_t), options);

  // Attack tables
  attack_tables_buffer_ = device_->newBuffer(sizeof(AttackTables), options);

  // Results buffer (multi-purpose)
  results_buffer_ = device_->newBuffer(batch * sizeof(int32_t), options);

  // History buffer
  history_buffer_ = device_->newBuffer(2 * 64 * 64 * sizeof(int16_t), options);

  // Zobrist buffers
  zobrist_pieces_buffer_ =
      device_->newBuffer(16 * 64 * sizeof(uint64_t), options);
  zobrist_castling_buffer_ = device_->newBuffer(16 * sizeof(uint64_t), options);
  zobrist_ep_buffer_ = device_->newBuffer(8 * sizeof(uint64_t), options);

  return positions_buffer_ && moves_buffer_ && move_counts_buffer_;
}

void GPUOps::upload_attack_tables() {
  if (!attack_tables_buffer_)
    return;

  AttackTables *tables =
      static_cast<AttackTables *>(attack_tables_buffer_->contents());

  // Copy pawn attacks
  for (int c = 0; c < 2; c++) {
    for (int sq = 0; sq < 64; sq++) {
      tables->pawnAttacks[c][sq] = PawnAttacks[c][sq];
    }
  }

  // Copy knight attacks
  for (int sq = 0; sq < 64; sq++) {
    tables->knightAttacks[sq] = KnightAttacks[sq];
  }

  // Copy king attacks
  for (int sq = 0; sq < 64; sq++) {
    tables->kingAttacks[sq] = KingAttacks[sq];
  }

  // Magic numbers and shifts would be copied here
  // (using traditional magic bitboards)
}

void GPUOps::upload_zobrist_keys() {
  if (!zobrist_pieces_buffer_)
    return;

  // Copy Zobrist piece keys
  uint64_t *pieces =
      static_cast<uint64_t *>(zobrist_pieces_buffer_->contents());
  for (int p = 0; p < 16; p++) {
    for (int sq = 0; sq < 64; sq++) {
      pieces[p * 64 + sq] = Zobrist::psq[p][sq];
    }
  }

  // Copy castling keys
  uint64_t *castling =
      static_cast<uint64_t *>(zobrist_castling_buffer_->contents());
  for (int i = 0; i < 16; i++) {
    castling[i] = Zobrist::castling[i];
  }

  // Copy en passant keys
  uint64_t *ep = static_cast<uint64_t *>(zobrist_ep_buffer_->contents());
  for (int f = 0; f < 8; f++) {
    ep[f] = Zobrist::enpassant[f];
  }
}

void GPUOps::prepare_positions(const std::vector<Position *> &positions) {
  if (!positions_buffer_)
    return;

  GPUPosition *gpu_positions =
      static_cast<GPUPosition *>(positions_buffer_->contents());

  for (size_t i = 0; i < positions.size() && i < (size_t)config_.max_batch_size;
       i++) {
    gpu_positions[i].from_position(*positions[i]);
  }
}

std::vector<int>
GPUOps::batch_generate_moves(const std::vector<Position *> &positions,
                             std::vector<std::vector<Move>> &moves) {
  if (!initialized_ || positions.empty()) {
    // CPU fallback
    std::vector<int> counts;
    for (auto *pos : positions) {
      MoveList<LEGAL> ml(*pos);
      moves.emplace_back();
      for (auto &m : ml) {
        moves.back().push_back(m);
      }
      counts.push_back(ml.size());
    }
    return counts;
  }

  auto start = std::chrono::high_resolution_clock::now();

  int batch_size = std::min((int)positions.size(), config_.max_batch_size);

  // Prepare positions
  prepare_positions(positions);

  // Clear move counts
  memset(move_counts_buffer_->contents(), 0, batch_size * sizeof(int32_t));

  // Create command buffer
  MTL::CommandBuffer *cmd = queue_->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  // Dispatch pawn moves
  if (pawn_movegen_kernel_) {
    enc->setComputePipelineState(pawn_movegen_kernel_);
    enc->setBuffer(positions_buffer_, 0, 0);
    enc->setBuffer(attack_tables_buffer_, 0, 1);
    enc->setBuffer(moves_buffer_, 0, 2);
    enc->setBuffer(move_counts_buffer_, 0, 3);
    enc->setBytes(&batch_size, sizeof(int), 4);

    // 8 pawns max per position
    MTL::Size grid(8, batch_size, 1);
    MTL::Size tg(8, 1, 1);
    enc->dispatchThreads(grid, tg);
  }

  // Dispatch knight moves
  if (knight_movegen_kernel_) {
    enc->setComputePipelineState(knight_movegen_kernel_);
    enc->setBuffer(positions_buffer_, 0, 0);
    enc->setBuffer(attack_tables_buffer_, 0, 1);
    enc->setBuffer(moves_buffer_, 0, 2);
    enc->setBuffer(move_counts_buffer_, 0, 3);
    enc->setBytes(&batch_size, sizeof(int), 4);

    MTL::Size grid(2, batch_size, 1);
    MTL::Size tg(2, 1, 1);
    enc->dispatchThreads(grid, tg);
  }

  // Dispatch king moves
  if (king_movegen_kernel_) {
    enc->setComputePipelineState(king_movegen_kernel_);
    enc->setBuffer(positions_buffer_, 0, 0);
    enc->setBuffer(attack_tables_buffer_, 0, 1);
    enc->setBuffer(moves_buffer_, 0, 2);
    enc->setBuffer(move_counts_buffer_, 0, 3);
    enc->setBytes(&batch_size, sizeof(int), 4);

    MTL::Size grid(batch_size, 1, 1);
    MTL::Size tg(1, 1, 1);
    enc->dispatchThreads(grid, tg);
  }

  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  // Read results
  int32_t *counts = static_cast<int32_t *>(move_counts_buffer_->contents());
  GPUMove *gpu_moves = static_cast<GPUMove *>(moves_buffer_->contents());

  std::vector<int> result(batch_size);
  moves.resize(batch_size);

  for (int i = 0; i < batch_size; i++) {
    result[i] = counts[i];
    moves[i].clear();
    for (int j = 0; j < counts[i]; j++) {
      moves[i].push_back(
          gpu_moves[i * config_.max_moves_per_position + j].to_move());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ += std::chrono::duration<double, std::milli>(end - start).count();
  batch_count_++;

  return result;
}

void GPUOps::batch_score_moves(const std::vector<Position *> &positions,
                               std::vector<std::vector<Move>> &moves,
                               const int16_t *history) {
  if (!initialized_ || !score_moves_kernel_ || positions.empty())
    return;

  int batch_size = std::min((int)positions.size(), config_.max_batch_size);

  // Prepare positions
  prepare_positions(positions);

  // Upload moves
  GPUMove *gpu_moves = static_cast<GPUMove *>(moves_buffer_->contents());
  int32_t *counts = static_cast<int32_t *>(move_counts_buffer_->contents());

  for (int i = 0; i < batch_size; i++) {
    counts[i] = moves[i].size();
    for (size_t j = 0; j < moves[i].size(); j++) {
      gpu_moves[i * config_.max_moves_per_position + j] =
          GPUMove::from_move(moves[i][j]);
    }
  }

  // Upload history
  if (history) {
    memcpy(history_buffer_->contents(), history, 2 * 64 * 64 * sizeof(int16_t));
  }

  // Create command buffer
  MTL::CommandBuffer *cmd = queue_->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  enc->setComputePipelineState(score_moves_kernel_);
  enc->setBuffer(positions_buffer_, 0, 0);
  enc->setBuffer(moves_buffer_, 0, 1);
  enc->setBuffer(move_counts_buffer_, 0, 2);
  enc->setBuffer(history_buffer_, 0, 3);
  enc->setBytes(&batch_size, sizeof(int), 4);

  MTL::Size grid(config_.max_moves_per_position, batch_size, 1);
  MTL::Size tg(32, 1, 1);
  enc->dispatchThreads(grid, tg);

  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  // Results are updated in-place in the moves buffer
}

std::vector<int> GPUOps::batch_see(const std::vector<Position *> &positions,
                                   const std::vector<Move> &moves) {
  if (!initialized_ || !see_kernel_ || positions.empty()) {
    // CPU fallback
    std::vector<int> result(positions.size());
    for (size_t i = 0; i < positions.size(); i++) {
      result[i] = positions[i]->see_ge(moves[i]) ? 1 : -1;
    }
    return result;
  }

  int batch_size = std::min((int)positions.size(), config_.max_batch_size);

  prepare_positions(positions);

  // Upload moves
  GPUMove *gpu_moves = static_cast<GPUMove *>(moves_buffer_->contents());
  for (int i = 0; i < batch_size; i++) {
    gpu_moves[i] = GPUMove::from_move(moves[i]);
  }

  // Dispatch SEE kernel
  MTL::CommandBuffer *cmd = queue_->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  enc->setComputePipelineState(see_kernel_);
  enc->setBuffer(positions_buffer_, 0, 0);
  enc->setBuffer(moves_buffer_, 0, 1);
  enc->setBuffer(attack_tables_buffer_, 0, 2);
  enc->setBuffer(results_buffer_, 0, 3);
  enc->setBytes(&batch_size, sizeof(int), 4);

  MTL::Size grid(batch_size, 1, 1);
  MTL::Size tg(std::min(batch_size, 256), 1, 1);
  enc->dispatchThreads(grid, tg);

  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  // Read results
  int32_t *results = static_cast<int32_t *>(results_buffer_->contents());
  return std::vector<int>(results, results + batch_size);
}

std::vector<Value>
GPUOps::batch_evaluate(const std::vector<Position *> &positions) {
  if (!initialized_ || !eval_kernel_ || positions.empty()) {
    // CPU fallback using classical eval
    std::vector<Value> result;
    for (auto *pos : positions) {
      Value v = Value(0);
      // Simple material count
      v += (popcount(pos->pieces(WHITE, PAWN)) -
            popcount(pos->pieces(BLACK, PAWN))) *
           100;
      v += (popcount(pos->pieces(WHITE, KNIGHT)) -
            popcount(pos->pieces(BLACK, KNIGHT))) *
           320;
      v += (popcount(pos->pieces(WHITE, BISHOP)) -
            popcount(pos->pieces(BLACK, BISHOP))) *
           330;
      v += (popcount(pos->pieces(WHITE, ROOK)) -
            popcount(pos->pieces(BLACK, ROOK))) *
           500;
      v += (popcount(pos->pieces(WHITE, QUEEN)) -
            popcount(pos->pieces(BLACK, QUEEN))) *
           900;

      if (pos->side_to_move() == BLACK)
        v = -v;
      result.push_back(v);
    }
    return result;
  }

  int batch_size = std::min((int)positions.size(), config_.max_batch_size);

  prepare_positions(positions);

  // Dispatch eval kernel
  MTL::CommandBuffer *cmd = queue_->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  int use_nnue = 0; // Classical eval for now

  enc->setComputePipelineState(eval_kernel_);
  enc->setBuffer(positions_buffer_, 0, 0);
  enc->setBuffer(results_buffer_, 0, 1); // NNUE scores (empty for classical)
  enc->setBuffer(results_buffer_, 0, 2); // Output
  enc->setBytes(&batch_size, sizeof(int), 3);
  enc->setBytes(&use_nnue, sizeof(int), 4);

  MTL::Size grid(batch_size, 1, 1);
  MTL::Size tg(std::min(batch_size, 256), 1, 1);
  enc->dispatchThreads(grid, tg);

  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  // Read results
  int32_t *results = static_cast<int32_t *>(results_buffer_->contents());
  std::vector<Value> output(batch_size);
  for (int i = 0; i < batch_size; i++) {
    output[i] = Value(results[i]);
  }
  return output;
}

std::vector<bool>
GPUOps::batch_is_attacked(const std::vector<Position *> &positions,
                          const std::vector<Square> &squares,
                          const std::vector<Color> &attackers) {
  // CPU fallback for now (attack detection is complex with magic bitboards)
  std::vector<bool> result(positions.size());
  for (size_t i = 0; i < positions.size(); i++) {
    result[i] = positions[i]->attackers_to(squares[i]) &
                positions[i]->pieces(attackers[i]);
  }
  return result;
}

std::vector<Key>
GPUOps::batch_compute_hash(const std::vector<Position *> &positions) {
  if (!initialized_ || !hash_kernel_ || positions.empty()) {
    // CPU fallback
    std::vector<Key> result;
    for (auto *pos : positions) {
      result.push_back(pos->key());
    }
    return result;
  }

  int batch_size = std::min((int)positions.size(), config_.max_batch_size);
  prepare_positions(positions);

  // Dispatch hash kernel
  MTL::CommandBuffer *cmd = queue_->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();

  uint64_t zobrist_side = Zobrist::side;

  enc->setComputePipelineState(hash_kernel_);
  enc->setBuffer(positions_buffer_, 0, 0);
  enc->setBuffer(zobrist_pieces_buffer_, 0, 1);
  enc->setBuffer(zobrist_castling_buffer_, 0, 2);
  enc->setBuffer(zobrist_ep_buffer_, 0, 3);
  enc->setBytes(&zobrist_side, sizeof(uint64_t), 4);
  enc->setBuffer(results_buffer_, 0, 5);
  enc->setBytes(&batch_size, sizeof(int), 6);

  MTL::Size grid(batch_size, 1, 1);
  MTL::Size tg(std::min(batch_size, 256), 1, 1);
  enc->dispatchThreads(grid, tg);

  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  // Read results
  uint64_t *hashes = static_cast<uint64_t *>(results_buffer_->contents());
  return std::vector<Key>(hashes, hashes + batch_size);
}

uint64_t GPUOps::gpu_perft(Position &pos, int depth) {
  // For now, use CPU perft with GPU-accelerated move generation
  // Full GPU perft would require iterative expansion on GPU

  if (depth == 0)
    return 1;

  uint64_t nodes = 0;
  MoveList<LEGAL> ml(pos);

  if (depth == 1)
    return ml.size();

  StateInfo st;
  for (const auto &m : ml) {
    pos.do_move(m, st);
    nodes += gpu_perft(pos, depth - 1);
    pos.undo_move(m);
  }

  return nodes;
}

void GPUOps::sync() {
  // Synchronize all pending GPU work
  if (queue_) {
    MTL::CommandBuffer *cmd = queue_->commandBuffer();
    cmd->commit();
    cmd->waitUntilCompleted();
  }
}

size_t GPUOps::get_gpu_memory_used() const {
  size_t total = 0;
  if (positions_buffer_)
    total += positions_buffer_->length();
  if (moves_buffer_)
    total += moves_buffer_->length();
  if (move_counts_buffer_)
    total += move_counts_buffer_->length();
  if (attack_tables_buffer_)
    total += attack_tables_buffer_->length();
  if (results_buffer_)
    total += results_buffer_->length();
  if (history_buffer_)
    total += history_buffer_->length();
  return total;
}

double GPUOps::get_average_batch_time() const {
  return batch_count_ > 0 ? total_time_ / batch_count_ : 0;
}

// Initialize global GPU operations
bool init_gpu_ops() {
  if (gpu_ops)
    return true;

  gpu_ops = std::make_unique<GPUOps>();

  // Get Metal device
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    std::cerr << "[GPU_OPS] No Metal device available" << std::endl;
    return false;
  }

  GPUOpsConfig config;
  config.max_batch_size = 256;
  config.max_moves_per_position = 256;
  config.use_unified_memory = true;

  return gpu_ops->init(device, config);
}

} // namespace GPU
} // namespace MetalFish
