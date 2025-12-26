/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/gpu_search.h"
#include "metal/device.h"
#include "core/bitboard.h"
#include <iostream>
#include <cstring>

namespace MetalFish {

// PositionData implementation
void PositionData::from_position(const Position& pos) {
    // Extract pieces
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Piece p = pos.piece_on(s);
        if (p == NO_PIECE) {
            pieces[s] = 0;
        } else {
            int8_t pt = int8_t(type_of(p));
            pieces[s] = (color_of(p) == WHITE) ? pt : -pt;
        }
    }
    
    // Extract occupancy
    occupancy[0] = pos.pieces(WHITE);
    occupancy[1] = pos.pieces(BLACK);
    occupancy[2] = pos.pieces();
    
    // Extract piece bitboards
    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt = NO_PIECE_TYPE; pt <= KING; ++pt) {
            piece_bb[c][pt] = pos.pieces(c, pt);
        }
    }
    
    side_to_move = pos.side_to_move();
    castling = pos.castling_rights();
    ep_square = pos.ep_square();
    ply = pos.game_ply();
}

// GPUSearch implementation
GPUSearch::GPUSearch() = default;

GPUSearch::~GPUSearch() {
    // Release kernels
    if (score_moves_kernel_) score_moves_kernel_->release();
    if (material_eval_kernel_) material_eval_kernel_->release();
    if (see_kernel_) see_kernel_->release();
    
    // Release buffers
    if (position_buffer_) position_buffer_->release();
    if (moves_buffer_) moves_buffer_->release();
    if (scores_buffer_) scores_buffer_->release();
    if (params_buffer_) params_buffer_->release();
    
    // Release library
    if (library_) library_->release();
}

bool GPUSearch::init() {
    try {
        Metal::Device& metal_device = Metal::get_device();
        device_ = metal_device.mtl_device();
        queue_ = metal_device.get_queue();
        
        if (!device_ || !queue_) {
            std::cerr << "[GPUSearch] Failed to get Metal device" << std::endl;
            return false;
        }
        
        if (!load_kernels()) {
            std::cerr << "[GPUSearch] Failed to load kernels" << std::endl;
            return false;
        }
        
        if (!create_buffers()) {
            std::cerr << "[GPUSearch] Failed to create buffers" << std::endl;
            return false;
        }
        
        available_ = true;
        std::cout << "[GPUSearch] GPU search acceleration initialized" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[GPUSearch] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool GPUSearch::load_kernels() {
    if (!device_) return false;
    
    // Inline shader for material evaluation
    std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;

constant int PIECE_VALUES[7] = { 0, 100, 320, 330, 500, 900, 10000 };

kernel void material_eval_batch(
    device const int8_t* pieces [[buffer(0)]],
    device int32_t* results [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    uint pos_idx [[thread_position_in_grid]])
{
    if ((int)pos_idx >= batch_size) return;
    
    int base = pos_idx * 64;
    int score = 0;
    
    for (int sq = 0; sq < 64; sq++) {
        int8_t piece = pieces[base + sq];
        if (piece == 0) continue;
        
        int piece_type = abs(piece);
        if (piece_type > 6) piece_type = 6;
        int sign = (piece > 0) ? 1 : -1;
        score += sign * PIECE_VALUES[piece_type];
    }
    
    results[pos_idx] = score;
}

constant int MAX_MOVES = 256;
constant int MVV_LVA_BASE = 10000;

kernel void score_moves_kernel(
    device const uint32_t* moves [[buffer(0)]],
    device const int32_t* move_counts [[buffer(1)]],
    device const int8_t* pieces [[buffer(2)]],
    device const int16_t* history [[buffer(3)]],
    device const uint32_t* killers [[buffer(4)]],
    device int32_t* scores [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    int pos_idx = tid.y;
    int move_idx = tid.x;
    
    if (pos_idx >= batch_size) return;
    
    int num_moves = move_counts[pos_idx];
    if (move_idx >= num_moves) return;
    
    uint32_t m = moves[pos_idx * MAX_MOVES + move_idx];
    
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    int promo = (m >> 12) & 0x7;
    
    int score = 0;
    
    int piece_base = pos_idx * 64;
    int8_t captured = pieces[piece_base + to];
    
    if (captured != 0) {
        int victim = abs(captured);
        if (victim > 6) victim = 6;
        int8_t moving = pieces[piece_base + from];
        int attacker = abs(moving);
        if (attacker > 6) attacker = 1;
        score = PIECE_VALUES[victim] - attacker + MVV_LVA_BASE;
    } else if (promo > 0) {
        score = PIECE_VALUES[promo < 7 ? promo : 5] + 5000;
    } else {
        uint32_t killer1 = killers[pos_idx * 2];
        uint32_t killer2 = killers[pos_idx * 2 + 1];
        
        if (m == killer1) score = 4000;
        else if (m == killer2) score = 3900;
        else score = history[from * 64 + to];
    }
    
    scores[pos_idx * MAX_MOVES + move_idx] = score;
}
)";
    
    NS::Error* error = nullptr;
    auto source = NS::String::string(shader_source.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    
    library_ = device_->newLibrary(source, options, &error);
    options->release();
    
    if (!library_) {
        if (error) {
            std::cerr << "[GPUSearch] Shader compilation failed: " 
                      << error->localizedDescription()->utf8String() << std::endl;
        }
        return false;
    }
    
    // Get kernel functions
    auto get_kernel = [this](const char* name) -> MTL::ComputePipelineState* {
        auto fn_name = NS::String::string(name, NS::UTF8StringEncoding);
        auto fn = library_->newFunction(fn_name);
        if (!fn) return nullptr;
        
        NS::Error* error = nullptr;
        auto pipeline = device_->newComputePipelineState(fn, &error);
        fn->release();
        return pipeline;
    };
    
    material_eval_kernel_ = get_kernel("material_eval_batch");
    score_moves_kernel_ = get_kernel("score_moves_kernel");
    
    return material_eval_kernel_ != nullptr;
}

bool GPUSearch::create_buffers() {
    if (!device_) return false;
    
    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    
    size_t max_batch = config_.batch_size;
    
    // Position data: 64 bytes per position (pieces array)
    position_buffer_ = device_->newBuffer(max_batch * 64, options);
    
    // Moves: 256 moves * 4 bytes per position
    moves_buffer_ = device_->newBuffer(max_batch * 256 * 4, options);
    
    // Scores: 256 scores * 4 bytes per position
    scores_buffer_ = device_->newBuffer(max_batch * 256 * 4, options);
    
    // Parameters buffer
    params_buffer_ = device_->newBuffer(64, options);
    
    return position_buffer_ && moves_buffer_ && scores_buffer_;
}

void GPUSearch::evaluate_batch(PositionBatch& batch) {
    if (!available_ || batch.size() == 0) {
        // Fallback to CPU evaluation
        for (size_t i = 0; i < batch.size(); ++i) {
            Value score = VALUE_ZERO;
            for (int sq = 0; sq < 64; ++sq) {
                int8_t piece = batch.positions[i].pieces[sq];
                if (piece == 0) continue;
                
                static const int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 10000};
                int pt = std::abs(piece);
                if (pt > 6) pt = 6;
                int sign = (piece > 0) ? 1 : -1;
                score += sign * PIECE_VALUES[pt];
            }
            batch.scores.push_back(score);
        }
        positions_evaluated_ += batch.size();
        return;
    }
    
    int num_positions = static_cast<int>(batch.size());
    
    // Copy position data to GPU buffer
    int8_t* pieces_ptr = static_cast<int8_t*>(position_buffer_->contents());
    for (int i = 0; i < num_positions; ++i) {
        memcpy(pieces_ptr + i * 64, batch.positions[i].pieces.data(), 64);
    }
    
    // Create command buffer
    auto cmd_buffer = queue_->commandBuffer();
    auto encoder = cmd_buffer->computeCommandEncoder();
    
    // Set kernel and buffers
    encoder->setComputePipelineState(material_eval_kernel_);
    encoder->setBuffer(position_buffer_, 0, 0);
    encoder->setBuffer(scores_buffer_, 0, 1);
    encoder->setBytes(&num_positions, sizeof(int), 2);
    
    // Dispatch
    MTL::Size grid(num_positions, 1, 1);
    MTL::Size group(std::min(num_positions, 256), 1, 1);
    encoder->dispatchThreads(grid, group);
    
    encoder->endEncoding();
    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();
    
    // Read results
    int32_t* results_ptr = static_cast<int32_t*>(scores_buffer_->contents());
    batch.scores.resize(num_positions);
    for (int i = 0; i < num_positions; ++i) {
        batch.scores[i] = Value(results_ptr[i]);
    }
    
    positions_evaluated_ += num_positions;
    gpu_batches_++;
}

void GPUSearch::score_moves_batch(const std::vector<Position*>& positions,
                                   std::vector<std::vector<Move>>& moves,
                                   std::vector<std::vector<int>>& scores) {
    // For each position, score its moves
    // This is a placeholder - actual implementation would batch on GPU
    
    scores.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        scores[i].resize(moves[i].size());
        for (size_t j = 0; j < moves[i].size(); ++j) {
            Move m = moves[i][j];
            Square to = m.to_sq();
            Piece captured = positions[i]->piece_on(to);
            
            if (captured != NO_PIECE) {
                // Capture - use MVV-LVA
                scores[i][j] = int(type_of(captured)) * 100 + 10000;
            } else {
                // Quiet move - use history (placeholder)
                scores[i][j] = 0;
            }
        }
    }
}

Value GPUSearch::parallel_search(Position& root, int depth, Value alpha, Value beta) {
    // Experimental parallel search on GPU
    // This would expand the search tree breadth-first and evaluate leaves on GPU
    
    if (depth == 0) {
        return Value(0);  // Would call evaluate
    }
    
    // Generate moves
    PositionBatch batch;
    batch.reserve(256);
    
    StateInfo st;
    MoveList<LEGAL> moves(root);
    
    // Create child positions
    for (Move m : moves) {
        PositionData pd;
        root.do_move(m, st);
        pd.from_position(root);
        root.undo_move(m);
        
        batch.positions.push_back(pd);
        batch.moves.push_back(m);
    }
    
    // Evaluate all children on GPU
    evaluate_batch(batch);
    
    // Return best score
    Value best = -VALUE_INFINITE;
    for (size_t i = 0; i < batch.scores.size(); ++i) {
        Value score = -batch.scores[i];  // Negamax
        if (score > best) {
            best = score;
        }
    }
    
    return best;
}

// Global instance
static std::unique_ptr<GPUSearch> g_gpu_search;

GPUSearch& get_gpu_search() {
    if (!g_gpu_search) {
        g_gpu_search = std::make_unique<GPUSearch>();
        g_gpu_search->init();
    }
    return *g_gpu_search;
}

} // namespace MetalFish


