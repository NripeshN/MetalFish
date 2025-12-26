/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/hybrid_search.h"
#include "metal/device.h"
#include "core/bitboard.h"
#include "core/movegen.h"
#include "eval/evaluate.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <random>

namespace MetalFish {

// PositionSnapshot implementation
PositionSnapshot::PositionSnapshot(const Position& pos) {
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Piece p = pos.piece_on(s);
        if (p == NO_PIECE) {
            pieces[s] = 0;
        } else {
            int8_t pt = int8_t(type_of(p));
            pieces[s] = (color_of(p) == WHITE) ? pt : -pt;
        }
    }
    side_to_move = pos.side_to_move();
    // Infer castling from can_castle queries
    castling = NO_CASTLING;
    if (pos.can_castle(WHITE_OO)) castling = CastlingRights(castling | WHITE_OO);
    if (pos.can_castle(WHITE_OOO)) castling = CastlingRights(castling | WHITE_OOO);
    if (pos.can_castle(BLACK_OO)) castling = CastlingRights(castling | BLACK_OO);
    if (pos.can_castle(BLACK_OOO)) castling = CastlingRights(castling | BLACK_OOO);
    ep_square = pos.ep_square();
    halfmove_clock = pos.rule50_count();
    game_ply = pos.game_ply();
}

// ============================================================================
// GPUEvaluationPipeline Implementation
// ============================================================================

GPUEvaluationPipeline::GPUEvaluationPipeline() = default;

GPUEvaluationPipeline::~GPUEvaluationPipeline() {
    running_ = false;
    queue_cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    // Release Metal resources
    if (policy_kernel_) policy_kernel_->release();
    if (value_kernel_) value_kernel_->release();
    if (feature_kernel_) feature_kernel_->release();
    if (position_buffer_) position_buffer_->release();
    if (feature_buffer_) feature_buffer_->release();
    if (policy_output_) policy_output_->release();
    if (value_output_) value_output_->release();
    if (weight_buffer_) weight_buffer_->release();
    if (library_) library_->release();
}

bool GPUEvaluationPipeline::init() {
    try {
        Metal::Device& metal_device = Metal::get_device();
        device_ = metal_device.mtl_device();
        queue_ = metal_device.get_queue();
        
        if (!device_ || !queue_) {
            std::cerr << "[GPUPipeline] Failed to get Metal device" << std::endl;
            return false;
        }
        
        // Compile the hybrid neural network shaders
        std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void batch_material_eval(
    device const int8_t* boards [[buffer(0)]],
    device int32_t* value_output [[buffer(1)]],
    device float* policy_output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    uint pos_idx [[thread_position_in_grid]])
{
    if ((int)pos_idx >= batch_size) return;
    
    device const int8_t* board = boards + pos_idx * 64;
    
    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        int8_t piece = board[sq];
        if (piece == 0) continue;
        
        int value = 0;
        int pt = abs(piece);
        switch (pt) {
            case 1: value = 100; break;
            case 2: value = 320; break;
            case 3: value = 330; break;
            case 4: value = 500; break;
            case 5: value = 900; break;
            case 6: value = 0; break;
        }
        
        if (piece > 0) score += value;
        else score -= value;
    }
    
    value_output[pos_idx] = score;
    
    // Initialize policy to uniform
    for (int i = 0; i < 256; i++) {
        policy_output[pos_idx * 256 + i] = 1.0f / 256.0f;
    }
}
)";
        
        NS::Error* error = nullptr;
        auto source = NS::String::string(shader_source.c_str(), NS::UTF8StringEncoding);
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
        
        library_ = device_->newLibrary(source, options, &error);
        options->release();
        
        if (!library_) {
            if (error) {
                std::cerr << "[GPUPipeline] Shader compilation failed: " 
                          << error->localizedDescription()->utf8String() << std::endl;
            }
            return false;
        }
        
        // Get kernel
        auto fn_name = NS::String::string("batch_material_eval", NS::UTF8StringEncoding);
        auto fn = library_->newFunction(fn_name);
        if (!fn) {
            std::cerr << "[GPUPipeline] Function not found" << std::endl;
            return false;
        }
        
        value_kernel_ = device_->newComputePipelineState(fn, &error);
        fn->release();
        
        if (!value_kernel_) {
            std::cerr << "[GPUPipeline] Pipeline creation failed" << std::endl;
            return false;
        }
        
        // Create buffers with shared storage (unified memory)
        MTL::ResourceOptions buffer_options = MTL::ResourceStorageModeShared;
        
        int max_batch = 256;
        position_buffer_ = device_->newBuffer(max_batch * 64, buffer_options);
        value_output_ = device_->newBuffer(max_batch * sizeof(int32_t), buffer_options);
        policy_output_ = device_->newBuffer(max_batch * 256 * sizeof(float), buffer_options);
        
        if (!position_buffer_ || !value_output_ || !policy_output_) {
            std::cerr << "[GPUPipeline] Buffer creation failed" << std::endl;
            return false;
        }
        
        // Start worker thread
        running_ = true;
        worker_thread_ = std::thread(&GPUEvaluationPipeline::worker_loop, this);
        
        std::cout << "[GPUPipeline] GPU evaluation pipeline initialized" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[GPUPipeline] Exception: " << e.what() << std::endl;
        return false;
    }
}

void GPUEvaluationPipeline::submit(const std::vector<EvalRequest>& requests) {
    if (requests.empty()) return;
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    pending_requests_.push(requests);
    queue_cv_.notify_one();
}

bool GPUEvaluationPipeline::get_results(std::vector<EvalResult>& results) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (completed_results_.empty()) {
        return false;
    }
    
    results = std::move(completed_results_.front());
    completed_results_.pop();
    return true;
}

void GPUEvaluationPipeline::flush() {
    // Wait for all pending work to complete
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return pending_requests_.empty(); });
}

void GPUEvaluationPipeline::worker_loop() {
    while (running_) {
        std::vector<EvalRequest> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !pending_requests_.empty() || !running_; 
            });
            
            if (!running_) break;
            
            batch = std::move(pending_requests_.front());
            pending_requests_.pop();
        }
        
        process_batch(batch);
    }
}

void GPUEvaluationPipeline::process_batch(const std::vector<EvalRequest>& batch) {
    int num_positions = static_cast<int>(batch.size());
    
    // Copy position data to GPU buffer (already in snapshot format)
    int8_t* pos_ptr = static_cast<int8_t*>(position_buffer_->contents());
    
    for (int i = 0; i < num_positions; ++i) {
        const PositionSnapshot& snap = batch[i].pos;
        for (int s = 0; s < 64; ++s) {
            pos_ptr[i * 64 + s] = snap.pieces[s];
        }
    }
    
    // Create command buffer and dispatch
    auto cmd_buffer = queue_->commandBuffer();
    auto encoder = cmd_buffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(value_kernel_);
    encoder->setBuffer(position_buffer_, 0, 0);
    encoder->setBuffer(value_output_, 0, 1);
    encoder->setBuffer(policy_output_, 0, 2);
    encoder->setBytes(&num_positions, sizeof(int), 3);
    
    MTL::Size grid(num_positions, 1, 1);
    MTL::Size group(std::min(num_positions, 256), 1, 1);
    encoder->dispatchThreads(grid, group);
    
    encoder->endEncoding();
    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();
    
    // Read results
    std::vector<EvalResult> results(num_positions);
    int32_t* value_ptr = static_cast<int32_t*>(value_output_->contents());
    float* policy_ptr = static_cast<float*>(policy_output_->contents());
    
    for (int i = 0; i < num_positions; ++i) {
        results[i].node_id = batch[i].node_id;
        results[i].eval = Value(value_ptr[i]);
        
        for (int j = 0; j < 256; ++j) {
            results[i].policy[j] = policy_ptr[i * 256 + j];
        }
        
        // Use a placeholder for num_moves since we can't generate from snapshot
        results[i].num_moves = 20;  // Average number of legal moves
    }
    
    // Store results
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        completed_results_.push(std::move(results));
    }
    
    positions_evaluated_ += num_positions;
    batches_processed_++;
}

// ============================================================================
// HybridSearchEngine Implementation
// ============================================================================

HybridSearchEngine::HybridSearchEngine()
    : gpu_pipeline_(std::make_unique<GPUEvaluationPipeline>()),
      root_(std::make_unique<SearchNode>()) {
}

HybridSearchEngine::~HybridSearchEngine() = default;

bool HybridSearchEngine::init() {
    if (!gpu_pipeline_->init()) {
        std::cerr << "[HybridSearch] GPU pipeline init failed, using CPU fallback" << std::endl;
        return false;
    }
    
    std::cout << "[HybridSearch] Hybrid search engine initialized" << std::endl;
    return true;
}

void HybridSearchEngine::clear() {
    root_ = std::make_unique<SearchNode>();
    stats_ = SearchStats{};
    stop_flag_ = false;
}

Move HybridSearchEngine::search(Position& pos, const SearchLimits& limits) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    clear();
    
    // Generate root moves
    MoveList<LEGAL> root_moves(pos);
    if (root_moves.size() == 0) {
        return Move::none();
    }
    if (root_moves.size() == 1) {
        return *root_moves.begin();
    }
    
    // Initialize root node
    root_->move = Move::none();
    root_->children.reserve(root_moves.size());
    
    for (Move m : root_moves) {
        SearchNode child;
        child.move = m;
        root_->children.push_back(child);
    }
    
    // Determine search parameters
    int max_depth = limits.depth > 0 ? limits.depth : 100;
    uint64_t max_nodes = limits.nodes > 0 ? limits.nodes : UINT64_MAX;
    
    Move best_move = *root_moves.begin();
    Value best_value = -VALUE_INFINITE;
    
    // Iterative deepening
    for (int depth = 1; depth <= max_depth && !stop_flag_; ++depth) {
        Value alpha = -VALUE_INFINITE;
        Value beta = VALUE_INFINITE;
        
        // Use aspiration windows for deeper searches
        if (depth >= 4 && best_value != VALUE_NONE) {
            alpha = best_value - Value(25);
            beta = best_value + Value(25);
        }
        
        Value value = alpha_beta(pos, depth, alpha, beta, root_.get(), true);
        
        // Find best move from root children
        SearchNode* best_child = nullptr;
        for (auto& child : root_->children) {
            if (!best_child || child.eval > best_child->eval) {
                best_child = &child;
            }
        }
        
        if (best_child && best_child->move != Move::none()) {
            best_move = best_child->move;
            best_value = best_child->eval;
        }
        
        // Check time limits
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        
        if (limits.movetime > 0 && elapsed >= limits.movetime) {
            break;
        }
        
        // Print UCI info
        std::cout << "info depth " << depth
                  << " score cp " << int(best_value)
                  << " nodes " << stats_.nodes_searched
                  << " nps " << (elapsed > 0 ? stats_.nodes_searched * 1000 / elapsed : 0)
                  << " pv " << best_move.from_sq() << best_move.to_sq()
                  << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.search_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    stats_.nps = stats_.search_time_ms > 0 ? 
                 stats_.nodes_searched * 1000.0 / stats_.search_time_ms : 0;
    
    return best_move;
}

Value HybridSearchEngine::alpha_beta(Position& pos, int depth, Value alpha, Value beta,
                                      SearchNode* node, bool is_pv) {
    if (stop_flag_ || depth <= 0) {
        // Leaf node - use GPU evaluation or quiescence
        if (depth <= 0) {
            Value eval = quiescence(pos, alpha, beta);
            node->eval = eval;
            return eval;
        }
    }
    
    stats_.nodes_searched++;
    stats_.ab_nodes++;
    
    // Generate moves if not already done
    if (node->children.empty()) {
        MoveList<LEGAL> moves(pos);
        node->children.reserve(moves.size());
        
        for (Move m : moves) {
            SearchNode child;
            child.move = m;
            node->children.push_back(child);
        }
    }
    
    if (node->children.empty()) {
        // No legal moves - checkmate or stalemate
        if (pos.checkers()) {
            return -VALUE_MATE + node->depth;
        }
        return VALUE_DRAW;
    }
    
    // Move ordering - use previous evaluation scores
    std::sort(node->children.begin(), node->children.end(),
              [](const SearchNode& a, const SearchNode& b) {
                  return a.eval > b.eval;
              });
    
    Value best_value = -VALUE_INFINITE;
    StateInfo st;
    
    for (auto& child : node->children) {
        child.depth = node->depth + 1;
        
        pos.do_move(child.move, st);
        Value value = -alpha_beta(pos, depth - 1, -beta, -alpha, &child, is_pv && (&child == &node->children.front()));
        pos.undo_move(child.move);
        
        child.eval = value;
        
        if (value > best_value) {
            best_value = value;
            
            if (value > alpha) {
                alpha = value;
                
                if (value >= beta) {
                    break;  // Beta cutoff
                }
            }
        }
    }
    
    node->eval = best_value;
    return best_value;
}

Value HybridSearchEngine::quiescence(Position& pos, Value alpha, Value beta) {
    stats_.nodes_searched++;
    
    // Stand pat
    Value stand_pat = Eval::evaluate(pos);
    
    if (stand_pat >= beta) {
        return beta;
    }
    
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }
    
    // Generate captures only
    MoveList<CAPTURES> captures(pos);
    
    StateInfo st;
    for (Move m : captures) {
        pos.do_move(m, st);
        Value value = -quiescence(pos, -beta, -alpha);
        pos.undo_move(m);
        
        if (value >= beta) {
            return beta;
        }
        
        if (value > alpha) {
            alpha = value;
        }
    }
    
    return alpha;
}

void HybridSearchEngine::mcts_expand(Position& pos, SearchNode* node) {
    // MCTS-style node expansion
    // This is called when we want to explore a node more thoroughly
    
    if (node->children.empty()) {
        MoveList<LEGAL> moves(pos);
        
        // Get policy from GPU if available
        std::vector<float> policy_scores(moves.size(), 1.0f / moves.size());
        
        int idx = 0;
        for (Move m : moves) {
            SearchNode child;
            child.move = m;
            child.policy = policy_scores[idx++];
            node->children.push_back(child);
        }
    }
    
    // Select best child using UCB
    SearchNode* best_child = select_best_child(node, true);
    
    if (best_child) {
        StateInfo st;
        pos.do_move(best_child->move, st);
        
        // Evaluate or recurse
        if (best_child->visits == 0) {
            // First visit - evaluate
            best_child->eval = Eval::evaluate(pos);
            best_child->Q = float(best_child->eval);
        } else {
            mcts_expand(pos, best_child);
        }
        
        pos.undo_move(best_child->move);
        
        // Backpropagate
        backpropagate(best_child, best_child->eval);
    }
    
    stats_.mcts_nodes++;
}

SearchNode* HybridSearchEngine::select_best_child(SearchNode* node, bool exploration) {
    if (node->children.empty()) return nullptr;
    
    float best_ucb = -std::numeric_limits<float>::max();
    SearchNode* best_child = nullptr;
    
    float exploration_constant = exploration ? 1.41f : 0.0f;
    
    for (auto& child : node->children) {
        float ucb = child.ucb(exploration_constant);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child = &child;
        }
    }
    
    return best_child;
}

void HybridSearchEngine::backpropagate(SearchNode* node, Value value) {
    node->visits++;
    
    // Running average update
    float delta = float(value) - node->Q;
    node->Q += delta / node->visits;
}

void HybridSearchEngine::order_moves_with_policy(Position& pos,
                                                  std::vector<Move>& moves,
                                                  std::vector<float>& scores) {
    // Use GPU to compute move scores
    // This leverages the policy network output
    
    scores.resize(moves.size());
    
    // For now, use simple MVV-LVA
    for (size_t idx = 0; idx < moves.size(); ++idx) {
        Move m = moves[idx];
        float score = 0.0f;
        
        Square to = m.to_sq();
        Piece captured = pos.piece_on(to);
        
        if (captured != NO_PIECE) {
            score = float(type_of(captured)) * 100.0f;
        }
        
        // Promotion bonus
        if (m.type_of() == PROMOTION) {
            score += float(m.promotion_type()) * 100.0f;
        }
        
        scores[idx] = score;
    }
}

void HybridSearchEngine::batch_evaluate_leaves(std::vector<SearchNode*>& leaves,
                                                std::vector<EvalResult>& results) {
    // Collect positions for batch evaluation
    std::vector<EvalRequest> requests;
    
    for (size_t i = 0; i < leaves.size(); ++i) {
        // Would need to reconstruct positions here
        // For now, skip batch evaluation
    }
    
    if (!requests.empty()) {
        gpu_pipeline_->submit(requests);
        gpu_pipeline_->get_results(results);
        stats_.gpu_evaluations += results.size();
    }
}

void HybridSearchEngine::prune_tree(int max_nodes) {
    // Simple pruning: keep only most visited nodes
    // This could be more sophisticated
    (void)max_nodes;
}

void HybridSearchEngine::reuse_subtree(Move best_move) {
    // Find the child corresponding to best_move and make it the new root
    for (auto& child : root_->children) {
        if (child.move == best_move) {
            auto new_root = std::make_unique<SearchNode>(std::move(child));
            root_ = std::move(new_root);
            return;
        }
    }
    
    // Not found - create fresh tree
    root_ = std::make_unique<SearchNode>();
}

// Global instance
static std::unique_ptr<HybridSearchEngine> g_hybrid_search;

HybridSearchEngine& get_hybrid_search() {
    if (!g_hybrid_search) {
        g_hybrid_search = std::make_unique<HybridSearchEngine>();
        g_hybrid_search->init();
    }
    return *g_hybrid_search;
}

} // namespace MetalFish

