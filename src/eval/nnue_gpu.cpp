/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "eval/nnue_gpu.h"
#include "metal/device.h"
#include "core/bitboard.h"
#include <iostream>
#include <fstream>
#include <cstring>

namespace MetalFish {
namespace NNUE {

// NNUEBuffers implementation
NNUEBuffers::~NNUEBuffers() {
    release();
}

void NNUEBuffers::release() {
    auto safe_release = [](MTL::Buffer*& buf) {
        if (buf) { buf->release(); buf = nullptr; }
    };
    
    safe_release(ft_weights);
    safe_release(ft_biases);
    safe_release(fc0_weights);
    safe_release(fc0_biases);
    safe_release(fc1_weights);
    safe_release(fc1_biases);
    safe_release(fc2_weights);
    safe_release(fc2_biases);
    safe_release(psqt_weights);
    safe_release(active_features);
    safe_release(feature_counts);
    safe_release(accumulators);
    safe_release(output);
}

// NNUEEvaluator implementation
NNUEEvaluator::NNUEEvaluator() 
    : buffers_(std::make_unique<NNUEBuffers>()) {
}

NNUEEvaluator::~NNUEEvaluator() {
    // Kernels are owned by the device/library, just nullify
    ft_kernel_ = nullptr;
    fc_kernel_ = nullptr;
    forward_kernel_ = nullptr;
    psqt_kernel_ = nullptr;
    
    if (library_) {
        library_->release();
        library_ = nullptr;
    }
    
    // command_queue_ and device_ are managed by Device singleton
}

bool NNUEEvaluator::init() {
    try {
        Metal::Device& metal_device = Metal::get_device();
        device_ = metal_device.mtl_device();
        command_queue_ = metal_device.get_queue();
        
        if (!device_ || !command_queue_) {
            std::cerr << "[NNUE] Failed to get Metal device" << std::endl;
            return false;
        }
        
        gpu_available_ = true;
        
        // Load kernels
        if (!load_kernels()) {
            std::cerr << "[NNUE] Failed to load Metal kernels" << std::endl;
            gpu_available_ = false;
            return false;
        }
        
        // Create buffers
        if (!create_buffers()) {
            std::cerr << "[NNUE] Failed to create Metal buffers" << std::endl;
            gpu_available_ = false;
            return false;
        }
        
        std::cout << "[NNUE] GPU acceleration initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[NNUE] Exception during init: " << e.what() << std::endl;
        gpu_available_ = false;
        return false;
    }
}

bool NNUEEvaluator::load_kernels() {
    if (!device_) return false;
    
    // Try to load the metallib
    NS::Error* error = nullptr;
    
    // First try loading from the build directory
    std::string metallib_path = "metalfish.metallib";
    auto path = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
    
    library_ = device_->newLibrary(path, &error);
    
    if (!library_) {
        // Try compiling from source
        std::string shader_source = R"(
#include <metal_stdlib>
using namespace metal;

constant int FEATURE_DIM_BIG = 1024;
constant int FC0_OUT = 15;
constant int FC1_OUT = 32;
constant int WEIGHT_SCALE_BITS = 6;
constant int OUTPUT_SCALE = 16;

typedef int16_t weight_t;
typedef int8_t  clipped_t;
typedef int32_t acc_t;

inline int8_t clipped_relu(int16_t x) {
    return int8_t(clamp(int(x) >> WEIGHT_SCALE_BITS, 0, 127));
}

inline int8_t sqr_clipped_relu(int16_t x) {
    int clamped = clamp(int(x) >> WEIGHT_SCALE_BITS, 0, 127);
    return int8_t((clamped * clamped) >> 7);
}

kernel void feature_transform(
    device const weight_t* weights [[buffer(0)]],
    device const int32_t* active_features [[buffer(1)]],
    device const int32_t* feature_counts [[buffer(2)]],
    device const weight_t* biases [[buffer(3)]],
    device acc_t* output [[buffer(4)]],
    constant int& hidden_dim [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    int pos_idx = tid.y;
    int hidden_idx = tid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim) return;
    
    acc_t acc = biases[hidden_idx];
    
    int feature_start = (pos_idx > 0) ? feature_counts[pos_idx - 1] : 0;
    int feature_end = feature_counts[pos_idx];
    
    for (int i = feature_start; i < feature_end; i++) {
        int feature_idx = active_features[i];
        acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
    
    output[pos_idx * hidden_dim + hidden_idx] = acc;
}

kernel void nnue_forward_batch(
    device const acc_t* transformed [[buffer(0)]],
    device const weight_t* fc0_weights [[buffer(1)]],
    device const weight_t* fc0_biases [[buffer(2)]],
    device const weight_t* fc1_weights [[buffer(3)]],
    device const weight_t* fc1_biases [[buffer(4)]],
    device const weight_t* fc2_weights [[buffer(5)]],
    device const weight_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant int& hidden_dim [[buffer(8)]],
    constant int& batch_size [[buffer(9)]],
    uint pos_idx [[thread_position_in_grid]])
{
    if (pos_idx >= (uint)batch_size) return;
    
    // Simple forward pass - each thread handles one position
    device const acc_t* white_acc = transformed + pos_idx * 2 * hidden_dim;
    device const acc_t* black_acc = white_acc + hidden_dim;
    
    // FC0 computation
    int8_t fc0_sqr[FC0_OUT * 2];
    int8_t fc0_clip[FC0_OUT + 1];
    
    for (int out = 0; out < FC0_OUT + 1; out++) {
        acc_t acc_w = fc0_biases[out];
        acc_t acc_b = fc0_biases[out];
        
        for (int i = 0; i < hidden_dim; i++) {
            int8_t w_val = clipped_relu(white_acc[i]);
            int8_t b_val = clipped_relu(black_acc[i]);
            acc_w += w_val * fc0_weights[i * (FC0_OUT + 1) + out];
            acc_b += b_val * fc0_weights[i * (FC0_OUT + 1) + out];
        }
        
        int16_t result_w = int16_t(acc_w >> WEIGHT_SCALE_BITS);
        int16_t result_b = int16_t(acc_b >> WEIGHT_SCALE_BITS);
        
        if (out < FC0_OUT) {
            fc0_sqr[out] = sqr_clipped_relu(result_w);
            fc0_sqr[FC0_OUT + out] = sqr_clipped_relu(result_b);
        }
        fc0_clip[out] = clipped_relu(result_w);
    }
    
    // FC1 computation
    int8_t fc1_out[FC1_OUT];
    for (int out = 0; out < FC1_OUT; out++) {
        acc_t acc = fc1_biases[out];
        
        for (int i = 0; i < FC0_OUT * 2; i++) {
            acc += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        
        fc1_out[out] = clipped_relu(int16_t(acc >> WEIGHT_SCALE_BITS));
    }
    
    // FC2 computation (output)
    acc_t final_acc = fc2_biases[0];
    for (int i = 0; i < FC1_OUT; i++) {
        final_acc += fc1_out[i] * fc2_weights[i];
    }
    
    // Add skip connection
    int32_t fwd_out = (fc0_clip[FC0_OUT] * (600 * OUTPUT_SCALE)) / 
                      (127 * (1 << WEIGHT_SCALE_BITS));
    
    output[pos_idx] = final_acc + fwd_out;
}
)";
        
        auto source = NS::String::string(shader_source.c_str(), NS::UTF8StringEncoding);
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
        
        library_ = device_->newLibrary(source, options, &error);
        options->release();
        
        if (!library_) {
            if (error) {
                std::cerr << "[NNUE] Shader compilation failed: " 
                          << error->localizedDescription()->utf8String() << std::endl;
            }
            return false;
        }
    }
    
    // Get kernel functions
    auto get_kernel = [this](const char* name) -> MTL::ComputePipelineState* {
        auto fn_name = NS::String::string(name, NS::UTF8StringEncoding);
        auto fn = library_->newFunction(fn_name);
        if (!fn) {
            std::cerr << "[NNUE] Function not found: " << name << std::endl;
            return nullptr;
        }
        
        NS::Error* error = nullptr;
        auto pipeline = device_->newComputePipelineState(fn, &error);
        fn->release();
        
        if (!pipeline && error) {
            std::cerr << "[NNUE] Pipeline creation failed for " << name << ": "
                      << error->localizedDescription()->utf8String() << std::endl;
        }
        return pipeline;
    };
    
    ft_kernel_ = get_kernel("feature_transform");
    forward_kernel_ = get_kernel("nnue_forward_batch");
    
    return ft_kernel_ != nullptr && forward_kernel_ != nullptr;
}

bool NNUEEvaluator::create_buffers() {
    if (!device_) return false;
    
    // Use shared storage mode for unified memory (zero-copy)
    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    
    // Feature transformer weights and biases
    size_t ft_weights_size = INPUT_FEATURES * hidden_dim_ * sizeof(weight_t);
    size_t ft_biases_size = hidden_dim_ * sizeof(weight_t);
    
    buffers_->ft_weights = device_->newBuffer(ft_weights_size, options);
    buffers_->ft_biases = device_->newBuffer(ft_biases_size, options);
    
    // FC layer weights and biases
    size_t fc0_weights_size = hidden_dim_ * (FC0_OUT + 1) * sizeof(weight_t);
    size_t fc0_biases_size = (FC0_OUT + 1) * sizeof(weight_t);
    
    buffers_->fc0_weights = device_->newBuffer(fc0_weights_size, options);
    buffers_->fc0_biases = device_->newBuffer(fc0_biases_size, options);
    
    size_t fc1_weights_size = FC0_OUT * 2 * FC1_OUT * sizeof(weight_t);
    size_t fc1_biases_size = FC1_OUT * sizeof(weight_t);
    
    buffers_->fc1_weights = device_->newBuffer(fc1_weights_size, options);
    buffers_->fc1_biases = device_->newBuffer(fc1_biases_size, options);
    
    size_t fc2_weights_size = FC1_OUT * sizeof(weight_t);
    size_t fc2_biases_size = sizeof(weight_t);
    
    buffers_->fc2_weights = device_->newBuffer(fc2_weights_size, options);
    buffers_->fc2_biases = device_->newBuffer(fc2_biases_size, options);
    
    // Dynamic buffers for batch processing
    size_t max_features = batch_size_ * MAX_ACTIVE_FEATURES * sizeof(int32_t);
    size_t feature_counts_size = batch_size_ * sizeof(int32_t);
    size_t acc_size = batch_size_ * 2 * hidden_dim_ * sizeof(acc_t);
    size_t output_size = batch_size_ * sizeof(int32_t);
    
    buffers_->active_features = device_->newBuffer(max_features, options);
    buffers_->feature_counts = device_->newBuffer(feature_counts_size, options);
    buffers_->accumulators = device_->newBuffer(acc_size, options);
    buffers_->output = device_->newBuffer(output_size, options);
    
    return buffers_->ft_weights && buffers_->fc0_weights && 
           buffers_->active_features && buffers_->output;
}

bool NNUEEvaluator::load_network(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[NNUE] Failed to open network file: " << path << std::endl;
        return false;
    }
    
    // Read network header and weights
    // This should match Stockfish's NNUE file format
    // For now, initialize with zeros (placeholder)
    
    // Initialize feature transformer with zeros
    if (buffers_->ft_weights) {
        memset(buffers_->ft_weights->contents(), 0, buffers_->ft_weights->length());
    }
    if (buffers_->ft_biases) {
        memset(buffers_->ft_biases->contents(), 0, buffers_->ft_biases->length());
    }
    
    // Initialize FC layers with zeros
    if (buffers_->fc0_weights) {
        memset(buffers_->fc0_weights->contents(), 0, buffers_->fc0_weights->length());
    }
    if (buffers_->fc0_biases) {
        memset(buffers_->fc0_biases->contents(), 0, buffers_->fc0_biases->length());
    }
    
    network_loaded_ = true;
    std::cout << "[NNUE] Network loaded (placeholder - needs real implementation)" << std::endl;
    
    return true;
}

void NNUEEvaluator::get_active_features(const Position& pos, Color perspective,
                                         std::vector<int>& features) {
    features.clear();
    
    // HalfKAv2_hm feature encoding
    // Each feature is encoded as: king_square * 641 + piece_square * 10 + piece_type
    
    Square ksq = pos.square<KING>(perspective);
    
    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
            Bitboard bb = pos.pieces(c, pt);
            while (bb) {
                Square sq = pop_lsb(bb);
                
                // Mirror for black perspective
                Square oriented_sq = (perspective == WHITE) ? sq : Square(sq ^ 56);
                Square oriented_ksq = (perspective == WHITE) ? ksq : Square(ksq ^ 56);
                
                // Feature index calculation (simplified)
                int piece_idx = (c == perspective) ? int(pt) : int(pt) + 6;
                int feature = int(oriented_ksq) * 641 + int(oriented_sq) * 12 + piece_idx;
                
                if (feature < INPUT_FEATURES) {
                    features.push_back(feature);
                }
            }
        }
    }
}

Value NNUEEvaluator::evaluate(const Position& pos) {
    if (!gpu_available_ || !network_loaded_) {
        return cpu_evaluate(pos);
    }
    
    // Single position evaluation using GPU
    std::vector<const Position*> positions = {&pos};
    std::vector<Value> results(1);
    evaluate_batch(positions, results);
    
    return results[0];
}

void NNUEEvaluator::evaluate_batch(const std::vector<const Position*>& positions,
                                    std::vector<Value>& results) {
    if (positions.empty()) return;
    
    int num_positions = static_cast<int>(positions.size());
    results.resize(num_positions);
    
    if (!gpu_available_ || !network_loaded_) {
        for (int i = 0; i < num_positions; i++) {
            results[i] = cpu_evaluate(*positions[i]);
        }
        return;
    }
    
    // Gather active features for all positions
    std::vector<int> all_features;
    std::vector<int> feature_counts;
    int total_features = 0;
    
    for (const Position* pos : positions) {
        std::vector<int> white_features, black_features;
        get_active_features(*pos, WHITE, white_features);
        get_active_features(*pos, BLACK, black_features);
        
        // Combine features
        for (int f : white_features) all_features.push_back(f);
        for (int f : black_features) all_features.push_back(f);
        
        total_features += white_features.size() + black_features.size();
        feature_counts.push_back(total_features);
    }
    
    // Copy features to GPU buffers
    memcpy(buffers_->active_features->contents(), 
           all_features.data(), 
           all_features.size() * sizeof(int));
    memcpy(buffers_->feature_counts->contents(),
           feature_counts.data(),
           feature_counts.size() * sizeof(int));
    
    // Create command buffer
    auto cmd_buffer = command_queue_->commandBuffer();
    auto encoder = cmd_buffer->computeCommandEncoder();
    
    // Dispatch feature transform
    encoder->setComputePipelineState(ft_kernel_);
    encoder->setBuffer(buffers_->ft_weights, 0, 0);
    encoder->setBuffer(buffers_->active_features, 0, 1);
    encoder->setBuffer(buffers_->feature_counts, 0, 2);
    encoder->setBuffer(buffers_->ft_biases, 0, 3);
    encoder->setBuffer(buffers_->accumulators, 0, 4);
    encoder->setBytes(&hidden_dim_, sizeof(int), 5);
    encoder->setBytes(&num_positions, sizeof(int), 6);
    
    MTL::Size ft_grid(hidden_dim_, num_positions, 1);
    MTL::Size ft_group(32, 1, 1);
    encoder->dispatchThreads(ft_grid, ft_group);
    
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    
    // Dispatch forward pass
    encoder->setComputePipelineState(forward_kernel_);
    encoder->setBuffer(buffers_->accumulators, 0, 0);
    encoder->setBuffer(buffers_->fc0_weights, 0, 1);
    encoder->setBuffer(buffers_->fc0_biases, 0, 2);
    encoder->setBuffer(buffers_->fc1_weights, 0, 3);
    encoder->setBuffer(buffers_->fc1_biases, 0, 4);
    encoder->setBuffer(buffers_->fc2_weights, 0, 5);
    encoder->setBuffer(buffers_->fc2_biases, 0, 6);
    encoder->setBuffer(buffers_->output, 0, 7);
    encoder->setBytes(&hidden_dim_, sizeof(int), 8);
    encoder->setBytes(&num_positions, sizeof(int), 9);
    
    MTL::Size fwd_grid(num_positions, 1, 1);
    MTL::Size fwd_group(1, 1, 1);
    encoder->dispatchThreads(fwd_grid, fwd_group);
    
    encoder->endEncoding();
    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();
    
    // Copy results back
    auto output_ptr = static_cast<int32_t*>(buffers_->output->contents());
    for (int i = 0; i < num_positions; i++) {
        // Convert raw score to Value
        results[i] = Value(output_ptr[i] / 16);  // Scale down
    }
}

Value NNUEEvaluator::cpu_evaluate(const Position& pos) {
    // Simple material evaluation as fallback
    Value score = VALUE_ZERO;
    
    for (Color c : {WHITE, BLACK}) {
        int sign = (c == WHITE) ? 1 : -1;
        score += sign * popcount(pos.pieces(c, PAWN)) * PawnValue;
        score += sign * popcount(pos.pieces(c, KNIGHT)) * KnightValue;
        score += sign * popcount(pos.pieces(c, BISHOP)) * BishopValue;
        score += sign * popcount(pos.pieces(c, ROOK)) * RookValue;
        score += sign * popcount(pos.pieces(c, QUEEN)) * QueenValue;
    }
    
    return pos.side_to_move() == WHITE ? score : -score;
}

// Global instance
static std::unique_ptr<NNUEEvaluator> g_gpu_nnue;

NNUEEvaluator& get_gpu_nnue() {
    if (!g_gpu_nnue) {
        g_gpu_nnue = std::make_unique<NNUEEvaluator>();
        g_gpu_nnue->init();
    }
    return *g_gpu_nnue;
}

} // namespace NNUE
} // namespace MetalFish


