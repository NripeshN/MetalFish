/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "eval/gpu_nnue.h"
#include "eval/nnue_loader.h"
#include "metal/device.h"
#include "core/bitboard.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <chrono>

namespace MetalFish {
namespace Eval {

// ============================================================================
// GPUNNUEWeights
// ============================================================================

GPUNNUEWeights::~GPUNNUEWeights() {
    if (ft_weights) ft_weights->release();
    if (ft_biases) ft_biases->release();
    if (psqt_weights) psqt_weights->release();
    if (fc0_weights) fc0_weights->release();
    if (fc0_biases) fc0_biases->release();
    if (fc1_weights) fc1_weights->release();
    if (fc1_biases) fc1_biases->release();
    if (fc2_weights) fc2_weights->release();
    if (fc2_bias) fc2_bias->release();
}

bool GPUNNUEWeights::load(MTL::Device* device, const std::string& path) {
    MTL::ResourceOptions options = MTL::ResourceStorageModeShared;
    
    // Try to load NNUE weights from file
    NNUE::NNUEWeights nnue;
    bool has_weights = false;
    
    if (!path.empty()) {
        has_weights = NNUE::load_network(path, nnue);
    }
    
    // Determine dimensions
    int half_dims = has_weights ? nnue.half_dimensions : FT_OUT_DIMS;
    
    // Allocate buffers
    ft_weights = device->newBuffer(FT_IN_DIMS * half_dims * sizeof(int16_t), options);
    ft_biases = device->newBuffer(half_dims * sizeof(int16_t), options);
    psqt_weights = device->newBuffer(FT_IN_DIMS * PSQT_BUCKETS * sizeof(int16_t), options);
    
    fc0_weights = device->newBuffer(half_dims * 2 * FC0_OUT * sizeof(int8_t), options);
    fc0_biases = device->newBuffer(FC0_OUT * sizeof(int32_t), options);
    fc1_weights = device->newBuffer(30 * FC1_OUT * sizeof(int8_t), options);
    fc1_biases = device->newBuffer(FC1_OUT * sizeof(int32_t), options);
    fc2_weights = device->newBuffer(FC1_OUT * sizeof(int8_t), options);
    fc2_bias = device->newBuffer(sizeof(int32_t), options);
    
    if (has_weights) {
        // Copy loaded weights to GPU buffers
        memcpy(ft_weights->contents(), nnue.ft_weights.data(), 
               nnue.ft_weights.size() * sizeof(int16_t));
        memcpy(ft_biases->contents(), nnue.ft_biases.data(),
               nnue.ft_biases.size() * sizeof(int16_t));
        memcpy(psqt_weights->contents(), nnue.psqt_weights.data(),
               std::min(nnue.psqt_weights.size() * sizeof(int16_t), psqt_weights->length()));
        
        // Copy first layer stack weights (stack 0 for now)
        size_t fc0_size = std::min(nnue.fc0_weights.size() * sizeof(int8_t), fc0_weights->length());
        memcpy(fc0_weights->contents(), nnue.fc0_weights.data(), fc0_size);
        
        size_t fc0_bias_size = std::min(nnue.fc0_biases.size() * sizeof(int32_t), fc0_biases->length());
        memcpy(fc0_biases->contents(), nnue.fc0_biases.data(), fc0_bias_size);
        
        size_t fc1_size = std::min(nnue.fc1_weights.size() * sizeof(int8_t), fc1_weights->length());
        memcpy(fc1_weights->contents(), nnue.fc1_weights.data(), fc1_size);
        
        size_t fc1_bias_size = std::min(nnue.fc1_biases.size() * sizeof(int32_t), fc1_biases->length());
        memcpy(fc1_biases->contents(), nnue.fc1_biases.data(), fc1_bias_size);
        
        size_t fc2_size = std::min(nnue.fc2_weights.size() * sizeof(int8_t), fc2_weights->length());
        memcpy(fc2_weights->contents(), nnue.fc2_weights.data(), fc2_size);
        
        memcpy(fc2_bias->contents(), nnue.fc2_biases.data(), sizeof(int32_t));
        
        std::cout << "[GPU_NNUE] Loaded weights from: " << path << std::endl;
    } else {
        // Initialize with zeros (placeholder for classical eval)
        memset(ft_weights->contents(), 0, ft_weights->length());
        memset(ft_biases->contents(), 0, ft_biases->length());
        memset(psqt_weights->contents(), 0, psqt_weights->length());
        memset(fc0_weights->contents(), 0, fc0_weights->length());
        memset(fc0_biases->contents(), 0, fc0_biases->length());
        memset(fc1_weights->contents(), 0, fc1_weights->length());
        memset(fc1_biases->contents(), 0, fc1_biases->length());
        memset(fc2_weights->contents(), 0, fc2_weights->length());
        memset(fc2_bias->contents(), 0, fc2_bias->length());
    }
    
    loaded_ = true;
    return true;
}

// ============================================================================
// GPUNNUEEvaluator
// ============================================================================

GPUNNUEEvaluator::GPUNNUEEvaluator()
    : weights_(std::make_unique<GPUNNUEWeights>()) {}

GPUNNUEEvaluator::~GPUNNUEEvaluator() {
    if (acc_buffer_) acc_buffer_->release();
    if (features_buffer_) features_buffer_->release();
    if (output_buffer_) output_buffer_->release();
    if (scratch_buffer_) scratch_buffer_->release();
    
    if (ft_kernel_) ft_kernel_->release();
    if (ft_update_kernel_) ft_update_kernel_->release();
    if (forward_kernel_) forward_kernel_->release();
    if (library_) library_->release();
}

bool GPUNNUEEvaluator::init() {
    try {
        Metal::Device& metal = Metal::get_device();
        device_ = metal.mtl_device();
        queue_ = metal.get_queue();
        
        if (!device_ || !queue_) {
            std::cerr << "[GPU_NNUE] Failed to get Metal device" << std::endl;
            return false;
        }
        
        // Compile shaders
        std::string shader_source = R"(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant int FT_OUT_DIMS = 1024;
constant int FC0_OUT = 16;
constant int FC1_OUT = 32;
constant int WEIGHT_SCALE = 6;

inline int8_t clipped_relu(int16_t x) {
    return int8_t(clamp(int(x) >> WEIGHT_SCALE, 0, 127));
}

inline int8_t sqr_clipped_relu(int16_t x) {
    int v = clamp(int(x) >> WEIGHT_SCALE, 0, 127);
    return int8_t((v * v) >> 7);
}

// Feature transformer kernel
kernel void feature_transform(
    device const int16_t* weights [[buffer(0)]],
    device const int16_t* biases [[buffer(1)]],
    device const int* features [[buffer(2)]],
    device int32_t* acc [[buffer(3)]],
    constant int& num_features [[buffer(4)]],
    constant int& ft_dims [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int h = gid;
    if (h >= ft_dims) return;
    
    int32_t sum = biases[h];
    for (int i = 0; i < num_features; i++) {
        int f = features[i];
        sum += weights[f * ft_dims + h];
    }
    acc[h] = sum;
}

// Forward pass kernel
kernel void nnue_forward(
    device const int32_t* acc_white [[buffer(0)]],
    device const int32_t* acc_black [[buffer(1)]],
    device const int8_t* fc0_w [[buffer(2)]],
    device const int32_t* fc0_b [[buffer(3)]],
    device const int8_t* fc1_w [[buffer(4)]],
    device const int32_t* fc1_b [[buffer(5)]],
    device const int8_t* fc2_w [[buffer(6)]],
    device const int32_t* fc2_b [[buffer(7)]],
    device int32_t* output [[buffer(8)]],
    constant int& ft_dims [[buffer(9)]],
    constant int& stm [[buffer(10)]],
    uint lid [[thread_position_in_threadgroup]])
{
    device const int32_t* acc_us = stm == 0 ? acc_white : acc_black;
    device const int32_t* acc_them = stm == 0 ? acc_black : acc_white;
    
    threadgroup int8_t fc0_sqr[15];
    threadgroup int8_t fc0_clip[15];
    threadgroup int16_t skip;
    threadgroup int8_t fc1_out[32];
    
    // FC0
    if (lid < FC0_OUT) {
        int32_t sum = fc0_b[lid];
        for (int i = 0; i < ft_dims; i++) {
            sum += clipped_relu(acc_us[i]) * fc0_w[i * FC0_OUT + lid];
            sum += clipped_relu(acc_them[i]) * fc0_w[(ft_dims + i) * FC0_OUT + lid];
        }
        int16_t r = int16_t(sum >> WEIGHT_SCALE);
        if (lid < 15) {
            fc0_sqr[lid] = sqr_clipped_relu(r);
            fc0_clip[lid] = clipped_relu(r);
        } else {
            skip = r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC1
    if (lid < FC1_OUT) {
        int32_t sum = fc1_b[lid];
        for (int i = 0; i < 15; i++) {
            sum += fc0_sqr[i] * fc1_w[i * FC1_OUT + lid];
        }
        fc1_out[lid] = clipped_relu(int16_t(sum >> WEIGHT_SCALE));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2
    if (lid == 0) {
        int32_t sum = *fc2_b;
        for (int i = 0; i < FC1_OUT; i++) {
            sum += fc1_out[i] * fc2_w[i];
        }
        int32_t fwd = skip * (600 * 16) / (127 * (1 << WEIGHT_SCALE));
        *output = sum + fwd;
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
                std::cerr << "[GPU_NNUE] Shader error: " 
                          << error->localizedDescription()->utf8String() << std::endl;
            }
            return false;
        }
        
        // Get kernels
        auto get_kernel = [this](const char* name) -> MTL::ComputePipelineState* {
            auto fn = library_->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
            if (!fn) return nullptr;
            NS::Error* err = nullptr;
            auto pso = device_->newComputePipelineState(fn, &err);
            fn->release();
            return pso;
        };
        
        ft_kernel_ = get_kernel("feature_transform");
        forward_kernel_ = get_kernel("nnue_forward");
        
        if (!ft_kernel_ || !forward_kernel_) {
            std::cerr << "[GPU_NNUE] Failed to create kernels" << std::endl;
            return false;
        }
        
        // Allocate working buffers
        MTL::ResourceOptions buf_opts = MTL::ResourceStorageModeShared;
        acc_buffer_ = device_->newBuffer(2 * FT_OUT_DIMS * sizeof(int32_t), buf_opts);
        features_buffer_ = device_->newBuffer(64 * sizeof(int), buf_opts);  // Max 64 features
        output_buffer_ = device_->newBuffer(sizeof(int32_t), buf_opts);
        scratch_buffer_ = device_->newBuffer(1024, buf_opts);
        
        // Load placeholder weights
        weights_->load(device_, "");
        
        ready_ = true;
        std::cout << "[GPU_NNUE] Initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[GPU_NNUE] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool GPUNNUEEvaluator::load_network(const std::string& path) {
    return weights_->load(device_, path);
}

void GPUNNUEEvaluator::get_features(const Position& pos, Color perspective,
                                     std::vector<int>& features) {
    features.clear();
    
    Square ksq = pos.square<KING>(perspective);
    
    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
            Bitboard bb = pos.pieces(c, pt);
            while (bb) {
                Square sq = pop_lsb(bb);
                
                // Mirror for black perspective
                Square oriented_sq = (perspective == WHITE) ? sq : Square(sq ^ 56);
                Square oriented_ksq = (perspective == WHITE) ? ksq : Square(ksq ^ 56);
                
                // HalfKAv2_hm index
                int piece_idx = (c == perspective) ? int(pt) - 1 : int(pt) + 5;
                int feature = int(oriented_ksq) * 641 + int(oriented_sq) * 10 + piece_idx;
                
                if (feature < FT_IN_DIMS) {
                    features.push_back(feature);
                }
            }
        }
    }
}

void GPUNNUEEvaluator::compute_accumulator(const Position& pos, GPUAccumulator& acc) {
    if (!ready_) {
        acc.computed = false;
        return;
    }
    
    // Get features for both perspectives
    std::vector<int> white_features, black_features;
    get_features(pos, WHITE, white_features);
    get_features(pos, BLACK, black_features);
    
    auto cmd = queue_->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    
    enc->setComputePipelineState(ft_kernel_);
    
    int ft_dims = FT_OUT_DIMS;
    
    // White accumulator
    int* feat_ptr = static_cast<int*>(features_buffer_->contents());
    memcpy(feat_ptr, white_features.data(), white_features.size() * sizeof(int));
    
    int32_t* acc_ptr = static_cast<int32_t*>(acc_buffer_->contents());
    
    enc->setBuffer(weights_->ft_weights, 0, 0);
    enc->setBuffer(weights_->ft_biases, 0, 1);
    enc->setBuffer(features_buffer_, 0, 2);
    enc->setBuffer(acc_buffer_, 0, 3);
    
    int num_feat = static_cast<int>(white_features.size());
    enc->setBytes(&num_feat, sizeof(int), 4);
    enc->setBytes(&ft_dims, sizeof(int), 5);
    
    enc->dispatchThreads(MTL::Size(FT_OUT_DIMS, 1, 1), MTL::Size(256, 1, 1));
    
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    // Copy to accumulator
    memcpy(acc.white.data(), acc_ptr, FT_OUT_DIMS * sizeof(int32_t));
    
    // Black accumulator
    memcpy(feat_ptr, black_features.data(), black_features.size() * sizeof(int));
    
    cmd = queue_->commandBuffer();
    enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(ft_kernel_);
    
    enc->setBuffer(weights_->ft_weights, 0, 0);
    enc->setBuffer(weights_->ft_biases, 0, 1);
    enc->setBuffer(features_buffer_, 0, 2);
    enc->setBuffer(acc_buffer_, FT_OUT_DIMS * sizeof(int32_t), 3);
    
    num_feat = static_cast<int>(black_features.size());
    enc->setBytes(&num_feat, sizeof(int), 4);
    enc->setBytes(&ft_dims, sizeof(int), 5);
    
    enc->dispatchThreads(MTL::Size(FT_OUT_DIMS, 1, 1), MTL::Size(256, 1, 1));
    
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    memcpy(acc.black.data(), acc_ptr + FT_OUT_DIMS, FT_OUT_DIMS * sizeof(int32_t));
    
    acc.computed = true;
}

Value GPUNNUEEvaluator::forward_pass(const GPUAccumulator& acc, Color stm) {
    if (!ready_ || !acc.computed) {
        return VALUE_ZERO;
    }
    
    // Copy accumulator to GPU buffer
    int32_t* buf = static_cast<int32_t*>(acc_buffer_->contents());
    memcpy(buf, acc.white.data(), FT_OUT_DIMS * sizeof(int32_t));
    memcpy(buf + FT_OUT_DIMS, acc.black.data(), FT_OUT_DIMS * sizeof(int32_t));
    
    auto cmd = queue_->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    
    enc->setComputePipelineState(forward_kernel_);
    
    enc->setBuffer(acc_buffer_, 0, 0);
    enc->setBuffer(acc_buffer_, FT_OUT_DIMS * sizeof(int32_t), 1);
    enc->setBuffer(weights_->fc0_weights, 0, 2);
    enc->setBuffer(weights_->fc0_biases, 0, 3);
    enc->setBuffer(weights_->fc1_weights, 0, 4);
    enc->setBuffer(weights_->fc1_biases, 0, 5);
    enc->setBuffer(weights_->fc2_weights, 0, 6);
    enc->setBuffer(weights_->fc2_bias, 0, 7);
    enc->setBuffer(output_buffer_, 0, 8);
    
    int ft_dims = FT_OUT_DIMS;
    int stm_int = (stm == WHITE) ? 0 : 1;
    enc->setBytes(&ft_dims, sizeof(int), 9);
    enc->setBytes(&stm_int, sizeof(int), 10);
    
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(64, 1, 1));
    
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    int32_t* out = static_cast<int32_t*>(output_buffer_->contents());
    
    evaluations_++;
    return Value(*out / 16);  // Scale down
}

Value GPUNNUEEvaluator::evaluate(const Position& pos, GPUAccumulator* acc) {
    if (!ready_) {
        return cpu_evaluate(pos);
    }
    
    GPUAccumulator local_acc;
    GPUAccumulator& use_acc = acc ? *acc : local_acc;
    
    if (!use_acc.computed) {
        compute_accumulator(pos, use_acc);
    }
    
    return forward_pass(use_acc, pos.side_to_move());
}

Value GPUNNUEEvaluator::cpu_evaluate(const Position& pos) {
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
static std::unique_ptr<GPUNNUEEvaluator> g_gpu_nnue;

GPUNNUEEvaluator& gpu_nnue() {
    if (!g_gpu_nnue) {
        g_gpu_nnue = std::make_unique<GPUNNUEEvaluator>();
        g_gpu_nnue->init();
    }
    return *g_gpu_nnue;
}

} // namespace Eval
} // namespace MetalFish


