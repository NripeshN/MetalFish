/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Kernel dispatch and management for Metal compute shaders.
*/

#include "metal/device.h"
#include "metal/allocator.h"
#include <vector>

namespace MetalFish {
namespace Metal {

/**
 * NNUEKernels manages the NNUE neural network evaluation on GPU.
 */
class NNUEKernels {
public:
    NNUEKernels() : device_(device()) {
        // Create command queue for NNUE operations
        device_.new_queue(NNUE_QUEUE_INDEX);
        
        // Load kernels from metallib
        load_kernels();
    }
    
    ~NNUEKernels() = default;
    
    /**
     * Evaluate multiple positions in a batch on GPU
     * @param accumulators Pre-computed feature accumulators [batch_size x 2 x hidden_dim]
     * @param batch_size Number of positions to evaluate
     * @return Vector of evaluation scores
     */
    std::vector<int32_t> evaluate_batch(
        const std::vector<int16_t>& accumulators,
        size_t batch_size,
        size_t hidden_dim) {
        
        std::vector<int32_t> results(batch_size);
        
        // Allocate GPU buffers
        Buffer acc_buffer = MetalAllocator::instance().allocate(
            accumulators.size() * sizeof(int16_t));
        Buffer out_buffer = MetalAllocator::instance().allocate(
            batch_size * sizeof(int32_t));
        
        // Copy accumulator data (unified memory - no explicit copy needed)
        std::memcpy(acc_buffer.contents(), accumulators.data(), 
                    accumulators.size() * sizeof(int16_t));
        
        // Dispatch kernel
        if (nnue_forward_kernel_) {
            auto& encoder = device_.get_command_encoder(NNUE_QUEUE_INDEX);
            
            encoder.set_compute_pipeline_state(nnue_forward_kernel_);
            encoder.set_buffer(acc_buffer.ptr, 0);
            encoder.set_buffer(fc0_weights_.ptr, 1);
            encoder.set_buffer(fc0_biases_.ptr, 2);
            encoder.set_buffer(fc1_weights_.ptr, 3);
            encoder.set_buffer(fc1_biases_.ptr, 4);
            encoder.set_buffer(fc2_weights_.ptr, 5);
            encoder.set_buffer(fc2_biases_.ptr, 6);
            encoder.set_buffer(out_buffer.ptr, 7);
            encoder.set_bytes(static_cast<int>(hidden_dim), 8);
            encoder.set_bytes(static_cast<int>(batch_size), 9);
            
            // Dispatch with one thread per position
            MTL::Size grid_size = MTL::Size::Make(batch_size, 1, 1);
            MTL::Size threadgroup_size = MTL::Size::Make(
                std::min(size_t(32), batch_size), 1, 1);
            
            encoder.dispatch_threadgroups(
                MTL::Size::Make((batch_size + 31) / 32, 1, 1),
                threadgroup_size);
            
            device_.end_encoding(NNUE_QUEUE_INDEX);
            device_.commit_command_buffer(NNUE_QUEUE_INDEX);
            
            // Wait for completion (in real implementation, use async)
            // Copy results back (unified memory - direct access)
            std::memcpy(results.data(), out_buffer.contents(), 
                        batch_size * sizeof(int32_t));
        }
        
        // Free buffers
        MetalAllocator::instance().free(acc_buffer);
        MetalAllocator::instance().free(out_buffer);
        
        return results;
    }
    
    /**
     * Load NNUE network weights from file
     */
    bool load_network(const std::string& path) {
        // TODO: Implement network loading
        // For now, allocate empty buffers
        
        const size_t hidden_dim = 1024;
        const size_t fc0_out = 16;
        const size_t fc1_out = 32;
        
        fc0_weights_ = MetalAllocator::instance().allocate(hidden_dim * fc0_out * sizeof(int16_t));
        fc0_biases_ = MetalAllocator::instance().allocate(fc0_out * sizeof(int16_t));
        fc1_weights_ = MetalAllocator::instance().allocate(fc0_out * 2 * fc1_out * sizeof(int16_t));
        fc1_biases_ = MetalAllocator::instance().allocate(fc1_out * sizeof(int16_t));
        fc2_weights_ = MetalAllocator::instance().allocate(fc1_out * sizeof(int16_t));
        fc2_biases_ = MetalAllocator::instance().allocate(sizeof(int16_t));
        
        network_loaded_ = true;
        return true;
    }
    
    bool is_network_loaded() const { return network_loaded_; }

private:
    static constexpr int NNUE_QUEUE_INDEX = 0;
    
    Device& device_;
    bool network_loaded_ = false;
    
    // Kernel pipeline states
    MTL::ComputePipelineState* nnue_forward_kernel_ = nullptr;
    MTL::ComputePipelineState* feature_transform_kernel_ = nullptr;
    MTL::ComputePipelineState* incremental_update_kernel_ = nullptr;
    
    // Network weight buffers
    Buffer fc0_weights_;
    Buffer fc0_biases_;
    Buffer fc1_weights_;
    Buffer fc1_biases_;
    Buffer fc2_weights_;
    Buffer fc2_biases_;
    
    void load_kernels() {
        try {
            nnue_forward_kernel_ = device_.get_kernel("nnue_forward_batch");
            feature_transform_kernel_ = device_.get_kernel("feature_transform");
            incremental_update_kernel_ = device_.get_kernel("incremental_update");
        } catch (const std::exception& e) {
            // Kernels not available - will fall back to CPU
        }
    }
};

/**
 * EvalKernels manages static evaluation on GPU.
 */
class EvalKernels {
public:
    EvalKernels() : device_(device()) {
        device_.new_queue(EVAL_QUEUE_INDEX);
        load_kernels();
    }
    
    /**
     * Evaluate material for a batch of positions
     */
    std::vector<int32_t> evaluate_material(
        const std::vector<uint8_t>& boards,
        size_t batch_size) {
        
        std::vector<int32_t> results(batch_size);
        
        if (material_kernel_) {
            Buffer board_buffer = MetalAllocator::instance().allocate(boards.size());
            Buffer out_buffer = MetalAllocator::instance().allocate(batch_size * sizeof(int32_t));
            
            std::memcpy(board_buffer.contents(), boards.data(), boards.size());
            
            auto& encoder = device_.get_command_encoder(EVAL_QUEUE_INDEX);
            encoder.set_compute_pipeline_state(material_kernel_);
            encoder.set_buffer(board_buffer.ptr, 0);
            encoder.set_buffer(out_buffer.ptr, 1);
            encoder.set_bytes(static_cast<int>(batch_size), 2);
            
            encoder.dispatch_threads(
                MTL::Size::Make(batch_size, 1, 1),
                MTL::Size::Make(std::min(size_t(256), batch_size), 1, 1));
            
            device_.end_encoding(EVAL_QUEUE_INDEX);
            device_.commit_command_buffer(EVAL_QUEUE_INDEX);
            
            std::memcpy(results.data(), out_buffer.contents(), batch_size * sizeof(int32_t));
            
            MetalAllocator::instance().free(board_buffer);
            MetalAllocator::instance().free(out_buffer);
        }
        
        return results;
    }

private:
    static constexpr int EVAL_QUEUE_INDEX = 1;
    
    Device& device_;
    
    MTL::ComputePipelineState* material_kernel_ = nullptr;
    MTL::ComputePipelineState* psqt_kernel_ = nullptr;
    MTL::ComputePipelineState* combine_eval_kernel_ = nullptr;
    
    void load_kernels() {
        try {
            material_kernel_ = device_.get_kernel("material_eval");
            psqt_kernel_ = device_.get_kernel("psqt_eval");
            combine_eval_kernel_ = device_.get_kernel("combine_eval");
        } catch (const std::exception& e) {
            // Kernels not available
        }
    }
};

/**
 * MoveGenKernels manages move generation on GPU.
 */
class MoveGenKernels {
public:
    MoveGenKernels() : device_(device()) {
        device_.new_queue(MOVEGEN_QUEUE_INDEX);
        load_kernels();
    }

private:
    static constexpr int MOVEGEN_QUEUE_INDEX = 2;
    
    Device& device_;
    
    MTL::ComputePipelineState* count_moves_kernel_ = nullptr;
    MTL::ComputePipelineState* generate_moves_kernel_ = nullptr;
    MTL::ComputePipelineState* check_legality_kernel_ = nullptr;
    
    void load_kernels() {
        try {
            count_moves_kernel_ = device_.get_kernel("count_moves");
            generate_moves_kernel_ = device_.get_kernel("generate_moves");
            check_legality_kernel_ = device_.get_kernel("check_legality");
        } catch (const std::exception& e) {
            // Kernels not available
        }
    }
};

// Global kernel managers
static std::unique_ptr<NNUEKernels> g_nnue_kernels;
static std::unique_ptr<EvalKernels> g_eval_kernels;
static std::unique_ptr<MoveGenKernels> g_movegen_kernels;

void init_kernels() {
    g_nnue_kernels = std::make_unique<NNUEKernels>();
    g_eval_kernels = std::make_unique<EvalKernels>();
    g_movegen_kernels = std::make_unique<MoveGenKernels>();
}

NNUEKernels& nnue_kernels() {
    if (!g_nnue_kernels) {
        g_nnue_kernels = std::make_unique<NNUEKernels>();
    }
    return *g_nnue_kernels;
}

EvalKernels& eval_kernels() {
    if (!g_eval_kernels) {
        g_eval_kernels = std::make_unique<EvalKernels>();
    }
    return *g_eval_kernels;
}

} // namespace Metal
} // namespace MetalFish

