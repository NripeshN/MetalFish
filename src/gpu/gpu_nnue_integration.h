/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Integration Header
  
  This module provides full integration between the CPU NNUE implementation
  and GPU acceleration. It handles:
  - Weight extraction from CPU networks
  - GPU buffer management
  - Batch evaluation dispatch
  - Seamless CPU fallback
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <string>
#include <array>

#include "backend.h"
#include "core/types.h"

namespace MetalFish {
class Position;

namespace Eval::NNUE {
struct Networks;
template<typename Arch, typename Transformer> class Network;
}
}

namespace MetalFish::GPU {

// ============================================================================
// Constants (use from gpu_nnue.h if already defined)
// ============================================================================

#ifndef GPU_NNUE_CONSTANTS_DEFINED
#define GPU_NNUE_CONSTANTS_DEFINED

// Network architecture
constexpr int GPU_FT_DIM_BIG = 1024;
constexpr int GPU_FT_DIM_SMALL = 128;
constexpr int GPU_FC0_OUT = 15;
constexpr int GPU_FC1_OUT = 32;
constexpr int GPU_PSQT_BUCKETS = 8;
constexpr int GPU_LAYER_STACKS = 8;

// Feature dimensions
constexpr int GPU_HALFKA_DIMS = 45056;
constexpr int GPU_THREAT_DIMS = 1536;

// Batch processing
constexpr int GPU_MAX_BATCH_SIZE = 256;
constexpr int GPU_MAX_FEATURES = 32;

#endif // GPU_NNUE_CONSTANTS_DEFINED

// ============================================================================
// GPU Position Representation
// ============================================================================

struct alignas(16) GPUPositionData {
    uint64_t pieces[2][7];  // [color][piece_type] bitboards
    uint8_t king_sq[2];     // King squares
    uint8_t stm;            // Side to move
    uint8_t piece_count;    // Total pieces for bucket selection
    uint8_t padding[4];
    
    void from_position(const Position& pos);
};

// ============================================================================
// Feature Update for Incremental Computation
// ============================================================================

struct GPUFeatureUpdate {
    std::array<int32_t, 32> added_features;
    std::array<int32_t, 32> removed_features;
    uint8_t num_added = 0;
    uint8_t num_removed = 0;
    uint8_t perspective = 0;
    uint8_t padding = 0;
};

// ============================================================================
// GPU Network Weights
// ============================================================================

struct GPULayerWeights {
    std::unique_ptr<Buffer> fc0_weights;  // [hidden_dim * (FC0_OUT+1)]
    std::unique_ptr<Buffer> fc0_biases;   // [FC0_OUT+1]
    std::unique_ptr<Buffer> fc1_weights;  // [FC0_OUT*2 * FC1_OUT]
    std::unique_ptr<Buffer> fc1_biases;   // [FC1_OUT]
    std::unique_ptr<Buffer> fc2_weights;  // [FC1_OUT]
    std::unique_ptr<Buffer> fc2_biases;   // [1]
    
    bool valid() const {
        return fc0_weights && fc0_biases && fc1_weights && 
               fc1_biases && fc2_weights && fc2_biases;
    }
};

struct GPUNetworkData {
    // Feature transformer
    std::unique_ptr<Buffer> ft_weights;   // [HALFKA_DIMS * hidden_dim]
    std::unique_ptr<Buffer> ft_biases;    // [hidden_dim]
    std::unique_ptr<Buffer> ft_psqt;      // [HALFKA_DIMS * PSQT_BUCKETS]
    
    // Threat weights (big network only)
    std::unique_ptr<Buffer> threat_weights;
    std::unique_ptr<Buffer> threat_psqt;
    
    // Per-bucket FC layers
    std::array<GPULayerWeights, GPU_LAYER_STACKS> layers;
    
    int hidden_dim = 0;
    bool has_threats = false;
    bool valid = false;
    
    size_t memory_usage() const;
};

// ============================================================================
// GPU Evaluation Batch
// ============================================================================

struct GPUEvalBatch {
    // Position data
    std::vector<GPUPositionData> positions;
    
    // Extracted features
    std::vector<int32_t> white_features;
    std::vector<int32_t> black_features;
    std::vector<uint32_t> feature_counts;
    std::vector<uint32_t> feature_offsets;
    
    // Bucket indices
    std::vector<int32_t> buckets;
    
    // Results
    std::vector<int32_t> psqt_scores;
    std::vector<int32_t> positional_scores;
    
    int count = 0;
    
    void clear();
    void reserve(int n);
    void add_position(const Position& pos);
    int get_bucket(int idx) const { return buckets[idx]; }
};

// ============================================================================
// GPU NNUE Manager
// ============================================================================

class GPUNNUEManager {
public:
    GPUNNUEManager();
    ~GPUNNUEManager();
    
    // Initialization
    bool initialize();
    bool load_networks(const Eval::NNUE::Networks& networks);
    bool is_ready() const { return initialized_ && (big_network_.valid || small_network_.valid); }
    
    // Batch evaluation
    bool evaluate_batch(GPUEvalBatch& batch, bool use_big_network = true);
    
    // Single position (falls back to CPU if batch size is 1)
    std::pair<int32_t, int32_t> evaluate_single(const Position& pos, bool use_big = true);
    
    // Configuration
    int min_batch_size() const { return min_batch_size_; }
    void set_min_batch_size(int size) { min_batch_size_ = size; }
    
    // Statistics
    size_t gpu_evaluations() const { return gpu_evals_; }
    size_t cpu_fallback_evaluations() const { return cpu_evals_; }
    size_t total_batches() const { return batch_count_; }
    double avg_batch_time_ms() const;
    double total_gpu_time_ms() const { return total_time_ms_; }
    void reset_stats();
    
    // Memory
    size_t gpu_memory_used() const;
    
    // Status
    std::string status_string() const;
    
private:
    bool initialized_ = false;
    int min_batch_size_ = 4;
    
    // Network weights
    GPUNetworkData big_network_;
    GPUNetworkData small_network_;
    
    // Compute kernels
    std::unique_ptr<ComputeKernel> extract_features_kernel_;
    std::unique_ptr<ComputeKernel> feature_transform_kernel_;
    std::unique_ptr<ComputeKernel> feature_transform_simd_kernel_;
    std::unique_ptr<ComputeKernel> psqt_kernel_;
    std::unique_ptr<ComputeKernel> forward_fused_kernel_;
    
    // Working buffers
    std::unique_ptr<Buffer> positions_buffer_;
    std::unique_ptr<Buffer> white_features_buffer_;
    std::unique_ptr<Buffer> black_features_buffer_;
    std::unique_ptr<Buffer> feature_counts_buffer_;
    std::unique_ptr<Buffer> feature_offsets_buffer_;
    std::unique_ptr<Buffer> accumulators_buffer_;
    std::unique_ptr<Buffer> psqt_buffer_;
    std::unique_ptr<Buffer> output_buffer_;
    
    // Statistics
    std::atomic<size_t> gpu_evals_{0};
    std::atomic<size_t> cpu_evals_{0};
    std::atomic<size_t> batch_count_{0};
    double total_time_ms_ = 0;
    
    // Internal methods
    bool compile_shaders();
    bool allocate_working_buffers();
    bool allocate_network_buffers(GPUNetworkData& net, int hidden_dim, bool has_threats);
    
    template<typename Network>
    bool extract_weights(const Network& network, GPUNetworkData& gpu_net, 
                        int hidden_dim, bool has_threats);
    
    bool dispatch_feature_transform(const GPUNetworkData& net, int batch_size);
    bool dispatch_forward_pass(const GPUNetworkData& net, int batch_size, int bucket);
};

// ============================================================================
// Global Interface
// ============================================================================

// Get the global GPU NNUE manager
GPUNNUEManager& gpu_nnue_manager();

// Initialize GPU NNUE with networks
bool initialize_gpu_nnue(const Eval::NNUE::Networks& networks);

// Check if GPU NNUE manager is available
bool gpu_nnue_manager_available();

// Evaluate a batch of positions
bool gpu_evaluate_batch(GPUEvalBatch& batch, bool use_big = true);

} // namespace MetalFish::GPU
