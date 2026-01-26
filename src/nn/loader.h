/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file loader.h
 * @brief MetalFish source file.
 */

  Lc0 Network Loader - Protobuf Weight Loading
  
  Loads Lc0-format neural network weights from .pb and .pb.gz files.
  Supports BT4 (Big Transformer 4) architecture with 1024 embedding,
  15 layers, and 32 attention heads.
  
  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {

// Forward declaration for protobuf weights
struct Lc0Weights;

// Transformer architecture parameters
struct TransformerConfig {
  int embedding_size = 1024;    // Embedding dimension
  int num_layers = 15;          // Number of transformer layers
  int num_heads = 32;           // Number of attention heads
  int head_dim = 32;            // Dimension per head (embedding_size / num_heads)
  int mlp_size = 2816;          // MLP hidden size
  int policy_outputs = 1858;    // Standard chess policy outputs
  
  // Activation and normalization
  bool use_layer_norm = true;
  float layer_norm_eps = 1e-6f;
};

// Weights for a single transformer layer
struct TransformerLayerWeights {
  // Multi-head self-attention
  std::vector<float> qkv_weights;  // [embedding_size, 3 * embedding_size]
  std::vector<float> qkv_bias;     // [3 * embedding_size]
  std::vector<float> attn_out_weights; // [embedding_size, embedding_size]
  std::vector<float> attn_out_bias;    // [embedding_size]
  
  // Layer normalization 1
  std::vector<float> ln1_gamma;  // [embedding_size]
  std::vector<float> ln1_beta;   // [embedding_size]
  
  // Feed-forward network (MLP)
  std::vector<float> mlp_fc1_weights;  // [embedding_size, mlp_size]
  std::vector<float> mlp_fc1_bias;     // [mlp_size]
  std::vector<float> mlp_fc2_weights;  // [mlp_size, embedding_size]
  std::vector<float> mlp_fc2_bias;     // [embedding_size]
  
  // Layer normalization 2
  std::vector<float> ln2_gamma;  // [embedding_size]
  std::vector<float> ln2_beta;   // [embedding_size]
};

// Complete network weights
struct Lc0NetworkWeights {
  TransformerConfig config;
  
  // Input embedding (position encoder to transformer)
  std::vector<float> input_embedding_weights;  // [112 planes, embedding_size]
  std::vector<float> input_embedding_bias;     // [embedding_size]
  
  // Transformer encoder layers
  std::vector<TransformerLayerWeights> transformer_layers;
  
  // Policy head
  std::vector<float> policy_weights;  // [embedding_size, policy_outputs]
  std::vector<float> policy_bias;     // [policy_outputs]
  
  // Value head (Win-Draw-Loss)
  std::vector<float> value_weights;   // [embedding_size, 3]
  std::vector<float> value_bias;      // [3] for W/D/L
  
  // Moves-left head
  std::vector<float> mlh_weights;     // [embedding_size, 1]
  std::vector<float> mlh_bias;        // [1]
};

// Lc0 Network Loader
class Lc0NetworkLoader {
public:
  Lc0NetworkLoader() = default;
  ~Lc0NetworkLoader() = default;
  
  // Load network from .pb or .pb.gz file
  // Returns nullptr on failure
  std::unique_ptr<Lc0NetworkWeights> load(const std::string& path);
  
  // Get error message from last load attempt
  const std::string& get_error() const { return error_message_; }
  
private:
  std::string error_message_;
  
  // Helper methods
  bool load_protobuf(const std::string& path, Lc0Weights& weights);
  bool extract_weights(const Lc0Weights& pb_weights, Lc0NetworkWeights& net_weights);
  bool decompress_gzip(const std::string& gz_path, std::vector<uint8_t>& output);
};

} // namespace NN
} // namespace MetalFish