/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Integration Implementation
*/

#include "gpu_nnue_integration.h"

#ifdef USE_METAL

#include "backend.h"
#include "core/bitboard.h"
#include "core/position.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_architecture.h"
#include "eval/nnue/nnue_feature_transformer.h"
#include "nnue_weight_accessor.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace MetalFish::GPU {

// ============================================================================
// GPUPositionData Implementation
// ============================================================================

void GPUPositionData::from_position(const Position &pos) {
  // Clear
  std::memset(this, 0, sizeof(GPUPositionData));

  // Copy piece bitboards
  for (int c = 0; c < 2; c++) {
    for (int pt = 0; pt <= 6; pt++) {
      pieces[c][pt] = pos.pieces(Color(c), PieceType(pt));
    }
  }

  // King squares
  king_sq[0] = pos.square<KING>(WHITE);
  king_sq[1] = pos.square<KING>(BLACK);

  // Side to move
  stm = pos.side_to_move();

  // Piece count
  piece_count = pos.count<ALL_PIECES>();
}

// ============================================================================
// GPUNetworkData Implementation
// ============================================================================

size_t GPUNetworkData::memory_usage() const {
  size_t total = 0;
  if (ft_weights)
    total += ft_weights->size();
  if (ft_biases)
    total += ft_biases->size();
  if (ft_psqt)
    total += ft_psqt->size();
  if (threat_weights)
    total += threat_weights->size();
  if (threat_psqt)
    total += threat_psqt->size();

  for (const auto &layer : layers) {
    if (layer.fc0_weights)
      total += layer.fc0_weights->size();
    if (layer.fc0_biases)
      total += layer.fc0_biases->size();
    if (layer.fc1_weights)
      total += layer.fc1_weights->size();
    if (layer.fc1_biases)
      total += layer.fc1_biases->size();
    if (layer.fc2_weights)
      total += layer.fc2_weights->size();
    if (layer.fc2_biases)
      total += layer.fc2_biases->size();
  }

  return total;
}

// ============================================================================
// GPUEvalBatch Implementation
// ============================================================================

void GPUEvalBatch::clear() {
  positions.clear();
  white_features.clear();
  black_features.clear();
  feature_counts.clear();
  feature_offsets.clear();
  buckets.clear();
  psqt_scores.clear();
  positional_scores.clear();
  count = 0;
}

void GPUEvalBatch::reserve(int n) {
  positions.reserve(n);
  white_features.reserve(n * GPU_MAX_FEATURES);
  black_features.reserve(n * GPU_MAX_FEATURES);
  feature_counts.reserve(n * 2);
  feature_offsets.reserve(n);
  buckets.reserve(n);
  psqt_scores.resize(n);
  positional_scores.resize(n);
}

void GPUEvalBatch::add_position(const Position &pos) {
  GPUPositionData data;
  data.from_position(pos);
  positions.push_back(data);

  // Calculate bucket based on piece count
  int bucket = (pos.count<ALL_PIECES>() - 1) / 4;
  bucket = std::clamp(bucket, 0, GPU_LAYER_STACKS - 1);
  buckets.push_back(bucket);

  // Track feature offset
  feature_offsets.push_back(white_features.size());

  count++;
}

// ============================================================================
// Embedded Shader Source
// ============================================================================

static const char *GPU_NNUE_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

constant uint FC0_OUT = 15;
constant uint FC1_OUT = 32;
constant uint WEIGHT_SCALE_BITS = 6;
constant uint OUTPUT_SCALE = 16;

typedef int16_t weight_t;
typedef int8_t layer_weight_t;
typedef int32_t accumulator_t;

inline int8_t clipped_relu(int16_t x) {
    return int8_t(clamp(int(x), 0, 127));
}

inline int8_t sqr_clipped_relu(int16_t x) {
    int clamped = clamp(int(x), 0, 127);
    return int8_t((clamped * clamped) >> 7);
}

struct GPUPositionData {
    uint64_t pieces[2][7];
    uint8_t king_sq[2];
    uint8_t stm;
    uint8_t piece_count;
    uint8_t padding[4];
};

// Feature transform kernel
kernel void gpu_feature_transform(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* features [[buffer(2)]],
    device const uint32_t* feature_counts [[buffer(3)]],
    device const uint32_t* feature_offsets [[buffer(4)]],
    device accumulator_t* accumulators [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    uint pos_idx = gid.y;
    uint hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
        return;
    
    accumulator_t acc = accumulator_t(biases[hidden_idx]);
    
    uint start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
    uint count = feature_counts[pos_idx];
    
    for (uint i = 0; i < count; i++) {
        int32_t feat_idx = features[start + i];
        if (feat_idx >= 0) {
            acc += weights[feat_idx * hidden_dim + hidden_idx];
        }
    }
    
    accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

// Fused forward pass kernel
kernel void gpu_nnue_forward(
    device const accumulator_t* accumulators [[buffer(0)]],
    device const layer_weight_t* fc0_weights [[buffer(1)]],
    device const int32_t* fc0_biases [[buffer(2)]],
    device const layer_weight_t* fc1_weights [[buffer(3)]],
    device const int32_t* fc1_biases [[buffer(4)]],
    device const layer_weight_t* fc2_weights [[buffer(5)]],
    device const int32_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    uint pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    threadgroup int8_t fc0_sqr[2 * 16];
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[32];
    
    device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const accumulator_t* black_acc = white_acc + hidden_dim;
    
    // FC0
    for (uint out = lid; out <= FC0_OUT; out += tg_size) {
        for (uint p = 0; p < 2; p++) {
            device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
            int32_t sum = fc0_biases[out];
            for (uint i = 0; i < hidden_dim; i++) {
                int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
            }
            
            int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
            if (out < FC0_OUT) {
                fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
            } else {
                fc0_skip[p] = clipped_relu(result);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC1
    for (uint out = lid; out < FC1_OUT; out += tg_size) {
        int32_t sum = fc1_biases[out];
        for (uint i = 0; i < 2 * FC0_OUT; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2
    if (lid == 0) {
        int32_t sum = fc2_biases[0];
        for (uint i = 0; i < FC1_OUT; i++) {
            sum += fc1_out[i] * fc2_weights[i];
        }
        
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * int32_t(OUTPUT_SCALE)) / 
                          (2 * 127 * (1 << WEIGHT_SCALE_BITS));
        
        output[pos_idx] = sum + skip_val;
    }
}

// PSQT accumulation kernel
kernel void gpu_psqt_accumulate(
    device const int32_t* psqt_weights [[buffer(0)]],
    device const int32_t* features [[buffer(1)]],
    device const uint32_t* feature_counts [[buffer(2)]],
    device const uint32_t* feature_offsets [[buffer(3)]],
    device int32_t* output [[buffer(4)]],
    constant uint& num_buckets [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    uint pos_idx = gid.y;
    uint bucket = gid.x;
    
    if (pos_idx >= batch_size || bucket >= num_buckets)
        return;
    
    uint start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
    uint count = feature_counts[pos_idx];
    
    int32_t acc = 0;
    for (uint i = 0; i < count; i++) {
        int32_t feat_idx = features[start + i];
        if (feat_idx >= 0) {
            acc += psqt_weights[feat_idx * num_buckets + bucket];
        }
    }
    
    output[pos_idx * num_buckets + bucket] = acc;
}
)";

// ============================================================================
// GPUNNUEManager Implementation
// ============================================================================

GPUNNUEManager::GPUNNUEManager() = default;
GPUNNUEManager::~GPUNNUEManager() = default;

bool GPUNNUEManager::initialize() {
  if (initialized_)
    return true;

  if (!gpu_available()) {
    std::cerr << "[GPU NNUE] GPU not available" << std::endl;
    return false;
  }

  if (!compile_shaders()) {
    std::cerr << "[GPU NNUE] Failed to compile shaders" << std::endl;
    return false;
  }

  if (!allocate_working_buffers()) {
    std::cerr << "[GPU NNUE] Failed to allocate working buffers" << std::endl;
    return false;
  }

  initialized_ = true;
  std::cout << "[GPU NNUE] Manager initialized" << std::endl;
  return true;
}

bool GPUNNUEManager::compile_shaders() {
  auto &backend = gpu();

  if (!backend.compile_library("gpu_nnue_integration",
                               GPU_NNUE_SHADER_SOURCE)) {
    std::cerr << "[GPU NNUE] Shader compilation failed" << std::endl;
    return false;
  }

  feature_transform_kernel_ =
      backend.create_kernel("gpu_feature_transform", "gpu_nnue_integration");
  forward_fused_kernel_ =
      backend.create_kernel("gpu_nnue_forward", "gpu_nnue_integration");
  psqt_kernel_ =
      backend.create_kernel("gpu_psqt_accumulate", "gpu_nnue_integration");

  if (!feature_transform_kernel_ || !feature_transform_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create feature_transform kernel"
              << std::endl;
    return false;
  }

  if (!forward_fused_kernel_ || !forward_fused_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create forward kernel" << std::endl;
    return false;
  }

  if (!psqt_kernel_ || !psqt_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create psqt kernel" << std::endl;
    return false;
  }

  std::cout << "[GPU NNUE] Shaders compiled successfully" << std::endl;
  return true;
}

bool GPUNNUEManager::allocate_working_buffers() {
  auto &backend = gpu();

  const size_t max_positions = GPU_MAX_BATCH_SIZE;
  const size_t max_features = max_positions * GPU_MAX_FEATURES;
  const size_t max_hidden = GPU_FT_DIM_BIG;

  positions_buffer_ =
      backend.create_buffer(max_positions * sizeof(GPUPositionData));
  white_features_buffer_ =
      backend.create_buffer(max_features * sizeof(int32_t));
  black_features_buffer_ =
      backend.create_buffer(max_features * sizeof(int32_t));
  feature_counts_buffer_ =
      backend.create_buffer(max_positions * 2 * sizeof(uint32_t));
  feature_offsets_buffer_ =
      backend.create_buffer(max_positions * sizeof(uint32_t));
  accumulators_buffer_ =
      backend.create_buffer(max_positions * 2 * max_hidden * sizeof(int32_t));
  psqt_buffer_ =
      backend.create_buffer(max_positions * GPU_PSQT_BUCKETS * sizeof(int32_t));
  output_buffer_ = backend.create_buffer(max_positions * sizeof(int32_t));

  if (!positions_buffer_ || !white_features_buffer_ ||
      !black_features_buffer_ || !feature_counts_buffer_ ||
      !feature_offsets_buffer_ || !accumulators_buffer_ || !psqt_buffer_ ||
      !output_buffer_) {
    return false;
  }

  std::cout << "[GPU NNUE] Working buffers allocated: "
            << backend.allocated_memory() / 1024 << " KB" << std::endl;
  return true;
}

bool GPUNNUEManager::allocate_network_buffers(GPUNetworkData &net,
                                              int hidden_dim,
                                              bool has_threats) {
  auto &backend = gpu();

  net.hidden_dim = hidden_dim;
  net.has_threats = has_threats;

  // Feature transformer
  net.ft_weights =
      backend.create_buffer(GPU_HALFKA_DIMS * hidden_dim * sizeof(int16_t));
  net.ft_biases = backend.create_buffer(hidden_dim * sizeof(int16_t));
  net.ft_psqt = backend.create_buffer(GPU_HALFKA_DIMS * GPU_PSQT_BUCKETS *
                                      sizeof(int32_t));

  if (!net.ft_weights || !net.ft_biases || !net.ft_psqt) {
    return false;
  }

  if (has_threats) {
    net.threat_weights =
        backend.create_buffer(GPU_THREAT_DIMS * hidden_dim * sizeof(int8_t));
    net.threat_psqt = backend.create_buffer(GPU_THREAT_DIMS * GPU_PSQT_BUCKETS *
                                            sizeof(int32_t));
    if (!net.threat_weights || !net.threat_psqt) {
      return false;
    }
  }

  // FC layers for each bucket
  for (int b = 0; b < GPU_LAYER_STACKS; b++) {
    auto &layer = net.layers[b];

    layer.fc0_weights = backend.create_buffer(
        hidden_dim * 2 * (GPU_FC0_OUT + 1) * sizeof(int8_t));
    layer.fc0_biases =
        backend.create_buffer((GPU_FC0_OUT + 1) * sizeof(int32_t));
    layer.fc1_weights =
        backend.create_buffer(GPU_FC0_OUT * 2 * GPU_FC1_OUT * sizeof(int8_t));
    layer.fc1_biases = backend.create_buffer(GPU_FC1_OUT * sizeof(int32_t));
    layer.fc2_weights = backend.create_buffer(GPU_FC1_OUT * sizeof(int8_t));
    layer.fc2_biases = backend.create_buffer(sizeof(int32_t));

    if (!layer.valid()) {
      return false;
    }
  }

  net.valid = true;
  return true;
}

bool GPUNNUEManager::load_networks(const Eval::NNUE::Networks &networks) {
  if (!initialized_ && !initialize()) {
    return false;
  }

  std::cout << "[GPU NNUE] Loading networks..." << std::endl;

  // Extract and print network info
  auto big_info = get_network_info<Eval::NNUE::NetworkBig>();
  auto small_info = get_network_info<Eval::NNUE::NetworkSmall>();

  print_network_info(big_info, "Big");
  print_network_info(small_info, "Small");

  std::cout << "[GPU NNUE] Total memory required: "
            << (get_network_memory_requirement<Eval::NNUE::NetworkBig>() +
                get_network_memory_requirement<Eval::NNUE::NetworkSmall>()) /
                   1024
            << " KB" << std::endl;

  // Allocate big network buffers
  if (!allocate_network_buffers(big_network_, GPU_FT_DIM_BIG, true)) {
    std::cerr << "[GPU NNUE] Failed to allocate big network buffers"
              << std::endl;
    return false;
  }

  // Allocate small network buffers
  if (!allocate_network_buffers(small_network_, GPU_FT_DIM_SMALL, false)) {
    std::cerr << "[GPU NNUE] Failed to allocate small network buffers"
              << std::endl;
    return false;
  }

  // Extract and upload big network weights
  auto big_weights =
      GPUNNUEWeightExtractor<Eval::NNUE::NetworkBig>::extract(networks.big);
  if (big_weights.valid) {
    std::cout << "[GPU NNUE] Uploading big network weights..." << std::endl;

    // Upload feature transformer
    if (big_weights.ft.biases && big_network_.ft_biases) {
      std::memcpy(
          big_network_.ft_biases->data(), big_weights.ft.biases,
          std::min(big_weights.ft.biases_size, big_network_.ft_biases->size()));
    }
    if (big_weights.ft.weights && big_network_.ft_weights) {
      std::memcpy(big_network_.ft_weights->data(), big_weights.ft.weights,
                  std::min(big_weights.ft.weights_size,
                           big_network_.ft_weights->size()));
    }
    if (big_weights.ft.psqt_weights && big_network_.ft_psqt) {
      std::memcpy(
          big_network_.ft_psqt->data(), big_weights.ft.psqt_weights,
          std::min(big_weights.ft.psqt_size, big_network_.ft_psqt->size()));
    }

    // Upload layer weights
    for (int b = 0; b < GPU_LAYER_STACKS; b++) {
      const auto &src = big_weights.layers[b];
      auto &dst = big_network_.layers[b];

      if (src.fc0_weights && dst.fc0_weights) {
        std::memcpy(dst.fc0_weights->data(), src.fc0_weights,
                    std::min(src.fc0_weights_size, dst.fc0_weights->size()));
      }
      if (src.fc0_biases && dst.fc0_biases) {
        std::memcpy(dst.fc0_biases->data(), src.fc0_biases,
                    std::min(src.fc0_biases_size, dst.fc0_biases->size()));
      }
      if (src.fc1_weights && dst.fc1_weights) {
        std::memcpy(dst.fc1_weights->data(), src.fc1_weights,
                    std::min(src.fc1_weights_size, dst.fc1_weights->size()));
      }
      if (src.fc1_biases && dst.fc1_biases) {
        std::memcpy(dst.fc1_biases->data(), src.fc1_biases,
                    std::min(src.fc1_biases_size, dst.fc1_biases->size()));
      }
      if (src.fc2_weights && dst.fc2_weights) {
        std::memcpy(dst.fc2_weights->data(), src.fc2_weights,
                    std::min(src.fc2_weights_size, dst.fc2_weights->size()));
      }
      if (src.fc2_biases && dst.fc2_biases) {
        std::memcpy(dst.fc2_biases->data(), src.fc2_biases,
                    std::min(src.fc2_biases_size, dst.fc2_biases->size()));
      }
    }
    std::cout << "[GPU NNUE] Big network weights uploaded" << std::endl;
  }

  // Extract and upload small network weights
  auto small_weights =
      GPUNNUEWeightExtractor<Eval::NNUE::NetworkSmall>::extract(networks.small);
  if (small_weights.valid) {
    std::cout << "[GPU NNUE] Uploading small network weights..." << std::endl;

    // Upload feature transformer
    if (small_weights.ft.biases && small_network_.ft_biases) {
      std::memcpy(small_network_.ft_biases->data(), small_weights.ft.biases,
                  std::min(small_weights.ft.biases_size,
                           small_network_.ft_biases->size()));
    }
    if (small_weights.ft.weights && small_network_.ft_weights) {
      std::memcpy(small_network_.ft_weights->data(), small_weights.ft.weights,
                  std::min(small_weights.ft.weights_size,
                           small_network_.ft_weights->size()));
    }
    if (small_weights.ft.psqt_weights && small_network_.ft_psqt) {
      std::memcpy(
          small_network_.ft_psqt->data(), small_weights.ft.psqt_weights,
          std::min(small_weights.ft.psqt_size, small_network_.ft_psqt->size()));
    }

    // Upload layer weights
    for (int b = 0; b < GPU_LAYER_STACKS; b++) {
      const auto &src = small_weights.layers[b];
      auto &dst = small_network_.layers[b];

      if (src.fc0_weights && dst.fc0_weights) {
        std::memcpy(dst.fc0_weights->data(), src.fc0_weights,
                    std::min(src.fc0_weights_size, dst.fc0_weights->size()));
      }
      if (src.fc0_biases && dst.fc0_biases) {
        std::memcpy(dst.fc0_biases->data(), src.fc0_biases,
                    std::min(src.fc0_biases_size, dst.fc0_biases->size()));
      }
      if (src.fc1_weights && dst.fc1_weights) {
        std::memcpy(dst.fc1_weights->data(), src.fc1_weights,
                    std::min(src.fc1_weights_size, dst.fc1_weights->size()));
      }
      if (src.fc1_biases && dst.fc1_biases) {
        std::memcpy(dst.fc1_biases->data(), src.fc1_biases,
                    std::min(src.fc1_biases_size, dst.fc1_biases->size()));
      }
      if (src.fc2_weights && dst.fc2_weights) {
        std::memcpy(dst.fc2_weights->data(), src.fc2_weights,
                    std::min(src.fc2_weights_size, dst.fc2_weights->size()));
      }
      if (src.fc2_biases && dst.fc2_biases) {
        std::memcpy(dst.fc2_biases->data(), src.fc2_biases,
                    std::min(src.fc2_biases_size, dst.fc2_biases->size()));
      }
    }
    std::cout << "[GPU NNUE] Small network weights uploaded" << std::endl;
  }

  std::cout << "[GPU NNUE] Networks loaded. Total GPU memory: "
            << gpu_memory_used() / 1024 << " KB" << std::endl;

  return true;
}

bool GPUNNUEManager::evaluate_batch(GPUEvalBatch &batch, bool use_big_network) {
  if (!is_ready() || batch.count == 0) {
    return false;
  }

  if (batch.count < min_batch_size_) {
    cpu_evals_ += batch.count;
    return false; // Fall back to CPU
  }

  auto start = std::chrono::high_resolution_clock::now();

  auto &backend = gpu();
  const GPUNetworkData &net = use_big_network ? big_network_ : small_network_;

  if (!net.valid) {
    cpu_evals_ += batch.count;
    return false;
  }

  int batch_size = batch.count;
  int hidden_dim = net.hidden_dim;

  // Upload position data
  std::memcpy(positions_buffer_->data(), batch.positions.data(),
              batch_size * sizeof(GPUPositionData));

  // Upload features
  if (!batch.white_features.empty()) {
    std::memcpy(white_features_buffer_->data(), batch.white_features.data(),
                batch.white_features.size() * sizeof(int32_t));
  }
  std::memcpy(feature_offsets_buffer_->data(), batch.feature_offsets.data(),
              batch_size * sizeof(uint32_t));

  auto encoder = backend.create_encoder();

  // Feature transform (white perspective)
  encoder->set_kernel(feature_transform_kernel_.get());
  encoder->set_buffer(net.ft_weights.get(), 0);
  encoder->set_buffer(net.ft_biases.get(), 1);
  encoder->set_buffer(white_features_buffer_.get(), 2);
  encoder->set_buffer(feature_counts_buffer_.get(), 3);
  encoder->set_buffer(feature_offsets_buffer_.get(), 4);
  encoder->set_buffer(accumulators_buffer_.get(), 5);
  encoder->set_value(static_cast<uint32_t>(hidden_dim), 6);
  encoder->set_value(static_cast<uint32_t>(batch_size), 7);
  encoder->dispatch_threads(hidden_dim, batch_size);
  encoder->barrier();

  // Forward pass (use bucket 0 for simplicity - full impl uses per-position
  // buckets)
  const auto &layer = net.layers[0];
  encoder->set_kernel(forward_fused_kernel_.get());
  encoder->set_buffer(accumulators_buffer_.get(), 0);
  encoder->set_buffer(layer.fc0_weights.get(), 1);
  encoder->set_buffer(layer.fc0_biases.get(), 2);
  encoder->set_buffer(layer.fc1_weights.get(), 3);
  encoder->set_buffer(layer.fc1_biases.get(), 4);
  encoder->set_buffer(layer.fc2_weights.get(), 5);
  encoder->set_buffer(layer.fc2_biases.get(), 6);
  encoder->set_buffer(output_buffer_.get(), 7);
  encoder->set_value(static_cast<uint32_t>(hidden_dim), 8);
  encoder->set_value(static_cast<uint32_t>(batch_size), 9);
  encoder->dispatch_threadgroups(batch_size, 1, 1, 64, 1, 1);

  backend.submit_and_wait(encoder.get());

  // Read results
  batch.positional_scores.resize(batch_size);
  std::memcpy(batch.positional_scores.data(), output_buffer_->data(),
              batch_size * sizeof(int32_t));

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
  batch_count_++;
  gpu_evals_ += batch_size;

  return true;
}

std::pair<int32_t, int32_t> GPUNNUEManager::evaluate_single(const Position &pos,
                                                            bool use_big) {
  // Single position evaluation is not efficient on GPU
  // Fall back to CPU
  cpu_evals_++;
  return {0, 0};
}

double GPUNNUEManager::avg_batch_time_ms() const {
  return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0;
}

void GPUNNUEManager::reset_stats() {
  gpu_evals_ = 0;
  cpu_evals_ = 0;
  batch_count_ = 0;
  total_time_ms_ = 0;
}

size_t GPUNNUEManager::gpu_memory_used() const {
  if (!gpu_available())
    return 0;
  return gpu().allocated_memory();
}

std::string GPUNNUEManager::status_string() const {
  std::stringstream ss;
  ss << "GPU NNUE Manager Status:\n";
  ss << "  Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
  ss << "  Big Network: " << (big_network_.valid ? "Ready" : "Not loaded")
     << "\n";
  ss << "  Small Network: " << (small_network_.valid ? "Ready" : "Not loaded")
     << "\n";
  ss << "  GPU Memory: " << gpu_memory_used() / 1024 << " KB\n";
  ss << "  GPU Evaluations: " << gpu_evals_.load() << "\n";
  ss << "  CPU Fallbacks: " << cpu_evals_.load() << "\n";
  ss << "  Total Batches: " << batch_count_.load() << "\n";
  if (batch_count_ > 0) {
    ss << "  Avg Batch Time: " << avg_batch_time_ms() << " ms\n";
  }
  return ss.str();
}

// ============================================================================
// Global Interface
// ============================================================================

static std::unique_ptr<GPUNNUEManager> g_gpu_nnue_manager;

GPUNNUEManager &gpu_nnue_manager() {
  if (!g_gpu_nnue_manager) {
    g_gpu_nnue_manager = std::make_unique<GPUNNUEManager>();
  }
  return *g_gpu_nnue_manager;
}

bool initialize_gpu_nnue(const Eval::NNUE::Networks &networks) {
  return gpu_nnue_manager().load_networks(networks);
}

bool gpu_nnue_manager_available() {
  return gpu_available() && gpu_nnue_manager().is_ready();
}

bool gpu_evaluate_batch(GPUEvalBatch &batch, bool use_big) {
  return gpu_nnue_manager().evaluate_batch(batch, use_big);
}

} // namespace MetalFish::GPU

#else // !USE_METAL

// Stub implementations when Metal is not available
namespace MetalFish::GPU {

void GPUPositionData::from_position(const Position &) {}
size_t GPUNetworkData::memory_usage() const { return 0; }
void GPUEvalBatch::clear() { count = 0; }
void GPUEvalBatch::reserve(int) {}
void GPUEvalBatch::add_position(const Position &) {}

GPUNNUEManager::GPUNNUEManager() = default;
GPUNNUEManager::~GPUNNUEManager() = default;
bool GPUNNUEManager::initialize() { return false; }
bool GPUNNUEManager::compile_shaders() { return false; }
bool GPUNNUEManager::allocate_working_buffers() { return false; }
bool GPUNNUEManager::allocate_network_buffers(GPUNetworkData &, int, bool) {
  return false;
}
bool GPUNNUEManager::load_networks(const Eval::NNUE::Networks &) {
  return false;
}
bool GPUNNUEManager::evaluate_batch(GPUEvalBatch &, bool) { return false; }
std::pair<int32_t, int32_t> GPUNNUEManager::evaluate_single(const Position &,
                                                            bool) {
  return {0, 0};
}
double GPUNNUEManager::avg_batch_time_ms() const { return 0; }
void GPUNNUEManager::reset_stats() {}
size_t GPUNNUEManager::gpu_memory_used() const { return 0; }
std::string GPUNNUEManager::status_string() const {
  return "GPU NNUE: Not available\n";
}

static std::unique_ptr<GPUNNUEManager> g_gpu_nnue_manager;

GPUNNUEManager &gpu_nnue_manager() {
  if (!g_gpu_nnue_manager) {
    g_gpu_nnue_manager = std::make_unique<GPUNNUEManager>();
  }
  return *g_gpu_nnue_manager;
}

bool initialize_gpu_nnue(const Eval::NNUE::Networks &) { return false; }
bool gpu_nnue_manager_available() { return false; }
bool gpu_evaluate_batch(GPUEvalBatch &, bool) { return false; }

} // namespace MetalFish::GPU

#endif // USE_METAL
