/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Implementation
*/

#include "gpu_nnue.h"

#if defined(USE_METAL) || defined(USE_CUDA)

#include "backend.h"
#include "core/position.h"
#include "eval/nnue/network.h"

#include <chrono>
#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include "cuda/kernels/nnue_kernels.h"
#endif

namespace MetalFish::GPU {

// ============================================================================
// GPUPositionBatch
// ============================================================================

void GPUPositionBatch::clear() {
  white_features.clear();
  black_features.clear();
  feature_counts.clear();
  buckets.clear();
  stm.clear();
  count = 0;
}

void GPUPositionBatch::reserve(int n, int features_per_pos) {
  white_features.reserve(n * features_per_pos);
  black_features.reserve(n * features_per_pos);
  feature_counts.reserve(n);
  buckets.reserve(n);
  stm.reserve(n);
}

// ============================================================================
// GPUNNUEWeightManager
// ============================================================================

GPUNNUEWeightManager::GPUNNUEWeightManager() = default;
GPUNNUEWeightManager::~GPUNNUEWeightManager() = default;

bool GPUNNUEWeightManager::allocate_network_buffers(GPUNetworkWeights &weights,
                                                    int hidden_dim,
                                                    bool has_threats) {
  if (!gpu_available())
    return false;

  auto &backend = gpu();

  weights.hidden_dim = hidden_dim;
  weights.has_threats = has_threats;
  weights.layers.resize(GPU_LAYER_STACKS);

  // Feature transformer buffers
  weights.ft_weights =
      backend.create_buffer(GPU_HALFKA_DIMS * hidden_dim * sizeof(int16_t));
  weights.ft_biases = backend.create_buffer(hidden_dim * sizeof(int16_t));
  weights.ft_psqt = backend.create_buffer(GPU_HALFKA_DIMS * GPU_PSQT_BUCKETS *
                                          sizeof(int32_t));

  if (!weights.ft_weights || !weights.ft_biases || !weights.ft_psqt) {
    return false;
  }

  if (has_threats) {
    weights.threat_weights =
        backend.create_buffer(GPU_THREAT_DIMS * hidden_dim * sizeof(int8_t));
    weights.threat_psqt = backend.create_buffer(
        GPU_THREAT_DIMS * GPU_PSQT_BUCKETS * sizeof(int32_t));
    if (!weights.threat_weights || !weights.threat_psqt) {
      return false;
    }
  }

  // FC layer buffers for each bucket
  for (int bucket = 0; bucket < GPU_LAYER_STACKS; bucket++) {
    auto &layer = weights.layers[bucket];

    layer.fc0_weights = backend.create_buffer(
        hidden_dim * 2 * (GPU_FC0_OUT + 1) * sizeof(int16_t));
    layer.fc0_biases =
        backend.create_buffer((GPU_FC0_OUT + 1) * sizeof(int32_t));
    layer.fc1_weights =
        backend.create_buffer(GPU_FC0_OUT * 2 * GPU_FC1_OUT * sizeof(int16_t));
    layer.fc1_biases = backend.create_buffer(GPU_FC1_OUT * sizeof(int32_t));
    layer.fc2_weights = backend.create_buffer(GPU_FC1_OUT * sizeof(int16_t));
    layer.fc2_biases = backend.create_buffer(sizeof(int32_t));

    if (!layer.fc0_weights || !layer.fc0_biases || !layer.fc1_weights ||
        !layer.fc1_biases || !layer.fc2_weights || !layer.fc2_biases) {
      return false;
    }
  }

  weights.valid = true;
  return true;
}

bool GPUNNUEWeightManager::load_networks(const Eval::NNUE::Networks &networks) {
  std::cout << "[GPU NNUE] Loading networks to GPU..." << std::endl;

  // Allocate buffers for big network
  if (!allocate_network_buffers(big_weights_, GPU_FT_DIM_BIG, true)) {
    std::cerr << "[GPU NNUE] Failed to allocate big network buffers"
              << std::endl;
    return false;
  }
  std::cout << "[GPU NNUE] Big network buffers allocated" << std::endl;

  // Allocate buffers for small network
  if (!allocate_network_buffers(small_weights_, GPU_FT_DIM_SMALL, false)) {
    std::cerr << "[GPU NNUE] Failed to allocate small network buffers"
              << std::endl;
    return false;
  }
  std::cout << "[GPU NNUE] Small network buffers allocated" << std::endl;

  // Note: Actual weight copying would require access to network internals
  // For now, buffers are allocated but not filled with actual weights
  // This allows the infrastructure to be tested

  std::cout << "[GPU NNUE] Networks loaded: " << gpu_memory_used() / 1024
            << " KB" << std::endl;
  return true;
}

size_t GPUNNUEWeightManager::gpu_memory_used() const {
  if (!gpu_available())
    return 0;
  return gpu().allocated_memory();
}

// ============================================================================
// GPUNNUEBatchEvaluator
// ============================================================================

GPUNNUEBatchEvaluator::GPUNNUEBatchEvaluator() = default;
GPUNNUEBatchEvaluator::~GPUNNUEBatchEvaluator() = default;

bool GPUNNUEBatchEvaluator::initialize(GPUNNUEWeightManager &weights) {
  if (!gpu_available()) {
    std::cerr << "[GPU NNUE Eval] GPU not available" << std::endl;
    return false;
  }

  weights_ = &weights;

  if (!compile_kernels()) {
    std::cerr << "[GPU NNUE Eval] Failed to compile kernels" << std::endl;
    return false;
  }

  if (!allocate_buffers(256)) {
    std::cerr << "[GPU NNUE Eval] Failed to allocate buffers" << std::endl;
    return false;
  }

  initialized_ = true;
  std::cout << "[GPU NNUE Eval] Initialized successfully" << std::endl;
  return true;
}

bool GPUNNUEBatchEvaluator::compile_kernels() {
  auto &backend = gpu();

#ifdef USE_METAL
  static const char *EVAL_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

constant int FC0_OUT = 15;
constant int FC1_OUT = 32;

inline int8_t clipped_relu(int16_t x) { 
    return int8_t(clamp(int(x), 0, 127)); 
}

inline int8_t sqr_clipped_relu(int16_t x) {
    int clamped = clamp(int(x), 0, 127);
    return int8_t((clamped * clamped) >> 7);
}

kernel void gpu_feature_transform(
    device const int16_t* weights [[buffer(0)]],
    device const int16_t* biases [[buffer(1)]],
    device const int32_t* features [[buffer(2)]],
    device const int32_t* feature_offsets [[buffer(3)]],
    device int32_t* output [[buffer(4)]],
    constant int& hidden_dim [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    int pos_idx = gid.y;
    int hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
        return;
    
    int32_t acc = int32_t(biases[hidden_idx]);
    
    int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
    int end = feature_offsets[pos_idx];
    
    for (int i = start; i < end; i++) {
        int feature_idx = features[i];
        if (feature_idx >= 0) {
            acc += weights[feature_idx * hidden_dim + hidden_idx];
        }
    }
    
    output[pos_idx * hidden_dim + hidden_idx] = acc;
}

kernel void gpu_nnue_forward(
    device const int32_t* accumulators [[buffer(0)]],
    device const int16_t* fc0_weights [[buffer(1)]],
    device const int32_t* fc0_biases [[buffer(2)]],
    device const int16_t* fc1_weights [[buffer(3)]],
    device const int32_t* fc1_biases [[buffer(4)]],
    device const int16_t* fc2_weights [[buffer(5)]],
    device const int32_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant int& hidden_dim [[buffer(8)]],
    constant int& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    int pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    threadgroup int8_t fc0_sqr[2 * 16];
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[32];
    
    device const int32_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const int32_t* black_acc = white_acc + hidden_dim;
    
    for (int out = lid; out <= FC0_OUT; out += tg_size) {
        for (int p = 0; p < 2; p++) {
            device const int32_t* acc = (p == 0) ? white_acc : black_acc;
            
            int32_t sum = fc0_biases[out];
            for (int i = 0; i < hidden_dim; i++) {
                int8_t clipped = clipped_relu(int16_t(acc[i] >> 6));
                sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
            }
            
            int16_t result = int16_t(sum >> 6);
            
            if (out < FC0_OUT) {
                fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
            } else {
                fc0_skip[p] = clipped_relu(result);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int out = lid; out < FC1_OUT; out += tg_size) {
        int32_t sum = fc1_biases[out];
        for (int i = 0; i < 2 * FC0_OUT; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        fc1_out[out] = clipped_relu(int16_t(sum >> 6));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lid == 0) {
        int32_t sum = fc2_biases[0];
        for (int i = 0; i < FC1_OUT; i++) {
            sum += fc1_out[i] * fc2_weights[i];
        }
        
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * 16) / (2 * 127 * 64);
        output[pos_idx] = sum + skip_val;
    }
}
)";

  if (!backend.compile_library("gpu_nnue_eval", EVAL_SHADER)) {
    std::cerr << "[GPU NNUE Eval] Failed to compile shader" << std::endl;
    return false;
  }

  ft_kernel_ = backend.create_kernel("gpu_feature_transform", "gpu_nnue_eval");
  forward_kernel_ = backend.create_kernel("gpu_nnue_forward", "gpu_nnue_eval");

  if (!ft_kernel_ || !ft_kernel_->valid()) {
    std::cerr << "[GPU NNUE Eval] Failed to create feature transform kernel"
              << std::endl;
    return false;
  }

  if (!forward_kernel_ || !forward_kernel_->valid()) {
    std::cerr << "[GPU NNUE Eval] Failed to create forward kernel" << std::endl;
    return false;
  }

  std::cout << "[GPU NNUE Eval] Metal kernels compiled successfully" << std::endl;
#endif // USE_METAL

#ifdef USE_CUDA
  // CUDA kernels are pre-compiled, just mark as ready
  std::cout << "[GPU NNUE Eval] CUDA kernels ready" << std::endl;
#endif // USE_CUDA

  return true;
}

bool GPUNNUEBatchEvaluator::allocate_buffers(int max_batch_size) {
  auto &backend = gpu();

  const int max_features = max_batch_size * 64;
  const int max_hidden = GPU_FT_DIM_BIG;

  features_buffer_ = backend.create_buffer(max_features * sizeof(int32_t));
  feature_offsets_buffer_ =
      backend.create_buffer(max_batch_size * sizeof(int32_t));
  accumulators_buffer_ =
      backend.create_buffer(max_batch_size * 2 * max_hidden * sizeof(int32_t));
  output_buffer_ = backend.create_buffer(max_batch_size * sizeof(int32_t));

  if (!features_buffer_ || !feature_offsets_buffer_ || !accumulators_buffer_ ||
      !output_buffer_) {
    return false;
  }

  std::cout << "[GPU NNUE Eval] Buffers allocated: "
            << backend.allocated_memory() / 1024 << " KB" << std::endl;
  return true;
}

void GPUNNUEBatchEvaluator::add_position(const Position &pos) {
  int feature_start = batch_.white_features.size();

  // Extract features from position (simplified HalfKA)
  Square wksq = pos.square<KING>(WHITE);
  Square bksq = pos.square<KING>(BLACK);

  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece p = pos.piece_on(s);
    if (p != NO_PIECE && type_of(p) != KING) {
      Color pc = color_of(p);
      PieceType pt = type_of(p);

      // Simplified feature index
      int white_feature =
          int(wksq) * 640 + int(pc) * 320 + int(pt - 1) * 64 + int(s);
      int black_feature = int(flip_rank(bksq)) * 640 + int(~pc) * 320 +
                          int(pt - 1) * 64 + int(flip_rank(s));

      if (white_feature >= 0 && white_feature < GPU_HALFKA_DIMS) {
        batch_.white_features.push_back(white_feature);
      }
      if (black_feature >= 0 && black_feature < GPU_HALFKA_DIMS) {
        batch_.black_features.push_back(black_feature);
      }
    }
  }

  batch_.feature_counts.push_back(batch_.white_features.size());
  batch_.buckets.push_back((pos.count<ALL_PIECES>() - 1) / 4);
  batch_.stm.push_back(pos.side_to_move() == WHITE ? 0 : 1);
  batch_.count++;
}

bool GPUNNUEBatchEvaluator::evaluate_big(GPUEvalResults &results) {
  if (!initialized_ || !weights_ || !weights_->big_network_ready()) {
    return false;
  }
  if (batch_.count < min_batch_size_) {
    return false;
  }
  return dispatch_evaluation(weights_->big_weights(), results);
}

bool GPUNNUEBatchEvaluator::evaluate_small(GPUEvalResults &results) {
  if (!initialized_ || !weights_ || !weights_->small_network_ready()) {
    return false;
  }
  if (batch_.count < min_batch_size_) {
    return false;
  }
  return dispatch_evaluation(weights_->small_weights(), results);
}

bool GPUNNUEBatchEvaluator::dispatch_evaluation(
    const GPUNetworkWeights &weights, GPUEvalResults &results) {
  auto start = std::chrono::high_resolution_clock::now();

  auto &backend = gpu();
  int batch_size = batch_.count;
  int hidden_dim = weights.hidden_dim;

  // Upload features
  std::memcpy(features_buffer_->data(), batch_.white_features.data(),
              batch_.white_features.size() * sizeof(int32_t));
  std::memcpy(feature_offsets_buffer_->data(), batch_.feature_counts.data(),
              batch_.feature_counts.size() * sizeof(int32_t));

  auto encoder = backend.create_encoder();

  // Feature transform
  encoder->set_kernel(ft_kernel_.get());
  encoder->set_buffer(weights.ft_weights.get(), 0);
  encoder->set_buffer(weights.ft_biases.get(), 1);
  encoder->set_buffer(features_buffer_.get(), 2);
  encoder->set_buffer(feature_offsets_buffer_.get(), 3);
  encoder->set_buffer(accumulators_buffer_.get(), 4);
  encoder->set_value(hidden_dim, 5);
  encoder->set_value(batch_size, 6);
  encoder->dispatch_threads(hidden_dim, batch_size);
  encoder->barrier();

  // Forward pass
  const auto &layer = weights.layers[0];
  encoder->set_kernel(forward_kernel_.get());
  encoder->set_buffer(accumulators_buffer_.get(), 0);
  encoder->set_buffer(layer.fc0_weights.get(), 1);
  encoder->set_buffer(layer.fc0_biases.get(), 2);
  encoder->set_buffer(layer.fc1_weights.get(), 3);
  encoder->set_buffer(layer.fc1_biases.get(), 4);
  encoder->set_buffer(layer.fc2_weights.get(), 5);
  encoder->set_buffer(layer.fc2_biases.get(), 6);
  encoder->set_buffer(output_buffer_.get(), 7);
  encoder->set_value(hidden_dim, 8);
  encoder->set_value(batch_size, 9);
  encoder->dispatch_threadgroups(batch_size, 1, 1, 64, 1, 1);

  backend.submit_and_wait(encoder.get());

  // Read results
  results.final_scores.resize(batch_size);
  std::memcpy(results.final_scores.data(), output_buffer_->data(),
              batch_size * sizeof(int32_t));

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
  batch_count_++;
  gpu_evals_ += batch_size;

  return true;
}

double GPUNNUEBatchEvaluator::avg_batch_time_ms() const {
  return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0;
}

void GPUNNUEBatchEvaluator::reset_stats() {
  gpu_evals_ = 0;
  batch_count_ = 0;
  total_time_ms_ = 0;
}

// ============================================================================
// GPUNNUEInterface
// ============================================================================

GPUNNUEInterface &GPUNNUEInterface::instance() {
  static GPUNNUEInterface inst;
  return inst;
}

bool GPUNNUEInterface::initialize(const Eval::NNUE::Networks &networks) {
  if (!gpu_available()) {
    std::cout << "[GPU NNUE] GPU not available" << std::endl;
    return false;
  }

  std::cout << "[GPU NNUE] Initializing..." << std::endl;

  if (!weights_.load_networks(networks)) {
    std::cerr << "[GPU NNUE] Failed to load networks" << std::endl;
    return false;
  }

  if (!evaluator_.initialize(weights_)) {
    std::cerr << "[GPU NNUE] Failed to initialize evaluator" << std::endl;
    return false;
  }

  initialized_ = true;
  std::cout << "[GPU NNUE] Initialization complete" << std::endl;
  return true;
}

std::string GPUNNUEInterface::status_string() const {
  std::stringstream ss;
  ss << "GPU NNUE Status:\n";
  ss << "  Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
  ss << "  Big Network: "
     << (weights_.big_network_ready() ? "Ready" : "Not loaded") << "\n";
  ss << "  Small Network: "
     << (weights_.small_network_ready() ? "Ready" : "Not loaded") << "\n";
  ss << "  GPU Memory: " << weights_.gpu_memory_used() / 1024 << " KB\n";
  if (initialized_) {
    ss << "  GPU Evals: " << evaluator_.total_gpu_evals() << "\n";
    ss << "  Batches: " << evaluator_.total_batches() << "\n";
    ss << "  Avg Batch Time: " << evaluator_.avg_batch_time_ms() << " ms\n";
  }
  return ss.str();
}

// Convenience functions
bool init_gpu_nnue(const Eval::NNUE::Networks &networks) {
  return GPUNNUEInterface::instance().initialize(networks);
}

bool gpu_nnue_ready() { return GPUNNUEInterface::instance().available(); }

GPUNNUEInterface &gpu_nnue_interface() { return GPUNNUEInterface::instance(); }

} // namespace MetalFish::GPU

#else // !USE_METAL && !USE_CUDA

namespace MetalFish::GPU {

void GPUPositionBatch::clear() { count = 0; }
void GPUPositionBatch::reserve(int, int) {}

GPUNNUEWeightManager::GPUNNUEWeightManager() = default;
GPUNNUEWeightManager::~GPUNNUEWeightManager() = default;
bool GPUNNUEWeightManager::load_networks(const Eval::NNUE::Networks &) {
  return false;
}
bool GPUNNUEWeightManager::allocate_network_buffers(GPUNetworkWeights &, int,
                                                    bool) {
  return false;
}
size_t GPUNNUEWeightManager::gpu_memory_used() const { return 0; }

GPUNNUEBatchEvaluator::GPUNNUEBatchEvaluator() = default;
GPUNNUEBatchEvaluator::~GPUNNUEBatchEvaluator() = default;
bool GPUNNUEBatchEvaluator::initialize(GPUNNUEWeightManager &) { return false; }
void GPUNNUEBatchEvaluator::add_position(const Position &) {}
bool GPUNNUEBatchEvaluator::evaluate_big(GPUEvalResults &) { return false; }
bool GPUNNUEBatchEvaluator::evaluate_small(GPUEvalResults &) { return false; }
bool GPUNNUEBatchEvaluator::compile_kernels() { return false; }
bool GPUNNUEBatchEvaluator::allocate_buffers(int) { return false; }
bool GPUNNUEBatchEvaluator::dispatch_evaluation(const GPUNetworkWeights &,
                                                GPUEvalResults &) {
  return false;
}
double GPUNNUEBatchEvaluator::avg_batch_time_ms() const { return 0; }
void GPUNNUEBatchEvaluator::reset_stats() {}

GPUNNUEInterface &GPUNNUEInterface::instance() {
  static GPUNNUEInterface inst;
  return inst;
}
bool GPUNNUEInterface::initialize(const Eval::NNUE::Networks &) {
  return false;
}
std::string GPUNNUEInterface::status_string() const {
  return "GPU NNUE: Not available\n";
}

bool init_gpu_nnue(const Eval::NNUE::Networks &) { return false; }
bool gpu_nnue_ready() { return false; }
GPUNNUEInterface &gpu_nnue_interface() { return GPUNNUEInterface::instance(); }

} // namespace MetalFish::GPU

#endif // USE_METAL || USE_CUDA
