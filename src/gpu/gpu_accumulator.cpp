/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Accumulator Implementation
*/

#include "gpu_accumulator.h"

#ifdef USE_METAL

#include "backend.h"
#include "core/bitboard.h"
#include "core/position.h"

#include <chrono>
#include <cstring>
#include <iostream>

namespace MetalFish::GPU {

// ============================================================================
// GPU Accumulator Entry
// ============================================================================

bool GPUAccumulatorEntry::allocate(int dim) {
  if (!gpu_available())
    return false;

  auto &backend = gpu();
  hidden_dim = dim;

  // Allocate accumulation buffer: [2 colors][hidden_dim] int32_t
  accumulation = backend.create_buffer(2 * dim * sizeof(int32_t));

  // Allocate PSQT buffer: [2 colors][PSQT_BUCKETS] int32_t
  psqt_accumulation =
      backend.create_buffer(2 * GPU_PSQT_BUCKETS * sizeof(int32_t));

  if (!accumulation || !psqt_accumulation) {
    return false;
  }

  valid = true;
  return true;
}

void GPUAccumulatorEntry::reset() {
  computed[0] = false;
  computed[1] = false;
}

// ============================================================================
// GPU Accumulator Stack
// ============================================================================

GPUAccumulatorStack::GPUAccumulatorStack() = default;
GPUAccumulatorStack::~GPUAccumulatorStack() = default;

bool GPUAccumulatorStack::initialize(int hidden_dim, bool has_threats) {
  if (initialized_)
    return true;

  if (!gpu_available()) {
    std::cerr << "[GPU AccStack] GPU not available" << std::endl;
    return false;
  }

  hidden_dim_ = hidden_dim;
  has_threats_ = has_threats;

  // Allocate entries
  for (int i = 0; i < MAX_PLY; i++) {
    if (!entries_[i].allocate(hidden_dim)) {
      std::cerr << "[GPU AccStack] Failed to allocate entry " << i << std::endl;
      return false;
    }
  }

  if (!compile_kernels()) {
    std::cerr << "[GPU AccStack] Failed to compile kernels" << std::endl;
    return false;
  }

  if (!allocate_buffers()) {
    std::cerr << "[GPU AccStack] Failed to allocate buffers" << std::endl;
    return false;
  }

  initialized_ = true;
  std::cout << "[GPU AccStack] Initialized with " << MAX_PLY << " entries"
            << std::endl;
  return true;
}

bool GPUAccumulatorStack::compile_kernels() {
  auto &backend = gpu();

  static const char *ACCUMULATOR_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

typedef int16_t weight_t;
typedef int32_t accumulator_t;

// Full feature transform
kernel void acc_full_transform(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* features [[buffer(2)]],
    device const uint32_t* feature_count [[buffer(3)]],
    device accumulator_t* output [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant uint& perspective [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid >= hidden_dim) return;
    
    accumulator_t acc = accumulator_t(biases[gid]);
    uint count = feature_count[0];
    
    for (uint i = 0; i < count; i++) {
        int32_t feat_idx = features[i];
        if (feat_idx >= 0) {
            acc += weights[feat_idx * hidden_dim + gid];
        }
    }
    
    output[perspective * hidden_dim + gid] = acc;
}

// Incremental update
kernel void acc_incremental_update(
    device const weight_t* weights [[buffer(0)]],
    device const int32_t* added_features [[buffer(1)]],
    device const int32_t* removed_features [[buffer(2)]],
    device const uint32_t* counts [[buffer(3)]],  // [num_added, num_removed]
    device accumulator_t* accumulator [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant uint& perspective [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid >= hidden_dim) return;
    
    uint num_added = counts[0];
    uint num_removed = counts[1];
    
    accumulator_t acc = accumulator[perspective * hidden_dim + gid];
    
    // Remove old features
    for (uint i = 0; i < num_removed; i++) {
        int32_t feat_idx = removed_features[i];
        if (feat_idx >= 0) {
            acc -= weights[feat_idx * hidden_dim + gid];
        }
    }
    
    // Add new features
    for (uint i = 0; i < num_added; i++) {
        int32_t feat_idx = added_features[i];
        if (feat_idx >= 0) {
            acc += weights[feat_idx * hidden_dim + gid];
        }
    }
    
    accumulator[perspective * hidden_dim + gid] = acc;
}

// Copy accumulator
kernel void acc_copy(
    device const accumulator_t* src [[buffer(0)]],
    device accumulator_t* dst [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        dst[gid] = src[gid];
    }
}
)";

  if (!backend.compile_library("gpu_accumulator", ACCUMULATOR_SHADER)) {
    return false;
  }

  full_transform_kernel_ =
      backend.create_kernel("acc_full_transform", "gpu_accumulator");
  incremental_kernel_ =
      backend.create_kernel("acc_incremental_update", "gpu_accumulator");
  copy_kernel_ = backend.create_kernel("acc_copy", "gpu_accumulator");

  return full_transform_kernel_ && full_transform_kernel_->valid() &&
         incremental_kernel_ && incremental_kernel_->valid() && copy_kernel_ &&
         copy_kernel_->valid();
}

bool GPUAccumulatorStack::allocate_buffers() {
  auto &backend = gpu();

  // Use the new larger feature limits
  features_buffer_ =
      backend.create_buffer(GPU_MAX_FEATURES_PER_PERSPECTIVE * sizeof(int32_t));
  feature_counts_buffer_ = backend.create_buffer(sizeof(uint32_t));
  update_buffer_ = backend.create_buffer(2 * GPU_MAX_FEATURES_PER_PERSPECTIVE *
                                             sizeof(int32_t) +
                                         2 * sizeof(uint32_t));

  return features_buffer_ && feature_counts_buffer_ && update_buffer_;
}

void GPUAccumulatorStack::reset() {
  size_ = 1;
  for (auto &entry : entries_) {
    entry.reset();
  }
}

void GPUAccumulatorStack::push() {
  if (size_ < MAX_PLY) {
    size_++;
    entries_[size_ - 1].reset();
  }
}

void GPUAccumulatorStack::pop() {
  if (size_ > 1) {
    size_--;
  }
}

GPUAccumulatorEntry &GPUAccumulatorStack::current() {
  return entries_[size_ - 1];
}

const GPUAccumulatorEntry &GPUAccumulatorStack::current() const {
  return entries_[size_ - 1];
}

GPUAccumulatorEntry &GPUAccumulatorStack::at(int ply) { return entries_[ply]; }

const GPUAccumulatorEntry &GPUAccumulatorStack::at(int ply) const {
  return entries_[ply];
}

bool GPUAccumulatorStack::compute_full(const Position &pos,
                                       const GPUNetworkData &network) {
  if (!initialized_ || !network.valid)
    return false;

  auto start = std::chrono::high_resolution_clock::now();

  auto &backend = gpu();
  auto &entry = current();

  // Extract features for both perspectives
  std::vector<int32_t> white_features, black_features;

  // Simple feature extraction (will be replaced by GPUFeatureExtractor)
  Square wksq = pos.square<KING>(WHITE);
  Square bksq = pos.square<KING>(BLACK);

  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece p = pos.piece_on(s);
    if (p != NO_PIECE && type_of(p) != KING) {
      Color pc = color_of(p);
      PieceType pt = type_of(p);

      // Simplified HalfKA index
      int white_feat =
          int(wksq) * 640 + int(pc) * 320 + int(pt - 1) * 64 + int(s);
      int black_feat = int(flip_rank(bksq)) * 640 + int(~pc) * 320 +
                       int(pt - 1) * 64 + int(flip_rank(s));

      if (white_feat >= 0 && white_feat < GPU_HALFKA_DIMS) {
        white_features.push_back(white_feat);
      }
      if (black_feat >= 0 && black_feat < GPU_HALFKA_DIMS) {
        black_features.push_back(black_feat);
      }
    }
  }

  auto encoder = backend.create_encoder();

  // Compute white perspective
  std::memcpy(features_buffer_->data(), white_features.data(),
              white_features.size() * sizeof(int32_t));
  uint32_t white_count = white_features.size();
  std::memcpy(feature_counts_buffer_->data(), &white_count, sizeof(uint32_t));

  encoder->set_kernel(full_transform_kernel_.get());
  encoder->set_buffer(network.ft_weights.get(), 0);
  encoder->set_buffer(network.ft_biases.get(), 1);
  encoder->set_buffer(features_buffer_.get(), 2);
  encoder->set_buffer(feature_counts_buffer_.get(), 3);
  encoder->set_buffer(entry.accumulation.get(), 4);
  encoder->set_value(static_cast<uint32_t>(hidden_dim_), 5);
  encoder->set_value(static_cast<uint32_t>(0), 6); // WHITE perspective
  encoder->dispatch_threads(hidden_dim_);
  encoder->barrier();

  // Compute black perspective
  std::memcpy(features_buffer_->data(), black_features.data(),
              black_features.size() * sizeof(int32_t));
  uint32_t black_count = black_features.size();
  std::memcpy(feature_counts_buffer_->data(), &black_count, sizeof(uint32_t));

  encoder->set_kernel(full_transform_kernel_.get());
  encoder->set_buffer(network.ft_weights.get(), 0);
  encoder->set_buffer(network.ft_biases.get(), 1);
  encoder->set_buffer(features_buffer_.get(), 2);
  encoder->set_buffer(feature_counts_buffer_.get(), 3);
  encoder->set_buffer(entry.accumulation.get(), 4);
  encoder->set_value(static_cast<uint32_t>(hidden_dim_), 5);
  encoder->set_value(static_cast<uint32_t>(1), 6); // BLACK perspective
  encoder->dispatch_threads(hidden_dim_);

  backend.submit_and_wait(encoder.get());

  entry.computed[0] = true;
  entry.computed[1] = true;

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
  full_computes_++;

  return true;
}

bool GPUAccumulatorStack::compute_incremental(const Position &pos,
                                              const GPUNetworkData &network,
                                              const GPUFeatureUpdate &update) {
  if (!initialized_ || !network.valid)
    return false;
  if (size_ < 2)
    return false; // Need previous entry

  auto start = std::chrono::high_resolution_clock::now();

  auto &backend = gpu();
  auto &curr = current();
  const auto &prev = at(size_ - 2);

  if (!prev.computed[update.perspective]) {
    return false; // Can't do incremental without computed source
  }

  // Copy from previous entry first
  auto encoder = backend.create_encoder();

  encoder->set_kernel(copy_kernel_.get());
  encoder->set_buffer(prev.accumulation.get(), 0);
  encoder->set_buffer(curr.accumulation.get(), 1);
  encoder->set_value(static_cast<uint32_t>(2 * hidden_dim_), 2);
  encoder->dispatch_threads(2 * hidden_dim_);
  encoder->barrier();

  // Apply incremental update
  // Pack update data:
  // [added_features...][removed_features...][num_added][num_removed]
  std::vector<int32_t> update_data;
  for (int i = 0; i < update.num_added; i++) {
    update_data.push_back(update.added_features[i]);
  }
  for (int i = 0; i < update.num_removed; i++) {
    update_data.push_back(update.removed_features[i]);
  }

  std::memcpy(update_buffer_->data(), update_data.data(),
              update_data.size() * sizeof(int32_t));

  uint32_t counts[2] = {update.num_added, update.num_removed};

  encoder->set_kernel(incremental_kernel_.get());
  encoder->set_buffer(network.ft_weights.get(), 0);
  encoder->set_buffer(update_buffer_.get(), 1); // added features
  encoder->set_buffer(update_buffer_.get(), 2,
                      update.num_added * sizeof(int32_t)); // removed features
  encoder->set_bytes(counts, sizeof(counts), 3);
  encoder->set_buffer(curr.accumulation.get(), 4);
  encoder->set_value(static_cast<uint32_t>(hidden_dim_), 5);
  encoder->set_value(static_cast<uint32_t>(update.perspective), 6);
  encoder->dispatch_threads(hidden_dim_);

  backend.submit_and_wait(encoder.get());

  curr.computed[update.perspective] = true;

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
  incremental_updates_++;

  return true;
}

bool GPUAccumulatorStack::copy_from(int src_ply, int dst_ply) {
  if (!initialized_)
    return false;
  if (src_ply < 0 || src_ply >= MAX_PLY || dst_ply < 0 || dst_ply >= MAX_PLY) {
    return false;
  }

  auto &backend = gpu();
  auto &src = entries_[src_ply];
  auto &dst = entries_[dst_ply];

  auto encoder = backend.create_encoder();

  encoder->set_kernel(copy_kernel_.get());
  encoder->set_buffer(src.accumulation.get(), 0);
  encoder->set_buffer(dst.accumulation.get(), 1);
  encoder->set_value(static_cast<uint32_t>(2 * hidden_dim_), 2);
  encoder->dispatch_threads(2 * hidden_dim_);

  backend.submit_and_wait(encoder.get());

  dst.computed[0] = src.computed[0];
  dst.computed[1] = src.computed[1];

  return true;
}

void GPUAccumulatorStack::reset_stats() {
  full_computes_ = 0;
  incremental_updates_ = 0;
  total_time_ms_ = 0;
}

// ============================================================================
// GPU Feature Extractor
// ============================================================================

GPUFeatureExtractor::GPUFeatureExtractor() = default;
GPUFeatureExtractor::~GPUFeatureExtractor() = default;

bool GPUFeatureExtractor::initialize() {
  if (initialized_)
    return true;

  if (!gpu_available()) {
    return false;
  }

  if (!compile_kernels()) {
    return false;
  }

  if (!create_lookup_tables()) {
    return false;
  }

  initialized_ = true;
  std::cout << "[GPU Features] Extractor initialized" << std::endl;
  return true;
}

bool GPUFeatureExtractor::compile_kernels() {
  // Feature extraction is currently done on CPU
  // GPU kernel compilation would go here for full GPU feature extraction
  return true;
}

bool GPUFeatureExtractor::create_lookup_tables() {
  // Create lookup tables for feature index calculation
  // This would include orientation tables, piece-to-index mappings, etc.
  return true;
}

bool GPUFeatureExtractor::extract(const Position &pos,
                                  std::vector<int32_t> &white_features,
                                  std::vector<int32_t> &black_features) {
  white_features.clear();
  black_features.clear();

  Square wksq = pos.square<KING>(WHITE);
  Square bksq = pos.square<KING>(BLACK);

  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    Piece p = pos.piece_on(s);
    if (p != NO_PIECE && type_of(p) != KING) {
      Color pc = color_of(p);
      PieceType pt = type_of(p);

      // Simplified HalfKA index calculation
      int white_feat =
          int(wksq) * 640 + int(pc) * 320 + int(pt - 1) * 64 + int(s);
      int black_feat = int(flip_rank(bksq)) * 640 + int(~pc) * 320 +
                       int(pt - 1) * 64 + int(flip_rank(s));

      if (white_feat >= 0 && white_feat < GPU_HALFKA_DIMS) {
        white_features.push_back(white_feat);
      }
      if (black_feat >= 0 && black_feat < GPU_HALFKA_DIMS) {
        black_features.push_back(black_feat);
      }
    }
  }

  return true;
}

bool GPUFeatureExtractor::extract_batch(
    const std::vector<const Position *> &positions,
    std::vector<int32_t> &features, std::vector<uint32_t> &feature_counts,
    std::vector<uint32_t> &feature_offsets) {
  features.clear();
  feature_counts.clear();
  feature_offsets.clear();

  for (const auto *pos : positions) {
    std::vector<int32_t> white_feats, black_feats;
    extract(*pos, white_feats, black_feats);

    feature_offsets.push_back(features.size());
    features.insert(features.end(), white_feats.begin(), white_feats.end());
    feature_counts.push_back(white_feats.size());
  }

  return true;
}

bool GPUFeatureExtractor::compute_delta(const Position &pos, Move move,
                                        GPUFeatureUpdate &white_update,
                                        GPUFeatureUpdate &black_update) {
  // This would compute the feature changes for a move
  // For now, return empty updates (will trigger full recompute)
  white_update.num_added = 0;
  white_update.num_removed = 0;
  white_update.perspective = 0;

  black_update.num_added = 0;
  black_update.num_removed = 0;
  black_update.perspective = 1;

  return true;
}

// ============================================================================
// Global Interface
// ============================================================================

static std::unique_ptr<GPUFeatureExtractor> g_feature_extractor;

GPUFeatureExtractor &gpu_feature_extractor() {
  if (!g_feature_extractor) {
    g_feature_extractor = std::make_unique<GPUFeatureExtractor>();
  }
  return *g_feature_extractor;
}

bool gpu_features_available() {
  return gpu_available() && gpu_feature_extractor().is_initialized();
}

void shutdown_gpu_feature_extractor() {
  if (g_feature_extractor) {
    g_feature_extractor.reset();
  }
}

} // namespace MetalFish::GPU

#else // !USE_METAL

namespace MetalFish::GPU {

bool GPUAccumulatorEntry::allocate(int) { return false; }
void GPUAccumulatorEntry::reset() {}

GPUAccumulatorStack::GPUAccumulatorStack() = default;
GPUAccumulatorStack::~GPUAccumulatorStack() = default;
bool GPUAccumulatorStack::initialize(int, bool) { return false; }
bool GPUAccumulatorStack::compile_kernels() { return false; }
bool GPUAccumulatorStack::allocate_buffers() { return false; }
void GPUAccumulatorStack::reset() {}
void GPUAccumulatorStack::push() {}
void GPUAccumulatorStack::pop() {}
GPUAccumulatorEntry &GPUAccumulatorStack::current() {
  static GPUAccumulatorEntry e;
  return e;
}
const GPUAccumulatorEntry &GPUAccumulatorStack::current() const {
  static GPUAccumulatorEntry e;
  return e;
}
GPUAccumulatorEntry &GPUAccumulatorStack::at(int) {
  static GPUAccumulatorEntry e;
  return e;
}
const GPUAccumulatorEntry &GPUAccumulatorStack::at(int) const {
  static GPUAccumulatorEntry e;
  return e;
}
bool GPUAccumulatorStack::compute_full(const Position &,
                                       const GPUNetworkData &) {
  return false;
}
bool GPUAccumulatorStack::compute_incremental(const Position &,
                                              const GPUNetworkData &,
                                              const GPUFeatureUpdate &) {
  return false;
}
bool GPUAccumulatorStack::copy_from(int, int) { return false; }
void GPUAccumulatorStack::reset_stats() {}

GPUFeatureExtractor::GPUFeatureExtractor() = default;
GPUFeatureExtractor::~GPUFeatureExtractor() = default;
bool GPUFeatureExtractor::initialize() { return false; }
bool GPUFeatureExtractor::compile_kernels() { return false; }
bool GPUFeatureExtractor::create_lookup_tables() { return false; }
bool GPUFeatureExtractor::extract(const Position &, std::vector<int32_t> &,
                                  std::vector<int32_t> &) {
  return false;
}
bool GPUFeatureExtractor::extract_batch(const std::vector<const Position *> &,
                                        std::vector<int32_t> &,
                                        std::vector<uint32_t> &,
                                        std::vector<uint32_t> &) {
  return false;
}
bool GPUFeatureExtractor::compute_delta(const Position &, Move,
                                        GPUFeatureUpdate &,
                                        GPUFeatureUpdate &) {
  return false;
}

static std::unique_ptr<GPUFeatureExtractor> g_feature_extractor;
GPUFeatureExtractor &gpu_feature_extractor() {
  if (!g_feature_extractor)
    g_feature_extractor = std::make_unique<GPUFeatureExtractor>();
  return *g_feature_extractor;
}
bool gpu_features_available() { return false; }

void shutdown_gpu_feature_extractor() {
  if (g_feature_extractor) {
    g_feature_extractor.reset();
  }
}

} // namespace MetalFish::GPU

#endif // USE_METAL
