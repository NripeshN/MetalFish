/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Evaluation (Legacy interface)
*/

#include "nnue_eval.h"
#include "gpu_nnue.h"

namespace MetalFish::GPU {

void EvalBatch::clear() {
    count = 0;
    features.clear();
    feature_counts.clear();
    psqt_scores.clear();
    positional_scores.clear();
}

void EvalBatch::reserve(int n) {
    features.reserve(n * MAX_FEATURES_PER_POSITION * 2);
    feature_counts.reserve(n);
    psqt_scores.resize(n);
    positional_scores.resize(n);
}

NNUEEvaluator::NNUEEvaluator() = default;
NNUEEvaluator::~NNUEEvaluator() = default;

bool NNUEEvaluator::initialize(const void*, size_t, const void*, size_t) {
    // Legacy interface - actual initialization is done via gpu_nnue.h
    initialized_ = gpu_nnue_ready();
    return initialized_;
}

bool NNUEEvaluator::evaluate_batch(EvalBatch& batch, bool) {
    if (!initialized_ || batch.count == 0) {
        return false;
    }
    // Legacy interface - not fully implemented
    // Use GPUNNUEBatchEvaluator from gpu_nnue.h instead
    cpu_evals_ += batch.count;
    return false;
}

int32_t NNUEEvaluator::evaluate(const Position&) {
    cpu_evals_++;
    return 0;
}

double NNUEEvaluator::avg_batch_time_ms() const {
    return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0;
}

void NNUEEvaluator::reset_stats() {
    gpu_evals_ = 0;
    cpu_evals_ = 0;
    total_time_ms_ = 0;
    batch_count_ = 0;
}

// Global instance
static std::unique_ptr<NNUEEvaluator> g_legacy_nnue;

bool gpu_nnue_available() {
    return gpu_nnue_ready();
}

NNUEEvaluator& gpu_nnue() {
    if (!g_legacy_nnue) {
        g_legacy_nnue = std::make_unique<NNUEEvaluator>();
    }
    return *g_legacy_nnue;
}

} // namespace MetalFish::GPU
