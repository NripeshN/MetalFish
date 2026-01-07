/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
*/

#include "search/batch_eval.h"
#include "eval/evaluate.h"
#include "metal/gpu_ops.h"
#include <iostream>

namespace MetalFish {
namespace Search {

// Global batch evaluator instance
BatchEvaluator g_batch_eval;

BatchEvaluator::BatchEvaluator() {}

BatchEvaluator::~BatchEvaluator() { flush(); }

void BatchEvaluator::init(const BatchEvalConfig &config) {
  config_ = config;

  // Initialize GPU operations if not already done
  if (!GPU::gpu_ops) {
    GPU::init_gpu_ops();
  }

  gpu_available_ = GPU::gpu_ops && GPU::gpu_ops->is_ready();

  if (gpu_available_) {
    std::cout << "[BatchEval] GPU evaluation enabled, batch size: "
              << config_.batch_size << std::endl;
  } else {
    std::cout << "[BatchEval] GPU unavailable, using CPU evaluation"
              << std::endl;
  }

  pending_.reserve(config_.max_pending);
  initialized_ = true;
}

int BatchEvaluator::submit(Position &pos, std::atomic<bool> &ready,
                           std::atomic<Value> &result) {
  if (!initialized_) {
    // Fallback: evaluate immediately
    result.store(Eval::evaluate(pos));
    ready.store(true);
    return -1;
  }

  EvalRequest req;
  req.pos = &pos;
  req.id = next_id_++;
  req.ready = &ready;
  req.result = &result;

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_.push_back(req);

    // Process batch if full
    if ((int)pending_.size() >= config_.batch_size) {
      evaluate_batch(pending_);
      pending_.clear();
    }
  }

  return req.id;
}

Value BatchEvaluator::evaluate_sync(Position &pos) {
  if (!initialized_ || !gpu_available_) {
    // CPU fallback
    return Eval::evaluate(pos);
  }

  // For single position, check if batch evaluation is worth it
  // If we have pending positions, add to batch and process
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (!pending_.empty()) {
      std::atomic<bool> ready{false};
      std::atomic<Value> result{VALUE_ZERO};

      EvalRequest req;
      req.pos = &pos;
      req.id = next_id_++;
      req.ready = &ready;
      req.result = &result;
      pending_.push_back(req);

      // Force batch processing
      evaluate_batch(pending_);
      pending_.clear();

      return result.load();
    }
  }

  // Single position - use CPU for low latency
  return Eval::evaluate(pos);
}

void BatchEvaluator::process_batch() {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  if (!pending_.empty()) {
    evaluate_batch(pending_);
    pending_.clear();
  }
}

void BatchEvaluator::flush() {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  if (!pending_.empty()) {
    evaluate_batch(pending_);
    pending_.clear();
  }
}

void BatchEvaluator::evaluate_batch(std::vector<EvalRequest> &batch) {
  if (batch.empty())
    return;

  batch_count_++;
  eval_count_ += batch.size();

  if (gpu_available_ && batch.size() >= 4) {
    // Use GPU batch evaluation
    std::vector<Position *> positions;
    positions.reserve(batch.size());
    for (auto &req : batch) {
      positions.push_back(req.pos);
    }

    std::vector<Value> results = GPU::gpu_ops->batch_evaluate(positions);

    // Store results
    for (size_t i = 0; i < batch.size(); i++) {
      batch[i].result->store(results[i]);
      batch[i].ready->store(true);
    }
  } else {
    // CPU fallback for small batches
    for (auto &req : batch) {
      req.result->store(Eval::evaluate(*req.pos));
      req.ready->store(true);
    }
  }
}

} // namespace Search
} // namespace MetalFish
