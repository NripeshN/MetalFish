/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Search Parameters

  Licensed under GPL-3.0
*/

#pragma once

#include "../nn/network.h"

#include <algorithm>
#include <string>
#include <thread>

namespace MetalFish {
namespace MCTS {

struct SearchParams {
  // PUCT exploration (Lc0 defaults)
  float cpuct = 1.745f;
  float cpuct_at_root = 1.745f;
  float cpuct_base = 38739.0f;
  float cpuct_factor = 3.894f;
  float cpuct_base_at_root = 38739.0f;
  float cpuct_factor_at_root = 3.894f;

  // First Play Urgency (Lc0 defaults: reduction strategy, same at root)
  bool fpu_absolute = false;
  float fpu_value = 0.33f;
  bool fpu_absolute_at_root = false;
  float fpu_value_at_root = 0.33f;
  float fpu_reduction = 0.33f;
  float fpu_reduction_at_root = 0.33f;

  // Policy softmax temperature
  float policy_softmax_temp = 1.359f;
  float root_policy_softmax_temp = 1.6f;

  // Dirichlet exploration noise (disabled for competitive play)
  bool add_dirichlet_noise = false;
  float noise_epsilon = 0.0f;
  float noise_alpha = 0.3f;

  // Moves left head utility (Lc0 defaults)
  float moves_left_max_effect = 0.0345f;
  float moves_left_threshold = 0.80f;
  float moves_left_slope = 0.0027f;
  float moves_left_constant_factor = 0.0f;
  float moves_left_scaled_factor = 1.6521f;
  float moves_left_quadratic_factor = -0.6521f;

  // Search control
  int max_concurrent_searchers = 1;
  int thread_idling_threshold = 1;
  float smart_pruning_factor = 1.33f;
  int smart_pruning_minimum_batches = 0;
  int solid_tree_threshold = 100;
  bool two_fold_draws = true;
  bool sticky_endgames = true;
  float draw_score = 0.0f;

  // WDL rescaling (ratio=1.0 means no rescaling)
  float wdl_rescale_ratio = 1.0f;
  float wdl_rescale_diff = 0.0f;

  // Temperature for final move selection (0 = always best, >0 = sample)
  float temperature = 0.0f;
  float temp_winpct_cutoff = 100.0f;
  bool high_policy_root_lever_selection = true;
  bool low_policy_root_lever_selection = true;
  bool root_tactical_capture_probe = true;
  bool low_visit_q_override_rescan = true;
  int fixed_movetime_q_override_cap = 0;

  // Contempt (positive = avoid draws, negative = prefer draws)
  float contempt = 0.0f;

  // Threading
  int num_threads = 2;
  int virtual_loss = 1;
  int minibatch_size = 32;
  bool minibatch_size_auto = false;

  // Time management (Lc0 defaults)
  std::string time_manager = "smooth";
  float slowmover = 2.2f;
  float move_overhead_ms = 10.0f;
  float alphazero_time_pct = 12.0f;

  // Minibatch gathering (Lc0 defaults)
  int max_prefetch = 32;
  float max_out_of_order_evals_factor = 4.0f;
  int max_collision_events = 917;
  int max_collision_visits = 80000;
  int max_collision_visits_scaling_start = 28;
  int max_collision_visits_scaling_end = 145000;
  float max_collision_visits_scaling_power = 1.25f;
  bool out_of_order_eval = true;

  // KLD gain stopper. Lc0 keeps this disabled by default, but MetalFish's
  // Apple Silicon tactical profile benefits from a small early-stop threshold.
  float kld_gain_min = 0.00005f;
  int kld_gain_average_interval = 100;

  // NNCache. Lc0 classic defaults to current-position cache keys.
  int cache_history_length = 0;
  int nn_cache_size = 2000000;

  // Backend
  std::string nn_weights_path;
  std::string nn_backend = "auto";
  std::string coreml_model_path;
  std::string coreml_compute_units = "cpu-ne";
  int cuda_device = -1;
  bool cuda_graph_execution = true;
  int cuda_stable_execution_batch_size = 0;
  bool cuda_deterministic_attention_softmax = true;
  bool cuda_full_buffer_clear = true;

  int GetNumThreads() const {
    if (num_threads <= 0) {
      int hw = static_cast<int>(std::thread::hardware_concurrency());
      return std::min(std::max(2, hw / 4), 4);
    }
    return num_threads;
  }

  float GetCpuct(bool is_root) const { return is_root ? cpuct_at_root : cpuct; }

  float GetCpuctBase(bool is_root) const {
    return is_root ? cpuct_base_at_root : cpuct_base;
  }

  float GetCpuctFactor(bool is_root) const {
    return is_root ? cpuct_factor_at_root : cpuct_factor;
  }

  bool GetFpuAbsolute(bool is_root) const {
    return is_root ? fpu_absolute_at_root : fpu_absolute;
  }

  float GetFpuValue(bool is_root) const {
    return is_root ? fpu_value_at_root : fpu_value;
  }

  NN::BackendConfig GetBackendConfig() const {
    NN::BackendConfig config;
    config.backend = nn_backend;
    config.coreml_model_path = coreml_model_path;
    config.coreml_compute_units = coreml_compute_units;
    config.cuda_device = cuda_device;
    config.cuda_graph_execution = cuda_graph_execution;
    config.cuda_stable_execution_batch_size = cuda_stable_execution_batch_size;
    config.cuda_deterministic_attention_softmax =
        cuda_deterministic_attention_softmax;
    config.cuda_full_buffer_clear = cuda_full_buffer_clear;
    return config;
  }
};

} // namespace MCTS
} // namespace MetalFish
