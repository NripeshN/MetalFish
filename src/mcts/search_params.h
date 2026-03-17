/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Search Parameters

  Licensed under GPL-3.0
*/

#pragma once

#include <algorithm>
#include <string>
#include <thread>

namespace MetalFish {
namespace MCTS {

struct SearchParams {
    // PUCT exploration (Lc0 defaults)
    float cpuct = 1.75f;
    float cpuct_at_root = 1.75f;
    float cpuct_base = 38739.0f;
    float cpuct_factor = 3.89f;

    // First Play Urgency (Lc0 defaults: reduction strategy, same at root)
    bool  fpu_absolute = false;
    float fpu_value = 0.33f;
    float fpu_reduction = 0.33f;
    float fpu_reduction_at_root = 1.0f;

    // Policy softmax temperature
    float policy_softmax_temp = 1.36f;

    // Dirichlet exploration noise (disabled for competitive play)
    bool  add_dirichlet_noise = false;
    float noise_epsilon = 0.0f;
    float noise_alpha = 0.3f;

    // Moves left head utility (Lc0 defaults)
    float moves_left_max_effect = 0.03f;
    float moves_left_threshold = 0.80f;
    float moves_left_slope = 0.00f;
    float moves_left_constant_factor = 0.0f;
    float moves_left_scaled_factor = 1.65f;
    float moves_left_quadratic_factor = -0.65f;

    // Search control
    int   max_concurrent_searchers = 1;
    int   thread_idling_threshold = 1;
    float smart_pruning_factor = 1.33f;
    int   solid_tree_threshold = 100;
    bool  two_fold_draws = true;
    bool  sticky_endgames = true;
    float draw_score = 0.0f;

    // Threading
    int num_threads = 2;
    int virtual_loss = 1;
    int minibatch_size = 256;

    // Time management (Lc0 defaults)
    float slowmover = 2.2f;
    float move_overhead_ms = 200.0f;

    // Minibatch gathering (Lc0 defaults)
    int max_prefetch = 32;
    float max_out_of_order_evals_factor = 2.4f;
    int max_collision_events = 917;
    int max_collision_visits = 80000;
    int max_collision_visits_scaling_start = 28;
    int max_collision_visits_scaling_end = 145000;
    float max_collision_visits_scaling_power = 1.25f;
    bool out_of_order_eval = true;

    // NNCache
    int nn_cache_size = 2000000;

    // Backend
    std::string nn_weights_path;

    int GetNumThreads() const {
        if (num_threads <= 0) {
            int hw = static_cast<int>(std::thread::hardware_concurrency());
            return std::min(std::max(2, hw / 4), 4);
        }
        return num_threads;
    }

    float GetCpuct(bool is_root) const {
        return is_root ? cpuct_at_root : cpuct;
    }
};

} // namespace MCTS
} // namespace MetalFish
