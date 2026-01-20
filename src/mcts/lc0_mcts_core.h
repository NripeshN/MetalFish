/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Lc0-style MCTS Core Algorithms
  
  This module implements the core MCTS algorithms from Leela Chess Zero (Lc0),
  adapted for MetalFish's architecture. These algorithms represent the 
  state-of-the-art in MCTS for chess.
  
  Key algorithms from Lc0:
  1. PUCT with logarithmic growth factor
  2. FPU (First Play Urgency) with reduction strategy  
  3. Moves Left Head (MLH) utility for preferring shorter wins
  4. WDL (Win/Draw/Loss) handling
  5. Dirichlet noise for exploration
  6. Policy softmax temperature
  
  Licensed under GPL-3.0
  
  Based on Leela Chess Zero (https://github.com/LeelaChessZero/lc0)
  Original Lc0 code is also GPL-3.0 licensed.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// Lc0-style Search Parameters
// ============================================================================

struct Lc0SearchParams {
  // PUCT parameters (from Lc0 defaults)
  float cpuct = 1.745f;              // Base PUCT constant
  float cpuct_base = 38739.0f;       // Base for logarithmic growth
  float cpuct_factor = 3.894f;       // Multiplier for log term
  
  // Root-specific PUCT (can be different from non-root)
  float cpuct_at_root = 1.745f;
  float cpuct_base_at_root = 38739.0f;
  float cpuct_factor_at_root = 3.894f;
  
  // FPU (First Play Urgency) parameters
  bool fpu_absolute = false;         // If true, use absolute FPU value
  float fpu_value = 0.330f;          // FPU reduction value
  bool fpu_absolute_at_root = false;
  float fpu_value_at_root = 1.0f;
  
  // Moves Left Head (MLH) parameters
  float moves_left_max_effect = 0.0345f;
  float moves_left_threshold = 0.8f;
  float moves_left_slope = 0.0027f;
  float moves_left_constant_factor = 0.0f;
  float moves_left_scaled_factor = 1.6521f;
  float moves_left_quadratic_factor = -0.6521f;
  
  // Dirichlet noise parameters
  float noise_epsilon = 0.0f;        // 0 = disabled, 0.25 = typical for training
  float noise_alpha = 0.3f;          // Dirichlet alpha
  
  // Policy softmax temperature
  float policy_softmax_temp = 1.0f;
  
  // Draw score (for contempt)
  float draw_score = 0.0f;
  
  // Virtual loss
  int virtual_loss = 3;
  
  // Two-fold draw detection
  bool two_fold_draws = true;
  
  // Sticky endgames (propagate terminal status)
  bool sticky_endgames = true;
};

// ============================================================================
// Fast Math Utilities (from Lc0)
// ============================================================================

namespace FastMath {

// Fast natural logarithm approximation
inline float FastLog(float x) {
  // Use standard log for accuracy - can optimize later if needed
  return std::log(x);
}

// Fast sign function
inline float FastSign(float x) {
  return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

// Fast logistic function: 1 / (1 + exp(-x))
inline float FastLogistic(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Fast tanh using standard library
inline float FastTanh(float x) {
  return std::tanh(x);
}

} // namespace FastMath

// ============================================================================
// PUCT Computation (from Lc0)
// ============================================================================

// Computes the PUCT exploration constant with logarithmic growth
// Formula: cpuct_init + cpuct_factor * log((N + cpuct_base) / cpuct_base)
inline float ComputeCpuct(const Lc0SearchParams& params, uint32_t N, bool is_root) {
  const float init = is_root ? params.cpuct_at_root : params.cpuct;
  const float k = is_root ? params.cpuct_factor_at_root : params.cpuct_factor;
  const float base = is_root ? params.cpuct_base_at_root : params.cpuct_base;
  
  if (k == 0.0f) {
    return init;
  }
  return init + k * FastMath::FastLog((static_cast<float>(N) + base) / base);
}

// ============================================================================
// FPU (First Play Urgency) Computation (from Lc0)
// ============================================================================

// Computes the FPU value for unvisited nodes
// Reduction strategy: parent_Q - fpu_value * sqrt(visited_policy)
// Absolute strategy: just return fpu_value
inline float ComputeFpu(const Lc0SearchParams& params, float parent_q, 
                        float visited_policy, bool is_root, float draw_score) {
  const bool use_absolute = is_root ? params.fpu_absolute_at_root : params.fpu_absolute;
  const float value = is_root ? params.fpu_value_at_root : params.fpu_value;
  
  if (use_absolute) {
    return value;
  }
  // Reduction strategy: start from parent's Q and reduce based on visited policy
  return parent_q - value * std::sqrt(visited_policy);
}

// Simplified FPU when visited_policy is not available
inline float ComputeFpuSimple(const Lc0SearchParams& params, float parent_q, 
                               bool is_root) {
  const bool use_absolute = is_root ? params.fpu_absolute_at_root : params.fpu_absolute;
  const float value = is_root ? params.fpu_value_at_root : params.fpu_value;
  
  if (use_absolute) {
    return value;
  }
  // Use a default visited_policy estimate
  return parent_q - value * 0.5f;
}

// ============================================================================
// Moves Left Head (MLH) Utility (from Lc0)
// ============================================================================

class MovesLeftEvaluator {
public:
  MovesLeftEvaluator() : enabled_(false) {}
  
  MovesLeftEvaluator(const Lc0SearchParams& params, float parent_m = 0.0f,
                     float parent_q = 0.0f)
      : enabled_(params.moves_left_max_effect > 0.0f),
        m_slope_(params.moves_left_slope),
        m_cap_(params.moves_left_max_effect),
        a_constant_(params.moves_left_constant_factor),
        a_linear_(params.moves_left_scaled_factor),
        a_square_(params.moves_left_quadratic_factor),
        q_threshold_(params.moves_left_threshold),
        parent_m_(parent_m),
        parent_within_threshold_(std::abs(parent_q) > params.moves_left_threshold) {}
  
  void SetParent(float parent_m, float parent_q) {
    parent_m_ = parent_m;
    parent_within_threshold_ = std::abs(parent_q) > q_threshold_;
  }
  
  // Calculates the utility for favoring shorter wins and longer losses
  // Returns a bonus/penalty to add to the Q value
  float GetMUtility(float child_m, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    
    // Calculate raw M effect
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    
    // Sign based on whether we're winning or losing
    m *= FastMath::FastSign(-q);
    
    // Scale based on Q magnitude if threshold is set
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      float q_scaled = std::max(0.0f, (std::abs(q) - q_threshold_)) / 
                       (1.0f - q_threshold_);
      m *= a_constant_ + a_linear_ * std::abs(q_scaled) + 
           a_square_ * q_scaled * q_scaled;
    } else {
      m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    }
    
    return m;
  }
  
  // Default M utility for unvisited nodes
  float GetDefaultMUtility() const { return 0.0f; }
  
  bool IsEnabled() const { return enabled_; }

private:
  bool enabled_;
  float m_slope_ = 0.0f;
  float m_cap_ = 0.0f;
  float a_constant_ = 0.0f;
  float a_linear_ = 0.0f;
  float a_square_ = 0.0f;
  float q_threshold_ = 0.0f;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

// ============================================================================
// Dirichlet Noise (from Lc0)
// ============================================================================

// Applies Dirichlet noise to policy priors at the root
// This encourages exploration during training/analysis
template<typename EdgeArray>
void ApplyDirichletNoise(EdgeArray& edges, int num_edges, float epsilon, 
                         float alpha, std::mt19937& rng) {
  if (epsilon <= 0.0f || num_edges == 0) return;
  
  std::gamma_distribution<float> gamma(alpha, 1.0f);
  
  std::vector<float> noise(num_edges);
  float total = 0.0f;
  
  for (int i = 0; i < num_edges; ++i) {
    noise[i] = gamma(rng);
    total += noise[i];
  }
  
  if (total < std::numeric_limits<float>::min()) return;
  
  // Mix noise with existing policy
  for (int i = 0; i < num_edges; ++i) {
    float current_p = edges[i].GetPolicy();
    float noisy_p = (1.0f - epsilon) * current_p + 
                    epsilon * (noise[i] / total);
    edges[i].SetPolicy(noisy_p);
  }
}

// ============================================================================
// PUCT Selection (from Lc0)
// ============================================================================

// Full PUCT selection with all Lc0 features
struct PuctSelectionResult {
  int best_idx = -1;
  float best_score = -1e9f;
  int second_best_idx = -1;
  float second_best_score = -1e9f;
  int visits_until_change = std::numeric_limits<int>::max();
};

// Computes PUCT score for a single edge
// Score = Q + U + M
// Where:
//   Q = value estimate (from child or FPU)
//   U = exploration bonus = cpuct * sqrt(parent_N) * P / (1 + child_N)
//   M = moves left utility (optional)
inline float ComputePuctScore(float q, float policy, float cpuct_sqrt_n, 
                               int child_n_started, float m_utility = 0.0f) {
  float u = cpuct_sqrt_n * policy / (1.0f + static_cast<float>(child_n_started));
  return q + u + m_utility;
}

// Select best child using PUCT with all features
template<typename Node, typename GetChildFunc, typename GetPolicyFunc, 
         typename GetNStartedFunc>
PuctSelectionResult SelectBestChildPuct(
    Node* parent,
    int num_edges,
    const Lc0SearchParams& params,
    bool is_root,
    float draw_score,
    GetChildFunc get_child,      // (int idx) -> child node or nullptr
    GetPolicyFunc get_policy,    // (int idx) -> float policy
    GetNStartedFunc get_n_started // (int idx) -> int n_started
) {
  PuctSelectionResult result;
  
  if (num_edges == 0) return result;
  
  // Get parent statistics
  uint32_t parent_n = parent->GetN() + parent->GetNInFlight();
  float parent_q = parent->GetN() > 0 ? parent->GetQ(draw_score) : 0.0f;
  
  // Compute CPUCT with logarithmic growth
  float cpuct = ComputeCpuct(params, parent_n, is_root);
  float cpuct_sqrt_n = cpuct * std::sqrt(static_cast<float>(
      std::max(parent->GetChildrenVisits(), 1u)));
  
  // Compute visited policy for FPU
  float visited_policy = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    auto* child = get_child(i);
    if (child && child->GetN() > 0) {
      visited_policy += get_policy(i);
    }
  }
  
  // Compute FPU
  float fpu = ComputeFpu(params, parent_q, visited_policy, is_root, draw_score);
  
  // Set up moves left evaluator if parent has M value
  MovesLeftEvaluator m_eval;
  if (params.moves_left_max_effect > 0.0f) {
    float parent_m = parent->GetM();
    m_eval = MovesLeftEvaluator(params, parent_m, parent_q);
  }
  
  // Find best and second-best children
  for (int i = 0; i < num_edges; ++i) {
    auto* child = get_child(i);
    float policy = get_policy(i);
    int n_started = get_n_started(i);
    
    float q, m_utility = 0.0f;
    
    if (child && child->GetN() > 0) {
      // Use child's Q value (negated because it's from opponent's perspective)
      q = -child->GetQ(draw_score);
      
      // Add moves left utility if enabled
      if (m_eval.IsEnabled()) {
        m_utility = m_eval.GetMUtility(child->GetM(), q);
      }
    } else {
      // Use FPU for unvisited nodes
      q = fpu;
      m_utility = m_eval.GetDefaultMUtility();
    }
    
    float score = ComputePuctScore(q, policy, cpuct_sqrt_n, n_started, m_utility);
    
    if (score > result.best_score) {
      result.second_best_idx = result.best_idx;
      result.second_best_score = result.best_score;
      result.best_idx = i;
      result.best_score = score;
    } else if (score > result.second_best_score) {
      result.second_best_idx = i;
      result.second_best_score = score;
    }
  }
  
  // Calculate visits until best child might change
  if (result.second_best_idx >= 0 && result.best_idx >= 0) {
    float best_q = result.best_score - 
                   cpuct_sqrt_n * get_policy(result.best_idx) / 
                   (1.0f + static_cast<float>(get_n_started(result.best_idx)));
    
    if (result.second_best_score > best_q) {
      int n1 = get_n_started(result.best_idx) + 1;
      float p1 = get_policy(result.best_idx);
      float visits_f = p1 * cpuct_sqrt_n / (result.second_best_score - best_q) - n1 + 1;
      result.visits_until_change = std::max(1, static_cast<int>(visits_f));
    }
  }
  
  return result;
}

// ============================================================================
// Best Move Selection (from Lc0)
// ============================================================================

enum class EdgeRank {
  kTerminalLoss,
  kTablebaseLoss,
  kNonTerminal,  // Non-terminal or terminal draw
  kTablebaseWin,
  kTerminalWin,
};

// Determines the rank of an edge for best move selection
template<typename Edge>
EdgeRank GetEdgeRank(const Edge& edge) {
  if (edge.GetN() == 0 || !edge.IsTerminal()) {
    return EdgeRank::kNonTerminal;
  }
  
  float wl = edge.GetWL();
  if (edge.IsTbTerminal()) {
    return wl < 0.0f ? EdgeRank::kTablebaseLoss : EdgeRank::kTablebaseWin;
  }
  return wl < 0.0f ? EdgeRank::kTerminalLoss : EdgeRank::kTerminalWin;
}

// Compare two edges for best move selection
// Returns true if 'a' is preferred over 'b'
template<typename Edge>
bool CompareEdgesForBestMove(const Edge& a, const Edge& b, float draw_score) {
  auto a_rank = GetEdgeRank(a);
  auto b_rank = GetEdgeRank(b);
  
  // Prefer better outcome
  if (a_rank != b_rank) return a_rank > b_rank;
  
  // Both terminal draws - prefer shorter
  if (a_rank == EdgeRank::kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
      a.IsTerminal() && b.IsTerminal()) {
    return a.GetM() < b.GetM();
  }
  
  // Neither terminal - use standard rules
  if (a_rank == EdgeRank::kNonTerminal) {
    // Prefer more visits
    if (a.GetN() != b.GetN()) return a.GetN() > b.GetN();
    // Then prefer better Q
    if (a.GetQ(draw_score) != b.GetQ(draw_score)) {
      return a.GetQ(draw_score) > b.GetQ(draw_score);
    }
    // Then prefer higher policy
    return a.GetP() > b.GetP();
  }
  
  // Both winning - prefer shorter win
  if (a_rank > EdgeRank::kNonTerminal) {
    return a.GetM() < b.GetM();
  }
  
  // Both losing - prefer longer loss
  return a.GetM() > b.GetM();
}

// ============================================================================
// WDL (Win/Draw/Loss) Utilities
// ============================================================================

// Convert Q value to centipawn score (Lc0 style)
inline int QToCentipawns(float q) {
  // Lc0 uses: 90 * tan(1.5637541897 * q)
  // Clamp q to avoid infinity
  q = std::clamp(q, -0.99f, 0.99f);
  return static_cast<int>(90.0f * std::tan(1.5637541897f * q));
}

// Convert centipawn score to Q value
inline float CentipawnsToQ(int cp) {
  // Inverse of QToCentipawns
  return std::atan(cp / 90.0f) / 1.5637541897f;
}

// Compute WDL probabilities from Q and D
struct WDL {
  float win;
  float draw;
  float loss;
};

inline WDL ComputeWDL(float q, float d) {
  WDL wdl;
  wdl.draw = std::max(0.0f, d);
  wdl.win = std::max(0.0f, (1.0f + q - d) / 2.0f);
  wdl.loss = std::max(0.0f, (1.0f - q - d) / 2.0f);
  
  // Normalize to ensure they sum to 1
  float sum = wdl.win + wdl.draw + wdl.loss;
  if (sum > 0.0f) {
    wdl.win /= sum;
    wdl.draw /= sum;
    wdl.loss /= sum;
  }
  return wdl;
}

// ============================================================================
// Score Transformation (NNUE to MCTS Q value)
// ============================================================================

// Convert NNUE centipawn score to MCTS Q value in [-1, 1]
// This uses a sigmoid-like transformation calibrated for chess
inline float NnueScoreToQ(int score) {
  // Clamp to reasonable range
  float cp = std::clamp(static_cast<float>(score), -3000.0f, 3000.0f);
  
  // Use Lc0-style conversion: Q = tanh(cp / 300)
  // This gives:
  //   100cp (1 pawn) -> Q ≈ 0.32
  //   300cp (3 pawns) -> Q ≈ 0.76
  //   500cp (5 pawns) -> Q ≈ 0.93
  return std::tanh(cp / 300.0f);
}

// Convert MCTS Q value back to centipawns
inline int QToNnueScore(float q) {
  // Clamp to avoid infinity
  q = std::clamp(q, -0.999f, 0.999f);
  // Inverse tanh: atanh(q) * 300
  return static_cast<int>(std::atanh(q) * 300.0f);
}

// ============================================================================
// WDL Rescaling (from Lc0) - For Contempt
// ============================================================================

// WDL rescaling parameters for contempt
struct WDLRescaleParams {
  float ratio = 1.0f;
  float diff = 0.0f;
  float max_s = 0.2f;  // Maximum reasonable s value
};

// Rescale WDL based on contempt settings (from Lc0)
// This adjusts the evaluation to prefer wins/draws based on contempt
inline void WDLRescale(float& v, float& d, const WDLRescaleParams& params,
                       float sign = 1.0f, bool invert = false) {
  float ratio = params.ratio;
  float diff = params.diff;
  
  if (invert) {
    diff = -diff;
    ratio = 1.0f / ratio;
  }
  
  float w = (1.0f + v - d) / 2.0f;
  float l = (1.0f - v - d) / 2.0f;
  
  // Safeguard against numerical issues
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && 
      d < (1.0f - eps) && l < (1.0f - eps)) {
    
    float a = FastMath::FastLog(1.0f / l - 1.0f);
    float b = FastMath::FastLog(1.0f / w - 1.0f);
    float s = 2.0f / (a + b);
    
    if (!invert) s = std::min(params.max_s, s);
    
    float mu = (a - b) / (a + b);
    float s_new = s * ratio;
    
    if (invert) {
      std::swap(s, s_new);
      s = std::min(params.max_s, s);
    }
    
    float mu_new = mu + sign * s * s * diff;
    float w_new = FastMath::FastLogistic((-1.0f + mu_new) / s_new);
    float l_new = FastMath::FastLogistic((-1.0f - mu_new) / s_new);
    
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
  }
}

// ============================================================================
// Collision Handling (from Lc0)
// ============================================================================

// Collision tracking for multi-threaded MCTS
struct CollisionStats {
  int max_collision_events = 32;
  int max_collision_visits = 9999;
  
  // Scaling for collision visits based on tree size
  int scaling_start = 28;
  int scaling_end = 145;
  float scaling_power = 1.56f;
  
  // Get max collision visits scaled by tree size
  int GetMaxCollisionVisits(uint32_t tree_size) const {
    if (tree_size < static_cast<uint32_t>(scaling_start)) {
      return 1;
    }
    if (tree_size >= static_cast<uint32_t>(scaling_end)) {
      return max_collision_visits;
    }
    
    float progress = static_cast<float>(tree_size - scaling_start) / 
                     (scaling_end - scaling_start);
    return 1 + static_cast<int>(
        (max_collision_visits - 1) * std::pow(progress, scaling_power));
  }
};

// ============================================================================
// Out-of-Order Evaluation Support (from Lc0)
// ============================================================================

// Node states for out-of-order evaluation
enum class NodeEvalState {
  kNotStarted,      // Not yet selected for evaluation
  kInFlight,        // Selected but not yet evaluated
  kEvaluated,       // Evaluation complete, ready for backprop
  kBackpropagated   // Backpropagation complete
};

// Batch item for out-of-order evaluation
struct EvalBatchItem {
  void* node = nullptr;           // Pointer to node being evaluated
  uint16_t depth = 0;             // Depth in tree
  int multivisit = 1;             // Number of visits to apply
  NodeEvalState state = NodeEvalState::kNotStarted;
  
  // Evaluation results
  float wl = 0.0f;                // Win-Loss value
  float d = 0.0f;                 // Draw probability
  float m = 0.0f;                 // Moves left estimate
  
  bool CanEvalOutOfOrder() const {
    // Can evaluate out of order if it's a cache hit or terminal
    return state == NodeEvalState::kEvaluated;
  }
};

// ============================================================================
// Policy Temperature (from Lc0)
// ============================================================================

// Apply temperature to policy for move selection
inline void ApplyPolicyTemperature(std::vector<float>& policy, float temperature) {
  if (temperature == 1.0f || policy.empty()) return;
  
  if (temperature == 0.0f) {
    // Temperature 0 = argmax
    float max_val = *std::max_element(policy.begin(), policy.end());
    for (float& p : policy) {
      p = (p == max_val) ? 1.0f : 0.0f;
    }
    return;
  }
  
  // Apply temperature: p_i = exp(log(p_i) / T) / sum(exp(log(p_j) / T))
  float max_log = -std::numeric_limits<float>::infinity();
  for (float p : policy) {
    if (p > 0.0f) {
      max_log = std::max(max_log, std::log(p) / temperature);
    }
  }
  
  float sum = 0.0f;
  for (float& p : policy) {
    if (p > 0.0f) {
      p = std::exp(std::log(p) / temperature - max_log);
      sum += p;
    }
  }
  
  if (sum > 0.0f) {
    for (float& p : policy) {
      p /= sum;
    }
  }
}

// ============================================================================
// Visit Count Temperature (from Lc0) - For Move Selection
// ============================================================================

// Apply temperature to visit counts for move selection
inline int SelectMoveWithTemperature(const std::vector<uint32_t>& visits,
                                     float temperature,
                                     std::mt19937& rng) {
  if (visits.empty()) return -1;
  
  if (temperature == 0.0f) {
    // Temperature 0 = argmax
    return static_cast<int>(std::distance(
        visits.begin(), std::max_element(visits.begin(), visits.end())));
  }
  
  // Convert visits to probabilities with temperature
  std::vector<float> probs(visits.size());
  float max_log = -std::numeric_limits<float>::infinity();
  
  for (size_t i = 0; i < visits.size(); ++i) {
    if (visits[i] > 0) {
      max_log = std::max(max_log, std::log(static_cast<float>(visits[i])) / temperature);
    }
  }
  
  float sum = 0.0f;
  for (size_t i = 0; i < visits.size(); ++i) {
    if (visits[i] > 0) {
      probs[i] = std::exp(std::log(static_cast<float>(visits[i])) / temperature - max_log);
      sum += probs[i];
    }
  }
  
  if (sum <= 0.0f) {
    // Fallback to uniform
    std::uniform_int_distribution<int> dist(0, static_cast<int>(visits.size()) - 1);
    return dist(rng);
  }
  
  // Sample from distribution
  std::uniform_real_distribution<float> dist(0.0f, sum);
  float r = dist(rng);
  
  for (size_t i = 0; i < probs.size(); ++i) {
    r -= probs[i];
    if (r <= 0.0f) return static_cast<int>(i);
  }
  
  return static_cast<int>(probs.size()) - 1;
}

// ============================================================================
// Node Statistics Update (from Lc0)
// ============================================================================

// Atomically update node statistics after evaluation
// This implements Lc0's finalize score update logic
struct NodeUpdateParams {
  float value;       // V from neural network (or terminal value)
  float draw;        // D from neural network
  float moves_left;  // M from neural network (or 0)
  int multivisit;    // Number of visits to apply
};

// Calculate new Q value after adding visits
inline float CalculateNewQ(float old_wl, uint32_t old_n, 
                           float new_v, int new_visits) {
  uint32_t total_n = old_n + new_visits;
  if (total_n == 0) return 0.0f;
  return (old_wl * old_n + new_v * new_visits) / total_n;
}

// ============================================================================
// Tree Reuse (from Lc0)
// ============================================================================

// Check if a subtree can be reused for the next position
inline bool CanReuseSubtree(uint64_t old_hash, uint64_t new_hash,
                            const std::vector<uint64_t>& move_hashes,
                            uint64_t& best_match_hash) {
  // Direct match
  if (old_hash == new_hash) {
    best_match_hash = old_hash;
    return true;
  }
  
  // Check if new position is reachable from old position
  for (uint64_t hash : move_hashes) {
    if (hash == new_hash) {
      best_match_hash = hash;
      return true;
    }
  }
  
  return false;
}

// ============================================================================
// Solid Tree Optimization (from Lc0)
// ============================================================================

// Threshold for converting linked list children to solid array
// This improves cache locality for frequently accessed subtrees
constexpr int SOLID_TREE_THRESHOLD = 100;

// Check if a node should be solidified
inline bool ShouldSolidify(uint32_t visits, int num_children) {
  return visits >= SOLID_TREE_THRESHOLD && num_children > 0;
}

} // namespace MCTS
} // namespace MetalFish
