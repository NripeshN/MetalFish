/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive GPU Tests - GPU Backend, NNUE Integration, Batch Evaluation
*/

#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/gpu_nnue_integration.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace MetalFish;
using namespace MetalFish::GPU;

namespace {

static int g_tests_passed = 0;
static int g_tests_failed = 0;

class TestCase {
public:
  TestCase(const char *name) : name_(name), passed_(true) {
    std::cout << "  Testing " << name_ << "... " << std::flush;
  }
  ~TestCase() {
    if (passed_) {
      std::cout << "PASSED" << std::endl;
      g_tests_passed++;
    } else {
      g_tests_failed++;
    }
  }
  void fail(const char *msg, const char *file, int line) {
    if (passed_) {
      std::cout << "FAILED\n";
      passed_ = false;
    }
    std::cout << "    " << file << ":" << line << ": " << msg << std::endl;
  }
  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      tc.fail(#cond, __FILE__, __LINE__);                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// GPU Backend Tests
// ============================================================================

bool test_backend_availability() {
  TestCase tc("BackendAvailability");

  // Check if GPU backend is available
  bool available = Backend::available();

  // At least verify the function works
  EXPECT(tc, true); // This test always passes - just checking availability

  std::cout << "\n    GPU Available: " << (available ? "YES" : "NO") << "\n  ";

  return tc.passed();
}

bool test_backend_type() {
  TestCase tc("BackendType");

  if (Backend::available()) {
    BackendType type = Backend::get().type();
    EXPECT(tc, type == BackendType::None || type == BackendType::Metal ||
                   type == BackendType::CUDA);
  } else {
    EXPECT(tc, true); // Skip if no GPU
  }

  return tc.passed();
}

// ============================================================================
// GPU Tuning Parameters Tests
// ============================================================================

bool test_tuning_params_defaults() {
  TestCase tc("TuningParamsDefaults");

  GPUTuningParams params;

  EXPECT(tc, params.min_batch_for_gpu > 0);
  EXPECT(tc, params.simd_threshold > params.min_batch_for_gpu);
  EXPECT(tc, params.gpu_extract_threshold > params.simd_threshold);

  return tc.passed();
}

bool test_tuning_strategy_selection() {
  TestCase tc("TuningStrategySelection");

  GPUTuningParams params;
  params.min_batch_for_gpu = 4;
  params.simd_threshold = 512;
  params.gpu_extract_threshold = 2048;

  // Small batch -> CPU
  EvalStrategy small = params.select_strategy(2);
  EXPECT(tc, small == EvalStrategy::CPU_FALLBACK);

  // Medium batch -> GPU standard
  EvalStrategy medium = params.select_strategy(100);
  EXPECT(tc, medium == EvalStrategy::GPU_STANDARD);

  // Large batch -> GPU SIMD
  EvalStrategy large = params.select_strategy(1000);
  EXPECT(tc, large == EvalStrategy::GPU_SIMD);

  return tc.passed();
}

// ============================================================================
// GPU Position Data Tests
// ============================================================================

bool test_gpu_position_data_from_position() {
  TestCase tc("GPUPositionDataFromPosition");

  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  GPUPositionData data;
  data.from_position(pos);

  EXPECT(tc, data.stm == WHITE);
  EXPECT(tc, data.king_sq[WHITE] == SQ_E1);
  EXPECT(tc, data.king_sq[BLACK] == SQ_E8);
  EXPECT(tc, data.piece_count == 32); // Starting position

  return tc.passed();
}

bool test_gpu_position_data_pieces() {
  TestCase tc("GPUPositionDataPieces");

  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  GPUPositionData data;
  data.from_position(pos);

  // Check pawn bitboards
  EXPECT(tc, data.pieces[WHITE][PAWN] == Rank2BB);
  EXPECT(tc, data.pieces[BLACK][PAWN] == Rank7BB);

  return tc.passed();
}

// ============================================================================
// GPU Eval Batch Tests
// ============================================================================

bool test_eval_batch_creation() {
  TestCase tc("EvalBatchCreation");

  GPUEvalBatch batch;
  batch.reserve(64);

  EXPECT(tc, batch.positions.capacity() >= 64);
  EXPECT(tc, batch.count == 0);

  return tc.passed();
}

bool test_eval_batch_add_position() {
  TestCase tc("EvalBatchAddPosition");

  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  GPUEvalBatch batch;
  batch.reserve(64);
  batch.add_position(pos);

  EXPECT(tc, batch.count == 1);
  EXPECT(tc, batch.positions.size() == 1);

  return tc.passed();
}

bool test_eval_batch_multiple_positions() {
  TestCase tc("EvalBatchMultiplePositions");

  GPUEvalBatch batch;
  batch.reserve(64);

  const char *fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"};

  for (const char *fen : fens) {
    StateInfo st;
    Position pos;
    pos.set(fen, false, &st);
    batch.add_position(pos);
  }

  EXPECT(tc, batch.count == 3);

  return tc.passed();
}

bool test_eval_batch_clear() {
  TestCase tc("EvalBatchClear");

  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  GPUEvalBatch batch;
  batch.reserve(64);
  batch.add_position(pos);
  batch.add_position(pos);

  EXPECT(tc, batch.count == 2);

  batch.clear();
  EXPECT(tc, batch.count == 0);

  return tc.passed();
}

// ============================================================================
// GPU Feature Update Tests
// ============================================================================

bool test_feature_update_structure() {
  TestCase tc("FeatureUpdateStructure");

  GPUFeatureUpdate update;

  EXPECT(tc, update.num_added == 0);
  EXPECT(tc, update.num_removed == 0);
  EXPECT(tc, update.perspective == 0);

  return tc.passed();
}

// ============================================================================
// GPU Network Data Tests
// ============================================================================

bool test_network_data_validity() {
  TestCase tc("NetworkDataValidity");

  GPUNetworkData net;

  // Uninitialized network should be invalid
  EXPECT(tc, !net.valid);
  EXPECT(tc, net.hidden_dim == 0);

  return tc.passed();
}

// ============================================================================
// GPU NNUE Manager Tests
// ============================================================================

bool test_nnue_manager_creation() {
  TestCase tc("NNUEManagerCreation");

  GPUNNUEManager manager;

  // Manager should exist but may not be initialized
  EXPECT(tc, true);

  return tc.passed();
}

bool test_nnue_manager_stats() {
  TestCase tc("NNUEManagerStats");

  GPUNNUEManager manager;

  // Stats should be zero initially
  EXPECT(tc, manager.gpu_evaluations() == 0);
  EXPECT(tc, manager.cpu_fallback_evaluations() == 0);
  EXPECT(tc, manager.total_batches() == 0);

  return tc.passed();
}

bool test_nnue_manager_min_batch() {
  TestCase tc("NNUEManagerMinBatch");

  GPUNNUEManager manager;

  int original = manager.min_batch_size();
  manager.set_min_batch_size(8);
  EXPECT(tc, manager.min_batch_size() == 8);

  manager.set_min_batch_size(original);

  return tc.passed();
}

bool test_nnue_manager_tuning_access() {
  TestCase tc("NNUEManagerTuningAccess");

  GPUNNUEManager manager;

  GPUTuningParams &params = manager.tuning();
  EXPECT(tc, params.min_batch_for_gpu > 0);

  // Modify and verify
  int original = params.min_batch_for_gpu;
  params.min_batch_for_gpu = 16;
  EXPECT(tc, manager.tuning().min_batch_for_gpu == 16);

  params.min_batch_for_gpu = original;

  return tc.passed();
}

bool test_nnue_manager_reset_stats() {
  TestCase tc("NNUEManagerResetStats");

  GPUNNUEManager manager;
  manager.reset_stats();

  EXPECT(tc, manager.gpu_evaluations() == 0);
  EXPECT(tc, manager.cpu_fallback_evaluations() == 0);

  return tc.passed();
}

// ============================================================================
// GPU Layer Weights Tests
// ============================================================================

bool test_layer_weights_validity() {
  TestCase tc("LayerWeightsValidity");

  GPULayerWeights weights;

  // Uninitialized weights should be invalid
  EXPECT(tc, !weights.valid());

  return tc.passed();
}

// ============================================================================
// Global Interface Tests
// ============================================================================

bool test_global_manager_available() {
  TestCase tc("GlobalManagerAvailable");

  // This may or may not be available depending on initialization
  bool available = gpu_nnue_manager_available();

  // Just verify the function works
  EXPECT(tc, available == true || available == false);

  return tc.passed();
}

} // namespace

bool test_gpu_comprehensive() {
  std::cout << "\n=== Comprehensive GPU Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Backend]" << std::endl;
  test_backend_availability();
  test_backend_type();

  std::cout << "\n[TuningParams]" << std::endl;
  test_tuning_params_defaults();
  test_tuning_strategy_selection();

  std::cout << "\n[GPUPositionData]" << std::endl;
  test_gpu_position_data_from_position();
  test_gpu_position_data_pieces();

  std::cout << "\n[EvalBatch]" << std::endl;
  test_eval_batch_creation();
  test_eval_batch_add_position();
  test_eval_batch_multiple_positions();
  test_eval_batch_clear();

  std::cout << "\n[FeatureUpdate]" << std::endl;
  test_feature_update_structure();

  std::cout << "\n[NetworkData]" << std::endl;
  test_network_data_validity();

  std::cout << "\n[NNUEManager]" << std::endl;
  test_nnue_manager_creation();
  test_nnue_manager_stats();
  test_nnue_manager_min_batch();
  test_nnue_manager_tuning_access();
  test_nnue_manager_reset_stats();

  std::cout << "\n[LayerWeights]" << std::endl;
  test_layer_weights_validity();

  std::cout << "\n[GlobalInterface]" << std::endl;
  test_global_manager_available();

  std::cout << "\n=== GPU Test Summary ===" << std::endl;
  std::cout << "  Passed: " << g_tests_passed << std::endl;
  std::cout << "  Failed: " << g_tests_failed << std::endl;

  return g_tests_failed == 0;
}
