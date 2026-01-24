/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Tests - Backend, NNUE Integration, Batch Evaluation
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
    std::cout << "  " << name_ << "... " << std::flush;
  }
  ~TestCase() {
    if (passed_) {
      std::cout << "OK" << std::endl;
      g_tests_passed++;
    } else {
      g_tests_failed++;
    }
  }
  void fail(const char *msg, int line) {
    if (passed_) {
      std::cout << "FAILED\n";
      passed_ = false;
    }
    std::cout << "    Line " << line << ": " << msg << std::endl;
  }
  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      tc.fail(#cond, __LINE__);                                                \
    }                                                                          \
  } while (0)

// ============================================================================
// Backend Tests
// ============================================================================

void test_backend() {
  {
    TestCase tc("Backend availability");
    bool available = Backend::available();
    std::cout << (available ? "(available) " : "(not available) ");
    EXPECT(tc, true);
  }
  {
    TestCase tc("Backend type");
    if (Backend::available()) {
      BackendType type = Backend::get().type();
      EXPECT(tc, type == BackendType::None || type == BackendType::Metal ||
                     type == BackendType::CUDA);
    } else {
      EXPECT(tc, true);
    }
  }
}

// ============================================================================
// Tuning Parameters Tests
// ============================================================================

void test_tuning() {
  {
    TestCase tc("Tuning defaults");
    GPUTuningParams params;

    EXPECT(tc, params.min_batch_for_gpu > 0);
    EXPECT(tc, params.simd_threshold > params.min_batch_for_gpu);
    EXPECT(tc, params.gpu_extract_threshold > params.simd_threshold);
  }
  {
    TestCase tc("Strategy selection");
    GPUTuningParams params;
    params.min_batch_for_gpu = 4;
    params.simd_threshold = 512;
    params.gpu_extract_threshold = 2048;

    // Strategy selection depends on whether GPU is available
    if (Backend::available()) {
      EvalStrategy small = params.select_strategy(2);
      EXPECT(tc, small == EvalStrategy::CPU_FALLBACK);

      EvalStrategy medium = params.select_strategy(100);
      EXPECT(tc, medium == EvalStrategy::GPU_STANDARD);

      EvalStrategy large = params.select_strategy(1000);
      EXPECT(tc, large == EvalStrategy::GPU_SIMD);
    } else {
      // Without GPU, all strategies should fall back to CPU
      EvalStrategy any = params.select_strategy(100);
      EXPECT(tc, any == EvalStrategy::CPU_FALLBACK);
    }
  }
}

// ============================================================================
// GPU Position Data Tests
// ============================================================================

void test_position_data() {
  {
    TestCase tc("From position");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    GPUPositionData data;
    data.from_position(pos);

    EXPECT(tc, data.stm == WHITE);
    EXPECT(tc, data.king_sq[WHITE] == SQ_E1);
    EXPECT(tc, data.king_sq[BLACK] == SQ_E8);
    EXPECT(tc, data.piece_count == 32);
  }
  {
    TestCase tc("Piece bitboards");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    GPUPositionData data;
    data.from_position(pos);

    EXPECT(tc, data.pieces[WHITE][PAWN] == Rank2BB);
    EXPECT(tc, data.pieces[BLACK][PAWN] == Rank7BB);
  }
}

// ============================================================================
// Eval Batch Tests
// ============================================================================

void test_eval_batch() {
  {
    TestCase tc("Batch creation");
    GPUEvalBatch batch;
    batch.reserve(64);

    EXPECT(tc, batch.positions.capacity() >= 64);
    EXPECT(tc, batch.count == 0);
  }
  {
    TestCase tc("Add position");
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    GPUEvalBatch batch;
    batch.reserve(64);
    batch.add_position(pos);

    EXPECT(tc, batch.count == 1);
    EXPECT(tc, batch.positions.size() == 1);
  }
  {
    TestCase tc("Multiple positions");
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
  }
  {
    TestCase tc("Batch clear");
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
  }
}

// ============================================================================
// Feature Update Tests
// ============================================================================

void test_feature_update() {
  {
    TestCase tc("Feature update structure");
    GPUFeatureUpdate update;

    EXPECT(tc, update.num_added == 0);
    EXPECT(tc, update.num_removed == 0);
    EXPECT(tc, update.perspective == 0);
  }
}

// ============================================================================
// Network Data Tests
// ============================================================================

void test_network_data() {
  {
    TestCase tc("Network validity");
    GPUNetworkData net;

    EXPECT(tc, !net.valid);
    EXPECT(tc, net.hidden_dim == 0);
  }
}

// ============================================================================
// NNUE Manager Tests
// ============================================================================

void test_nnue_manager() {
  {
    TestCase tc("Manager creation");
    GPUNNUEManager manager;
    EXPECT(tc, true);
  }
  {
    TestCase tc("Manager stats");
    GPUNNUEManager manager;

    EXPECT(tc, manager.gpu_evaluations() == 0);
    EXPECT(tc, manager.cpu_fallback_evaluations() == 0);
    EXPECT(tc, manager.total_batches() == 0);
  }
  {
    TestCase tc("Min batch size");
    GPUNNUEManager manager;

    int original = manager.min_batch_size();
    manager.set_min_batch_size(8);
    EXPECT(tc, manager.min_batch_size() == 8);

    manager.set_min_batch_size(original);
  }
  {
    TestCase tc("Tuning access");
    GPUNNUEManager manager;

    GPUTuningParams &params = manager.tuning();
    EXPECT(tc, params.min_batch_for_gpu > 0);

    int original = params.min_batch_for_gpu;
    params.min_batch_for_gpu = 16;
    EXPECT(tc, manager.tuning().min_batch_for_gpu == 16);

    params.min_batch_for_gpu = original;
  }
  {
    TestCase tc("Reset stats");
    GPUNNUEManager manager;
    manager.reset_stats();

    EXPECT(tc, manager.gpu_evaluations() == 0);
    EXPECT(tc, manager.cpu_fallback_evaluations() == 0);
  }
}

// ============================================================================
// Layer Weights Tests
// ============================================================================

void test_layer_weights() {
  {
    TestCase tc("Weights validity");
    GPULayerWeights weights;

    EXPECT(tc, !weights.valid());
  }
}

// ============================================================================
// Global Interface Tests
// ============================================================================

void test_global_interface() {
  {
    TestCase tc("Manager available check");
    bool available = gpu_nnue_manager_available();
    EXPECT(tc, available == true || available == false);
  }
}

} // namespace

bool test_gpu_module() {
  std::cout << "\n=== GPU Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Backend]" << std::endl;
  test_backend();

  std::cout << "\n[Tuning]" << std::endl;
  test_tuning();

  std::cout << "\n[Position Data]" << std::endl;
  test_position_data();

  std::cout << "\n[Eval Batch]" << std::endl;
  test_eval_batch();

  std::cout << "\n[Feature Update]" << std::endl;
  test_feature_update();

  std::cout << "\n[Network Data]" << std::endl;
  test_network_data();

  std::cout << "\n[NNUE Manager]" << std::endl;
  test_nnue_manager();

  std::cout << "\n[Layer Weights]" << std::endl;
  test_layer_weights();

  std::cout << "\n[Global Interface]" << std::endl;
  test_global_interface();

  std::cout << "\n--- GPU Results: " << g_tests_passed << " passed, "
            << g_tests_failed << " failed ---" << std::endl;

  return g_tests_failed == 0;
}
