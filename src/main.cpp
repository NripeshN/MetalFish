/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include <cstdlib>
#include <iostream>
#include <memory>

#include "core/bitboard.h"
#include "core/misc.h"
#include "core/position.h"
#include "gpu/backend.h"
#include "gpu/gpu_nnue_integration.h"
#include "mcts/ab_integration.h"
#include "search/tune.h"
#include "uci/uci.h"

using namespace MetalFish;

// Global flag to track if cleanup has been done
static std::atomic<bool> g_cleanup_done{false};

// Global cleanup function registered with atexit
// This ensures GPU resources are cleaned up before static destruction
// The order is critical: MCTS components -> GPU NNUE -> GPU Backend
static void cleanup_gpu_resources() {
  // Only cleanup once - prevent double cleanup
  bool expected = false;
  if (!g_cleanup_done.compare_exchange_strong(expected, true)) {
    return; // Already cleaned up
  }

  // 0. Cleanup parallel hybrid search first (it holds GPU resources)
  cleanup_parallel_hybrid_search();

  // 1. Shutdown MCTS components first (they use GPU)
  MCTS::shutdown_hybrid_bridge();

  // 2. Shutdown GPU NNUE manager (this also shuts down the GPU backend)
  GPU::shutdown_gpu_nnue();
}

int main(int argc, char *argv[]) {
  std::cout << engine_info() << std::endl;

  Bitboards::init();
  Position::init();

  // Register cleanup function with atexit
  // This ensures cleanup happens before static destruction
  std::atexit(cleanup_gpu_resources);

  {
    // Scope the UCIEngine to ensure it's destroyed before GPU cleanup
    auto uci = std::make_unique<UCIEngine>(argc, argv);

    Tune::init(uci->engine_options());

    uci->loop();

    // Explicitly destroy the UCI engine before GPU cleanup
    uci.reset();
  }

  // Call cleanup explicitly here to ensure it happens before
  // any other static destructors run
  cleanup_gpu_resources();

  return 0;
}
