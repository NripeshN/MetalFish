/*
  MetalFish - Comprehensive Test Suite
  Copyright (C) 2025 Nripesh Niketan

  Test runner for all engine subsystems.
  Tests are organized by module:
    - core:   Bitboard, position, move generation
    - search: Alpha-Beta search, TT, move ordering, time management
    - eval:   NNUE evaluation, GPU integration
    - mcts:   MCTS tree, PUCT, batched evaluation
    - hybrid: Parallel hybrid search, PV injection
    - metal:  Metal GPU backend, shaders, buffers
*/

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include "../src/core/bitboard.h"
#include "../src/core/position.h"

// Test module declarations
bool test_core();
bool test_search();
bool test_eval_gpu();
bool test_mcts_all();
bool test_hybrid_module();

int main(int argc, char *argv[]) {
  MetalFish::Bitboards::init();
  MetalFish::Position::init();

  std::cout << "=== MetalFish Test Suite ===" << std::endl;

  // Filter: if a test name is passed as argument, run only that test
  std::string filter = (argc > 1) ? argv[1] : "";

  struct TestEntry {
    std::string name;
    std::function<bool()> fn;
  };

  std::vector<TestEntry> tests = {
      {"core", test_core},
      {"search", test_search},
      {"eval_gpu", test_eval_gpu},
      {"mcts", test_mcts_all},
      {"hybrid", test_hybrid_module},
  };

  int passed = 0, failed = 0, skipped = 0;

  for (const auto &t : tests) {
    if (!filter.empty() && t.name != filter) {
      skipped++;
      continue;
    }

    std::cout << "\n========== " << t.name << " ==========" << std::endl;
    try {
      if (t.fn()) {
        std::cout << ">> " << t.name << ": PASSED" << std::endl;
        passed++;
      } else {
        std::cout << ">> " << t.name << ": FAILED" << std::endl;
        failed++;
      }
    } catch (const std::exception &e) {
      std::cout << ">> " << t.name << ": CRASHED (" << e.what() << ")"
                << std::endl;
      failed++;
    }
  }

  std::cout << "\n===================="
            << "\nResults: " << passed << " passed, " << failed << " failed";
  if (skipped > 0)
    std::cout << ", " << skipped << " skipped";
  std::cout << "\n====================" << std::endl;

  return failed > 0 ? 1 : 0;
}
