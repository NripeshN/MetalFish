/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive Test Suite for MetalFish
*/

#include <cassert>
#include <iostream>
#include <string>

// Test declarations - Core tests
bool test_bitboard();
bool test_position();
bool test_movegen();
bool test_search();
bool test_metal();
bool test_cuda();
bool run_all_gpu_tests();
bool test_mcts();

// Test declarations - Comprehensive tests
bool test_core_comprehensive();
bool test_search_comprehensive();
bool test_mcts_comprehensive();
bool test_hybrid_comprehensive();
bool test_gpu_comprehensive();

int main(int argc, char *argv[]) {
  std::cout << "MetalFish Comprehensive Test Suite\n";
  std::cout << "===================================\n\n";

  int passed = 0;
  int failed = 0;

  // Check for specific test to run
  std::string specific_test = "";
  if (argc > 1) {
    specific_test = argv[1];
    std::cout << "Running specific test: " << specific_test << "\n\n";
  }

  // Run tests
  struct Test {
    const char *name;
    bool (*func)();
  };

  // Original tests
  Test original_tests[] = {
      {"Bitboard", test_bitboard},
      {"Position", test_position},
      {"Move Generation", test_movegen},
      {"Search", test_search},
      {"Metal GPU", test_metal},
      {"CUDA GPU", test_cuda},
      {"GPU NNUE", run_all_gpu_tests},
      {"MCTS Hybrid", test_mcts}};

  // Comprehensive tests
  Test comprehensive_tests[] = {
      {"Core Comprehensive", test_core_comprehensive},
      {"Search Comprehensive", test_search_comprehensive},
      {"MCTS Comprehensive", test_mcts_comprehensive},
      {"Hybrid Comprehensive", test_hybrid_comprehensive},
      {"GPU Comprehensive", test_gpu_comprehensive}};

  // Run original tests
  std::cout << "--- Original Tests ---\n\n";

  for (const auto &test : original_tests) {
    if (!specific_test.empty() && specific_test != test.name &&
        specific_test != "original" && specific_test != "all") {
      continue;
    }

    std::cout << "Running " << test.name << " tests... ";
    std::cout.flush();

    try {
      if (test.func()) {
        std::cout << "PASSED\n";
        passed++;
      } else {
        std::cout << "FAILED\n";
        failed++;
      }
    } catch (const std::exception &e) {
      std::cout << "ERROR: " << e.what() << "\n";
      failed++;
    }
  }

  // Run comprehensive tests
  std::cout << "\n--- Comprehensive Tests ---\n";

  for (const auto &test : comprehensive_tests) {
    if (!specific_test.empty() && specific_test != test.name &&
        specific_test != "comprehensive" && specific_test != "all") {
      continue;
    }

    std::cout << "\nRunning " << test.name << " tests...\n";
    std::cout.flush();

    try {
      if (test.func()) {
        std::cout << "  " << test.name << ": PASSED\n";
        passed++;
      } else {
        std::cout << "  " << test.name << ": FAILED\n";
        failed++;
      }
    } catch (const std::exception &e) {
      std::cout << "  " << test.name << ": ERROR: " << e.what() << "\n";
      failed++;
    }
  }

  std::cout << "\n===================================\n";
  std::cout << "Results: " << passed << " passed, " << failed << " failed\n";

  if (failed > 0) {
    std::cout << "\nSOME TESTS FAILED!\n";
  } else {
    std::cout << "\nALL TESTS PASSED!\n";
  }

  std::cout << "\nUsage: " << (argc > 0 ? argv[0] : "metalfish_tests")
            << " [test_name]\n";
  std::cout << "  test_name: specific test to run, or 'original', "
               "'comprehensive', 'all'\n";

  return failed > 0 ? 1 : 0;
}
