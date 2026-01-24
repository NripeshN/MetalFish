/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Test Suite Entry Point
*/

#include <iostream>
#include <string>

// Module test declarations
bool test_core();
bool test_search_module();
bool test_mcts_module();
bool test_hybrid_module();
bool test_gpu_module();

// Hardware-specific tests
bool test_metal();
bool test_cuda();
bool run_all_gpu_tests();

// Lc0 NN tests
bool run_all_lc0_tests();

int main(int argc, char *argv[]) {
  std::cout << "MetalFish Test Suite\n";
  std::cout << "====================\n";

  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
    std::cout << "Filter: " << filter << "\n";
  }

  int passed = 0;
  int failed = 0;

  auto run_test = [&](const char *name, bool (*func)()) {
    if (!filter.empty() && filter != name && filter != "all") {
      return;
    }
    std::cout << "\nRunning " << name << " tests...\n";
    try {
      if (func()) {
        passed++;
      } else {
        failed++;
      }
    } catch (const std::exception &e) {
      std::cout << "ERROR: " << e.what() << "\n";
      failed++;
    }
  };

  // Core module tests
  run_test("core", test_core);
  run_test("search", test_search_module);
  run_test("mcts", test_mcts_module);
  run_test("hybrid", test_hybrid_module);
  run_test("gpu", test_gpu_module);

  // Hardware-specific tests
  run_test("metal", test_metal);
  run_test("cuda", test_cuda);
  run_test("gpu_nnue", run_all_gpu_tests);
  
  // Lc0 NN module tests
  run_test("lc0_nn", run_all_lc0_tests);

  std::cout << "\n====================\n";
  std::cout << "Results: " << passed << " passed, " << failed << " failed\n";

  if (failed > 0) {
    std::cout << "\nSOME TESTS FAILED!\n";
    return 1;
  }

  std::cout << "\nALL TESTS PASSED!\n";
  return 0;
}
