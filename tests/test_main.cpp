/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Test suite for MetalFish
*/

#include <iostream>
#include <cassert>
#include <string>

// Test declarations
bool test_bitboard();
bool test_position();
bool test_movegen();
bool test_search();
bool test_metal();

int main() {
    std::cout << "MetalFish Test Suite\n";
    std::cout << "====================\n\n";
    
    int passed = 0;
    int failed = 0;
    
    // Run tests
    struct Test {
        const char* name;
        bool (*func)();
    };
    
    Test tests[] = {
        {"Bitboard", test_bitboard},
        {"Position", test_position},
        {"Move Generation", test_movegen},
        {"Search", test_search},
        {"Metal GPU", test_metal}
    };
    
    for (const auto& test : tests) {
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
        }
        catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what() << "\n";
            failed++;
        }
    }
    
    std::cout << "\n====================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    
    return failed > 0 ? 1 : 0;
}

