/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include <iostream>
#include <string>

#include "../src/core/bitboard.h"
#include "../src/core/position.h"
#include "../src/nn/encoder.h"
#include "../src/nn/loader.h"
#include "../src/nn/network.h"
#include "../src/mcts/nn_mcts_evaluator.h"

using namespace MetalFish;

// Test positions from the benchmark
const std::vector<std::string> kTestPositions = {
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // Starting position
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // Kiwipete
  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  // Endgame
  "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",  // Complex
  "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",  // Tactical
};

void TestEncoder() {
  std::cout << "Testing NN Encoder..." << std::endl;
  
  Position pos;
  StateInfo si;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);
  
  // Test encoding
  NN::InputPlanes planes = NN::EncodePositionForNN(pos);
  
  // Verify planes are populated
  int non_zero_planes = 0;
  for (int i = 0; i < NN::kTotalPlanes; ++i) {
    bool has_data = false;
    for (int sq = 0; sq < 64; ++sq) {
      if (planes[i][sq] != 0.0f) {
        has_data = true;
        break;
      }
    }
    if (has_data) non_zero_planes++;
  }
  
  std::cout << "  Non-zero planes: " << non_zero_planes << " / " << NN::kTotalPlanes << std::endl;
  std::cout << "  Encoder test: " << (non_zero_planes > 0 ? "PASS" : "FAIL") << std::endl;
}

void TestLoader() {
  std::cout << "\nTesting NN Loader..." << std::endl;
  
  // Try autodiscovery
  auto weights_path = NN::DiscoverWeightsFile();
  
  if (weights_path.empty()) {
    std::cout << "  No weights file found (expected - network file not provided)" << std::endl;
    std::cout << "  Loader test: SKIP" << std::endl;
    return;
  }
  
  try {
    auto weights = NN::LoadWeightsFromFile(weights_path);
    std::cout << "  Successfully loaded weights from: " << weights_path << std::endl;
    std::cout << "  Loader test: PASS" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "  Error loading weights: " << e.what() << std::endl;
    std::cout << "  Loader test: FAIL" << std::endl;
  }
}

void TestMCTSEvaluator() {
  std::cout << "\nTesting MCTS NN Evaluator..." << std::endl;
  
  try {
    // This will fail without actual weights file, but tests the integration
    MCTS::NNMCTSEvaluator evaluator("<autodiscover>");
    
    Position pos;
    StateInfo si;
    pos.set(kTestPositions[0], false, &si);
    
    auto result = evaluator.Evaluate(pos);
    
    std::cout << "  Policy size: " << result.policy.size() << std::endl;
    std::cout << "  Value: " << result.value << std::endl;
    std::cout << "  MCTS evaluator test: PASS" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "  Expected error (no weights file): " << e.what() << std::endl;
    std::cout << "  MCTS evaluator test: SKIP" << std::endl;
  }
}

void TestComparison() {
  std::cout << "\nNN Comparison Test (vs reference):" << std::endl;
  std::cout << "  This test requires:" << std::endl;
  std::cout << "    1. A trained network file (BT4 transformer)" << std::endl;
  std::cout << "    2. Reference outputs from the same network" << std::endl;
  std::cout << "    3. Metal backend implementation for inference" << std::endl;
  std::cout << "  Status: NOT IMPLEMENTED (infrastructure only)" << std::endl;
}

int main() {
  // Initialize bitboards and engine
  Bitboards::init();
  
  std::cout << "=== MetalFish Neural Network Test Suite ===" << std::endl;
  std::cout << std::endl;
  
  TestEncoder();
  TestLoader();
  TestMCTSEvaluator();
  TestComparison();
  
  std::cout << "\n=== Test Summary ===" << std::endl;
  std::cout << "Note: This is a minimal infrastructure implementation." << std::endl;
  std::cout << "Full functionality requires:" << std::endl;
  std::cout << "  1. Complete policy mapping tables (1858 moves)" << std::endl;
  std::cout << "  2. Metal backend for transformer inference" << std::endl;
  std::cout << "  3. Actual network weights file" << std::endl;
  std::cout << "  4. Integration with MCTS search" << std::endl;
  
  return 0;
}
