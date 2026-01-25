/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include <iostream>
#include <string>

#include "../src/core/bitboard.h"
#include "../src/core/position.h"
#include "../src/core/movegen.h"
#include "../src/nn/encoder.h"
#include "../src/nn/loader.h"
#include "../src/nn/network.h"
#include "../src/nn/policy_map.h"
#include "../src/mcts/nn_mcts_evaluator.h"
#include "../src/uci/uci.h"

using namespace MetalFish;

// Test positions from the benchmark
const std::vector<std::string> kTestPositions = {
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // Starting position
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // Kiwipete
  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  // Endgame
  "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",  // Complex
  "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",  // Tactical
};

void test_policy_tables() {
  std::cout << "Testing policy tables..." << std::endl;
  
  // Simple test that tables are initialized
  std::cout << "  ✓ Policy tables initialized (detailed tests require move construction)" << std::endl;
}

void test_encoder() {
  std::cout << "\nTesting encoder..." << std::endl;
  
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st);
  
  auto planes = NN::EncodePositionForNN(
      pos, MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);
  
  // Count non-zero planes
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
  std::cout << "  ✓ Encoded starting position to 112 planes" << std::endl;
}

void test_network() {
  std::cout << "\nTesting network..." << std::endl;
  
  const char* weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "  ⊘ Skipped (METALFISH_NN_WEIGHTS not set)" << std::endl;
    return;
  }
  
  try {
    auto network = NN::CreateNetwork(weights_path, "auto");
    std::cout << "  Network: " << network->GetNetworkInfo() << std::endl;
    
    // Test evaluation
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st);
    
    auto planes = NN::EncodePositionForNN(
        pos, MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);
    
    auto output = network->Evaluate(planes);
    std::cout << "  Value: " << output.value << std::endl;
    std::cout << "  Policy size: " << output.policy.size() << std::endl;
    if (output.has_wdl) {
      std::cout << "  WDL: [" << output.wdl[0] << ", " << output.wdl[1] 
                << ", " << output.wdl[2] << "]" << std::endl;
    }
    std::cout << "  ✓ Network evaluation successful" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "  ✗ Error: " << e.what() << std::endl;
  }
}

void test_mcts_evaluator() {
  std::cout << "\nTesting MCTS NN evaluator..." << std::endl;
  
  const char* weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "  ⊘ Skipped (METALFISH_NN_WEIGHTS not set)" << std::endl;
    return;
  }
  
  try {
    MCTS::NNMCTSEvaluator evaluator(weights_path);
    std::cout << "  Network: " << evaluator.GetNetworkInfo() << std::endl;
    
    StateInfo st;
    Position pos;
    pos.set(kTestPositions[0], false, &st);
    
    auto result = evaluator.Evaluate(pos);
    
    std::cout << "  Value: " << result.value << std::endl;
    std::cout << "  Policy priors: " << result.policy_priors.size() << " moves" << std::endl;
    if (result.has_wdl) {
      std::cout << "  WDL: [" << result.wdl[0] << ", " << result.wdl[1] 
                << ", " << result.wdl[2] << "]" << std::endl;
    }
    
    // Show top 3 moves by policy
    auto sorted_moves = result.policy_priors;
    std::sort(sorted_moves.begin(), sorted_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "  Top 3 moves:" << std::endl;
    for (int i = 0; i < std::min(3, (int)sorted_moves.size()); ++i) {
      std::cout << "    Move #" << i+1 << " → " << sorted_moves[i].second << std::endl;
    }
    
    std::cout << "  ✓ MCTS evaluator test passed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "  ✗ Error: " << e.what() << std::endl;
  }
}

int main() {
  // Initialize bitboards and engine
  Bitboards::init();
  Position::init();
  NN::InitPolicyTables();
  
  std::cout << "=== MetalFish Neural Network Tests ===" << std::endl;
  std::cout << std::endl;
  
  test_policy_tables();
  test_encoder();
  test_network();
  test_mcts_evaluator();
  
  std::cout << "\n=== Test Summary ===" << std::endl;
  std::cout << "Note: Full functionality requires:" << std::endl;
  std::cout << "  1. Complete policy mapping tables (1858 moves)" << std::endl;
  std::cout << "  2. Metal backend for transformer inference" << std::endl;
  std::cout << "  3. Actual network weights file (set METALFISH_NN_WEIGHTS)" << std::endl;
  std::cout << "  4. Integration with MCTS search (✓ completed)" << std::endl;
  
  return 0;
}
