/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS smoke tests for the current Search/Node pipeline.
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "mcts/backend_adapter.h"
#include "mcts/core.h"
#include "mcts/search.h"
#include "syzygy/tbprobe.h"

#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace MetalFish;
using namespace MetalFish::MCTS;

namespace {

struct TestCounter {
  int passed = 0;
  int failed = 0;
};

void expect(bool cond, const char *msg, TestCounter &tc) {
  if (cond) {
    tc.passed++;
  } else {
    tc.failed++;
    std::cout << "    FAIL: " << msg << std::endl;
  }
}

void test_node_basics(TestCounter &tc) {
  std::cout << "  Node basics..." << std::endl;
  constexpr size_t node_size_budget =
      CACHE_LINE_SIZE == 128 ? CACHE_LINE_SIZE : 2 * CACHE_LINE_SIZE;
  expect(alignof(Node) == CACHE_LINE_SIZE, "node cache-line alignment", tc);
  expect(sizeof(Node) <= node_size_budget, "node fits size budget", tc);

  Node n;
  expect(n.GetN() == 0, "initial N", tc);
  expect(n.GetNInFlight() == 0, "initial N_in_flight", tc);
  expect(!n.IsTerminal(), "initial non-terminal", tc);

  expect(n.TryStartScoreUpdate(2), "start score update", tc);
  expect(n.GetNInFlight() >= 2, "virtual visits set", tc);
  n.FinalizeScoreUpdate(0.5f, 0.1f, 20.0f, 2);
  expect(n.GetN() == 2, "finalized multivisit", tc);
  expect(n.GetNInFlight() == 0, "in-flight cleared", tc);
}

void test_tablebase_wdl_conversion(TestCounter &tc) {
  std::cout << "  Tablebase WDL conversion..." << std::endl;

  expect(TablebaseWDLToParentWL(Tablebases::WDLWin) == -1.0f,
         "side-to-move TB win is parent loss", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLLoss) == 1.0f,
         "side-to-move TB loss is parent win", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLDraw) == 0.0f,
         "TB draw has neutral WL", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLCursedWin) == 0.0f,
         "cursed TB win is draw-equivalent", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLBlessedLoss) == 0.0f,
         "blessed TB loss is draw-equivalent", tc);

  expect(TablebaseWDLToDraw(Tablebases::WDLWin) == 0.0f,
         "decisive TB win has no draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLLoss) == 0.0f,
         "decisive TB loss has no draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLDraw) == 1.0f,
         "TB draw has full draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLCursedWin) == 1.0f,
         "cursed TB win has full draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLBlessedLoss) == 1.0f,
         "blessed TB loss has full draw mass", tc);
}

void test_root_search_smoke(TestCounter &tc) {
  std::cout << "  Search smoke..." << std::endl;
  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path.clear(); // Smoke path without NN backend.
  auto search = CreateSearch(params);
  expect(static_cast<bool>(search), "search object created", tc);

  MetalFish::Search::LimitsType limits;
  limits.nodes = 8;
  search->StartSearch("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                      limits, nullptr, nullptr);
  search->Wait();

  const auto &stats = search->Stats();
  expect(stats.total_nodes.load() > 0, "search produced visits", tc);
}

void test_search_params_defaults(TestCounter &tc) {
  std::cout << "  Search params..." << std::endl;
  SearchParams params;
  expect(params.fpu_reduction_at_root == 0.33f,
         "root FPU default aligned", tc);
  expect(params.GetCpuctBase(true) == params.cpuct_base_at_root,
         "root cpuct base getter", tc);
  expect(params.GetFpuValue(true) == params.fpu_value_at_root,
         "root fpu value getter", tc);
}

void test_nn_cache_policy_capacity(TestCounter &tc) {
  std::cout << "  NN cache policy capacity..." << std::endl;

  NNCache cache(8);
  EvaluationResult small;
  small.value = 0.25f;
  for (int i = 0; i < 8; ++i) {
    small.policy_priors.emplace_back(Move(static_cast<std::uint16_t>(100 + i)),
                                     0.01f * static_cast<float>(i + 1));
  }

  cache.Insert(1234, small);
  EvaluationResult small_out;
  bool small_hit = cache.Lookup(1234, 8, small_out);
  expect(small_hit, "small policy entry should be cached", tc);
  expect(small_out.policy_priors.size() == small.policy_priors.size(),
         "small policy entry should round-trip all moves", tc);

  EvaluationResult large;
  large.value = -0.10f;
  for (int i = 0; i < 128; ++i) {
    large.policy_priors.emplace_back(Move(static_cast<std::uint16_t>(500 + i)),
                                     0.001f * static_cast<float>(i + 1));
  }

  cache.Insert(5678, large);
  EvaluationResult large_out;
  bool large_hit = cache.Lookup(5678, 128, large_out);
  expect(!large_hit || large_out.policy_priors.size() == large.policy_priors.size(),
         "cache must not return truncated policy entries", tc);
}

void test_history_buffer_ownership(TestCounter &tc) {
  std::cout << "  History buffer ownership..." << std::endl;

  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const std::vector<Move> line = {
      Move(SQ_E2, SQ_E4), Move(SQ_E7, SQ_E5),
      Move(SQ_G1, SQ_F3), Move(SQ_B8, SQ_C6),
  };

  SearchWorkerCtx ctx;
  ctx.SetRootFen(fen);
  for (Move move : line) {
    ctx.DoMove(move);
  }

  std::vector<std::unique_ptr<SearchWorkerCtx::HistoryBuffer>> keepalive;
  keepalive.push_back(ctx.BuildHistory());

  const auto &history = *keepalive.back();
  expect(history.depth == static_cast<int>(line.size()) + 1,
         "history should include root plus played moves", tc);

  Position replay;
  StateInfo root_state;
  replay.set(fen, false, &root_state);
  expect(history.ptrs[0]->raw_key() == replay.raw_key(),
         "root history state remains valid after owner move", tc);

  std::vector<StateInfo> states(line.size());
  for (size_t i = 0; i < line.size(); ++i) {
    replay.do_move(line[i], states[i]);
    expect(history.ptrs[i + 1]->raw_key() == replay.raw_key(),
           "played history state remains valid after owner move", tc);
  }
}

void test_deterministic_repro(TestCounter &tc) {
  std::cout << "  Deterministic reproducibility..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;
  params.out_of_order_eval = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 128;
  const std::string fen =
      "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3";

  auto s1 = CreateSearch(params);
  s1->StartSearch(fen, limits, nullptr, nullptr);
  s1->Wait();
  Move b1 = s1->GetBestMove();

  auto s2 = CreateSearch(params);
  s2->StartSearch(fen, limits, nullptr, nullptr);
  s2->Wait();
  Move b2 = s2->GetBestMove();

  expect(b1 == b2, "bestmove should match across deterministic runs", tc);
}

void test_evaluator_legal_move_view_parity(TestCounter &tc) {
  std::cout << "  Evaluator legal-move view parity..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  Position pos;
  StateInfo st;
  pos.set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
          false, &st);

  std::array<const Position *, 1> history = {&pos};
  MoveList<LEGAL> moves(pos);
  NNMCTSEvaluator evaluator(weights);

  auto generated = evaluator.EvaluateWithHistory(history);
  auto provided = evaluator.EvaluateWithHistoryAndMoves(
      history, NNMCTSEvaluator::LegalMovesView(moves.begin(), moves.size()));

  expect(generated.policy_priors.size() == provided.policy_priors.size(),
         "provided move list should preserve policy size", tc);
  expect(std::abs(generated.value - provided.value) < 1e-4f,
         "provided move list should preserve value", tc);

  size_t common =
      std::min(generated.policy_priors.size(), provided.policy_priors.size());
  for (size_t i = 0; i < common; ++i) {
    expect(generated.policy_priors[i].first == provided.policy_priors[i].first,
           "provided move list should preserve move order", tc);
    expect(std::abs(generated.policy_priors[i].second -
                    provided.policy_priors[i].second) < 1e-4f,
           "provided move list should preserve policy logits", tc);
  }
}

void test_nodes_limit_with_callback(TestCounter &tc) {
  std::cout << "  Nodes limit with callback..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 16;

  bool callback_called = false;
  auto search = CreateSearch(params);
  search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      [&](Move best, Move) {
        callback_called = true;
        expect(best != Move::none(), "callback best move should be legal", tc);
      },
      nullptr);
  search->Wait();

  expect(callback_called, "bestmove callback should fire", tc);
  expect(search->Stats().total_nodes.load() >= limits.nodes,
         "MCTS should honor node limit before callback", tc);

  callback_called = false;
  auto async_search = CreateSearch(params);
  async_search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      [&](Move best, Move) {
        callback_called = true;
        expect(best != Move::none(), "async callback best move should be legal",
               tc);
      },
      [](const std::string &) {});
  std::thread waiter([&]() { async_search->Wait(); });
  waiter.join();

  expect(callback_called, "async bestmove callback should fire", tc);
  expect(async_search->Stats().total_nodes.load() >= limits.nodes,
         "MCTS should honor node limit with asynchronous waiter", tc);
}

void test_cache_hit_rate(TestCounter &tc) {
  std::cout << "  Cache hit rate..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;
  params.out_of_order_eval = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 256;
  const std::string fen =
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";

  auto search = CreateSearch(params);
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();

  const auto &stats = search->Stats();
  uint64_t hits = stats.cache_hits.load();
  uint64_t misses = stats.cache_misses.load();
  uint64_t total_lookups = hits + misses;
  double hit_rate = total_lookups > 0 ? 100.0 * hits / total_lookups : 0.0;
  std::cout << "    Cache hits: " << hits << " / " << total_lookups
            << " lookups (" << hit_rate << "%)" << std::endl;
  expect(hit_rate > 5.0, "cache hit rate > 5%", tc);
}

} // namespace

bool test_mcts_all() {
  std::cout << "\n[MCTS]" << std::endl;
  TestCounter tc;

  test_node_basics(tc);
  test_tablebase_wdl_conversion(tc);
  test_search_params_defaults(tc);
  test_nn_cache_policy_capacity(tc);
  test_history_buffer_ownership(tc);
  test_root_search_smoke(tc);
  test_evaluator_legal_move_view_parity(tc);
  test_deterministic_repro(tc);
  test_nodes_limit_with_callback(tc);
  test_cache_hit_rate(tc);

  std::cout << "  Passed: " << tc.passed << ", Failed: " << tc.failed
            << std::endl;
  return tc.failed == 0;
}
