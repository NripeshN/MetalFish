/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "uci/uci.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "core/memory.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "eval/evaluate.h"
#include "eval/gpu_backend.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_accumulator.h"
#include "eval/score.h"
#include "hybrid/classifier.h"
#include "hybrid/hybrid_search.h"
#include "hybrid/position_adapter.h"
#include "mcts/search.h"
#include "search/search.h"
#include "uci/benchmark.h"
#include "uci/engine.h"
#include "uci/ucioption.h"

namespace MetalFish {

static void stop_active_searches();
static void ponderhit_active_searches(Engine &engine);
static void wait_active_searches();
static void join_search_waiter();
static void preload_search_objects(Engine &engine);
static void reset_cached_search_objects();

constexpr auto BenchmarkCommand = "speedtest";

constexpr auto StartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
template <typename... Ts> struct overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> overload(Ts...) -> overload<Ts...>;

bool uci_trace_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("METALFISH_UCI_TRACE");
    if (!env || !*env)
      return false;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return value != "0" && value != "false" && value != "off" && value != "no";
  }();
  return enabled;
}

void UCIEngine::print_info_string(std::string_view str) {
  sync_cout_start();
  for (auto &line : split(str, "\n")) {
    if (!is_whitespace(line)) {
      std::cout << "info string " << line << '\n';
    }
  }
  sync_cout_end();
}

UCIEngine::UCIEngine(int argc, char **argv) : engine(argv[0]), cli(argc, argv) {

  engine.get_options().add_info_listener(
      [](const std::optional<std::string> &str) {
        if (str.has_value())
          print_info_string(*str);
      });

  init_search_update_listeners();
}

void UCIEngine::init_search_update_listeners() {
  engine.set_on_iter([](const auto &i) { on_iter(i); });
  engine.set_on_update_no_moves([](const auto &i) { on_update_no_moves(i); });
  engine.set_on_update_full([this](const auto &i) {
    on_update_full(i, engine.get_options()["UCI_ShowWDL"]);
  });
  engine.set_on_bestmove(
      [](const auto &bm, const auto &p) { on_bestmove(bm, p); });
  engine.set_on_verify_networks([](const auto &s) { print_info_string(s); });
}

void UCIEngine::loop() {
  std::string token, cmd;

  for (int i = 1; i < cli.argc; ++i)
    cmd += std::string(cli.argv[i]) + " ";

  do {
    if (cli.argc == 1 &&
        !getline(std::cin,
                 cmd)) // Wait for an input or an end-of-file (EOF) indication
      cmd = "quit";

    std::istringstream is(cmd);

    token.clear();
    is >> std::skipws >> token;

    // Debug command trace is opt-in via METALFISH_UCI_TRACE=1.
    if (uci_trace_enabled() && !token.empty()) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
      std::cerr << "[UCI:" << ms << "] " << cmd << std::endl;
    }

    if (token == "quit" || token == "stop") {
      stop_active_searches();
      if (token == "quit")
        wait_active_searches();
      engine.stop();
    }

    else if (token == "uci") {
      sync_cout << "id name " << engine_info(true) << "\n"
                << engine.get_options() << sync_endl;
      sync_cout << "uciok" << sync_endl;
    }

    else if (token == "isready") {
      preload_search_objects(engine);
      sync_cout << "readyok" << sync_endl;
    }

    else if (token == "ucinewgame") {
      stop_active_searches();
      wait_active_searches();
      engine.search_clear();
      reset_cached_search_objects();
    }

    else if (token == "position")
      position(is);

    else if (token == "setoption")
      setoption(is);

    else if (token == "ponderhit")
      ponderhit_active_searches(engine);

    else if (token == "go") {
      if (engine.get_options()["UseMCTS"])
        mcts_mt_go(is);
      else if (engine.get_options()["UseHybridSearch"])
        parallel_hybrid_go(is);
      else
        go(is);
    }

    else if (token == "d")
      sync_cout << engine.visualize() << sync_endl;
    else if (token == "eval")
      engine.trace_eval();
    else if (token == "flip")
      engine.flip();
    else if (token == "bench")
      bench(is);
    else if (token == BenchmarkCommand)
      benchmark(is);
    else if (token == "compiler")
      sync_cout << compiler_info() << sync_endl;

    // Direct engine mode commands (CLI shortcuts)
    else if (token == "mctsmt")
      mcts_mt_go(is);
    else if (token == "hybrid" || token == "parallel_hybrid")
      parallel_hybrid_go(is);

    // GPU diagnostics
    else if (token == "gpu")
      gpu_info();
    else if (token == "gpubench")
      gpu_benchmark();
    else if (token == "nnuebench")
      nnue_benchmark(is);
    else if (token == "mctsbench")
      mcts_batch_benchmark(is);

    // Network export
    else if (token == "export_net") {
      std::pair<std::optional<std::string>, std::string> files[2];
      if (is >> std::skipws >> files[0].second)
        files[0].first = files[0].second;
      if (is >> std::skipws >> files[1].second)
        files[1].first = files[1].second;
      engine.save_network(files);
    }

    else if (token == "help" || token == "--help" || token == "license" ||
             token == "--license")
      sync_cout
          << "\nMetalFish is a powerful chess engine for playing and analyzing."
             "\nIt is released as free software licensed under the GNU GPLv3 "
             "License."
             "\nMetalFish is normally used with a graphical user interface "
             "(GUI) and implements"
             "\nthe Universal Chess Interface (UCI) protocol to communicate "
             "with a GUI, an API, etc."
             "\nFor any further information, visit "
             "https://github.com/NripeshN/MetalFish#readme"
             "\nor read the corresponding README.md and Copying.txt files "
             "distributed along with this program.\n"
          << sync_endl;

    else if (!token.empty() && token[0] != '#')
      sync_cout << "Unknown command: '" << cmd
                << "'. Type help for more information." << sync_endl;

  } while (token != "quit" && cli.argc == 1);

  stop_active_searches();
  wait_active_searches();
  join_search_waiter();
}

Search::LimitsType UCIEngine::parse_limits(std::istream &is) {
  Search::LimitsType limits;
  std::string token;

  limits.startTime = now();

  while (is >> token)
    if (token == "searchmoves") // Needs to be the last command on the line
      while (is >> token)
        limits.searchmoves.push_back(to_lower(token));

    else if (token == "wtime")
      is >> limits.time[WHITE];
    else if (token == "btime")
      is >> limits.time[BLACK];
    else if (token == "winc")
      is >> limits.inc[WHITE];
    else if (token == "binc")
      is >> limits.inc[BLACK];
    else if (token == "movestogo")
      is >> limits.movestogo;
    else if (token == "depth")
      is >> limits.depth;
    else if (token == "nodes")
      is >> limits.nodes;
    else if (token == "movetime")
      is >> limits.movetime;
    else if (token == "mate")
      is >> limits.mate;
    else if (token == "perft")
      is >> limits.perft;
    else if (token == "infinite")
      limits.infinite = 1;
    else if (token == "ponder")
      limits.ponderMode = true;

  return limits;
}

void UCIEngine::go(std::istringstream &is) {

  Search::LimitsType limits = parse_limits(is);

  if (limits.perft)
    perft(limits);
  else
    engine.go(limits);
}

void UCIEngine::bench(std::istream &args) {
  std::string token;
  uint64_t num, nodes = 0, cnt = 1;
  uint64_t nodesSearched = 0;
  const auto &options = engine.get_options();

  engine.set_on_update_full([&](const auto &i) {
    nodesSearched = i.nodes;
    on_update_full(i, options["UCI_ShowWDL"]);
  });

  std::vector<std::string> list = Benchmark::setup_bench(engine.fen(), args);

  num = count_if(list.begin(), list.end(), [](const std::string &s) {
    return s.find("go ") == 0 || s.find("eval") == 0;
  });

  TimePoint elapsed = now();

  for (const auto &cmd : list) {
    std::istringstream is(cmd);
    is >> std::skipws >> token;

    if (token == "go" || token == "eval") {
      std::cerr << "\nPosition: " << cnt++ << '/' << num << " (" << engine.fen()
                << ")" << std::endl;
      if (token == "go") {
        Search::LimitsType limits = parse_limits(is);

        if (limits.perft)
          nodesSearched = perft(limits);
        else {
          engine.go(limits);
          engine.wait_for_search_finished();
        }

        nodes += nodesSearched;
        nodesSearched = 0;
      } else
        engine.trace_eval();
    } else if (token == "setoption")
      setoption(is);
    else if (token == "position")
      position(is);
    else if (token == "ucinewgame") {
      engine.search_clear();
      elapsed = now();
    }
  }

  elapsed = now() - elapsed + 1;

  dbg_print();

  std::cerr << "\n==========================="   //
            << "\nTotal time (ms) : " << elapsed //
            << "\nNodes searched  : " << nodes   //
            << "\nNodes/second    : " << 1000 * nodes / elapsed << std::endl;

  // reset callback to avoid dangling reference to nodesSearched
  engine.set_on_update_full(
      [&](const auto &i) { on_update_full(i, options["UCI_ShowWDL"]); });
}

void UCIEngine::benchmark(std::istream &args) {
  static constexpr int NUM_WARMUP_POSITIONS = 3;

  std::string token;
  uint64_t nodes = 0, cnt = 1;
  uint64_t nodesSearched = 0;

  engine.set_on_update_full(
      [&](const Engine::InfoFull &i) { nodesSearched = i.nodes; });

  engine.set_on_iter([](const auto &) {});
  engine.set_on_update_no_moves([](const auto &) {});
  engine.set_on_bestmove([](const auto &, const auto &) {});
  engine.set_on_verify_networks([](const auto &) {});

  Benchmark::BenchmarkSetup setup = Benchmark::setup_benchmark(args);

  const auto numGoCommands =
      count_if(setup.commands.begin(), setup.commands.end(),
               [](const std::string &s) { return s.find("go ") == 0; });

  TimePoint totalTime = 0;

  auto ss =
      std::istringstream("name Threads value " + std::to_string(setup.threads));
  setoption(ss);
  ss = std::istringstream("name Hash value " + std::to_string(setup.ttSize));
  setoption(ss);
  ss = std::istringstream("name UCI_Chess960 value false");
  setoption(ss);

  for (const auto &cmd : setup.commands) {
    std::istringstream is(cmd);
    is >> std::skipws >> token;

    if (token == "go") {
      std::cerr << "\rWarmup position " << cnt++ << '/' << NUM_WARMUP_POSITIONS;

      Search::LimitsType limits = parse_limits(is);

      engine.go(limits);
      engine.wait_for_search_finished();
    } else if (token == "position")
      position(is);
    else if (token == "ucinewgame") {
      engine.search_clear();
    }

    if (cnt > NUM_WARMUP_POSITIONS)
      break;
  }

  std::cerr << "\n";

  cnt = 1;
  nodes = 0;

  int numHashfullReadings = 0;
  constexpr int hashfullAges[] = {
      0, 999}; // Only normal hashfull and touched hash.
  constexpr int hashfullAgeCount = std::size(hashfullAges);
  int totalHashfull[hashfullAgeCount] = {0};
  int maxHashfull[hashfullAgeCount] = {0};

  auto updateHashfullReadings = [&]() {
    numHashfullReadings += 1;

    for (int i = 0; i < hashfullAgeCount; ++i) {
      const int hashfull = engine.get_hashfull(hashfullAges[i]);
      maxHashfull[i] = std::max(maxHashfull[i], hashfull);
      totalHashfull[i] += hashfull;
    }
  };

  engine.search_clear();

  for (const auto &cmd : setup.commands) {
    std::istringstream is(cmd);
    is >> std::skipws >> token;

    if (token == "go") {
      std::cerr << "\rPosition " << cnt++ << '/' << numGoCommands;

      Search::LimitsType limits = parse_limits(is);

      nodesSearched = 0;
      TimePoint elapsed = now();

      engine.go(limits);
      engine.wait_for_search_finished();

      totalTime += now() - elapsed;

      updateHashfullReadings();

      nodes += nodesSearched;
    } else if (token == "position")
      position(is);
    else if (token == "ucinewgame") {
      engine.search_clear();
    }
  }

  totalTime = std::max<TimePoint>(totalTime, 1);

  dbg_print();

  std::cerr << "\n";

  static_assert(std::size(hashfullAges) == 2 && hashfullAges[0] == 0 &&
                    hashfullAges[1] == 999,
                "Hardcoded for display. Would complicate the code needlessly "
                "in the current state.");

  std::string threadBinding = engine.thread_binding_information_as_string();
  if (threadBinding.empty())
    threadBinding = "none";

  // clang-format off

    std::cerr << "==========================="
              << "\nVersion                    : "
              << engine_version_info()
              // "\nCompiled by                : "
              << compiler_info()
              << "Large pages                : " << (has_large_pages() ? "yes" : "no")
              << "\nUser invocation            : " << BenchmarkCommand << " "
              << setup.originalInvocation << "\nFilled invocation          : " << BenchmarkCommand
              << " " << setup.filledInvocation
              << "\nAvailable processors       : " << engine.get_numa_config_as_string()
              << "\nThread count               : " << setup.threads
              << "\nThread binding             : " << threadBinding
              << "\nTT size [MiB]              : " << setup.ttSize
              << "\nHash max, avg [per mille]  : "
              << "\n    single search          : " << maxHashfull[0] << ", "
              << totalHashfull[0] / numHashfullReadings
              << "\n    single game            : " << maxHashfull[1] << ", "
              << totalHashfull[1] / numHashfullReadings
              << "\nTotal nodes searched       : " << nodes
              << "\nTotal search time [s]      : " << totalTime / 1000.0
              << "\nNodes/second               : " << 1000 * nodes / totalTime << std::endl;

  // clang-format on

  init_search_update_listeners();
}

void UCIEngine::setoption(std::istringstream &is) {
  engine.wait_for_search_finished();
  engine.get_options().setoption(is);
}

std::uint64_t UCIEngine::perft(const Search::LimitsType &limits) {
  auto nodes = engine.perft(engine.fen(), limits.perft,
                            engine.get_options()["UCI_Chess960"]);
  sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl;
  return nodes;
}

void UCIEngine::position(std::istringstream &is) {
  std::string token, fen;

  stop_active_searches();
  wait_active_searches();
  engine.stop();
  engine.wait_for_search_finished();

  is >> token;

  if (token == "startpos") {
    fen = StartFEN;
    is >> token; // Consume the "moves" token, if any
  } else if (token == "fen")
    while (is >> token && token != "moves")
      fen += token + " ";
  else
    return;

  std::vector<std::string> moves;

  while (is >> token) {
    moves.push_back(token);
  }

  engine.set_position(fen, moves);
}

namespace {

struct WinRateParams {
  double a;
  double b;
};

WinRateParams win_rate_params(const Position &pos) {

  int material = pos.count<PAWN>() + 3 * pos.count<KNIGHT>() +
                 3 * pos.count<BISHOP>() + 5 * pos.count<ROOK>() +
                 9 * pos.count<QUEEN>();

  // The fitted model only uses data for material counts in [17, 78], and is
  // anchored at count 58.
  double m = std::clamp(material, 17, 78) / 58.0;

  // Return a = p_a(material) and b = p_b(material), see
  // WDL model calibration parameters
  constexpr double as[] = {-13.50030198, 40.92780883, -36.82753545,
                           386.83004070};
  constexpr double bs[] = {96.53354896, -165.79058388, 90.89679019,
                           49.29561889};

  double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
  double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

  return {a, b};
}

// The win rate model is 1 / (1 + exp((a - eval) / b)), where a = p_a(material)
// and b = p_b(material). It fits the LTC fishtest statistics rather accurately.
int win_rate_model(Value v, const Position &pos) {

  auto [a, b] = win_rate_params(pos);

  // Return the win rate in per mille units, rounded to the nearest integer.
  return int(0.5 + 1000 / (1 + std::exp((a - double(v)) / b)));
}
} // namespace

std::string UCIEngine::format_score(const Score &s) {
  constexpr int TB_CP = 20000;
  const auto format = overload{
      [](Score::Mate mate) -> std::string {
        auto m = (mate.plies > 0 ? (mate.plies + 1) : mate.plies) / 2;
        return std::string("mate ") + std::to_string(m);
      },
      [](Score::Tablebase tb) -> std::string {
        return std::string("cp ") +
               std::to_string((tb.win ? TB_CP - tb.plies : -TB_CP - tb.plies));
      },
      [](Score::InternalUnits units) -> std::string {
        return std::string("cp ") + std::to_string(units.value);
      }};

  return s.visit(format);
}

// Turns a Value to an integer centipawn number,
// without treatment of mate and similar special scores.
int UCIEngine::to_cp(Value v, const Position &pos) {

  // In general, the score can be defined via the WDL as
  // (log(1/L - 1) - log(1/W - 1)) / (log(1/L - 1) + log(1/W - 1)).
  // Based on our win_rate_model, this simply yields v / a.

  auto [a, b] = win_rate_params(pos);

  return int(std::round(100 * int(v) / a));
}

std::string UCIEngine::wdl(Value v, const Position &pos) {
  std::stringstream ss;

  int wdl_w = win_rate_model(v, pos);
  int wdl_l = win_rate_model(-v, pos);
  int wdl_d = 1000 - wdl_w - wdl_l;
  ss << wdl_w << " " << wdl_d << " " << wdl_l;

  return ss.str();
}

std::string UCIEngine::square(Square s) {
  return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string UCIEngine::move(Move m, bool chess960) {
  if (m == Move::none())
    return "(none)";

  if (m == Move::null())
    return "0000";

  Square from = m.from_sq();
  Square to = m.to_sq();

  if (m.type_of() == CASTLING && !chess960)
    to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

  std::string move = square(from) + square(to);

  if (m.type_of() == PROMOTION)
    move += " pnbrqk"[m.promotion_type()];

  return move;
}

std::string UCIEngine::to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](auto c) { return std::tolower(c); });

  return str;
}

Move UCIEngine::to_move(const Position &pos, std::string str) {
  str = to_lower(str);

  for (const auto &m : MoveList<LEGAL>(pos))
    if (str == move(m, pos.is_chess960()))
      return m;

  return Move::none();
}

void UCIEngine::on_update_no_moves(const Engine::InfoShort &info) {
  sync_cout << "info depth " << info.depth << " score "
            << format_score(info.score) << sync_endl;
}

void UCIEngine::on_update_full(const Engine::InfoFull &info, bool showWDL) {
  std::stringstream ss;

  ss << "info";
  ss << " depth " << info.depth                //
     << " seldepth " << info.selDepth          //
     << " multipv " << info.multiPV            //
     << " score " << format_score(info.score); //

  if (!info.bound.empty())
    ss << " " << info.bound;

  if (showWDL)
    ss << " wdl " << info.wdl;

  ss << " nodes " << info.nodes       //
     << " nps " << info.nps           //
     << " hashfull " << info.hashfull //
     << " tbhits " << info.tbHits     //
     << " time " << info.timeMs       //
     << " pv " << info.pv;            //

  sync_cout << ss.str() << sync_endl;
}

void UCIEngine::on_iter(const Engine::InfoIter &info) {
  std::stringstream ss;

  ss << "info";
  ss << " depth " << info.depth                    //
     << " currmove " << info.currmove              //
     << " currmovenumber " << info.currmovenumber; //

  sync_cout << ss.str() << sync_endl;
}

void UCIEngine::on_bestmove(std::string_view bestmove,
                            std::string_view ponder) {
  sync_cout << "bestmove " << bestmove;
  if (!ponder.empty())
    std::cout << " ponder " << ponder;
  std::cout << sync_endl;
}

void UCIEngine::gpu_info() {
  std::stringstream ss;

  ss << "\nGPU Information\n";
  ss << "===============\n";

  if (GPU::gpu_available()) {
    auto &backend = GPU::gpu();

    ss << "Status: Available\n";
    ss << "Backend: ";
    switch (backend.type()) {
    case GPU::BackendType::Metal:
      ss << "Metal";
      break;
    case GPU::BackendType::CUDA:
      ss << "CUDA";
      break;
    default:
      ss << "None";
      break;
    }
    ss << "\n";

    ss << "Device: " << backend.device_name() << "\n";
    ss << "Unified Memory: " << (backend.has_unified_memory() ? "Yes" : "No")
       << "\n";
    ss << "Max Buffer Size: " << (backend.max_buffer_size() / (1024 * 1024))
       << " MB\n";
    ss << "Max Threadgroup Memory: " << backend.max_threadgroup_memory()
       << " bytes\n";
    ss << "Allocated Memory: " << (backend.allocated_memory() / 1024)
       << " KB\n";
    ss << "Peak Memory: " << (backend.peak_memory() / 1024) << " KB\n";
    ss << "\nNNUE GPU: Disabled by design\n";
    ss << "GPU role: transformer/MCTS inference only\n";
  } else {
    ss << "Status: Not available\n";
    ss << "Reason: No compatible GPU backend found\n";
    ss << "Note: AB search still uses CPU NNUE\n";
  }

  sync_cout << ss.str() << sync_endl;
}

void UCIEngine::gpu_benchmark() {
  sync_cout << "info string GPU NNUE benchmark disabled: NNUE is CPU-only. "
               "Use MCTS/Hybrid searches to exercise transformer GPU inference."
            << sync_endl;
}

static float get_float_option(Engine &engine, const char *name,
                              float fallback) {
  if (!engine.get_options().count(name))
    return fallback;
  try {
    return std::stof(std::string(engine.get_options()[name]));
  } catch (...) {
    return fallback;
  }
}

static bool float_option_is_non_default(Engine &engine, const char *name,
                                        float default_value) {
  return std::abs(get_float_option(engine, name, default_value) -
                  default_value) > 1e-6f;
}

static float get_float_option_alias(Engine &engine, const char *preferred,
                                    const char *legacy, float fallback) {
  const float preferred_value = get_float_option(engine, preferred, fallback);
  if (float_option_is_non_default(engine, preferred, fallback))
    return preferred_value;
  if (float_option_is_non_default(engine, legacy, fallback))
    return get_float_option(engine, legacy, fallback);
  return preferred_value;
}

static int auto_mcts_minibatch_size(int num_threads) {
#ifdef __APPLE__
  // The current BT4/MPSGraph path is strongest and most predictable with
  // direct single-position evals. Explicit MCTSMinibatchSize values remain
  // available for throughput experiments.
  (void)num_threads;
  return 1;
#else
  return num_threads >= 8 ? 64 : 32;
#endif
}

static std::optional<std::string>
transformer_low_time_fallback_reason(Engine &engine,
                                     const Search::LimitsType &limits) {
  if (limits.infinite || limits.nodes > 0)
    return std::nullopt;

  const TimePoint low_time_fallback = static_cast<TimePoint>(
      static_cast<int>(engine.get_options()["TransformerLowTimeFallbackMs"]));

  if (low_time_fallback > 0 && limits.movetime > 0 &&
      limits.movetime < low_time_fallback) {
    return "fixed movetime " + std::to_string(limits.movetime) + "ms";
  }

  const bool has_clock_time = limits.time[WHITE] > 0 || limits.time[BLACK] > 0;
  if (!has_clock_time)
    return std::nullopt;

  Position pos;
  StateInfo st;
  pos.set(engine.fen(), static_cast<bool>(engine.get_options()["UCI_Chess960"]),
          &st);

  const Color us = pos.side_to_move();
  const TimePoint time_left = limits.time[us];
  const TimePoint increment = limits.inc[us];
  const TimePoint move_overhead = static_cast<TimePoint>(
      static_cast<int>(engine.get_options()["Move Overhead"]));
  const TimePoint increment_backed_clock =
      time_left + std::max<TimePoint>(0, increment) - move_overhead;
  if (low_time_fallback > 0 && time_left < low_time_fallback &&
      increment_backed_clock < low_time_fallback) {
    return std::string(us == WHITE ? "white" : "black") + " clock " +
           std::to_string(time_left) + "ms";
  }

  const int moves_to_go = limits.movestogo > 0 ? limits.movestogo : 30;
  const TimePoint base_budget = time_left / std::max(1, moves_to_go);
  const TimePoint inc_bonus = std::max<TimePoint>(0, increment) * 3 / 4;
  const TimePoint budget = base_budget + inc_bonus;
  const TimePoint hard_cap = std::max<TimePoint>(500, time_left / 4);
  const TimePoint reserve_cap =
      std::max<TimePoint>(1, time_left - move_overhead);
  const TimePoint estimated_move_budget =
      std::max<TimePoint>(250, std::min({budget, hard_cap, reserve_cap}));
  const TimePoint min_transformer_move_budget = static_cast<TimePoint>(
      static_cast<int>(engine.get_options()["TransformerMinMoveBudgetMs"]));
  if (min_transformer_move_budget > 0 &&
      estimated_move_budget < min_transformer_move_budget) {
    return "estimated move budget " + std::to_string(estimated_move_budget) +
           "ms";
  }

  return std::nullopt;
}

static bool should_preload_transformer_search() {
  const char *env = std::getenv("METALFISH_PRELOAD_TRANSFORMER");
  if (!env || !*env)
    return true;

  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value != "0" && value != "false" && value != "off" && value != "no";
}

static MCTS::SearchParams make_mcts_config(Engine &engine,
                                           const std::string &nn_weights,
                                           int num_threads) {
  MCTS::SearchParams config;
  config.nn_weights_path = nn_weights;
  config.num_threads = num_threads;
  config.nn_backend = std::string(engine.get_options()["NNBackend"]);
  config.coreml_model_path =
      std::string(engine.get_options()["NNCoreMLModelPath"]);
  config.coreml_compute_units =
      std::string(engine.get_options()["NNCoreMLComputeUnits"]);

  config.cpuct = get_float_option(engine, "MCTSCPuct", config.cpuct);
  config.cpuct_at_root =
      get_float_option(engine, "MCTSCPuctAtRoot", config.cpuct_at_root);
  config.cpuct_base =
      get_float_option(engine, "MCTSCPuctBase", config.cpuct_base);
  config.cpuct_factor =
      get_float_option(engine, "MCTSCPuctFactor", config.cpuct_factor);
  config.cpuct_base_at_root = get_float_option(engine, "MCTSCPuctBaseAtRoot",
                                               config.cpuct_base_at_root);
  config.cpuct_factor_at_root = get_float_option(
      engine, "MCTSCPuctFactorAtRoot", config.cpuct_factor_at_root);

  config.fpu_absolute = engine.get_options()["MCTSFpuAbsolute"];
  config.fpu_absolute_at_root = engine.get_options()["MCTSFpuAbsoluteAtRoot"];
  config.fpu_value = get_float_option(engine, "MCTSFpuValue", config.fpu_value);
  config.fpu_value_at_root =
      get_float_option(engine, "MCTSFpuValueAtRoot", config.fpu_value_at_root);
  config.fpu_reduction =
      get_float_option(engine, "MCTSFpuReduction", config.fpu_reduction);
  config.fpu_reduction_at_root = get_float_option(
      engine, "MCTSFpuReductionAtRoot", config.fpu_reduction_at_root);

  config.policy_softmax_temp = get_float_option_alias(
      engine, "MCTSPolicyTemperature", "MCTSPolicySoftmaxTemp",
      config.policy_softmax_temp);
  config.moves_left_max_effect = get_float_option(
      engine, "MCTSMovesLeftMaxEffect", config.moves_left_max_effect);
  config.moves_left_threshold = get_float_option(
      engine, "MCTSMovesLeftThreshold", config.moves_left_threshold);
  config.moves_left_slope =
      get_float_option(engine, "MCTSMovesLeftSlope", config.moves_left_slope);
  config.moves_left_constant_factor = get_float_option(
      engine, "MCTSMovesLeftConstantFactor", config.moves_left_constant_factor);
  config.moves_left_scaled_factor = get_float_option(
      engine, "MCTSMovesLeftScaledFactor", config.moves_left_scaled_factor);
  config.moves_left_quadratic_factor =
      get_float_option(engine, "MCTSMovesLeftQuadraticFactor",
                       config.moves_left_quadratic_factor);
  config.temperature =
      get_float_option(engine, "MCTSTemperature", config.temperature);
  config.temp_winpct_cutoff = get_float_option(engine, "MCTSTempValueCutoff",
                                               config.temp_winpct_cutoff);
  config.smart_pruning_factor = get_float_option(
      engine, "MCTSSmartPruningFactor", config.smart_pruning_factor);
  config.smart_pruning_minimum_batches =
      static_cast<int>(engine.get_options()["MCTSSmartPruningMinimumBatches"]);
  config.kld_gain_min = get_float_option(engine, "MCTSMinimumKLDGainPerNode",
                                         config.kld_gain_min);
  config.kld_gain_average_interval =
      static_cast<int>(engine.get_options()["MCTSKLDGainAverageInterval"]);
  config.move_overhead_ms = static_cast<float>(
      static_cast<int>(engine.get_options()["Move Overhead"]));
  config.time_manager = std::string(engine.get_options()["MCTSTimeManager"]);
  config.cache_history_length =
      static_cast<int>(engine.get_options()["MCTSCacheHistoryLength"]);
  config.nn_cache_size =
      static_cast<int>(engine.get_options()["MCTSNNCacheSize"]);
  config.solid_tree_threshold =
      static_cast<int>(engine.get_options()["MCTSSolidTreeThreshold"]);
  config.max_prefetch =
      static_cast<int>(engine.get_options()["MCTSMaxPrefetch"]);
  config.max_collision_events =
      static_cast<int>(engine.get_options()["MCTSMaxCollisionEvents"]);
  config.max_collision_visits =
      static_cast<int>(engine.get_options()["MCTSMaxCollisionVisits"]);
  config.max_collision_visits_scaling_start = static_cast<int>(
      engine.get_options()["MCTSMaxCollisionVisitsScalingStart"]);
  config.max_collision_visits_scaling_end = static_cast<int>(
      engine.get_options()["MCTSMaxCollisionVisitsScalingEnd"]);
  config.max_collision_visits_scaling_power =
      get_float_option(engine, "MCTSMaxCollisionVisitsScalingPower",
                       config.max_collision_visits_scaling_power);
  config.virtual_loss =
      std::max(1, static_cast<int>(engine.get_options()["MCTSVirtualLoss"]));
  const int requested_minibatch =
      static_cast<int>(engine.get_options()["MCTSMinibatchSize"]);
  config.minibatch_size = requested_minibatch > 0
                              ? requested_minibatch
                              : auto_mcts_minibatch_size(num_threads);
  config.max_out_of_order_evals_factor = get_float_option_alias(
      engine, "MCTSMaxOutOfOrderEvalsFactor", "MCTSMaxOutOfOrderFactor",
      config.max_out_of_order_evals_factor);
  if (config.max_out_of_order_evals_factor <= 0.0f)
    config.out_of_order_eval = false;

  config.add_dirichlet_noise = engine.get_options()["MCTSAddDirichletNoise"];
  config.noise_epsilon =
      get_float_option(engine, "MCTSNoiseEpsilon", config.noise_epsilon);
  config.noise_alpha =
      get_float_option(engine, "MCTSNoiseAlpha", config.noise_alpha);

  if (engine.get_options()["MCTSParityPreset"]) {
    config.add_dirichlet_noise = false;
    config.out_of_order_eval = false;
    config.fpu_reduction_at_root = config.fpu_reduction;
    config.fpu_absolute_at_root = config.fpu_absolute;
    config.fpu_value_at_root = config.fpu_value;
  }

  return config;
}

struct HybridThreadSplit {
  int mcts_threads = 1;
  int ab_threads = 1;
};

static bool is_fixed_budget_hybrid_search(const Search::LimitsType *limits) {
  return limits && (limits->movetime > 0 || limits->nodes > 0 ||
                    limits->depth > 0 || limits->mate > 0 || limits->infinite);
}

static int auto_hybrid_ab_threads_cap(int available) {
#ifdef __APPLE__
  if (available >= 8)
    return available - 2;
#endif
  (void)available;
  return 0;
}

static HybridThreadSplit
compute_hybrid_thread_split(Engine &engine,
                            const Search::LimitsType *limits = nullptr) {
  const int total_threads =
      std::max(1, static_cast<int>(engine.get_options()["Threads"]));
  const int mcts_override =
      static_cast<int>(engine.get_options()["HybridMCTSThreads"]);
  const int ab_override =
      static_cast<int>(engine.get_options()["HybridABThreads"]);
  const int auto_ab_cap =
      static_cast<int>(engine.get_options()["HybridAutoABThreadsCap"]);
  // UCI Threads should describe search workers. The hybrid coordinator is a
  // lightweight sleeping monitor, so keep it outside the AB+MCTS worker budget
  // instead of stealing a core from the engines on Apple Silicon.
  const int available = total_threads;
  int mcts_threads = 0;
  if (mcts_override > 0) {
    mcts_threads = std::clamp(mcts_override, 1, available);
  } else {
    // One transformer worker keeps the GPU side active while leaving the CPU
    // search enough cores to finish tactical verification.
    mcts_threads = 1;
  }

  int ab_threads = 0;
  if (ab_override > 0) {
    ab_threads = std::clamp(ab_override, 1, available);
  } else {
    if (is_fixed_budget_hybrid_search(limits)) {
      ab_threads = std::max(1, available - mcts_threads);
    } else {
      const int effective_ab_cap =
          auto_ab_cap > 0 ? auto_ab_cap : auto_hybrid_ab_threads_cap(available);
      ab_threads = std::max(1, available - mcts_threads);
      if (effective_ab_cap > 0)
        ab_threads = std::min(effective_ab_cap, ab_threads);
    }
  }

  while (mcts_threads + ab_threads > available) {
    if (ab_threads > 1)
      --ab_threads;
    else if (mcts_threads > 1)
      --mcts_threads;
    else
      break;
  }

  return {mcts_threads, ab_threads};
}

static MCTS::ParallelHybridConfig
make_hybrid_config(Engine &engine, const std::string &nn_weights,
                   const Search::LimitsType *limits = nullptr) {
  MCTS::ParallelHybridConfig config;
  auto split = compute_hybrid_thread_split(engine, limits);

  config.mcts_config = make_mcts_config(engine, nn_weights, split.mcts_threads);
  config.mcts_config.kld_gain_min =
      get_float_option(engine, "HybridMCTSMinimumKLDGainPerNode", 0.0f);
  config.mcts_threads = split.mcts_threads;
  config.ab_threads = split.ab_threads;

  config.ab_min_depth = 10;
  config.ab_use_time = true;
  config.ab_policy_weight = std::clamp(
      get_float_option(engine, "HybridABPolicyWeight", 0.0f), 0.0f, 1.0f);
  config.agreement_threshold = 0.3f;
  config.override_threshold = 1.0f;
  config.policy_update_interval_ms = 50;
  config.use_position_classifier = true;
  config.decision_mode = MCTS::ParallelHybridConfig::DecisionMode::DYNAMIC;
  config.transformer_batch_size = 128;
  config.use_transformer_prefetch = true;
  config.ab_root_reject_mcts = engine.get_options()["HybridABRootRejectMCTS"];
  config.mcts_root_reject = engine.get_options()["HybridMCTSRootReject"];
  config.use_shared_tt = engine.get_options()["HybridMCTSUseSharedTT"];
  config.mcts_ab_root_hints = engine.get_options()["HybridMCTSABRootHints"];
  config.mcts_ab_root_hint_delay_ms =
      static_cast<int>(engine.get_options()["HybridMCTSABRootHintDelayMs"]);
  config.mcts_ab_root_hint_count =
      static_cast<int>(engine.get_options()["HybridMCTSABRootHintCount"]);
  config.ab_candidate_verify_ms =
      static_cast<int>(engine.get_options()["HybridABCandidateVerifyMs"]);
  config.ab_candidate_verify_count =
      static_cast<int>(engine.get_options()["HybridABCandidateVerifyCount"]);
  config.root_pawn_lever_tiebreak =
      engine.get_options()["HybridRootPawnLeverTieBreak"];
  config.ane_root_probe = engine.get_options()["HybridANERootProbe"];
  config.ane_confirm_mcts_override =
      engine.get_options()["HybridANEConfirmMCTSOverride"];
  config.ane_weights_path =
      std::string(engine.get_options()["HybridANEWeights"]);
  config.ane_model_path =
      std::string(engine.get_options()["HybridANEModelPath"]);
  config.ane_compute_units =
      std::string(engine.get_options()["HybridANEComputeUnits"]);
  config.ane_root_hint_count =
      static_cast<int>(engine.get_options()["HybridANERootHintCount"]);
  config.ane_root_hint_wait_ms =
      static_cast<int>(engine.get_options()["HybridANERootHintWaitMs"]);
  config.ane_min_budget_ms =
      static_cast<int>(engine.get_options()["HybridANEMinBudgetMs"]);
  config.trace_decisions = engine.get_options()["HybridTrace"];
  return config;
}

static std::string make_mcts_cache_key(const std::string &nn_weights,
                                       const MCTS::SearchParams &config);

static std::string
make_hybrid_cache_key(const std::string &nn_weights,
                      const MCTS::ParallelHybridConfig &config) {
  std::ostringstream key;
  key << make_mcts_cache_key(nn_weights, config.mcts_config) << "|hybrid"
      << "|" << config.mcts_threads << "|" << config.ab_threads << "|"
      << config.ab_min_depth << "|" << config.ab_max_depth << "|"
      << config.ab_use_time << "|" << config.ab_policy_weight << "|"
      << config.agreement_threshold << "|" << config.override_threshold << "|"
      << config.policy_update_interval_ms << "|"
      << config.use_position_classifier << "|"
      << static_cast<int>(config.decision_mode) << "|"
      << config.transformer_batch_size << "|"
      << config.transformer_batch_timeout_us << "|"
      << config.use_transformer_prefetch << "|" << config.ab_root_reject_mcts
      << "|" << config.mcts_root_reject << "|" << config.use_shared_tt << "|"
      << config.mcts_ab_root_hints << "|" << config.mcts_ab_root_hint_delay_ms
      << "|" << config.mcts_ab_root_hint_count << "|"
      << config.ab_candidate_verify_ms << "|"
      << config.ab_candidate_verify_count << "|"
      << config.root_pawn_lever_tiebreak << "|" << config.ane_root_probe << "|"
      << config.ane_confirm_mcts_override << "|"
      << config.ane_weights_path << "|" << config.ane_model_path << "|"
      << config.ane_compute_units << "|" << config.ane_root_hint_count << "|"
      << config.ane_root_hint_wait_ms
      << "|" << config.ane_min_budget_ms << "|" << config.trace_decisions;
  return key.str();
}

static std::filesystem::path executable_dir() {
  namespace fs = std::filesystem;
  fs::path dir = fs::current_path();
#ifdef __APPLE__
  char buf[4096];
  uint32_t sz = sizeof(buf);
  if (_NSGetExecutablePath(buf, &sz) == 0)
    dir = fs::path(buf).parent_path();
#elif defined(__linux__)
  std::error_code ec;
  auto exe = fs::read_symlink("/proc/self/exe", ec);
  if (!ec)
    dir = exe.parent_path();
#endif
  return dir;
}

static std::string resolve_nn_weights_path(const std::string &raw_path) {
  namespace fs = std::filesystem;
  if (raw_path.empty())
    return {};

  fs::path path(raw_path);
  if (path.is_absolute())
    return raw_path;

  const fs::path exe_dir = executable_dir();
  const std::vector<fs::path> candidates = {
      fs::current_path() / path,
      exe_dir / path,
      exe_dir.parent_path() / path,
  };

  for (const auto &candidate : candidates) {
    std::error_code ec;
    if (!fs::is_regular_file(candidate, ec))
      continue;
    auto canonical = fs::weakly_canonical(candidate, ec);
    return ec ? candidate.string() : canonical.string();
  }

  return raw_path;
}

static std::string get_nn_weights_path(Engine &engine) {
  std::string nn_weights =
      resolve_nn_weights_path(std::string(engine.get_options()["NNWeights"]));
  if (nn_weights.empty()) {
    const char *env_path = std::getenv("METALFISH_NN_WEIGHTS");
    if (env_path)
      nn_weights = resolve_nn_weights_path(env_path);
  }
  if (nn_weights.empty()) {
    namespace fs = std::filesystem;
    auto try_dir = [&](const fs::path &dir) -> std::string {
      if (!fs::is_directory(dir))
        return {};
      std::string best;
      for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file())
          continue;
        auto ext = entry.path().extension().string();
        if (ext == ".pb" || ext == ".gz" || ext == ".onnx") {
          auto name = entry.path().filename().string();
          if (best.empty() || name > best)
            best = entry.path().string();
        }
      }
      return best;
    };

    auto exe_dir = executable_dir();

    nn_weights = try_dir(exe_dir / "networks");
    if (nn_weights.empty())
      nn_weights = try_dir(exe_dir.parent_path() / "networks");
    if (nn_weights.empty())
      nn_weights = try_dir(fs::current_path() / "networks");
    if (nn_weights.empty())
      nn_weights = try_dir("networks");

    if (!nn_weights.empty())
      sync_cout << "info string Auto-detected NN weights: " << nn_weights
                << sync_endl;
  }
  return nn_weights;
}

static std::unique_ptr<MCTS::ParallelHybridSearch> g_parallel_hybrid_search;
static std::string g_parallel_hybrid_key;
static std::shared_ptr<MCTS::Search> g_active_mcts;
static std::mutex g_active_mcts_mutex;
static std::shared_ptr<MCTS::Search> g_cached_mcts;
static std::string g_cached_mcts_key;
static std::mutex g_cached_mcts_mutex;
static std::thread g_search_waiter;
static std::mutex g_search_waiter_mutex;

static void reset_cached_search_objects() {
  if (g_parallel_hybrid_search)
    g_parallel_hybrid_search->new_game();

  std::lock_guard<std::mutex> lock(g_cached_mcts_mutex);
  if (g_cached_mcts)
    g_cached_mcts->NewGame();
}

static int resolve_mcts_thread_count(Engine &engine, bool explicit_threads_arg,
                                     int requested_threads, bool announce_cap) {
  int num_threads = requested_threads;
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0)
      num_threads = 4;
  }

  const bool allow_parallel_mcts =
      static_cast<bool>(engine.get_options()["MCTSParallelSearch"]);
#ifdef __APPLE__
  if (!allow_parallel_mcts && num_threads > 1) {
    if (announce_cap) {
      sync_cout << "info string Capping pure MCTS threads from " << num_threads
                << " to 1 for Apple Silicon strength stability" << sync_endl;
    }
    num_threads = 1;
  }
#endif

  int mcts_thread_cap =
      static_cast<int>(engine.get_options()["MCTSMaxThreads"]);
  if (!explicit_threads_arg && mcts_thread_cap <= 0) {
    // Strength-first auto mode:
    // Apple Silicon MPSGraph latency is better with one MCTS worker for the
    // current transformer. Higher worker counts require MCTSParallelSearch for
    // explicit throughput tests.
    mcts_thread_cap = 1;
  }

  if (!explicit_threads_arg && mcts_thread_cap > 0 &&
      num_threads > mcts_thread_cap) {
    if (announce_cap) {
      sync_cout << "info string Capping MCTS threads from " << num_threads
                << " to " << mcts_thread_cap << " for tactical stability"
                << sync_endl;
    }
    num_threads = mcts_thread_cap;
  }

  return std::max(1, num_threads);
}

static std::string make_mcts_cache_key(const std::string &nn_weights,
                                       const MCTS::SearchParams &config) {
  std::ostringstream key;
  key << nn_weights << "|" << config.nn_backend << "|"
      << config.coreml_model_path << "|" << config.coreml_compute_units << "|"
      << config.num_threads << "|" << config.cpuct << "|"
      << config.cpuct_at_root << "|" << config.cpuct_base << "|"
      << config.cpuct_factor << "|" << config.cpuct_base_at_root << "|"
      << config.cpuct_factor_at_root << "|" << config.fpu_absolute << "|"
      << config.fpu_absolute_at_root << "|" << config.fpu_value << "|"
      << config.fpu_value_at_root << "|" << config.fpu_reduction << "|"
      << config.fpu_reduction_at_root << "|" << config.policy_softmax_temp
      << "|" << config.moves_left_max_effect << "|"
      << config.moves_left_threshold << "|" << config.moves_left_slope << "|"
      << config.moves_left_constant_factor << "|"
      << config.moves_left_scaled_factor << "|"
      << config.moves_left_quadratic_factor << "|" << config.temperature << "|"
      << config.temp_winpct_cutoff << "|" << config.draw_score << "|"
      << config.wdl_rescale_ratio << "|" << config.wdl_rescale_diff << "|"
      << config.two_fold_draws << "|" << config.sticky_endgames << "|"
      << config.virtual_loss << "|" << config.minibatch_size << "|"
      << config.max_out_of_order_evals_factor << "|"
      << config.add_dirichlet_noise << "|" << config.noise_epsilon << "|"
      << config.noise_alpha << "|" << config.out_of_order_eval << "|"
      << config.nn_cache_size << "|" << config.smart_pruning_factor << "|"
      << config.smart_pruning_minimum_batches << "|" << config.kld_gain_min
      << "|" << config.kld_gain_average_interval << "|" << config.time_manager
      << "|" << config.cache_history_length << "|"
      << config.solid_tree_threshold << "|" << config.max_prefetch << "|"
      << config.max_collision_events << "|" << config.max_collision_visits
      << "|" << config.max_collision_visits_scaling_start << "|"
      << config.max_collision_visits_scaling_end << "|"
      << config.max_collision_visits_scaling_power;
  return key.str();
}

static std::shared_ptr<MCTS::Search>
get_or_create_cached_mcts(const MCTS::SearchParams &config,
                          const std::string &cache_key, bool *created) {
  std::lock_guard<std::mutex> lock(g_cached_mcts_mutex);
  if (created)
    *created = false;

  if (g_cached_mcts && g_cached_mcts_key == cache_key)
    return g_cached_mcts;

  if (g_cached_mcts) {
    g_cached_mcts->Stop();
    g_cached_mcts->ClearCallbacks();
    {
      std::lock_guard<std::mutex> active_lock(g_active_mcts_mutex);
      if (g_active_mcts == g_cached_mcts)
        g_active_mcts.reset();
    }
    g_cached_mcts.reset();
    g_cached_mcts_key.clear();
  }

  g_cached_mcts.reset(MCTS::CreateSearch(config).release());
  g_cached_mcts_key = cache_key;
  if (created)
    *created = static_cast<bool>(g_cached_mcts);
  return g_cached_mcts;
}

static void preload_search_objects(Engine &engine) {
  bool need_hybrid = engine.get_options()["UseHybridSearch"];
  bool need_mcts = engine.get_options()["UseMCTS"];
  if (!need_hybrid && !need_mcts)
    return;

  if (!should_preload_transformer_search())
    return;

  std::string nn_weights = get_nn_weights_path(engine);
  if (nn_weights.empty())
    return;

  if (need_hybrid) {
    const int hybrid_mcts_override =
        static_cast<int>(engine.get_options()["HybridMCTSThreads"]);
    const int hybrid_ab_override =
        static_cast<int>(engine.get_options()["HybridABThreads"]);
    const bool stable_hybrid_split =
        hybrid_mcts_override > 0 && hybrid_ab_override > 0;
    const bool can_preload_hybrid =
        stable_hybrid_split || !g_parallel_hybrid_search;
    if (can_preload_hybrid) {
      auto config = make_hybrid_config(engine, nn_weights);
      const std::string cache_key = make_hybrid_cache_key(nn_weights, config);
      const bool needs_reinit =
          !g_parallel_hybrid_search || g_parallel_hybrid_key != cache_key;

      if (needs_reinit && (!g_parallel_hybrid_search ||
                           !g_parallel_hybrid_search->is_searching())) {
        if (g_parallel_hybrid_search) {
          g_parallel_hybrid_search->stop();
          g_parallel_hybrid_search->wait();
          g_parallel_hybrid_search.reset();
          g_parallel_hybrid_key.clear();
        }
        g_parallel_hybrid_search =
            MCTS::create_parallel_hybrid_search(&engine, config);
        g_parallel_hybrid_key = g_parallel_hybrid_search ? cache_key : "";
      }

      if (g_parallel_hybrid_search && needs_reinit) {
        sync_cout << "info string Hybrid search preloaded (transformer ready)"
                  << sync_endl;
      }
    }
  }

  if (need_mcts) {
    {
      std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
      if (g_active_mcts)
        return;
    }
    {
      std::lock_guard<std::mutex> lock(g_search_waiter_mutex);
      if (g_search_waiter.joinable())
        return;
    }

    const int requested_threads =
        static_cast<int>(engine.get_options()["Threads"]);
    const int num_threads =
        resolve_mcts_thread_count(engine, false, requested_threads, false);
    MCTS::SearchParams config =
        make_mcts_config(engine, nn_weights, num_threads);
    const std::string cache_key = make_mcts_cache_key(nn_weights, config);

    bool created = false;
    auto mcts = get_or_create_cached_mcts(config, cache_key, &created);
    if (mcts && created) {
      sync_cout << "info string MCTS search preloaded (transformer ready)"
                << sync_endl;
    }
  }
}

static void join_search_waiter() {
  std::lock_guard<std::mutex> lock(g_search_waiter_mutex);
  if (g_search_waiter.joinable())
    g_search_waiter.join();
}

static void stop_active_searches() {
  if (g_parallel_hybrid_search && g_parallel_hybrid_search->is_searching())
    g_parallel_hybrid_search->stop();
  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    if (g_active_mcts)
      g_active_mcts->Stop();
  }
}

static void ponderhit_active_searches(Engine &engine) {
  if (g_parallel_hybrid_search && g_parallel_hybrid_search->is_searching()) {
    g_parallel_hybrid_search->ponderhit();
    return;
  }
  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    if (g_active_mcts) {
      g_active_mcts->PonderHit();
      return;
    }
  }
  engine.set_ponderhit(false);
}

static void wait_active_searches() {
  if (g_parallel_hybrid_search && g_parallel_hybrid_search->is_searching())
    g_parallel_hybrid_search->wait();
  join_search_waiter();
}

} // namespace MetalFish

// Cleanup function to be called before GPU shutdown (in MetalFish namespace)
void MetalFish::cleanup_parallel_hybrid_search() {
  join_search_waiter();
  if (g_parallel_hybrid_search) {
    g_parallel_hybrid_search->stop();
    g_parallel_hybrid_search->wait();
    g_parallel_hybrid_search.reset();
    g_parallel_hybrid_key.clear();
  }
  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    g_active_mcts.reset();
  }
  {
    std::lock_guard<std::mutex> lock(g_cached_mcts_mutex);
    if (g_cached_mcts) {
      g_cached_mcts->Stop();
      g_cached_mcts->ClearCallbacks();
      g_cached_mcts.reset();
      g_cached_mcts_key.clear();
    }
  }
}

namespace MetalFish {

void UCIEngine::parallel_hybrid_go(std::istringstream &is) {
  Search::LimitsType limits = parse_limits(is);

  if (!limits.ponderMode) {
    if (auto reason = transformer_low_time_fallback_reason(engine, limits)) {
      sync_cout << "info string Time safety: " << *reason
                << "; using Alpha-Beta without transformer MCTS" << sync_endl;
      engine.go(limits);
      return;
    }
  }

  sync_cout << "info string Starting Parallel Hybrid Search (MCTS + AB)..."
            << sync_endl;

  const int total_threads =
      std::max(1, static_cast<int>(engine.get_options()["Threads"]));
  if (total_threads < 3) {
    sync_cout << "info string Hybrid search requires at least 3 threads for "
                 "separate AB, MCTS, and coordinator workers; falling back "
                 "to Alpha-Beta"
              << sync_endl;
    engine.go(limits);
    return;
  }

  std::string nn_weights = get_nn_weights_path(engine);
  if (nn_weights.empty()) {
    sync_cout << "info string ERROR: No transformer weights. Set UCI option "
                 "NNWeights."
              << sync_endl;
    sync_cout << "info string Falling back to Alpha-Beta search" << sync_endl;
    engine.go(limits);
    return;
  }

  auto config = make_hybrid_config(engine, nn_weights, &limits);
  sync_cout << "info string Hybrid thread split: MCTS=" << config.mcts_threads
            << " AB=" << config.ab_threads
            << " (search workers=" << (config.mcts_threads + config.ab_threads)
            << ", +1 coordinator, total="
            << (config.mcts_threads + config.ab_threads + 1) << ")"
            << sync_endl;

  const std::string cache_key = make_hybrid_cache_key(nn_weights, config);
  bool need_reinit =
      !g_parallel_hybrid_search || g_parallel_hybrid_key != cache_key;

  if (need_reinit) {
    if (g_parallel_hybrid_search) {
      g_parallel_hybrid_search->stop();
      g_parallel_hybrid_search->wait();
      g_parallel_hybrid_search.reset();
      g_parallel_hybrid_key.clear();
    }
    g_parallel_hybrid_search =
        MCTS::create_parallel_hybrid_search(&engine, config);
    g_parallel_hybrid_key = g_parallel_hybrid_search ? cache_key : "";

    if (!g_parallel_hybrid_search) {
      sync_cout << "info string ERROR: Failed to create parallel hybrid search"
                << sync_endl;
      sync_cout << "info string Falling back to Alpha-Beta search" << sync_endl;
      engine.go(limits);
      return;
    }
    sync_cout << "info string Parallel hybrid search initialized" << sync_endl;
  } else {
    g_parallel_hybrid_search->set_config(config);
  }

  Position pos;
  StateInfo st;
  pos.set(engine.fen(), static_cast<bool>(engine.get_options()["UCI_Chess960"]),
          &st);

  auto best_move_cb = [](Move best, Move ponder) {
    std::string best_str = UCIEngine::move(best, false);
    std::string ponder_str = ponder != Move::none()
                                 ? " ponder " + UCIEngine::move(ponder, false)
                                 : "";
    sync_cout << "bestmove " << best_str << ponder_str << sync_endl;
  };

  auto info_cb = [](const std::string &info) {
    sync_cout << info << sync_endl;
  };

  join_search_waiter();
  g_parallel_hybrid_search->start_search(pos, limits, best_move_cb, info_cb);
}

void UCIEngine::mcts_mt_go(std::istringstream &is) {
  std::string args;
  std::getline(is, args, '\0');

  std::string token;
  int num_threads = static_cast<int>(engine.get_options()["Threads"]);
  bool explicit_threads_arg = false;

  std::istringstream option_args(args);
  while (option_args >> token) {
    if (token.find("threads=") == 0) {
      num_threads = std::stoi(token.substr(8));
      explicit_threads_arg = true;
    }
  }

  std::istringstream limit_args(args);
  Search::LimitsType limits = parse_limits(limit_args);

  if (auto reason = transformer_low_time_fallback_reason(engine, limits)) {
    sync_cout << "info string Time safety: " << *reason
              << "; using Alpha-Beta without transformer MCTS" << sync_endl;
    engine.go(limits);
    return;
  }

  num_threads = resolve_mcts_thread_count(engine, explicit_threads_arg,
                                          num_threads, true);

#ifdef __APPLE__
  constexpr std::uint64_t kLowNodeMctsThreadCap = 1024;
  if (limits.nodes > 0 && limits.nodes <= kLowNodeMctsThreadCap &&
      num_threads > 1) {
    sync_cout << "info string Capping low-node pure MCTS search from "
              << num_threads
              << " to 1 thread for Apple Silicon tactical stability"
              << sync_endl;
    num_threads = 1;
  }
#endif

  sync_cout << "info string Starting Multi-Threaded MCTS Search with "
            << num_threads << " threads..." << sync_endl;

  std::string nn_weights = get_nn_weights_path(engine);
  if (nn_weights.empty()) {
    sync_cout << "info string ERROR: No transformer weights. Set UCI option "
                 "NNWeights."
              << sync_endl;
    sync_cout << "info string Falling back to Alpha-Beta search" << sync_endl;
    engine.go(limits);
    return;
  }

  MCTS::SearchParams config = make_mcts_config(engine, nn_weights, num_threads);

  std::shared_ptr<MCTS::Search> mcts;
  const std::string cache_key = make_mcts_cache_key(nn_weights, config);

  join_search_waiter();
  mcts = get_or_create_cached_mcts(config, cache_key, nullptr);

  if (!mcts) {
    sync_cout << "info string ERROR: Failed to create multi-threaded MCTS"
              << sync_endl;
    sync_cout << "info string Falling back to Alpha-Beta search" << sync_endl;
    engine.go(limits);
    return;
  }

  sync_cout << "info string Multi-threaded MCTS initialized with "
            << num_threads << " threads" << sync_endl;

  std::string fen = engine.fen();

  auto best_move_cb = [](Move best, Move ponder) {
    std::string best_str = UCIEngine::move(best, false);
    std::string ponder_str = ponder != Move::none()
                                 ? " ponder " + UCIEngine::move(ponder, false)
                                 : "";
    sync_cout << "bestmove " << best_str << ponder_str << sync_endl;
  };

  auto info_cb = [](const std::string &info) {
    sync_cout << info << sync_endl;
  };

  auto start_time = std::chrono::steady_clock::now();
  mcts->StartSearch(fen, limits, best_move_cb, info_cb);

  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    g_active_mcts = mcts;
  }

  join_search_waiter();
  {
    std::lock_guard<std::mutex> wlock(g_search_waiter_mutex);
    g_search_waiter = std::thread([mcts, start_time, num_threads]() {
      mcts->Wait();

      {
        std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
        if (g_active_mcts == mcts)
          g_active_mcts.reset();
      }

      auto end_time = std::chrono::steady_clock::now();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time)
                            .count();

      const auto &stats = mcts->Stats();
      uint64_t nodes = stats.total_nodes.load();
      uint64_t nn_evals = stats.nn_evaluations.load();
      uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

      sync_cout << "info string Final stats:" << sync_endl;
      sync_cout << "info string   Nodes: " << nodes << sync_endl;
      sync_cout << "info string   NPS: " << nps << sync_endl;
      sync_cout << "info string   Time: " << elapsed_ms << "ms" << sync_endl;
      sync_cout << "info string   Threads: " << num_threads << sync_endl;
      sync_cout << "info string   NN evals: " << nn_evals << sync_endl;
      sync_cout << "info string   Cache hits: " << stats.cache_hits.load()
                << " misses: " << stats.cache_misses.load() << sync_endl;
    });
  }
}

void UCIEngine::mcts_batch_benchmark(std::istringstream &is) {
  (void)is;
  sync_cout << "info string MCTS NNUE batch benchmark disabled: hybrid MCTS "
               "uses the transformer backend, not GPU NNUE."
            << sync_endl;
}

void UCIEngine::nnue_benchmark(std::istream &is) {
  (void)is;
  sync_cout << "info string NNUE benchmark policy: CPU-only. GPU NNUE is "
               "disabled so Apple GPU time stays reserved for transformer MCTS."
            << sync_endl;
  sync_cout << "info string Use 'bench' or 'speedtest' for end-to-end CPU NNUE "
               "search throughput."
            << sync_endl;
}

} // namespace MetalFish
