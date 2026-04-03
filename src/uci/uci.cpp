/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "uci/uci.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <filesystem>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#endif

#include "core/memory.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "eval/evaluate.h"
#include "eval/gpu_backend.h"
#include "eval/gpu_integration.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_accumulator.h"
#include "eval/score.h"
#include "hybrid/classifier.h"
#include "hybrid/hybrid_search.h"
#include "hybrid/position_adapter.h"
#include "mcts/gpu_backend.h"
#include "mcts/search.h"
#include "search/search.h"
#include "uci/benchmark.h"
#include "uci/engine.h"
#include "uci/ucioption.h"

namespace MetalFish {

// Forward declarations for search synchronization helpers (defined below)
static void stop_active_searches();
static void wait_active_searches();
static void join_search_waiter();
static void preload_search_objects(Engine &engine);

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
    return value != "0" && value != "false" && value != "off" &&
           value != "no";
  }();
  return enabled;
}

int detect_apple_perf_cores() {
#ifdef __APPLE__
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname("hw.perflevel0.physicalcpu_max", &value, &size, nullptr,
                   0) == 0 &&
      value > 0) {
    return value;
  }
#endif
  return 0;
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

    token.clear(); // Avoid a stale if getline() returns nothing or a blank line
    is >> std::skipws >> token;

    // Debug command trace is opt-in via METALFISH_UCI_TRACE=1.
    if (uci_trace_enabled() && !token.empty()) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
      std::cerr << "[UCI:" << ms << "] " << cmd << std::endl;
    }

    // ======================================================================
    // Standard UCI Protocol Commands
    // See: https://backscattering.de/chess/uci/
    // ======================================================================

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
      // Don't wait for active MCTS/Hybrid searches -- they're non-blocking
      // and fire bestmove via callback. Just preload if needed and respond.
      preload_search_objects(engine);
      sync_cout << "readyok" << sync_endl;
    }

    else if (token == "ucinewgame") {
      stop_active_searches();
      wait_active_searches();
      engine.search_clear();
    }

    else if (token == "position")
      position(is);

    else if (token == "setoption")
      setoption(is);

    else if (token == "ponderhit")
      engine.set_ponderhit(false);

    else if (token == "go") {
      // The standard `go` command routes to the active engine mode
      // based on UCI options. GUIs set UseMCTS or UseHybridSearch
      // via `setoption` before sending `go`.
      if (engine.get_options()["UseMCTS"])
        mcts_mt_go(is);
      else if (engine.get_options()["UseHybridSearch"])
        parallel_hybrid_go(is);
      else
        go(is);
      }

    // ======================================================================
    // MetalFish Extensions (debugging / CLI only -- GUIs never send these)
    // ======================================================================

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

  } while (token != "quit" &&
           cli.argc == 1); // The command-line arguments are one-shot

  // Clean up background search threads before exiting
  stop_active_searches();
  wait_active_searches();
  join_search_waiter();
}

Search::LimitsType UCIEngine::parse_limits(std::istream &is) {
  Search::LimitsType limits;
  std::string token;

  limits.startTime = now(); // The search starts as early as possible

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
      engine.search_clear(); // search_clear may take a while
      elapsed = now();
    }
  }

  elapsed =
      now() - elapsed + 1; // Ensure positivity to avoid a 'divide by zero'

  dbg_print();

  std::cerr << "\n==========================="   //
            << "\nTotal time (ms) : " << elapsed //
            << "\nNodes searched  : " << nodes   //
            << "\nNodes/second    : " << 1000 * nodes / elapsed << std::endl;

  // reset callback, to not capture a dangling reference to nodesSearched
  engine.set_on_update_full(
      [&](const auto &i) { on_update_full(i, options["UCI_ShowWDL"]); });
}

void UCIEngine::benchmark(std::istream &args) {
  // Probably not very important for a test this long, but include for
  // completeness and sanity.
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

  // Set options once at the start.
  auto ss =
      std::istringstream("name Threads value " + std::to_string(setup.threads));
  setoption(ss);
  ss = std::istringstream("name Hash value " + std::to_string(setup.ttSize));
  setoption(ss);
  ss = std::istringstream("name UCI_Chess960 value false");
  setoption(ss);

  // Warmup
  for (const auto &cmd : setup.commands) {
    std::istringstream is(cmd);
    is >> std::skipws >> token;

    if (token == "go") {
      // One new line is produced by the search, so omit it here
      std::cerr << "\rWarmup position " << cnt++ << '/' << NUM_WARMUP_POSITIONS;

      Search::LimitsType limits = parse_limits(is);

      // Run with silenced network verification
      engine.go(limits);
      engine.wait_for_search_finished();
    } else if (token == "position")
      position(is);
    else if (token == "ucinewgame") {
      engine.search_clear(); // search_clear may take a while
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

  engine.search_clear(); // search_clear may take a while

  for (const auto &cmd : setup.commands) {
    std::istringstream is(cmd);
    is >> std::skipws >> token;

    if (token == "go") {
      // One new line is produced by the search, so omit it here
      std::cerr << "\rPosition " << cnt++ << '/' << numGoCommands;

      Search::LimitsType limits = parse_limits(is);

      nodesSearched = 0;
      TimePoint elapsed = now();

      // Run with silenced network verification
      engine.go(limits);
      engine.wait_for_search_finished();

      totalTime += now() - elapsed;

      updateHashfullReadings();

      nodes += nodesSearched;
    } else if (token == "position")
      position(is);
    else if (token == "ucinewgame") {
      engine.search_clear(); // search_clear may take a while
    }
  }

  totalTime = std::max<TimePoint>(
      totalTime, 1); // Ensure positivity to avoid a 'divide by zero'

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

    // New GPU NNUE interface
    if (GPU::gpu_nnue_manager_available()) {
      ss << "\n" << GPU::gpu_nnue_manager().status_string();
    } else if (GPU::gpu_nnue_manager().is_ready()) {
      ss << "\n" << GPU::gpu_nnue_manager().status_string();
    } else {
      ss << "\nGPU NNUE: Not initialized\n";
    }
  } else {
    ss << "Status: Not available\n";
    ss << "Reason: No compatible GPU backend found\n";
    ss << "Note: Engine will use CPU-only evaluation\n";
  }

  sync_cout << ss.str() << sync_endl;
}

void UCIEngine::gpu_benchmark() {
  // Use cout directly for incremental output
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     MetalFish GPU NNUE Benchmark Suite                   ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════╝\n\n";
  std::cout.flush();

  if (!GPU::gpu_available()) {
    std::cout << "GPU not available\n";
    return;
  }

  auto &manager = GPU::gpu_nnue_manager();
  if (!manager.is_ready()) {
    std::cout << "GPU NNUE not initialized (networks not loaded)\n";
    return;
  }

  auto &backend = GPU::gpu();

  std::cout << "Creating test positions...\n" << std::flush;

  // Create test positions - diverse set from real games and test suites
  // Includes opening, middlegame, endgame, and tactical positions
  const char *fens[] = {
      // Opening positions
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
      // Middlegame positions (complex)
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 "
      "10",
      "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 9",
      "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 8",
      // Tactical positions
      "r2qkb1r/pp2nppp/3p4/2pNn1B1/2B1P1b1/3P4/PPP2PPP/R2QK2R w KQkq - 0 1",
      "1rbq1rk1/p1b1nppp/1p2p3/8/1B1pN3/P2B4/1P3PPP/2RQ1R1K w - - 0 1",
      "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/2N1BN2/PPB1QPPP/R3R1K1 w - - 0 1",
      "2rr2k1/1p2qppp/p1n1pn2/8/3P4/1QN1PN2/PP3PPP/R4RK1 w - - 0 1",
      // Endgame positions
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "8/8/8/8/8/8/8/4K2k w - - 0 1",
      "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
      "8/8/p1p5/1p5p/1P5p/8/PPP2K1k/4R3 w - - 0 1",
      "8/pp2r1k1/2p1p3/3pP2p/6pP/1P1R2P1/P1P2PK1/8 w - - 0 1",
      "8/3k4/8/8/8/8/3K4/1R6 w - - 0 1",
      // More middlegame (diverse piece counts)
      "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      "r2qkbnr/ppp2ppp/2n1p3/3pPb2/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq d6 0 5",
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4",
      "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5",
      // Positions with imbalances
      "r1bq1rk1/ppp1bppp/2n2n2/3pp3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 w - - 0 7",
      "r2qkb1r/pp1bpppp/2n2n2/2pp4/3P1B2/2N1PN2/PPP2PPP/R2QKB1R w KQkq - 0 6",
      "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
      "r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
      // More endgames
      "8/5k2/8/8/8/8/5K2/4R3 w - - 0 1",
      "8/8/8/3k4/8/3K4/8/3R4 w - - 0 1",
      "8/8/8/8/8/k7/8/KR6 w - - 0 1",
      "8/8/8/8/8/8/6k1/4K2R w - - 0 1",
  };
  constexpr int NUM_FENS = 32;

  std::vector<std::unique_ptr<std::deque<StateInfo>>> states_vec;
  std::vector<Position> positions(2048);
  for (int i = 0; i < 2048; i++) {
    states_vec.push_back(std::make_unique<std::deque<StateInfo>>(1));
    positions[i].set(fens[i % NUM_FENS], false, &states_vec.back()->back());
  }

  std::cout << "Positions created: 2048 (from " << NUM_FENS
            << " unique FENs)\n\n"
            << std::flush;

  manager.set_min_batch_size(1);

  // ========================================================================
  // BENCHMARK 1: CPU Simple Eval (baseline - material only)
  // ========================================================================
  std::cout << "=== BENCHMARK 1: CPU Simple Eval (material + PST) ===\n";
  std::cout << "Scope: Eval::simple_eval() - material + piece-square tables\n";
  std::cout << "Iterations: 100,000\n" << std::flush;

  std::vector<double> cpu_simple_samples;
  cpu_simple_samples.reserve(100000);
  for (int i = 0; i < 100; i++) { // warmup
    volatile Value v = Eval::simple_eval(positions[i % NUM_FENS]);
    (void)v;
  }
  for (int i = 0; i < 100000; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile Value v = Eval::simple_eval(positions[i % NUM_FENS]);
    (void)v;
    auto end = std::chrono::high_resolution_clock::now();
    cpu_simple_samples.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  std::sort(cpu_simple_samples.begin(), cpu_simple_samples.end());
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nCPU simple_eval:\n";
  std::cout << "  Median: " << cpu_simple_samples[50000] << " µs\n";
  std::cout << "  P95:    " << cpu_simple_samples[95000] << " µs\n";
  std::cout << "  P99:    " << cpu_simple_samples[99000] << " µs\n";
  std::cout << "  Min:    " << cpu_simple_samples.front() << " µs\n";
  std::cout << "  Max:    " << cpu_simple_samples.back() << " µs\n\n"
            << std::flush;

  // ========================================================================
  // BENCHMARK 1b: CPU NNUE Full Evaluation (CRITICAL BASELINE)
  // ========================================================================
  std::cout << "=== BENCHMARK 1b: CPU NNUE Full Evaluation ===\n";
  std::cout << "Note: CPU NNUE benchmark requires separate network loading.\n";
  std::cout << "For scope-matched comparison, use 'bench' command NPS.\n";
  std::cout << "GPU NNUE latency should be compared against CPU NNUE latency\n";
  std::cout
      << "from engine search, which is approximately 1-5 µs per position.\n\n"
      << std::flush;

  // We store placeholder scores for correctness comparison
  // The GPU scores will be compared for consistency only
  std::vector<int> valid_pos_indices;
  for (int i = 0; i < 2048; i++) {
    if (!positions[i].checkers()) {
      valid_pos_indices.push_back(i);
    }
  }
  std::cout << "Valid positions (no check): " << valid_pos_indices.size()
            << " / 2048\n\n"
            << std::flush;

  // ========================================================================
  // BENCHMARK 2: CPU Feature Extraction (via batch creation)
  // ========================================================================
  std::cout << "=== BENCHMARK 2: Batch Creation Benchmark ===\n";
  std::cout << "Scope: Create GPU batch with position features\n";
  std::cout << "Iterations: 100,000\n" << std::flush;

  {
    std::vector<double> cpu_feat_samples;
    cpu_feat_samples.reserve(100000);

    for (int i = 0; i < 100; i++) { // warmup
      GPU::GPUEvalBatch batch;
      batch.add_position(positions[i % NUM_FENS]);
    }
    for (int i = 0; i < 100000; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      GPU::GPUEvalBatch batch;
      batch.add_position(positions[i % NUM_FENS]);
      auto end = std::chrono::high_resolution_clock::now();
      cpu_feat_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(cpu_feat_samples.begin(), cpu_feat_samples.end());
    std::cout << "\nBatch creation (includes feature extraction):\n";
    std::cout << "  Median: " << cpu_feat_samples[50000] << " µs\n";
    std::cout << "  P95:    " << cpu_feat_samples[95000] << " µs\n";
    std::cout << "  P99:    " << cpu_feat_samples[99000] << " µs\n\n"
              << std::flush;
  }

  // Note: CPU NNUE full eval requires access to engine's networks which are
  // private. The CPU baseline for comparison is simple_eval (material + PST).
  // For scope-matched comparison, see search benchmark which uses full NNUE.

  // ========================================================================
  // BENCHMARK 3: Feature Count Distribution
  // ========================================================================
  std::cout << "=== BENCHMARK 3: Feature Count Distribution ===\n";
  std::cout << "Checking if 64-feature-per-perspective cap is ever hit\n"
            << std::flush;

  {
    int total_positions = 0;
    int capped_positions = 0;
    int max_features_seen = 0;
    int max_per_perspective = 0;
    std::vector<int> feature_histogram(129, 0); // Up to 128 total features

    for (int i = 0; i < 2048; i++) {
      GPU::GPUEvalBatch batch;
      batch.add_position(positions[i]);

      // Estimate features from piece count (each non-king piece = 2 features)
      int piece_count = positions[i].count<ALL_PIECES>() - 2; // exclude kings
      int estimated_features = piece_count * 2; // white + black perspective

      max_features_seen = std::max(max_features_seen, estimated_features);
      max_per_perspective = std::max(max_per_perspective, piece_count);
      if (estimated_features < 129)
        feature_histogram[estimated_features]++;
      // Check against GPU limit: 64 per perspective
      if (piece_count > GPU::GPU_MAX_FEATURES_PER_PERSPECTIVE)
        capped_positions++;
      total_positions++;
    }

    std::cout << "\nPositions analyzed: " << total_positions << "\n";
    std::cout << "Max features seen (total):  " << max_features_seen << "\n";
    std::cout << "Max features per perspective: " << max_per_perspective
              << "\n";
    std::cout << "GPU limit per perspective: "
              << GPU::GPU_MAX_FEATURES_PER_PERSPECTIVE << "\n";
    std::cout << "Positions exceeding GPU limit: " << capped_positions;
    std::cout << " (" << std::fixed << std::setprecision(2)
              << (100.0 * capped_positions / total_positions) << "%)\n";
    std::cout << "Feature distribution (top 5):\n";

    std::vector<std::pair<int, int>> sorted_hist;
    for (int i = 0; i < 129; i++) {
      if (feature_histogram[i] > 0)
        sorted_hist.push_back({feature_histogram[i], i});
    }
    std::sort(sorted_hist.rbegin(), sorted_hist.rend());
    for (int i = 0; i < std::min(5, (int)sorted_hist.size()); i++) {
      std::cout << "  " << sorted_hist[i].second
                << " features: " << sorted_hist[i].first << " positions\n";
    }
    std::cout << "\n" << std::flush;
  }

  // ========================================================================
  // BENCHMARK 4: GPU Dispatch Overhead (Minimal Kernel)
  // ========================================================================
  std::cout << "=== BENCHMARK 4: GPU Dispatch Overhead (Minimal Kernel) ===\n";
  std::cout << "Scope: create_encoder + dispatch(1) + submit_and_wait\n";
  std::cout << "Iterations: 1,000\n" << std::flush;

  const char *shader = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void minimal_kernel(device int* out [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
      if (gid == 0) out[0] = 1;
    }
  )";

  if (backend.compile_library("bench_minimal", shader)) {
    auto kernel = backend.create_kernel("minimal_kernel", "bench_minimal");
    auto buffer = backend.create_buffer(sizeof(int));

    std::vector<double> dispatch_samples;
    for (int i = 0; i < 100; i++) { // warmup
      auto enc = backend.create_encoder();
      enc->set_kernel(kernel.get());
      enc->set_buffer(buffer.get(), 0);
      enc->dispatch_threads(1);
      backend.submit_and_wait(enc.get());
    }
    for (int i = 0; i < 1000; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      auto enc = backend.create_encoder();
      enc->set_kernel(kernel.get());
      enc->set_buffer(buffer.get(), 0);
      enc->dispatch_threads(1);
      backend.submit_and_wait(enc.get());
      auto end = std::chrono::high_resolution_clock::now();
      dispatch_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(dispatch_samples.begin(), dispatch_samples.end());
    std::cout << "\nGPU minimal dispatch:\n";
    std::cout << "  Median: " << std::fixed << std::setprecision(1)
              << dispatch_samples[500] << " µs\n";
    std::cout << "  P95:    " << dispatch_samples[950] << " µs\n";
    std::cout << "  P99:    " << dispatch_samples[990] << " µs\n\n"
              << std::flush;
  }

  // ========================================================================
  // BENCHMARK 5: GPU Stage Breakdown (N=1, 8, 512)
  // ========================================================================
  std::cout << "=== BENCHMARK 5: GPU Stage Breakdown ===\n";
  std::cout << "Breaking down end-to-end latency into stages\n";
  std::cout << "Iterations: 100 per batch size\n\n" << std::flush;

  std::vector<int> breakdown_sizes = {1, 8, 512};
  for (int N : breakdown_sizes) {
    std::vector<double> stage_prep, stage_gpu;

    for (int iter = 0; iter < 100; iter++) {
      // Stage 1: Batch preparation (CPU)
      auto t1 = std::chrono::high_resolution_clock::now();
      GPU::GPUEvalBatch batch;
      batch.reserve(N);
      for (int i = 0; i < N; i++) {
        batch.add_position(positions[i]);
      }
      auto t2 = std::chrono::high_resolution_clock::now();

      // Stage 2: GPU evaluation (buffer + dispatch + kernel + sync)
      manager.evaluate_batch(batch, true);
      auto t3 = std::chrono::high_resolution_clock::now();

      stage_prep.push_back(
          std::chrono::duration<double, std::micro>(t2 - t1).count());
      stage_gpu.push_back(
          std::chrono::duration<double, std::micro>(t3 - t2).count());
    }

    std::sort(stage_prep.begin(), stage_prep.end());
    std::sort(stage_gpu.begin(), stage_gpu.end());

    std::cout << "Batch Size " << N << ":\n";
    std::cout << "  CPU prep (batch creation):  " << std::fixed
              << std::setprecision(1) << stage_prep[50] << " µs ("
              << (stage_prep[50] / N) << " µs/pos)\n";
    std::cout << "  GPU eval (buf+disp+kernel): " << stage_gpu[50] << " µs ("
              << (stage_gpu[50] / N) << " µs/pos)\n";
    std::cout << "  Total:                      "
              << (stage_prep[50] + stage_gpu[50]) << " µs ("
              << ((stage_prep[50] + stage_gpu[50]) / N) << " µs/pos)\n";
    std::cout << "  GPU fraction:               " << std::setprecision(1)
              << (100.0 * stage_gpu[50] / (stage_prep[50] + stage_gpu[50]))
              << "%\n\n"
              << std::flush;
  }

  // ========================================================================
  // BENCHMARK 6: GPU Batch Latency Table (extended)
  // ========================================================================
  std::cout << "=== BENCHMARK 6: GPU End-to-End Batch Latency Table ===\n";
  std::cout << "Including batch sizes 768, 1536, 2048 to explain 1024 jump\n";
  std::cout << "Iterations: 100 per batch size\n\n";

  std::cout << "Batch   Median    P95       P99       Per-Pos\n";
  std::cout << "Size    (µs)      (µs)      (µs)      (µs)\n";
  std::cout << "------------------------------------------------\n"
            << std::flush;

  std::vector<int> batch_sizes = {1,   2,   4,   8,    16,   32,   64,   128,
                                  256, 512, 768, 1024, 1536, 2048, 3072, 4096};

  for (int batch_size : batch_sizes) {
    std::vector<double> samples;
    for (int iter = 0; iter < 100; iter++) {
      GPU::GPUEvalBatch batch;
      batch.reserve(batch_size);
      for (int i = 0; i < batch_size; i++) {
        batch.add_position(positions[i % 2048]);
      }
      auto start = std::chrono::high_resolution_clock::now();
      manager.evaluate_batch(batch, true);
      auto end = std::chrono::high_resolution_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    double median = samples[50];
    double p95 = samples[95];
    double p99 = samples[99];

    std::cout << std::setw(5) << batch_size << std::fixed
              << std::setprecision(1) << std::setw(10) << median
              << std::setw(10) << p95 << std::setw(10) << p99 << std::setw(10)
              << (median / batch_size) << "\n"
              << std::flush;
  }

  // ========================================================================
  // BENCHMARK 7: True Batching Verification
  // ========================================================================
  std::cout << "\n=== BENCHMARK 7: True Batching Verification ===\n";
  std::cout << "Sequential: N separate command buffers\n";
  std::cout
      << "Batched: 1 command buffer with 2 dispatches (feature + forward)\n\n";

  std::cout << "N       Sequential  Batched     Speedup   CB Ratio\n";
  std::cout << "        (N×1 CB)    (1×1 CB)              (N:1)\n";
  std::cout << "----------------------------------------------------\n"
            << std::flush;

  std::vector<int> verify_sizes = {16, 64, 256, 1024};
  for (int N : verify_sizes) {
    std::vector<double> seq_samples;
    for (int iter = 0; iter < 50; iter++) {
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < N; i++) {
        GPU::GPUEvalBatch batch;
        batch.reserve(1);
        batch.add_position(positions[i]);
        manager.evaluate_batch(batch, true);
      }
      auto end = std::chrono::high_resolution_clock::now();
      seq_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::vector<double> batch_samples;
    for (int iter = 0; iter < 50; iter++) {
      GPU::GPUEvalBatch batch;
      batch.reserve(N);
      for (int i = 0; i < N; i++) {
        batch.add_position(positions[i]);
      }
      auto start = std::chrono::high_resolution_clock::now();
      manager.evaluate_batch(batch, true);
      auto end = std::chrono::high_resolution_clock::now();
      batch_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(seq_samples.begin(), seq_samples.end());
    std::sort(batch_samples.begin(), batch_samples.end());
    double seq_med = seq_samples[25];
    double batch_med = batch_samples[25];

    std::cout << std::setw(5) << N << std::fixed << std::setprecision(1)
              << std::setw(12) << seq_med << std::setw(12) << batch_med
              << std::setw(10) << (seq_med / batch_med) << "×" << std::setw(10)
              << N << ":1\n"
              << std::flush;
  }

  // ========================================================================
  // BENCHMARK 8: GPU Evaluation Consistency Check
  // ========================================================================
  std::cout << "\n=== BENCHMARK 8: GPU Evaluation Consistency Check ===\n";
  std::cout << "Verifying GPU produces consistent, non-zero scores\n";
  std::cout << "Running same batch twice to check reproducibility\n\n"
            << std::flush;

  if (valid_pos_indices.empty()) {
    std::cout << "ERROR: No valid positions for comparison\n\n";
  } else {
    const int CHECK_SIZE = std::min(1000, (int)valid_pos_indices.size());

    // First evaluation
    GPU::GPUEvalBatch batch1;
    batch1.reserve(CHECK_SIZE);
    for (int i = 0; i < CHECK_SIZE; i++) {
      batch1.add_position(positions[valid_pos_indices[i]]);
    }
    manager.evaluate_batch(batch1, true);

    // Second evaluation (same positions)
    GPU::GPUEvalBatch batch2;
    batch2.reserve(CHECK_SIZE);
    for (int i = 0; i < CHECK_SIZE; i++) {
      batch2.add_position(positions[valid_pos_indices[i]]);
    }
    manager.evaluate_batch(batch2, true);

    // Check consistency
    int non_zero = 0;
    int consistent = 0;
    int64_t sum_abs_score = 0;
    int min_score = INT_MAX, max_score = INT_MIN;

    for (int i = 0; i < CHECK_SIZE; i++) {
      int score1 = batch1.positional_scores[i];
      int score2 = batch2.positional_scores[i];

      if (score1 != 0)
        non_zero++;
      if (score1 == score2)
        consistent++;

      sum_abs_score += std::abs(score1);
      min_score = std::min(min_score, score1);
      max_score = std::max(max_score, score1);
    }

    std::cout << "Positions checked: " << CHECK_SIZE << "\n";
    std::cout << "Non-zero GPU scores: " << non_zero << " (" << std::fixed
              << std::setprecision(1) << (100.0 * non_zero / CHECK_SIZE)
              << "%)\n";
    std::cout << "Consistent across runs: " << consistent << " ("
              << (100.0 * consistent / CHECK_SIZE) << "%)\n";
    std::cout << "Mean |GPU score|: " << (double(sum_abs_score) / CHECK_SIZE)
              << "\n";
    std::cout << "Score range: [" << min_score << ", " << max_score << "]\n";

    std::cout << "\nSample scores (first 10 positions):\n";
    std::cout << "  Pos  Run 1      Run 2      Match\n";
    for (int i = 0; i < std::min(10, CHECK_SIZE); i++) {
      int score1 = batch1.positional_scores[i];
      int score2 = batch2.positional_scores[i];
      std::cout << "  " << std::setw(3) << i << "  " << std::setw(8) << score1
                << "  " << std::setw(8) << score2 << "  "
                << (score1 == score2 ? "Yes" : "NO") << "\n";
    }
    std::cout << "\n" << std::flush;
  }

  // ========================================================================
  // BENCHMARK 9: Dataset Description
  // ========================================================================
  std::cout << "=== BENCHMARK 9: Dataset Description ===\n";
  std::cout << "Position source: Standard benchmark FENs (diverse "
               "openings/middlegames)\n";
  std::cout << "Total unique positions: " << NUM_FENS << "\n";
  std::cout << "Positions used in benchmarks: 2048 (cycled from " << NUM_FENS
            << " unique)\n\n";

  // Piece count distribution
  std::vector<int> piece_counts(33, 0);
  for (int i = 0; i < NUM_FENS && i < 2048; i++) {
    int pc = positions[i].count<ALL_PIECES>();
    if (pc < 33)
      piece_counts[pc]++;
  }

  std::cout << "Piece count distribution:\n";
  int min_pc = 33, max_pc = 0;
  for (int i = 2; i < 33; i++) {
    if (piece_counts[i] > 0) {
      min_pc = std::min(min_pc, i);
      max_pc = std::max(max_pc, i);
    }
  }
  std::cout << "  Range: " << min_pc << " - " << max_pc << " pieces\n";

  // Show top 5 most common piece counts
  std::vector<std::pair<int, int>> pc_sorted;
  for (int i = 2; i < 33; i++) {
    if (piece_counts[i] > 0)
      pc_sorted.push_back({piece_counts[i], i});
  }
  std::sort(pc_sorted.rbegin(), pc_sorted.rend());
  std::cout << "  Most common piece counts:\n";
  for (int i = 0; i < std::min(5, (int)pc_sorted.size()); i++) {
    std::cout << "    " << pc_sorted[i].second
              << " pieces: " << pc_sorted[i].first << " positions\n";
  }
  std::cout << "\n" << std::flush;

  std::cout << "\nBenchmark complete.\n" << std::flush;
}

// ============================================================================
// Preload transformer weights and initialize search objects during isready.
// This ensures the first 'go' command responds instantly without weight
// loading.
// ============================================================================

static float get_float_option(Engine &engine, const char *name, float fallback) {
  if (!engine.get_options().count(name))
    return fallback;
  try {
    return std::stof(std::string(engine.get_options()[name]));
  } catch (...) {
    return fallback;
  }
}

static MCTS::SearchParams make_mcts_config(Engine &engine,
                                           const std::string &nn_weights,
                                           int num_threads) {
  MCTS::SearchParams config;
  config.nn_weights_path = nn_weights;
  config.num_threads = num_threads;

  config.cpuct = get_float_option(engine, "MCTSCPuct", config.cpuct);
  config.cpuct_at_root =
      get_float_option(engine, "MCTSCPuctAtRoot", config.cpuct_at_root);
  config.cpuct_base =
      get_float_option(engine, "MCTSCPuctBase", config.cpuct_base);
  config.cpuct_factor =
      get_float_option(engine, "MCTSCPuctFactor", config.cpuct_factor);
  config.cpuct_base_at_root =
      get_float_option(engine, "MCTSCPuctBaseAtRoot", config.cpuct_base_at_root);
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

  config.policy_softmax_temp = get_float_option(
      engine, "MCTSPolicySoftmaxTemp", config.policy_softmax_temp);
  config.virtual_loss = std::max(1, static_cast<int>(
                                        engine.get_options()["MCTSVirtualLoss"]));
  config.minibatch_size = std::max(
      1, static_cast<int>(engine.get_options()["MCTSMinibatchSize"]));
  config.max_out_of_order_evals_factor = get_float_option(
      engine, "MCTSMaxOutOfOrderFactor", config.max_out_of_order_evals_factor);

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

static HybridThreadSplit compute_hybrid_thread_split(Engine &engine) {
  const int total_threads =
      std::max(1, static_cast<int>(engine.get_options()["Threads"]));
  const int mcts_override =
      static_cast<int>(engine.get_options()["HybridMCTSThreads"]);
  const int ab_override =
      static_cast<int>(engine.get_options()["HybridABThreads"]);
  constexpr int coordinator_threads = 1;
  const int available = std::max(1, total_threads - coordinator_threads);

  int mcts_threads = 0;
  if (mcts_override > 0) {
    mcts_threads = std::clamp(mcts_override, 1, available);
  } else {
    // Strength-first default: keep MCTS single-threaded unless explicitly
    // overridden, and use additional cores for AB.
    mcts_threads = 1;
  }

  int ab_threads = 0;
  if (ab_override > 0) {
    ab_threads = std::clamp(ab_override, 1, available);
  } else {
    ab_threads = std::max(1, available - mcts_threads);
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
make_hybrid_config(Engine &engine, const std::string &nn_weights) {
  MCTS::ParallelHybridConfig config;
  auto split = compute_hybrid_thread_split(engine);

  // Use the same UCI-configurable MCTS params as pure MCTS mode,
  // so all improvements (cache key fix, KLD stopper, solidification,
  // prefetching, TB probing, etc.) apply to hybrid too.
  config.mcts_config = make_mcts_config(engine, nn_weights, split.mcts_threads);
  config.mcts_threads = split.mcts_threads;
  config.ab_threads = split.ab_threads;

  // Hybrid-specific override: slightly higher exploration at root
  config.mcts_config.cpuct_at_root = 2.15f;

  config.ab_min_depth = 10;
  config.ab_use_time = true;
  config.ab_policy_weight = 0.3f;
  config.agreement_threshold = 0.3f;
  config.override_threshold = 1.0f;
  config.policy_update_interval_ms = 50;
  config.use_position_classifier = true;
  config.decision_mode = MCTS::ParallelHybridConfig::DecisionMode::DYNAMIC;
  config.gpu_batch_size = 128;
  config.use_async_gpu_eval = true;
  config.use_gpu_resident_batches = true;
  config.use_simd_kernels = true;
  return config;
}

static std::string get_nn_weights_path(Engine &engine) {
  std::string nn_weights = std::string(engine.get_options()["NNWeights"]);
  if (nn_weights.empty()) {
    const char *env_path = std::getenv("METALFISH_NN_WEIGHTS");
    if (env_path)
      nn_weights = env_path;
  }
  if (nn_weights.empty()) {
    namespace fs = std::filesystem;
    auto try_dir = [&](const fs::path &dir) -> std::string {
      if (!fs::is_directory(dir)) return {};
      std::string best;
      for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext == ".pb" || ext == ".gz" || ext == ".onnx") {
          auto name = entry.path().filename().string();
          if (best.empty() || name > best)
            best = entry.path().string();
        }
      }
      return best;
    };

    auto exe_dir = fs::path("/proc/self/exe").parent_path();
    #ifdef __APPLE__
    {
      char buf[4096];
      uint32_t sz = sizeof(buf);
      if (_NSGetExecutablePath(buf, &sz) == 0)
        exe_dir = fs::path(buf).parent_path();
    }
    #endif

    nn_weights = try_dir(exe_dir / "networks");
    if (nn_weights.empty())
      nn_weights = try_dir(exe_dir.parent_path() / "networks");
    if (nn_weights.empty())
      nn_weights = try_dir(fs::current_path() / "networks");
    if (nn_weights.empty())
      nn_weights = try_dir("networks");

    if (!nn_weights.empty())
      sync_cout << "info string Auto-detected NN weights: " << nn_weights << sync_endl;
  }
  return nn_weights;
}

// Called from isready to preload transformer weights and compile MPSGraph.
// This makes the first 'go' instant -- no weight loading delay.
// Forward declarations for static globals defined below.
static std::unique_ptr<MCTS::ParallelHybridSearch> g_parallel_hybrid_search;
static GPU::GPUNNUEManager *g_hybrid_gpu_manager = nullptr;
static std::shared_ptr<MCTS::Search> g_active_mcts;
static std::mutex g_active_mcts_mutex;
static std::thread g_search_waiter;
static std::mutex g_search_waiter_mutex;

static void preload_search_objects(Engine &engine) {
  std::string nn_weights = get_nn_weights_path(engine);
  if (nn_weights.empty())
    return; // No weights configured -- nothing to preload

  bool need_hybrid = engine.get_options()["UseHybridSearch"];
  bool need_mcts = engine.get_options()["UseMCTS"];
  if (!need_hybrid && !need_mcts)
    return; // AB mode -- no transformer needed

  // Preload hybrid search object (includes transformer weight loading)
  if (need_hybrid && !g_parallel_hybrid_search) {
    GPU::GPUNNUEManager *gpu_manager = nullptr;
    if (GPU::gpu_nnue_manager_available())
      gpu_manager = &GPU::gpu_nnue_manager();

    auto config = make_hybrid_config(engine, nn_weights);
    g_parallel_hybrid_search =
        MCTS::create_parallel_hybrid_search(gpu_manager, &engine, config);
    g_hybrid_gpu_manager = gpu_manager;

    if (g_parallel_hybrid_search) {
      sync_cout << "info string Hybrid search preloaded (transformer ready)"
                << sync_endl;
    }
  }
}

// ============================================================================
// Parallel Hybrid Search Command (MCTS + AB running simultaneously)
// Optimized for Apple Silicon with unified memory
// ============================================================================

// Wait for any background search waiter thread to complete
static void join_search_waiter() {
  std::lock_guard<std::mutex> lock(g_search_waiter_mutex);
  if (g_search_waiter.joinable())
    g_search_waiter.join();
}

// Stop any active MCTS/Hybrid search (called from UCI stop command)
static void stop_active_searches() {
  if (g_parallel_hybrid_search && g_parallel_hybrid_search->is_searching())
    g_parallel_hybrid_search->stop();
  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    if (g_active_mcts)
      g_active_mcts->Stop();
  }
}

// Wait for any active MCTS/Hybrid search to finish (called from UCI isready)
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
    g_hybrid_gpu_manager = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    g_active_mcts.reset();
  }
}

namespace MetalFish {

void UCIEngine::parallel_hybrid_go(std::istringstream &is) {
  sync_cout << "info string Starting Parallel Hybrid Search (MCTS + AB)..."
            << sync_endl;

  // Parse search limits
  Search::LimitsType limits = parse_limits(is);

  // Get transformer weights path
  std::string nn_weights = get_nn_weights_path(engine);
  if (nn_weights.empty()) {
    sync_cout << "info string ERROR: No transformer weights. Set UCI option "
                 "NNWeights."
              << sync_endl;
    return;
  }

  // GPU NNUE manager is optional
  GPU::GPUNNUEManager *gpu_manager = nullptr;
  if (GPU::gpu_nnue_manager_available()) {
    gpu_manager = &GPU::gpu_nnue_manager();
  }

  auto config = make_hybrid_config(engine, nn_weights);
  sync_cout << "info string Hybrid thread split: MCTS="
            << config.mcts_threads << " AB=" << config.ab_threads
            << " (+1 coordinator, total="
            << (config.mcts_threads + config.ab_threads + 1) << ")"
            << sync_endl;

  // Reuse preloaded search object, or create if not yet initialized
  bool need_reinit =
      !g_parallel_hybrid_search || g_hybrid_gpu_manager != gpu_manager;

  if (need_reinit) {
    if (g_parallel_hybrid_search) {
      g_parallel_hybrid_search->stop();
      g_parallel_hybrid_search->wait();
      g_parallel_hybrid_search.reset();
    }
    g_parallel_hybrid_search =
        MCTS::create_parallel_hybrid_search(gpu_manager, &engine, config);
    g_hybrid_gpu_manager = gpu_manager;

    if (!g_parallel_hybrid_search) {
      sync_cout << "info string ERROR: Failed to create parallel hybrid search"
                << sync_endl;
      return;
    }
    sync_cout << "info string Parallel hybrid search initialized" << sync_endl;
  } else {
    g_parallel_hybrid_search->set_config(config);
  }

  // Get current position from engine
  Position pos;
  StateInfo st;
  pos.set(engine.fen(), false, &st);

  // Callbacks for UCI output
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

  // Start search - AB uses search_silent internally so no duplicate bestmove
  // The search runs asynchronously; the UCI loop must remain free to process
  // stop/quit commands. The bestmove callback fires when the search finishes.
  join_search_waiter(); // Clean up any previous waiter
  g_parallel_hybrid_search->start_search(pos, limits, best_move_cb, info_cb);

  // Note: We do NOT wait() here. The UCI loop must keep reading stdin so it
  // can process 'stop' and 'quit' commands. The bestmove callback handles
  // output when the search completes.
}

// ============================================================================
// Multi-Threaded MCTS Search Command (Pure GPU MCTS)
// ============================================================================
void UCIEngine::mcts_mt_go(std::istringstream &is) {
  // Parse threads option
  std::string token;
  int num_threads = static_cast<int>(engine.get_options()["Threads"]);
  bool explicit_threads_arg = false;

  // Parse additional options (threads=N)
  while (is >> token) {
    if (token.find("threads=") == 0) {
      num_threads = std::stoi(token.substr(8));
      explicit_threads_arg = true;
    }
  }

  // Handle threads=-1 (use all available threads)
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) {
      num_threads = 4; // Fallback if hardware_concurrency() returns 0
    }
  }

  // Reset stream for limit parsing
  is.clear();
  is.seekg(0);
  Search::LimitsType limits = parse_limits(is);

  int mcts_thread_cap = static_cast<int>(engine.get_options()["MCTSMaxThreads"]);
  if (!explicit_threads_arg && mcts_thread_cap <= 0) {
    // Strength-first auto mode:
    // - For standard timed play, cap to 1 thread to avoid tactical
    //   regressions observed with higher MCTS thread counts.
    // - For fixed-node runs, use a throughput-oriented cap.
    if (limits.nodes > 0) {
      int perf_cores = detect_apple_perf_cores();
      if (perf_cores > 0) {
        mcts_thread_cap = perf_cores + 2;
      }
    } else {
      mcts_thread_cap = 1;
    }
  }
  if (!explicit_threads_arg && mcts_thread_cap > 0 && num_threads > mcts_thread_cap) {
    sync_cout << "info string Capping MCTS threads from " << num_threads
              << " to " << mcts_thread_cap << " for tactical stability"
              << sync_endl;
    num_threads = mcts_thread_cap;
  }

  sync_cout << "info string Starting Multi-Threaded MCTS Search with "
            << num_threads << " threads..." << sync_endl;

  // Get transformer weights path from UCI option
  std::string nn_weights = get_nn_weights_path(engine);
  if (nn_weights.empty()) {
    sync_cout << "info string ERROR: No transformer weights. Set UCI option "
                 "NNWeights."
              << sync_endl;
    return;
  }

  MCTS::SearchParams config = make_mcts_config(engine, nn_weights, num_threads);

  static std::shared_ptr<MCTS::Search> s_cached_mcts;
  static std::string s_cached_key;

  std::shared_ptr<MCTS::Search> mcts;
  std::ostringstream config_key;
  config_key << nn_weights
             << "|" << config.num_threads
             << "|" << config.cpuct
             << "|" << config.cpuct_at_root
             << "|" << config.cpuct_base
             << "|" << config.cpuct_factor
             << "|" << config.fpu_reduction
             << "|" << config.fpu_reduction_at_root
             << "|" << config.policy_softmax_temp
             << "|" << config.virtual_loss
             << "|" << config.minibatch_size
             << "|" << config.max_out_of_order_evals_factor
             << "|" << config.add_dirichlet_noise;

  if (s_cached_mcts && s_cached_key == config_key.str()) {
    mcts = s_cached_mcts;
  } else {
    if (s_cached_mcts) {
      s_cached_mcts->Stop();
      s_cached_mcts->ClearCallbacks();
      {
        std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
        if (g_active_mcts == s_cached_mcts)
          g_active_mcts.reset();
      }
      join_search_waiter();
      s_cached_mcts.reset();
    }
    mcts.reset(MCTS::CreateSearch(config).release());
    s_cached_mcts = mcts;
    s_cached_key = config_key.str();
  }

  if (!mcts) {
    sync_cout << "info string ERROR: Failed to create multi-threaded MCTS"
              << sync_endl;
    return;
  }

  // Ensure any previous search waiter thread is joined before starting a new
  // search on the same cached object. start_search() handles stop/wait
  // internally so we only need to join the background waiter.
  join_search_waiter();

  sync_cout << "info string Multi-threaded MCTS initialized with "
            << num_threads << " threads" << sync_endl;

  // Get current position from engine
  std::string fen = engine.fen();

  // Callbacks for UCI output
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

  // Start search
  auto start_time = std::chrono::steady_clock::now();
  mcts->StartSearch(fen, limits, best_move_cb, info_cb);

  // Store a reference so the stop command can reach it
  {
    std::lock_guard<std::mutex> lock(g_active_mcts_mutex);
    g_active_mcts = mcts;
  }

  // Spawn a background waiter thread for post-search stats.
  // The UCI loop must remain free to process stop/quit commands.
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

// ============================================================================
// MCTS Batch vs Direct Benchmark
// ============================================================================
void UCIEngine::mcts_batch_benchmark(std::istringstream &is) {
  sync_cout << "info string MCTS Batched vs Direct Evaluation Benchmark"
            << sync_endl;
  sync_cout << "info string ==========================================="
            << sync_endl;

  // Get GPU NNUE manager
  GPU::GPUNNUEManager *gpu_manager = nullptr;
  if (GPU::gpu_nnue_manager_available()) {
    gpu_manager = &GPU::gpu_nnue_manager();
  }

  if (!gpu_manager) {
    sync_cout << "info string ERROR: GPU NNUE not available" << sync_endl;
    return;
  }

  // Parse options
  std::string token;
  int num_threads = 4;
  int duration_ms = 5000;

  while (is >> token) {
    if (token.find("threads=") == 0) {
      num_threads = std::stoi(token.substr(8));
    } else if (token.find("time=") == 0) {
      duration_ms = std::stoi(token.substr(5));
    }
  }

  // Handle threads=-1 (use all available threads)
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) {
      num_threads = 4;
    }
  }

  std::string fen = engine.fen();
  Search::LimitsType limits;
  limits.movetime = duration_ms;

  sync_cout << "info string Position: " << fen << sync_endl;
  sync_cout << "info string Threads: " << num_threads << sync_endl;
  sync_cout << "info string Duration: " << duration_ms << "ms per test"
            << sync_endl;
  sync_cout << "" << sync_endl;

  // ============================================
  // Test 1: Direct evaluation (old approach)
  // ============================================
  sync_cout << "info string --- Test 1: Direct Evaluation (mutex per eval) ---"
            << sync_endl;

  {
    MCTS::SearchParams config;
    config.num_threads = num_threads;
    config.cpuct_at_root = 2.5f;
    config.add_dirichlet_noise = true;

    auto mcts = MCTS::CreateSearch(config);

    auto start = std::chrono::steady_clock::now();
    mcts->StartSearch(fen, limits, nullptr, nullptr);
    mcts->Wait();
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    const auto &stats = mcts->Stats();
    uint64_t nodes = stats.total_nodes.load();
    uint64_t nn_evals = stats.nn_evaluations.load();
    uint64_t nps = elapsed_ms > 0 ? (nn_evals * 1000) / elapsed_ms : 0;

    sync_cout << "info string   Nodes: " << nodes << sync_endl;
    sync_cout << "info string   NPS: " << nps << sync_endl;
    sync_cout << "info string   NN evals: " << nn_evals << sync_endl;
    sync_cout << "info string   Cache hits: " << stats.cache_hits.load()
              << " misses: " << stats.cache_misses.load() << sync_endl;
  }

  sync_cout << "" << sync_endl;

  // ============================================
  // Test 2: Batched evaluation (new approach)
  // ============================================
  sync_cout
      << "info string --- Test 2: Batched Evaluation (dedicated thread) ---"
      << sync_endl;

  {
    MCTS::SearchParams config;
    config.num_threads = num_threads;
    config.cpuct_at_root = 2.5f;
    config.add_dirichlet_noise = true;

    auto mcts = MCTS::CreateSearch(config);

    auto start = std::chrono::steady_clock::now();
    mcts->StartSearch(fen, limits, nullptr, nullptr);
    mcts->Wait();
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    const auto &stats = mcts->Stats();
    uint64_t nodes = stats.total_nodes.load();
    uint64_t nn_evals = stats.nn_evaluations.load();
    uint64_t nps = elapsed_ms > 0 ? (nn_evals * 1000) / elapsed_ms : 0;

    sync_cout << "info string   Nodes: " << nodes << sync_endl;
    sync_cout << "info string   NPS: " << nps << sync_endl;
    sync_cout << "info string   NN evals: " << nn_evals << sync_endl;
    sync_cout << "info string   Cache hits: " << stats.cache_hits.load()
              << " misses: " << stats.cache_misses.load() << sync_endl;
  }

  sync_cout << "" << sync_endl;
  sync_cout << "info string Benchmark complete!" << sync_endl;
}

// ============================================================================
// NNUE Benchmark - Compare CPU vs GPU NNUE Performance
// ============================================================================
void UCIEngine::nnue_benchmark(std::istream &is) {
  sync_cout << "info string =================================================="
            << sync_endl;
  sync_cout << "info string      CPU vs GPU NNUE Benchmark" << sync_endl;
  sync_cout << "info string =================================================="
            << sync_endl;

  // Parse options
  std::string token;
  int num_positions = 1000;
  int batch_size = 64;

  while (is >> token) {
    if (token == "positions")
      is >> num_positions;
    else if (token == "batch")
      is >> batch_size;
  }

  // Test positions (diverse positions from different game phases)
  std::vector<std::string> test_fens = {
      // Starting position
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      // Italian Game
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      // Sicilian Defense
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
      // Queen's Gambit
      "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
      // Ruy Lopez
      "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
      // King's Indian
      "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
      // Complex middlegame
      "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
      // Endgame
      "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
      // Complex position
      "r2qr1k1/pp1nbppp/2p2n2/3p4/3P4/2NBPN2/PP3PPP/R1BQR1K1 w - - 0 11",
      // Tactical position
      "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"};

  sync_cout << "info string Test configuration:" << sync_endl;
  sync_cout << "info string   Positions to evaluate: " << num_positions
            << sync_endl;
  sync_cout << "info string   GPU batch size: " << batch_size << sync_endl;
  sync_cout << "info string   Test FENs: " << test_fens.size() << sync_endl;

  // Check GPU availability
  bool gpu_available = GPU::gpu_nnue_manager_available();
  if (gpu_available) {
    sync_cout << "info string   GPU NNUE: Available ("
              << GPU::gpu().device_name() << ")" << sync_endl;
    sync_cout << "info string   Unified Memory: "
              << (GPU::gpu().has_unified_memory() ? "Yes" : "No") << sync_endl;
  } else {
    sync_cout << "info string   GPU NNUE: Not available" << sync_endl;
  }
  sync_cout << "" << sync_endl;

  // Create positions using pointers to avoid copy issues
  std::vector<std::unique_ptr<Position>> positions;
  std::vector<std::unique_ptr<StateInfo>> states;
  positions.reserve(num_positions);
  states.reserve(num_positions);

  for (int i = 0; i < num_positions; i++) {
    states.push_back(std::make_unique<StateInfo>());
    positions.push_back(std::make_unique<Position>());
    positions.back()->set(test_fens[i % test_fens.size()], false,
                          states.back().get());
  }

  // =========================================================================
  // CPU NNUE Benchmark (using NNUE evaluation via engine search)
  // We'll use simple_eval as a baseline for CPU since we can't access networks
  // directly
  // =========================================================================
  sync_cout << "info string [CPU Simple Eval Benchmark (baseline)]"
            << sync_endl;

  // Warm up CPU
  for (int i = 0; i < 100 && i < num_positions; i++) {
    int v = Eval::simple_eval(*positions[i]);
    (void)v;
  }

  // CPU simple eval benchmark
  std::vector<int32_t> cpu_simple_results(num_positions);
  auto cpu_simple_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_positions; i++) {
    cpu_simple_results[i] = Eval::simple_eval(*positions[i]);
  }

  auto cpu_simple_end = std::chrono::high_resolution_clock::now();
  double cpu_simple_time_ms = std::chrono::duration<double, std::milli>(
                                  cpu_simple_end - cpu_simple_start)
                                  .count();
  double cpu_simple_pos_per_sec = (num_positions * 1000.0) / cpu_simple_time_ms;

  sync_cout << "info string   Time: " << std::fixed << std::setprecision(2)
            << cpu_simple_time_ms << " ms" << sync_endl;
  sync_cout << "info string   Positions/sec: " << std::fixed
            << std::setprecision(0) << cpu_simple_pos_per_sec << sync_endl;
  sync_cout << "" << sync_endl;

  // =========================================================================
  // GPU NNUE Benchmark (if available)
  // =========================================================================
  if (gpu_available) {
    sync_cout << "info string [GPU NNUE Benchmark]" << sync_endl;

    auto &gpu_manager = GPU::gpu_nnue_manager();
    gpu_manager.reset_stats();

    // Warm up GPU
    {
      GPU::GPUEvalBatch warmup_batch;
      warmup_batch.reserve(std::min(batch_size, 100));
      for (int i = 0; i < std::min(batch_size, 100); i++) {
        warmup_batch.add_position(*positions[i]);
      }
      gpu_manager.evaluate_batch(warmup_batch, true);
    }

    // GPU benchmark - single position evaluation
    sync_cout << "info string   [Single Position Mode]" << sync_endl;
    std::vector<int32_t> gpu_single_results(num_positions);
    auto gpu_single_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_positions; i++) {
      auto [psqt, pos_score] = gpu_manager.evaluate_single(*positions[i], true);
      gpu_single_results[i] = psqt + pos_score;
    }

    auto gpu_single_end = std::chrono::high_resolution_clock::now();
    double gpu_single_time_ms = std::chrono::duration<double, std::milli>(
                                    gpu_single_end - gpu_single_start)
                                    .count();
    double gpu_single_pos_per_sec =
        (num_positions * 1000.0) / gpu_single_time_ms;

    sync_cout << "info string     Time: " << std::fixed << std::setprecision(2)
              << gpu_single_time_ms << " ms" << sync_endl;
    sync_cout << "info string     Positions/sec: " << std::fixed
              << std::setprecision(0) << gpu_single_pos_per_sec << sync_endl;

    // GPU benchmark - batched evaluation
    gpu_manager.reset_stats();
    sync_cout << "info string   [Batched Mode (batch=" << batch_size << ")]"
              << sync_endl;
    std::vector<int32_t> gpu_batch_results(num_positions);
    auto gpu_batch_start = std::chrono::high_resolution_clock::now();

    for (int start = 0; start < num_positions; start += batch_size) {
      int end = std::min(start + batch_size, num_positions);
      int count = end - start;

      GPU::GPUEvalBatch batch;
      batch.reserve(count);

      for (int i = start; i < end; i++) {
        batch.add_position(*positions[i]);
      }

      gpu_manager.evaluate_batch(batch, true);

      // Store results
      for (int i = 0; i < count; i++) {
        int32_t psqt = batch.psqt_scores.size() > static_cast<size_t>(i)
                           ? batch.psqt_scores[i]
                           : 0;
        int32_t pos = batch.positional_scores.size() > static_cast<size_t>(i)
                          ? batch.positional_scores[i]
                          : 0;
        gpu_batch_results[start + i] = psqt + pos;
      }
    }

    auto gpu_batch_end = std::chrono::high_resolution_clock::now();
    double gpu_batch_time_ms = std::chrono::duration<double, std::milli>(
                                   gpu_batch_end - gpu_batch_start)
                                   .count();
    double gpu_batch_pos_per_sec = (num_positions * 1000.0) / gpu_batch_time_ms;

    sync_cout << "info string     Time: " << std::fixed << std::setprecision(2)
              << gpu_batch_time_ms << " ms" << sync_endl;
    sync_cout << "info string     Positions/sec: " << std::fixed
              << std::setprecision(0) << gpu_batch_pos_per_sec << sync_endl;
    sync_cout << "info string     GPU evaluations: "
              << gpu_manager.gpu_evaluations() << sync_endl;
    sync_cout << "info string     Batches: " << gpu_manager.total_batches()
              << sync_endl;
    sync_cout << "" << sync_endl;

    // =========================================================================
    // Performance Comparison
    // =========================================================================
    sync_cout << "info string [Performance Summary]" << sync_endl;
    sync_cout << "info string   CPU Simple Eval: " << std::fixed
              << std::setprecision(0) << cpu_simple_pos_per_sec << " pos/sec"
              << sync_endl;
    sync_cout << "info string   GPU Single:      " << std::fixed
              << std::setprecision(0) << gpu_single_pos_per_sec << " pos/sec"
              << sync_endl;
    sync_cout << "info string   GPU Batched:     " << std::fixed
              << std::setprecision(0) << gpu_batch_pos_per_sec << " pos/sec"
              << sync_endl;

    double batch_vs_single = gpu_batch_pos_per_sec / gpu_single_pos_per_sec;
    sync_cout << "info string   Batch speedup over single: " << std::fixed
              << std::setprecision(2) << batch_vs_single << "x" << sync_endl;
    sync_cout << "" << sync_endl;

    // Verify consistency between single and batch modes
    sync_cout << "info string [Consistency Check (Single vs Batch)]"
              << sync_endl;
    int matches = 0;
    for (int i = 0; i < num_positions; i++) {
      if (gpu_single_results[i] == gpu_batch_results[i]) {
        matches++;
      }
    }
    double match_pct = (matches * 100.0) / num_positions;
    sync_cout << "info string   Exact matches: " << matches << "/"
              << num_positions << " (" << std::fixed << std::setprecision(1)
              << match_pct << "%)" << sync_endl;

    // Show sample results
    sync_cout << "" << sync_endl;
    sync_cout << "info string [Sample GPU NNUE Results (first 5 positions)]"
              << sync_endl;
    for (int i = 0; i < std::min(5, num_positions); i++) {
      int32_t psqt = 0, pos = 0;
      auto [p, s] = gpu_manager.evaluate_single(*positions[i], true);
      psqt = p;
      pos = s;
      sync_cout << "info string   Pos " << (i + 1) << ": PSQT=" << psqt
                << " Positional=" << pos << " Total=" << (psqt + pos)
                << " (Simple=" << cpu_simple_results[i] << ")" << sync_endl;
    }
  } else {
    sync_cout << "info string GPU NNUE not available - skipping GPU benchmark"
              << sync_endl;
  }

  sync_cout << "" << sync_endl;
  sync_cout << "info string =================================================="
            << sync_endl;
  sync_cout << "info string NNUE Benchmark Complete!" << sync_endl;
  sync_cout << "info string =================================================="
            << sync_endl;
}

} // namespace MetalFish
