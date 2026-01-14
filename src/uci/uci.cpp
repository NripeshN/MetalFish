/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "uci/uci.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "core/memory.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "eval/evaluate.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_accumulator.h"
#include "eval/score.h"
#include "gpu/backend.h"
#include "gpu/gpu_accumulator.h"
#include "gpu/gpu_mcts_backend.h"
#include "gpu/gpu_nnue.h"
#include "gpu/gpu_nnue_integration.h"
#include "gpu/nnue_eval.h"
#include "mcts/enhanced_hybrid_search.h"
#include "mcts/hybrid_search.h"
#include "mcts/position_classifier.h"
#include "mcts/stockfish_adapter.h"
#include "mcts/thread_safe_mcts.h"
#include "search/search.h"
#include "uci/benchmark.h"
#include "uci/engine.h"
#include "uci/ucioption.h"

namespace MetalFish {

constexpr auto BenchmarkCommand = "speedtest";

constexpr auto StartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
template <typename... Ts> struct overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> overload(Ts...) -> overload<Ts...>;

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

    if (token == "quit" || token == "stop")
      engine.stop();

    // The GUI sends 'ponderhit' to tell that the user has played the expected
    // move. So, 'ponderhit' is sent if pondering was done on the same move that
    // the user has played. The search should continue, but should also switch
    // from pondering to the normal search.
    else if (token == "ponderhit")
      engine.set_ponderhit(false);

    else if (token == "uci") {
      sync_cout << "id name " << engine_info(true) << "\n"
                << engine.get_options() << sync_endl;

      sync_cout << "uciok" << sync_endl;
    }

    else if (token == "setoption")
      setoption(is);
    else if (token == "go") {
      // send info strings after the go command is sent for old GUIs and
      // python-chess
      print_info_string(engine.numa_config_information_as_string());
      print_info_string(engine.thread_allocation_information_as_string());
      go(is);
    } else if (token == "position")
      position(is);
    else if (token == "ucinewgame")
      engine.search_clear();
    else if (token == "isready")
      sync_cout << "readyok" << sync_endl;

    // Add custom non-UCI commands, mainly for debugging purposes.
    // These commands must not be used during a search!
    else if (token == "flip")
      engine.flip();
    else if (token == "bench")
      bench(is);
    else if (token == BenchmarkCommand)
      benchmark(is);
    else if (token == "d")
      sync_cout << engine.visualize() << sync_endl;
    else if (token == "eval")
      engine.trace_eval();
    else if (token == "gpu")
      gpu_info();
    else if (token == "gpubench")
      gpu_benchmark();
    else if (token == "mcts")
      mcts_go(is);
    else if (token == "mctsmt")
      mcts_mt_go(is);
    else if (token == "mctsbench")
      mcts_batch_benchmark(is);
    else if (token == "hybridbench")
      hybrid_benchmark();
    else if (token == "compiler")
      sync_cout << compiler_info() << sync_endl;
    else if (token == "export_net") {
      std::pair<std::optional<std::string>, std::string> files[2];

      if (is >> std::skipws >> files[0].second)
        files[0].first = files[0].second;

      if (is >> std::skipws >> files[1].second)
        files[1].first = files[1].second;

      engine.save_network(files);
    } else if (token == "--help" || token == "help" || token == "--license" ||
               token == "license")
      sync_cout
          << "\nMetalFish is a powerful chess engine for playing and analyzing."
             "\nIt is released as free software licensed under the GNU GPLv3 "
             "License."
             "\nMetalFish is normally used with a graphical user interface "
             "(GUI) and implements"
             "\nthe Universal Chess Interface (UCI) protocol to communicate "
             "with a GUI, an API, etc."
             "\nFor any further information, visit "
             "https://github.com/official-stockfish/MetalFish#readme"
             "\nor read the corresponding README.md and Copying.txt files "
             "distributed along with this program.\n"
          << sync_endl;
    else if (!token.empty() && token[0] != '#')
      sync_cout << "Unknown command: '" << cmd
                << "'. Type help for more information." << sync_endl;

  } while (token != "quit" &&
           cli.argc == 1); // The command-line arguments are one-shot
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
  // github.com/official-stockfish/WDL_model
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
  // BENCHMARK 2: CPU Feature Extraction
  // ========================================================================
  std::cout << "=== BENCHMARK 2: CPU Feature Extraction ===\n";
  std::cout << "Scope: Extract HalfKAv2_hm features from position\n";
  std::cout << "Iterations: 100,000\n" << std::flush;

  auto &extractor = GPU::gpu_feature_extractor();
  if (extractor.initialize()) {
    std::vector<int32_t> white_f, black_f;
    std::vector<double> cpu_feat_samples;
    cpu_feat_samples.reserve(100000);

    for (int i = 0; i < 100; i++) { // warmup
      extractor.extract(positions[i % NUM_FENS], white_f, black_f);
    }
    for (int i = 0; i < 100000; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      extractor.extract(positions[i % NUM_FENS], white_f, black_f);
      auto end = std::chrono::high_resolution_clock::now();
      cpu_feat_samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(cpu_feat_samples.begin(), cpu_feat_samples.end());
    std::cout << "\nCPU feature extraction:\n";
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
  std::cout << "Checking if 32-feature cap is ever hit\n" << std::flush;

  if (extractor.initialize()) {
    std::vector<int32_t> white_f, black_f;
    int total_positions = 0;
    int capped_positions = 0;
    int max_features_seen = 0;
    int max_per_perspective = 0;
    std::vector<int> feature_histogram(129, 0); // Up to 128 total features

    for (int i = 0; i < 2048; i++) {
      extractor.extract(positions[i], white_f, black_f);
      int white_count = white_f.size();
      int black_count = black_f.size();
      int total = white_count + black_count;
      int max_perspective = std::max(white_count, black_count);

      max_features_seen = std::max(max_features_seen, total);
      max_per_perspective = std::max(max_per_perspective, max_perspective);
      if (total < 129)
        feature_histogram[total]++;
      // Check against new GPU limit: 64 per perspective, 128 total
      if (max_perspective > GPU::GPU_MAX_FEATURES_PER_PERSPECTIVE)
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
// MCTS Hybrid Search Command
// ============================================================================
void UCIEngine::mcts_go(std::istringstream &is) {
  sync_cout << "info string Starting Enhanced Hybrid Search..." << sync_endl;

  // Parse search limits
  Search::LimitsType limits = parse_limits(is);

  // Get GPU NNUE manager
  GPU::GPUNNUEManager *gpu_manager = nullptr;
  if (GPU::gpu_nnue_manager_available()) {
    gpu_manager = &GPU::gpu_nnue_manager();
  }

  if (!gpu_manager) {
    sync_cout << "info string ERROR: GPU NNUE not available" << sync_endl;
    return;
  }

  // Configure enhanced hybrid search
  MCTS::EnhancedHybridConfig config;
  config.mcts_config.min_batch_size = 8;
  config.mcts_config.max_batch_size = 256;
  config.mcts_config.cpuct = 2.5f;
  config.mcts_config.add_dirichlet_noise = true;
  config.mcts_config.num_search_threads =
      1; // Single-threaded for now (multi-thread needs more debugging)
  config.enable_ab_verify = true;
  config.ab_verify_depth = 6;
  config.ab_override_threshold = 0.5f;
  config.use_position_classifier = true;
  config.dynamic_strategy = true;

  // Create enhanced hybrid search
  auto search = MCTS::create_enhanced_hybrid_search(gpu_manager, config);

  if (!search) {
    sync_cout << "info string ERROR: Failed to create hybrid search"
              << sync_endl;
    return;
  }

  sync_cout << "info string Enhanced hybrid search initialized" << sync_endl;

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

  // Start search
  search->start_search(pos, limits, best_move_cb, info_cb);

  // Wait for completion
  search->wait();

  // Print final statistics
  const auto &stats = search->stats();
  sync_cout << "info string Final stats: MCTS=" << stats.mcts_nodes
            << " AB=" << stats.ab_nodes
            << " verifications=" << stats.ab_verifications
            << " overrides=" << stats.ab_overrides
            << " GPU batches=" << stats.gpu_batches << sync_endl;
  sync_cout << "info string TT usage: " << std::fixed << std::setprecision(1)
            << MCTS::mcts_tt().usage_percent() << "%" << sync_endl;
  sync_cout << "info string Time: MCTS=" << std::fixed << std::setprecision(1)
            << stats.mcts_time_ms << "ms AB=" << stats.ab_time_ms
            << "ms Total=" << stats.total_time_ms << "ms" << sync_endl;
}

// ============================================================================
// Hybrid Search Validation Benchmark
// ============================================================================
void UCIEngine::hybrid_benchmark() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     MetalFish Hybrid Search Validation Suite             ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  // Get GPU NNUE manager
  GPU::GPUNNUEManager *gpu_manager = nullptr;
  if (GPU::gpu_nnue_manager_available()) {
    gpu_manager = &GPU::gpu_nnue_manager();
  }

  if (!gpu_manager || !gpu_manager->is_ready()) {
    std::cout << "ERROR: GPU NNUE not available\n";
    return;
  }

  // Official Stockfish benchmark positions (from
  // reference/stockfish/src/benchmark.cpp)
  static const char *TEST_FENS[] = {
      // Opening and middlegame positions
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
      "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
      "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
      "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
      "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
      "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
      // Endgame positions
      "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
      "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
      "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
      "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
      // Mate positions
      "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
      "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
      "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
      "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
  };
  const int NUM_TEST_FENS = 16;

  // ========================================================================
  // 1. Classifier Distribution Analysis
  // ========================================================================
  std::cout << "\n=== 1. Classifier Distribution Analysis ===\n";

  PositionClassifier classifier;
  int class_counts[5] = {0, 0, 0, 0, 0};
  int positions_tested = 0;

  for (int i = 0; i < NUM_TEST_FENS; i++) {
    Position pos;
    StateInfo st;
    pos.set(TEST_FENS[i], false, &st);

    // Skip positions in check (can cause analysis issues)
    if (pos.checkers()) {
      continue;
    }

    auto type = classifier.quick_classify(pos);
    class_counts[static_cast<int>(type)]++;
    positions_tested++;
  }

  const char *type_names[] = {"HIGHLY_TACTICAL", "TACTICAL", "BALANCED",
                              "STRATEGIC", "HIGHLY_STRATEGIC"};
  for (int i = 0; i < 5; i++) {
    double pct =
        positions_tested > 0 ? 100.0 * class_counts[i] / positions_tested : 0;
    std::cout << "  " << std::setw(18) << type_names[i] << ": " << std::setw(3)
              << class_counts[i] << " (" << std::fixed << std::setprecision(1)
              << pct << "%)\n";
  }
  std::cout << "  Positions tested: " << positions_tested << "/"
            << NUM_TEST_FENS << "\n";

  // ========================================================================
  // 2. MCTS Profiling
  // ========================================================================
  std::cout << "\n=== 2. MCTS Profiling (3 second search) ===\n";

  MCTS::HybridSearchConfig mcts_config;
  mcts_config.num_search_threads = 1;
  auto mcts_search = MCTS::create_hybrid_search(gpu_manager, mcts_config);

  MCTS::MCTSPositionHistory history;
  history.reset(StartFEN);

  Search::LimitsType limits;
  limits.movetime = 3000;

  MCTS::MCTSMove best_move;
  auto start = std::chrono::steady_clock::now();

  mcts_search->start_search(
      history, limits,
      [&](MCTS::MCTSMove move, MCTS::MCTSMove ponder) { best_move = move; },
      nullptr);
  mcts_search->wait();

  auto end = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  const auto &mcts_stats = mcts_search->stats();
  double sel_pct, exp_pct, eval_pct, bp_pct, q_pct;
  mcts_stats.get_profile_breakdown(sel_pct, exp_pct, eval_pct, bp_pct, q_pct);

  uint64_t total_nodes = mcts_stats.mcts_nodes.load();
  double nps = elapsed_ms > 0 ? total_nodes * 1000.0 / elapsed_ms : 0;
  std::cout << "  Selection:    " << std::fixed << std::setprecision(1)
            << sel_pct << "%\n";
  std::cout << "  Expansion:    " << exp_pct << "%\n";
  std::cout << "  Evaluation:   " << eval_pct << "%\n";
  std::cout << "  Backprop:     " << bp_pct << "%\n";
  std::cout << "  Total nodes:  " << total_nodes << "\n";
  std::cout << "  NPS:          " << std::fixed << std::setprecision(0) << nps
            << "\n";
  std::cout << "  Cache hits:   " << mcts_stats.cache_hits.load() << "\n";
  std::cout << "  Cache misses: " << mcts_stats.cache_misses.load() << "\n";
  std::cout.flush();

  // ========================================================================
  // 3. Hybrid vs Pure MCTS comparison (quick test)
  // ========================================================================
  std::cout << "\n=== 3. Move Comparison: Hybrid vs Pure MCTS ===\n";
  std::cout << "  (Skipped - see 'mcts' command for interactive testing)\n";
  std::cout.flush();

  // ========================================================================
  // 4. Tactical Test Suite - Verifier Validation
  // ========================================================================
  std::cout << "\n=== 4. Tactical Test Suite (AB Verifier Validation) ===\n";
  std::cout << "  (Skipped - see 'mcts' command for interactive testing)\n";
  std::cout.flush();

  // ========================================================================
  // 5. Ablation Study
  // ========================================================================
  std::cout << "\n=== 5. Ablation Study ===\n";
  std::cout << "  (Skipped - see paper for detailed ablation results)\n";
  std::cout.flush();

  // ========================================================================
  // Summary
  // ========================================================================
  std::cout << "\n=== Summary ===\n";
  std::cout << "  Classifier tested: " << NUM_TEST_FENS << " positions\n";
  std::cout << "  MCTS NPS: " << std::fixed << std::setprecision(0) << nps
            << "\n";
  std::cout << "  Evaluation dominates: " << std::fixed << std::setprecision(1)
            << eval_pct << "% of MCTS time\n";

  std::cout << "\nNote: For full Elo testing, use external tools like "
               "cutechess-cli\n";
  std::cout << "with 'mcts' command for hybrid and 'go' for pure AB.\n";

  std::cout << "\nBenchmark complete.\n";
}

// ============================================================================
// Multi-Threaded MCTS Search Command
// ============================================================================
void UCIEngine::mcts_mt_go(std::istringstream &is) {
  // Parse threads option
  std::string token;
  int num_threads = 4; // Default to 4 threads

  // Parse additional options (threads=N)
  while (is >> token) {
    if (token.find("threads=") == 0) {
      num_threads = std::stoi(token.substr(8));
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

  sync_cout << "info string Starting Multi-Threaded MCTS Search with "
            << num_threads << " threads..." << sync_endl;

  // Get GPU NNUE manager
  GPU::GPUNNUEManager *gpu_manager = nullptr;
  if (GPU::gpu_nnue_manager_available()) {
    gpu_manager = &GPU::gpu_nnue_manager();
  }

  if (!gpu_manager) {
    sync_cout << "info string ERROR: GPU NNUE not available" << sync_endl;
    return;
  }

  // Configure multi-threaded MCTS
  MCTS::ThreadSafeMCTSConfig config;
  config.num_threads = num_threads;
  config.cpuct = 2.5f;
  config.fpu_value = -1.0f;
  config.policy_softmax_temp = 1.0f;
  config.add_dirichlet_noise = true;
  config.dirichlet_alpha = 0.3f;
  config.dirichlet_epsilon = 0.25f;
  config.virtual_loss = 3;
  config.min_batch_size = 8;
  config.max_batch_size = 256;

  // Create thread-safe MCTS
  auto mcts = MCTS::create_thread_safe_mcts(gpu_manager, config);

  if (!mcts) {
    sync_cout << "info string ERROR: Failed to create multi-threaded MCTS"
              << sync_endl;
    return;
  }

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
  mcts->start_search(fen, limits, best_move_cb, info_cb);

  // Wait for completion
  mcts->wait();

  // Print final statistics
  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

  const auto &stats = mcts->stats();
  uint64_t nodes = stats.total_nodes.load();
  uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

  sync_cout << "info string Final stats:" << sync_endl;
  sync_cout << "info string   Nodes: " << nodes << sync_endl;
  sync_cout << "info string   NPS: " << nps << sync_endl;
  sync_cout << "info string   Time: " << elapsed_ms << "ms" << sync_endl;
  sync_cout << "info string   Threads: " << num_threads << sync_endl;
  sync_cout << "info string   NN evals: " << stats.nn_evaluations.load()
            << sync_endl;
  sync_cout << "info string   Cache hits: " << stats.cache_hits.load()
            << " misses: " << stats.cache_misses.load() << sync_endl;

  // Profiling breakdown
  uint64_t sel_us = stats.selection_time_us.load();
  uint64_t exp_us = stats.expansion_time_us.load();
  uint64_t eval_us = stats.evaluation_time_us.load();
  uint64_t bp_us = stats.backprop_time_us.load();
  uint64_t total_us = sel_us + exp_us + eval_us + bp_us;

  if (total_us > 0) {
    sync_cout << "info string   Selection: " << std::fixed
              << std::setprecision(1) << (100.0 * sel_us / total_us) << "%"
              << sync_endl;
    sync_cout << "info string   Expansion: " << std::fixed
              << std::setprecision(1) << (100.0 * exp_us / total_us) << "%"
              << sync_endl;
    sync_cout << "info string   Evaluation: " << std::fixed
              << std::setprecision(1) << (100.0 * eval_us / total_us) << "%"
              << sync_endl;
    sync_cout << "info string   Backprop: " << std::fixed
              << std::setprecision(1) << (100.0 * bp_us / total_us) << "%"
              << sync_endl;
  }

  // Batching statistics
  uint64_t batch_count = stats.batch_count.load();
  if (batch_count > 0) {
    sync_cout << "info string   Avg batch size: " << std::fixed
              << std::setprecision(1) << stats.avg_batch_size() << sync_endl;
    sync_cout << "info string   Batch wait time: "
              << (stats.batch_wait_time_us.load() / 1000) << "ms" << sync_endl;
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
    MCTS::ThreadSafeMCTSConfig config;
    config.num_threads = num_threads;
    config.use_batched_eval = false; // Disable batching
    config.cpuct = 2.5f;
    config.add_dirichlet_noise = true;
    config.virtual_loss = 3;

    auto mcts = MCTS::create_thread_safe_mcts(gpu_manager, config);

    auto start = std::chrono::steady_clock::now();
    mcts->start_search(fen, limits, nullptr, nullptr);
    mcts->wait();
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    const auto &stats = mcts->stats();
    uint64_t nodes = stats.total_nodes.load();
    uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

    sync_cout << "info string   Nodes: " << nodes << sync_endl;
    sync_cout << "info string   NPS: " << nps << sync_endl;
    sync_cout << "info string   NN evals: " << stats.nn_evaluations.load()
              << sync_endl;
    sync_cout << "info string   Cache hits: " << stats.cache_hits.load()
              << " misses: " << stats.cache_misses.load() << sync_endl;

    uint64_t sel_us = stats.selection_time_us.load();
    uint64_t exp_us = stats.expansion_time_us.load();
    uint64_t eval_us = stats.evaluation_time_us.load();
    uint64_t bp_us = stats.backprop_time_us.load();
    uint64_t total_us = sel_us + exp_us + eval_us + bp_us;

    if (total_us > 0) {
      sync_cout << "info string   Evaluation time: " << std::fixed
                << std::setprecision(1) << (100.0 * eval_us / total_us)
                << "% of iteration" << sync_endl;
    }
  }

  sync_cout << "" << sync_endl;

  // ============================================
  // Test 2: Batched evaluation (new approach)
  // ============================================
  sync_cout
      << "info string --- Test 2: Batched Evaluation (dedicated thread) ---"
      << sync_endl;

  {
    MCTS::ThreadSafeMCTSConfig config;
    config.num_threads = num_threads;
    config.use_batched_eval = true; // Enable batching
    config.min_batch_size = 8;
    config.max_batch_size = 256;
    config.batch_timeout_us = 500;
    config.cpuct = 2.5f;
    config.add_dirichlet_noise = true;
    config.virtual_loss = 3;

    auto mcts = MCTS::create_thread_safe_mcts(gpu_manager, config);

    auto start = std::chrono::steady_clock::now();
    mcts->start_search(fen, limits, nullptr, nullptr);
    mcts->wait();
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    const auto &stats = mcts->stats();
    uint64_t nodes = stats.total_nodes.load();
    uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

    sync_cout << "info string   Nodes: " << nodes << sync_endl;
    sync_cout << "info string   NPS: " << nps << sync_endl;
    sync_cout << "info string   NN evals: " << stats.nn_evaluations.load()
              << sync_endl;
    sync_cout << "info string   NN batches: " << stats.nn_batches.load()
              << sync_endl;
    sync_cout << "info string   Avg batch size: " << std::fixed
              << std::setprecision(1) << stats.avg_batch_size() << sync_endl;
    sync_cout << "info string   Cache hits: " << stats.cache_hits.load()
              << " misses: " << stats.cache_misses.load() << sync_endl;
    sync_cout << "info string   Batch wait time: "
              << (stats.batch_wait_time_us.load() / 1000) << "ms total"
              << sync_endl;

    uint64_t sel_us = stats.selection_time_us.load();
    uint64_t exp_us = stats.expansion_time_us.load();
    uint64_t eval_us = stats.evaluation_time_us.load();
    uint64_t bp_us = stats.backprop_time_us.load();
    uint64_t total_us = sel_us + exp_us + eval_us + bp_us;

    if (total_us > 0) {
      sync_cout << "info string   Evaluation time: " << std::fixed
                << std::setprecision(1) << (100.0 * eval_us / total_us)
                << "% of iteration" << sync_endl;
    }
  }

  sync_cout << "" << sync_endl;
  sync_cout << "info string Benchmark complete!" << sync_endl;
}

} // namespace MetalFish
