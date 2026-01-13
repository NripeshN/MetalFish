/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include "core/misc.h"
#include "search/search.h"
#include "uci/engine.h"

namespace MetalFish {

class Position;
class Move;
class Score;
enum Square : int8_t;
using Value = int;

class UCIEngine {
public:
  UCIEngine(int argc, char **argv);

  void loop();

  static int to_cp(Value v, const Position &pos);
  static std::string format_score(const Score &s);
  static std::string square(Square s);
  static std::string move(Move m, bool chess960);
  static std::string wdl(Value v, const Position &pos);
  static std::string to_lower(std::string str);
  static Move to_move(const Position &pos, std::string str);

  static Search::LimitsType parse_limits(std::istream &is);

  auto &engine_options() { return engine.get_options(); }

private:
  Engine engine;
  CommandLine cli;

  static void print_info_string(std::string_view str);

  void go(std::istringstream &is);
  void bench(std::istream &args);
  void benchmark(std::istream &args);
  void position(std::istringstream &is);
  void setoption(std::istringstream &is);
  std::uint64_t perft(const Search::LimitsType &);
  void gpu_info();
  void gpu_benchmark();
  void mcts_go(std::istringstream &is); // Hybrid MCTS search

  static void on_update_no_moves(const Engine::InfoShort &info);
  static void on_update_full(const Engine::InfoFull &info, bool showWDL);
  static void on_iter(const Engine::InfoIter &info);
  static void on_bestmove(std::string_view bestmove, std::string_view ponder);

  void init_search_update_listeners();
};

} // namespace MetalFish

#endif // #ifndef UCI_H_INCLUDED
