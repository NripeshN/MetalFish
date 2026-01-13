/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include <iostream>
#include <memory>

#include "core/bitboard.h"
#include "core/misc.h"
#include "core/position.h"
#include "search/tune.h"
#include "uci/uci.h"

using namespace MetalFish;

int main(int argc, char *argv[]) {
  std::cout << engine_info() << std::endl;

  Bitboards::init();
  Position::init();

  auto uci = std::make_unique<UCIEngine>(argc, argv);

  Tune::init(uci->engine_options());

  uci->loop();

  return 0;
}
