/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

  Syzygy Tablebase Probing
  ========================

  Provides access to Syzygy endgame tablebases for perfect endgame play.
  Supports WDL (Win/Draw/Loss) and DTZ (Distance to Zeroing) probing.
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include <functional>
#include <string>
#include <vector>

namespace MetalFish {

// Forward declarations
namespace Search {
struct RootMove;
using RootMoves = std::vector<RootMove>;
} // namespace Search

namespace Tablebases {

// Tablebase configuration
struct Config {
  int cardinality = 0;    // Max number of pieces in available tablebases
  bool rootInTB = false;  // Is root position in tablebase?
  bool useRule50 = false; // Use 50-move rule in TB probing
  int probeDepth = 0;     // Minimum depth to probe TB
};

// WDL (Win/Draw/Loss) scores from tablebase
enum WDLScore {
  WDLLoss = -2,        // Loss
  WDLBlessedLoss = -1, // Loss, but draw under 50-move rule
  WDLDraw = 0,         // Draw
  WDLCursedWin = 1,    // Win, but draw under 50-move rule
  WDLWin = 2,          // Win
};

// Probe state after a probing operation
enum ProbeState {
  FAIL = 0,             // Probe failed (missing file/table)
  OK = 1,               // Probe successful
  CHANGE_STM = -1,      // DTZ should check the other side
  ZEROING_BEST_MOVE = 2 // Best move zeroes DTZ (capture or pawn move)
};

// Maximum cardinality of loaded tablebases
extern int MaxCardinality;

// Initialize tablebases from given paths
// Paths are separated by ":" on Unix and ";" on Windows
void init(const std::string &paths);

// Probe WDL table for a position
// Returns WDL score from the side to move's perspective
// Sets result to FAIL if probe failed
WDLScore probe_wdl(Position &pos, ProbeState *result);

// Probe DTZ table for a position
// Returns DTZ (distance to zeroing move) from side to move's perspective
// Positive = win, negative = loss, 0 = draw
// Sets result to FAIL if probe failed
int probe_dtz(Position &pos, ProbeState *result);

// Probe tablebases at root and rank root moves by DTZ
// Returns true if probe was successful
bool root_probe(Position &pos, Search::RootMoves &rootMoves, bool rule50,
                bool rankDTZ, const std::function<bool()> &time_abort);

// Probe WDL tablebases at root (fallback when DTZ unavailable)
// Returns true if probe was successful
bool root_probe_wdl(Position &pos, Search::RootMoves &rootMoves, bool rule50);

// Rank root moves using tablebase information
Config rank_root_moves(
    Position &pos, Search::RootMoves &rootMoves, bool rankDTZ = false,
    const std::function<bool()> &time_abort = []() { return false; });

} // namespace Tablebases

} // namespace MetalFish
