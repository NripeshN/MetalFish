/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  Syzygy Tablebase Probing Implementation
  =======================================

  This is a simplified implementation of Syzygy tablebase probing.
  The full implementation requires memory-mapped file access and
  complex indexing algorithms. This version provides the interface
  and basic structure for future complete implementation.
*/

#include "syzygy/tbprobe.h"
#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "search/search.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace MetalFish {
namespace Tablebases {

int MaxCardinality = 0;

namespace {

// Paths to tablebase files
std::string TBPaths;

// Mutex for thread-safe initialization
std::mutex tbMutex;

// Whether tablebases have been initialized
bool tbInitialized = false;

// WDL to Value conversion
constexpr Value WDL_to_value[] = {
    -VALUE_MATE + MAX_PLY + 1, // WDLLoss
    VALUE_DRAW - 2,            // WDLBlessedLoss
    VALUE_DRAW,                // WDLDraw
    VALUE_DRAW + 2,            // WDLCursedWin
    VALUE_MATE - MAX_PLY - 1   // WDLWin
};

// Check if a file exists (for future use)
[[maybe_unused]] bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

// Search for tablebase files in the given paths
void scan_for_tablebases(const std::string &paths) {
  if (paths.empty())
    return;

#ifndef _WIN32
  constexpr char SepChar = ':';
#else
  constexpr char SepChar = ';';
#endif

  std::stringstream ss(paths);
  std::string path;

  while (std::getline(ss, path, SepChar)) {
    // Check for WDL files (*.rtbw)
    // In a full implementation, we would scan the directory
    // and load all available tablebase files

    // For now, just record that we have a valid path
    if (!path.empty()) {
      // Check if path exists
      struct stat st;
      if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        // Valid directory - in full implementation, scan for .rtbw files
        // For now, we just note that tablebases might be available
      }
    }
  }
}

// DTZ before zeroing move based on WDL score
int dtz_before_zeroing(WDLScore wdl) {
  return wdl == WDLWin         ? 1
         : wdl == WDLCursedWin ? 101
         : wdl == WDLBlessedLoss ? -101
         : wdl == WDLLoss        ? -1
                                 : 0;
}

// Sign of a value
template <typename T> int sign_of(T val) {
  return (T(0) < val) - (val < T(0));
}

} // anonymous namespace

void init(const std::string &paths) {
  std::lock_guard<std::mutex> lock(tbMutex);

  TBPaths = paths;
  MaxCardinality = 0;
  tbInitialized = false;

  if (paths.empty()) {
    std::cout << "info string No Syzygy path specified" << std::endl;
    return;
  }

  scan_for_tablebases(paths);

  // In a full implementation, we would:
  // 1. Scan directories for .rtbw and .rtbz files
  // 2. Parse file names to determine piece configurations
  // 3. Build hash tables for quick lookup
  // 4. Set MaxCardinality based on available files

  // For now, indicate that TB support is available but no files loaded
  std::cout << "info string Syzygy path: " << paths << std::endl;
  std::cout << "info string Syzygy tablebase support enabled (files not loaded)"
            << std::endl;

  tbInitialized = true;
}

WDLScore probe_wdl(Position &pos, ProbeState *result) {
  *result = FAIL;

  // Cannot probe if:
  // 1. Tablebases not initialized
  // 2. Too many pieces on the board
  // 3. Position has castling rights
  // 4. Position has en passant (complex to handle)

  if (!tbInitialized || MaxCardinality == 0) {
    return WDLDraw;
  }

  int pieceCount = popcount(pos.pieces());
  if (pieceCount > MaxCardinality) {
    return WDLDraw;
  }

  if (pos.can_castle(ANY_CASTLING)) {
    return WDLDraw;
  }

  // In a full implementation, we would:
  // 1. Compute the material key
  // 2. Look up the appropriate tablebase
  // 3. Compute the index into the tablebase
  // 4. Decompress and return the WDL value

  // For now, return FAIL to indicate probe not available
  return WDLDraw;
}

int probe_dtz(Position &pos, ProbeState *result) {
  *result = FAIL;

  if (!tbInitialized || MaxCardinality == 0) {
    return 0;
  }

  int pieceCount = popcount(pos.pieces());
  if (pieceCount > MaxCardinality) {
    return 0;
  }

  if (pos.can_castle(ANY_CASTLING)) {
    return 0;
  }

  // First probe WDL to determine if position is won/lost/drawn
  WDLScore wdl = probe_wdl(pos, result);
  if (*result == FAIL || wdl == WDLDraw) {
    return 0;
  }

  // In a full implementation, we would:
  // 1. Look up the DTZ tablebase
  // 2. Compute the index
  // 3. Return the DTZ value

  // For now, return a default value based on WDL
  *result = OK;
  return dtz_before_zeroing(wdl);
}

bool root_probe(Position &pos, Search::RootMoves &rootMoves, bool /*rule50*/,
                bool /*rankDTZ*/, const std::function<bool()> &time_abort) {
  if (!tbInitialized || MaxCardinality == 0 || rootMoves.empty()) {
    return false;
  }

  int pieceCount = popcount(pos.pieces());
  if (pieceCount > MaxCardinality || pos.can_castle(ANY_CASTLING)) {
    return false;
  }

  ProbeState result;
  StateInfo st;

  // Probe each root move
  for (auto &m : rootMoves) {
    if (time_abort())
      return false;

    pos.do_move(m.pv[0], st);

    // Get DTZ for position after move
    int dtz;
    if (pos.rule50_count() == 0) {
      // Zeroing move - probe WDL
      WDLScore wdl = probe_wdl(pos, &result);
      dtz = dtz_before_zeroing(WDLScore(-int(wdl)));
    } else {
      dtz = -probe_dtz(pos, &result);
      if (dtz > 0)
        dtz++;
      else if (dtz < 0)
        dtz--;
    }

    pos.undo_move(m.pv[0]);

    if (result == FAIL)
      return false;

    // Rank moves by DTZ
    m.tbRank = dtz > 0   ? (1000 - dtz)
               : dtz < 0 ? (-1000 - dtz)
                         : 0;
    m.tbScore = WDL_to_value[2 + sign_of(dtz)]; // Convert to approximate score
  }

  // Sort moves by TB rank
  std::stable_sort(rootMoves.begin(), rootMoves.end(),
                   [](const Search::RootMove &a, const Search::RootMove &b) {
                     return a.tbRank > b.tbRank;
                   });

  return true;
}

bool root_probe_wdl(Position &pos, Search::RootMoves &rootMoves, bool /*rule50*/) {
  if (!tbInitialized || MaxCardinality == 0 || rootMoves.empty()) {
    return false;
  }

  int pieceCount = popcount(pos.pieces());
  if (pieceCount > MaxCardinality || pos.can_castle(ANY_CASTLING)) {
    return false;
  }

  ProbeState result;
  StateInfo st;

  // Probe each root move
  for (auto &m : rootMoves) {
    pos.do_move(m.pv[0], st);

    WDLScore wdl = probe_wdl(pos, &result);
    wdl = WDLScore(-int(wdl)); // Negate for opponent's perspective

    pos.undo_move(m.pv[0]);

    if (result == FAIL)
      return false;

    // Set TB rank and score
    m.tbRank = int(wdl) * 1000;
    m.tbScore = WDL_to_value[wdl + 2];
  }

  // Sort moves by TB rank
  std::stable_sort(rootMoves.begin(), rootMoves.end(),
                   [](const Search::RootMove &a, const Search::RootMove &b) {
                     return a.tbRank > b.tbRank;
                   });

  return true;
}

Config rank_root_moves(Position &pos, Search::RootMoves &rootMoves,
                       bool rankDTZ, const std::function<bool()> &time_abort) {
  Config config;

  if (rootMoves.empty())
    return config;

  config.rootInTB = false;
  config.useRule50 = true; // Default to using 50-move rule
  config.probeDepth = 0;
  config.cardinality = MaxCardinality;

  if (MaxCardinality == 0)
    return config;

  int pieceCount = popcount(pos.pieces());
  if (pieceCount > MaxCardinality || pos.can_castle(ANY_CASTLING))
    return config;

  // Try DTZ probe first
  config.rootInTB = root_probe(pos, rootMoves, config.useRule50, rankDTZ, time_abort);

  if (!config.rootInTB && !time_abort()) {
    // Fall back to WDL probe
    config.rootInTB = root_probe_wdl(pos, rootMoves, config.useRule50);
  }

  if (config.rootInTB) {
    // Disable further TB probing during search if we have DTZ
    config.cardinality = 0;
  } else {
    // Clean up if probes failed
    for (auto &m : rootMoves)
      m.tbRank = 0;
  }

  return config;
}

} // namespace Tablebases
} // namespace MetalFish

