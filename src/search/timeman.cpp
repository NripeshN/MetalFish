/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file timeman.cpp
 * @brief MetalFish source file.
 */

  Licensed under GPL-3.0
*/

#include "search/timeman.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "search/search.h"
#include "uci/ucioption.h"

namespace MetalFish {

TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }

void TimeManagement::clear() {
  availableNodes = -1; // When in 'nodes as time' mode
}

void TimeManagement::advance_nodes_time(std::int64_t nodes) {
  assert(useNodesTime);
  availableNodes = std::max(int64_t(0), availableNodes - nodes);
}

// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)
void TimeManagement::init(Search::LimitsType &limits, Color us, int ply,
                          const OptionsMap &options,
                          double &originalTimeAdjust) {
  TimePoint npmsec = TimePoint(options["nodestime"]);

  // If we have no time, we don't need to fully initialize TM.
  // startTime is used by movetime and useNodesTime is used in elapsed calls.
  startTime = limits.startTime;
  useNodesTime = npmsec != 0;

  if (limits.time[us] == 0)
    return;

  TimePoint moveOverhead = TimePoint(options["Move Overhead"]);

  // optScale is a percentage of available time to use for the current move.
  // maxScale is a multiplier applied to optimumTime.
  double optScale, maxScale;

  // If we have to play in 'nodes as time' mode, then convert from time
  // to nodes, and use resulting values in time management formulas.
  // WARNING: to avoid time losses, the given npmsec (nodes per millisecond)
  // must be much lower than the real engine speed.
  if (useNodesTime) {
    if (availableNodes == -1)                    // Only once at game start
      availableNodes = npmsec * limits.time[us]; // Time is in msec

    // Convert from milliseconds to nodes
    limits.time[us] = TimePoint(availableNodes);
    limits.inc[us] *= npmsec;
    limits.npmsec = npmsec;
    moveOverhead *= npmsec;
  }

  // These numbers are used where multiplications, divisions or comparisons
  // with constants are involved.
  const int64_t scaleFactor = useNodesTime ? npmsec : 1;
  const TimePoint scaledTime = limits.time[us] / scaleFactor;

  // Maximum move horizon
  int centiMTG =
      limits.movestogo ? std::min(limits.movestogo * 100, 5000) : 5051;

  // If less than one second, gradually reduce mtg
  if (scaledTime < 1000)
    centiMTG = int(scaledTime * 5.051);

  // Make sure timeLeft is > 0 since we may use it as a divisor
  TimePoint timeLeft = std::max(
      TimePoint(1), limits.time[us] + (limits.inc[us] * (centiMTG - 100) -
                                       moveOverhead * (200 + centiMTG)) /
                                          100);

  // x basetime (+ z increment)
  // If there is a healthy increment, timeLeft can exceed the actual available
  // game time for the current move, so also cap to a percentage of available
  // game time.
  if (limits.movestogo == 0) {
    // Extra time according to timeLeft
    if (originalTimeAdjust < 0)
      originalTimeAdjust = 0.3128 * std::log10(timeLeft) - 0.4354;

    // Calculate time constants based on current time left.
    double logTimeInSec = std::log10(scaledTime / 1000.0);
    double optConstant =
        std::min(0.0032116 + 0.000321123 * logTimeInSec, 0.00508017);
    double maxConstant = std::max(3.3977 + 3.03950 * logTimeInSec, 2.94761);

    optScale =
        std::min(0.0121431 + std::pow(ply + 2.94693, 0.461073) * optConstant,
                 0.213035 * limits.time[us] / timeLeft) *
        originalTimeAdjust;

    maxScale = std::min(6.67704, maxConstant + ply / 11.9847);
  }

  // x moves in y seconds (+ z increment)
  else {
    optScale = std::min((0.88 + ply / 116.4) / (centiMTG / 100.0),
                        0.88 * limits.time[us] / timeLeft);
    maxScale = 1.3 + 0.11 * (centiMTG / 100.0);
  }

  // Limit the maximum possible time for this move
  optimumTime = TimePoint(optScale * timeLeft);
  maximumTime = TimePoint(std::min(0.825179 * limits.time[us] - moveOverhead,
                                   maxScale * optimumTime)) -
                10;

  if (options["Ponder"])
    optimumTime += optimumTime / 4;
}

} // namespace MetalFish