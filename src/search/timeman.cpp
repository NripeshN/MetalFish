/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

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

void TimeManagement::clear() { availableNodes = -1; }

void TimeManagement::advance_nodes_time(std::int64_t nodes) {
  assert(useNodesTime);
  availableNodes = std::max(int64_t(0), availableNodes - nodes);
}

void TimeManagement::init(Search::LimitsType &limits, Color us, int ply,
                          const OptionsMap &options,
                          double &originalTimeAdjust) {
  TimePoint npmsec = TimePoint(options["nodestime"]);

  startTime = limits.startTime;
  useNodesTime = npmsec != 0;

  if (limits.time[us] == 0)
    return;

  TimePoint moveOverhead = TimePoint(options["Move Overhead"]);

  double optScale, maxScale;

  if (useNodesTime) {
    if (availableNodes == -1)
      availableNodes = npmsec * limits.time[us];

    limits.time[us] = TimePoint(availableNodes);
    limits.inc[us] *= npmsec;
    limits.npmsec = npmsec;
    moveOverhead *= npmsec;
  }

  const int64_t scaleFactor = useNodesTime ? npmsec : 1;
  const TimePoint scaledTime = limits.time[us] / scaleFactor;

  int centiMTG =
      limits.movestogo ? std::min(limits.movestogo * 100, 5000) : 5051;

  if (scaledTime < 1000)
    centiMTG = int(scaledTime * 5.051);

  TimePoint timeLeft = std::max(
      TimePoint(1), limits.time[us] + (limits.inc[us] * (centiMTG - 100) -
                                       moveOverhead * (200 + centiMTG)) /
                                          100);

  if (limits.movestogo == 0) {
    if (originalTimeAdjust < 0)
      originalTimeAdjust = 0.3128 * std::log10(timeLeft) - 0.4354;

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

  else {
    optScale = std::min((0.88 + ply / 116.4) / (centiMTG / 100.0),
                        0.88 * limits.time[us] / timeLeft);
    maxScale = 1.3 + 0.11 * (centiMTG / 100.0);
  }

  optimumTime = TimePoint(optScale * timeLeft);
  maximumTime = TimePoint(std::min(0.825179 * limits.time[us] - moveOverhead,
                                   maxScale * optimumTime)) -
                10;

  if (options["Ponder"])
    optimumTime += optimumTime / 4;
}

} // namespace MetalFish
