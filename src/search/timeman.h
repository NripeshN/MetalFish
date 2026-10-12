/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef TIMEMAN_H_INCLUDED
#define TIMEMAN_H_INCLUDED

#include <limits>

#include "core/misc.h"

namespace MetalFish {

class OptionsMap;
enum Color : u8;

namespace Search {
struct LimitsType;
}

// The TimeManagement class computes the optimal time to think depending on
// the maximum available time, the game move number, and other parameters.
class TimeManagement {
public:
  void init(Search::LimitsType &limits, Color us, int ply,
            const OptionsMap &options, double &originalTimeAdjust);

  TimePoint optimum() const;
  TimePoint maximum() const;
  template <typename FUNC> TimePoint elapsed(FUNC nodes) const {
    return useNodesTime ? TimePoint(nodes()) : elapsed_time();
  }
  TimePoint elapsed_time() const { return now() - startTime; };

  void clear();
  void advance_nodes_time(i64 nodes);

private:
  static constexpr TimePoint NoBound =
      std::numeric_limits<TimePoint>::max() / 2;

  TimePoint startTime;
  TimePoint optimumTime = NoBound;
  TimePoint maximumTime = NoBound;

  i64 availableNodes = -1;   // When in 'nodes as time' mode
  bool useNodesTime = false; // True if we are in 'nodes as time' mode
};

} // namespace MetalFish

#endif // #ifndef TIMEMAN_H_INCLUDED
