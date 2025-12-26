#pragma once

#include "position.h"
#include "types.h"
#include <cstdint>

namespace MetalFish {

// Perft - Performance Test for move generation verification
// Returns the number of leaf nodes at the given depth
uint64_t perft(Position &pos, int depth, bool root = false);

// Divide - Like perft, but shows node counts for each root move
void divide(Position &pos, int depth);

} // namespace MetalFish
