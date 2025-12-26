/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "core/types.h"
#include <atomic>
#include <cstdint>

namespace MetalFish {

// TTEntry struct stores information about a position in the transposition table
struct TTEntry {
  Move move() const { return Move(move16); }
  Value value() const { return Value(value16); }
  Value eval() const { return Value(eval16); }
  Depth depth() const { return Depth(depth8 + DEPTH_ENTRY_OFFSET); }
  bool is_pv() const { return bool(genBound8 & 0x4); }
  Bound bound() const { return Bound(genBound8 & 0x3); }
  void save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev,
            uint8_t generation);

private:
  friend class TranspositionTable;

  uint16_t key16;
  int16_t value16;
  int16_t eval16;
  uint8_t genBound8; // generation (5 bits) + PV (1 bit) + bound (2 bits)
  int8_t depth8;
  uint16_t move16;
};

// TTCluster: a cluster of entries (for better cache usage)
constexpr int ClusterSize = 3;

struct TTCluster {
  TTEntry entry[ClusterSize];
  char padding[2]; // Pad to 32 bytes
};

static_assert(sizeof(TTCluster) == 32, "TTCluster must be 32 bytes");

// TranspositionTable: the main hash table
class TranspositionTable {
public:
  TranspositionTable() = default;
  ~TranspositionTable();

  // Disable copy
  TranspositionTable(const TranspositionTable &) = delete;
  TranspositionTable &operator=(const TranspositionTable &) = delete;

  void resize(size_t mbSize);
  void clear();
  void new_search() { generation8 += 8; }

  TTEntry *probe(Key key, bool &found) const;
  int hashfull() const;

  uint8_t generation() const { return generation8; }

  // For GPU-accelerated access
  void *data() const { return table; }
  size_t size() const { return clusterCount * sizeof(TTCluster); }

private:
  TTCluster *table = nullptr;
  size_t clusterCount = 0;
  uint8_t generation8 = 0;
};

extern TranspositionTable TT;

} // namespace MetalFish
