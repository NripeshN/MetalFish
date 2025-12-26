/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/tt.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace MetalFish {

TranspositionTable TT;

// Save data to a TT entry
void TTEntry::save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev,
                   uint8_t generation) {
  // Preserve any existing move if we don't have a new one
  if (m || uint16_t(k) != key16)
    move16 = uint16_t(m.raw());

  // Overwrite less valuable entries
  if (b == BOUND_EXACT || uint16_t(k) != key16 ||
      d - DEPTH_ENTRY_OFFSET + 2 * pv > depth8 - 4) {
    key16 = uint16_t(k);
    depth8 = int8_t(d - DEPTH_ENTRY_OFFSET);
    genBound8 = uint8_t(generation | (pv << 2) | b);
    value16 = int16_t(v);
    eval16 = int16_t(ev);
  }
}

TranspositionTable::~TranspositionTable() {
  if (table) {
#ifdef _WIN32
    _aligned_free(table);
#else
    std::free(table);
#endif
  }
}

// Resize the transposition table
void TranspositionTable::resize(size_t mbSize) {
  if (table) {
#ifdef _WIN32
    _aligned_free(table);
#else
    std::free(table);
#endif
    table = nullptr;
  }

  clusterCount = mbSize * 1024 * 1024 / sizeof(TTCluster);

  // Allocate aligned memory for cache efficiency
#ifdef _WIN32
  table = static_cast<TTCluster *>(
      _aligned_malloc(clusterCount * sizeof(TTCluster), 64));
#else
  void *mem = nullptr;
  if (posix_memalign(&mem, 64, clusterCount * sizeof(TTCluster)) == 0) {
    table = static_cast<TTCluster *>(mem);
  }
#endif

  if (!table) {
    clusterCount = 0;
    return;
  }

  clear();
}

// Clear the transposition table using multiple threads
void TranspositionTable::clear() {
  if (!table || clusterCount == 0)
    return;

  // Use multiple threads to clear faster
  unsigned threadCount = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> threads;

  for (unsigned i = 0; i < threadCount; ++i) {
    threads.emplace_back([this, i, threadCount]() {
      size_t start = i * clusterCount / threadCount;
      size_t end = (i + 1) * clusterCount / threadCount;
      std::memset(&table[start], 0, (end - start) * sizeof(TTCluster));
    });
  }

  for (auto &t : threads)
    t.join();

  generation8 = 0;
}

// Probe the transposition table
TTEntry *TranspositionTable::probe(Key key, bool &found) const {
  TTEntry *const tte = &table[key & (clusterCount - 1)].entry[0];
  const uint16_t key16 = uint16_t(key);

  for (int i = 0; i < ClusterSize; ++i) {
    if (tte[i].key16 == key16 || !tte[i].depth8) {
      // Refresh entry age
      tte[i].genBound8 = uint8_t(generation8 | (tte[i].genBound8 & 0x7));
      found = tte[i].depth8 != 0 && tte[i].key16 == key16;
      return &tte[i];
    }
  }

  // Find replacement entry (lowest depth/oldest)
  TTEntry *replace = tte;
  for (int i = 1; i < ClusterSize; ++i) {
    if (replace->depth8 - ((generation8 - replace->genBound8) & 0xF8) >
        tte[i].depth8 - ((generation8 - tte[i].genBound8) & 0xF8))
      replace = &tte[i];
  }

  found = false;
  return replace;
}

// Calculate how full the hash table is (per mille)
int TranspositionTable::hashfull() const {
  int count = 0;
  for (size_t i = 0; i < 1000; ++i) {
    for (int j = 0; j < ClusterSize; ++j) {
      if ((table[i].entry[j].genBound8 & 0xF8) == generation8)
        ++count;
    }
  }
  return count * 1000 / (1000 * ClusterSize);
}

} // namespace MetalFish
