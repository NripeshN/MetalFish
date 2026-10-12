/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include <tuple>

#include "core/memory.h"
#include "core/misc.h"
#include "core/types.h"

namespace MetalFish {

class ThreadPool;
struct TTEntry;
struct Cluster;

// There is only one global hash table for the engine and all its threads.
// For chess in particular, we even allow racy updates between threads to and
// from the TT, as taking the time to synchronize access would cost thinking
// time and thus Elo. As a hash table, collisions are possible and may cause
// chess playing issues (bizarre blunders, faulty mate reports, etc). Fixing
// these also loses Elo; however such risk decreases with larger TT size.
//
// We clearly separate TTData, a local copy of an entry, from TTWriter, which
// writes to the global table.

// A copy of the data already in an entry (possibly collided). Probes and reads
// are racy and non-atomic, possibly resulting in inconsistent data.
struct TTData {
  Move move;
  Value value, eval;
  Depth depth;
  Bound bound;
  bool is_pv;

  TTData() = delete;

  // clang-format off
    TTData(Move m, Value v, Value ev, Depth d, Bound b, bool pv) :
        move(m),
        value(v),
        eval(ev),
        depth(d),
        bound(b),
        is_pv(pv) {};
  // clang-format on
};

// This is used to make racy, non-atomic writes to the global TT. Writes are
// not "guaranteed": for chess reasons, we may decide the new data is less
// important than the old.
struct TTWriter {
public:
  void write(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev,
             u8 generation8);
  void penalize(int penalty); // decrement stored depth by the penalty

private:
  friend class TranspositionTable;
  TTEntry *entry;
  TTWriter(TTEntry *tte);
};

class TranspositionTable {

public:
  ~TranspositionTable() { aligned_large_pages_free(table); }

  // Set TT size in MiB
  void resize(usize mbSize, ThreadPool &threads);

  // Re-initialize memory, multithreaded
  void clear(ThreadPool &threads);

  bool is_allocated() const { return table != nullptr; }

  // Must be called at the beginning of each root search to track entry aging
  void new_search();

  // The current age, used when writing new data to the TT
  u8 generation() const;

  // Approximate what fraction of entries (permille) have been written to
  // during this root search.
  int hashfull(int maxAge = 0) const;

  // `probe(key)` is the primary method: given a board position, we lookup
  //  its entry in the table, and return a tuple of:
  //   1) whether the entry already had data on this position
  //   2) a copy of the prior data, if any (may be self-inconsistent due to
  //   races) 3) a writer object to the entry
  std::tuple<bool, TTData, TTWriter> probe(const Key key) const;

  // The hash function; its only external use is memory prefetching
  TTEntry *first_entry(const Key key) const;

private:
  friend struct TTEntry;

  usize clusterCount;
  Cluster *table = nullptr;

  u8 generation8 = 0;
};

} // namespace MetalFish

#endif // #ifndef TT_H_INCLUDED
