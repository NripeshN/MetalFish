/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstring>

#include "core/misc.h"
#include "core/types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::Eval::NNUE {

struct alignas(CacheLineSize) Accumulator;

class FeatureTransformer;

// Class that holds the result of affine transformation of input features,
// combined HalfKA + Threats
struct alignas(CacheLineSize) Accumulator {
  std::array<std::array<i16, L1>, COLOR_NB> accumulation;
  std::array<std::array<i32, PSQTBuckets>, COLOR_NB> psqtAccumulation;
  std::array<bool, COLOR_NB> computed = {};
};

// AccumulatorCaches struct provides per-thread accumulator caches, where each
// cache contains multiple entries for each of the possible king squares.
// When the accumulator needs to be refreshed, the cached entry is used to more
// efficiently update the accumulator, instead of rebuilding it from scratch.
// This idea, was first described by Luecx (author of Koivisto) and
// is commonly referred to as "Finny Tables".
struct AccumulatorCaches {
  template <typename Network> AccumulatorCaches(const Network &network) {
    clear(network);
  }

  struct alignas(CacheLineSize) Entry {
    std::array<BiasType, L1> accumulation;
    std::array<PSQTWeightType, PSQTBuckets> psqtAccumulation;
    std::array<Piece, SQUARE_NB> pieces;
    Bitboard pieceBB;

    // To initialize a refresh entry, we set all its bitboards empty,
    // so we put the biases in the accumulation, without any weights on top
    void clear(const std::array<BiasType, L1> &biases) {
      accumulation = biases;
      std::memset(reinterpret_cast<std::byte *>(this) +
                      offsetof(Entry, psqtAccumulation),
                  0, sizeof(Entry) - offsetof(Entry, psqtAccumulation));
    }
  };

  template <typename Network> void clear(const Network &network) {
    for (auto &entries1D : entries)
      for (auto &entry : entries1D)
        entry.clear(network.featureTransformer.biases);
  }

  std::array<Entry, COLOR_NB> &operator[](Square sq) { return entries[sq]; }

  std::array<std::array<Entry, COLOR_NB>, SQUARE_NB> entries;
};

struct AccumulatorState : public Accumulator, Dirties {};

class AccumulatorStack {
public:
  static constexpr usize MaxSize = MAX_PLY + 1;

  [[nodiscard]] const AccumulatorState &latest() const noexcept;

  void reset() noexcept;
  Dirties &push() noexcept;
  void pop() noexcept;

  void evaluate(const Position &pos,
                const FeatureTransformer &featureTransformer,
                // Silence spurious warning on GCC 10
                [[maybe_unused]] AccumulatorCaches &cache) noexcept;

private:
  [[nodiscard]] AccumulatorState &mut_latest() noexcept;

  void evaluate_side(Color perspective, const Position &pos,
                     const FeatureTransformer &featureTransformer,
                     // Silence spurious warning on GCC 10
                     [[maybe_unused]] AccumulatorCaches &cache,
                     usize last_usable_accum) noexcept;

  [[nodiscard]] usize
  find_last_usable_accumulator(Color perspective) const noexcept;

  void forward_update_incremental(Color perspective, const Position &pos,
                                  const FeatureTransformer &featureTransformer,
                                  const usize begin) noexcept;

  void backward_update_incremental(Color perspective, const Position &pos,
                                   const FeatureTransformer &featureTransformer,
                                   const usize end) noexcept;

  void forward_update_incremental_both(
      const Position &pos, const FeatureTransformer &featureTransformer,
      usize white_begin, usize black_begin) noexcept;

  std::array<AccumulatorState, MaxSize> accumulators;
  usize size = 1;
};

} // namespace MetalFish::Eval::NNUE

#endif // NNUE_ACCUMULATOR_H_INCLUDED
