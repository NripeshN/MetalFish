/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace MetalFish::Eval::NNUE::Layers {

// Clipped ReLU
template <IndexType InDims> class SqrClippedReLU {
public:
  // Input/output type
  using InputType = std::int32_t;
  using OutputType = std::uint8_t;

  // Number of input/output dimensions
  static constexpr IndexType InputDimensions = InDims;
  static constexpr IndexType OutputDimensions = InputDimensions;
  static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

  using OutputBuffer = OutputType[PaddedOutputDimensions];

  // Hash value embedded in the evaluation file
  static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
    std::uint32_t hashValue = 0x538D24C7u;
    hashValue += prevHash;
    return hashValue;
  }

  // Read network parameters
  bool read_parameters(std::istream &) { return true; }

  // Write network parameters
  bool write_parameters(std::ostream &) const { return true; }

  std::size_t get_content_hash() const {
    std::size_t h = 0;
    hash_combine(h, get_hash_value(0));
    return h;
  }

  // Forward propagation
  void propagate(const InputType *input, OutputType *output) const {

#if defined(USE_SSE2)
    constexpr IndexType NumChunks = InputDimensions / 16;

    static_assert(WeightScaleBits == 6);
    const auto in = reinterpret_cast<const __m128i *>(input);
    const auto out = reinterpret_cast<__m128i *>(output);
    for (IndexType i = 0; i < NumChunks; ++i) {
      __m128i words0 = _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]),
                                       _mm_load_si128(&in[i * 4 + 1]));
      __m128i words1 = _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]),
                                       _mm_load_si128(&in[i * 4 + 3]));

      // We shift by WeightScaleBits * 2 = 12 and divide by 128
      // which is an additional shift-right of 7, meaning 19 in total.
      // MulHi strips the lower 16 bits so we need to shift out 3 more to match.
      words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
      words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);

      _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
    }
    constexpr IndexType Start = NumChunks * 16;

#else
    constexpr IndexType Start = 0;
#endif

    for (IndexType i = Start; i < InputDimensions; ++i) {
      output[i] = static_cast<OutputType>(
          // Really should be /127 but we need to make it fast so we right-shift
          // by an extra 7 bits instead. Needs to be accounted for in the
          // trainer.
          std::min(127ll, ((long long)(input[i]) * input[i]) >>
                              (2 * WeightScaleBits + 7)));
    }
  }
};

} // namespace MetalFish::Eval::NNUE::Layers

#endif // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
