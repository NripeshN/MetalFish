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

#elif defined(USE_NEON)
    constexpr IndexType NumChunks = InputDimensions / 16;

    static_assert(WeightScaleBits == 6);
    const auto in = reinterpret_cast<const int32x4_t *>(input);
    const auto out = reinterpret_cast<int8x16_t *>(output);
    for (IndexType i = 0; i < NumChunks; ++i) {
      // Pack 4x int32 -> int16 (saturate)
      int16x4_t lo0 = vqmovn_s32(in[i * 4 + 0]);
      int16x4_t hi0 = vqmovn_s32(in[i * 4 + 1]);
      int16x4_t lo1 = vqmovn_s32(in[i * 4 + 2]);
      int16x4_t hi1 = vqmovn_s32(in[i * 4 + 3]);

      // Square and shift: (a*a) >> 19 using vmull (widening multiply)
      // vmull_s16 gives int32x4, then shift right by 19 and narrow
      int32x4_t sq0 = vmull_s16(lo0, lo0);
      int32x4_t sq1 = vmull_s16(hi0, hi0);
      int32x4_t sq2 = vmull_s16(lo1, lo1);
      int32x4_t sq3 = vmull_s16(hi1, hi1);

      // Shift right by 19 then narrow to int16
      int16x4_t r0 = vmovn_s32(vshrq_n_s32(sq0, 19));
      int16x4_t r1 = vmovn_s32(vshrq_n_s32(sq1, 19));
      int16x4_t r2 = vmovn_s32(vshrq_n_s32(sq2, 19));
      int16x4_t r3 = vmovn_s32(vshrq_n_s32(sq3, 19));

      int16x8_t words0 = vcombine_s16(r0, r1);
      int16x8_t words1 = vcombine_s16(r2, r3);

      // Pack int16 -> uint8 (saturate)
      uint8x8_t bytes0 = vqmovun_s16(words0);
      uint8x8_t bytes1 = vqmovun_s16(words1);
      vst1q_u8(reinterpret_cast<uint8_t *>(&out[i]),
               vcombine_u8(bytes0, bytes1));
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
