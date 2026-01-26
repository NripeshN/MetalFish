/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file nnue_common.h
 * @brief MetalFish source file.
 */

  Licensed under GPL-3.0
*/

// Constants used in NNUE evaluation function

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "core/misc.h"

#if defined(USE_AVX2)
#include <immintrin.h>

#elif defined(USE_SSE41)
#include <smmintrin.h>

#elif defined(USE_SSSE3)
#include <tmmintrin.h>

#elif defined(USE_SSE2)
#include <emmintrin.h>

#elif defined(USE_NEON)
#include <arm_neon.h>
#endif

namespace MetalFish::Eval::NNUE {

using BiasType = std::int16_t;
using ThreatWeightType = std::int8_t;
using WeightType = std::int16_t;
using PSQTWeightType = std::int32_t;
using IndexType = std::uint32_t;

// Version of the evaluation file
constexpr std::uint32_t Version = 0x7AF32F20u;

// Constant used in evaluation value calculation
constexpr int OutputScale = 16;
constexpr int WeightScaleBits = 6;

// Size of cache line (in bytes)
constexpr std::size_t CacheLineSize = 64;

constexpr const char Leb128MagicString[] = "COMPRESSED_LEB128";
constexpr const std::size_t Leb128MagicStringSize =
    sizeof(Leb128MagicString) - 1;

// SIMD width (in bytes)
#if defined(USE_AVX2)
constexpr std::size_t SimdWidth = 32;

#elif defined(USE_SSE2)
constexpr std::size_t SimdWidth = 16;

#elif defined(USE_NEON)
constexpr std::size_t SimdWidth = 16;
#endif

constexpr std::size_t MaxSimdWidth = 32;

// Type of input feature after conversion
using TransformedFeatureType = std::uint8_t;

// Round n up to be a multiple of base
template <typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
  return (n + base - 1) / base * base;
}

// Utility to read an integer (signed or unsigned, any size)
// from a stream in little-endian order. We swap the byte order after the read
// if necessary to return a result with the byte ordering of the compiling
// machine.
template <typename IntType>
inline IntType read_little_endian(std::istream &stream) {
  IntType result;

  if (IsLittleEndian)
    stream.read(reinterpret_cast<char *>(&result), sizeof(IntType));
  else {
    std::uint8_t u[sizeof(IntType)];
    std::make_unsigned_t<IntType> v = 0;

    stream.read(reinterpret_cast<char *>(u), sizeof(IntType));
    for (std::size_t i = 0; i < sizeof(IntType); ++i)
      v = (v << 8) | u[sizeof(IntType) - i - 1];

    std::memcpy(&result, &v, sizeof(IntType));
  }

  return result;
}

// Utility to write an integer (signed or unsigned, any size)
// to a stream in little-endian order. We swap the byte order before the write
// if necessary to always write in little-endian order, independently of the
// byte ordering of the compiling machine.
template <typename IntType>
inline void write_little_endian(std::ostream &stream, IntType value) {

  if (IsLittleEndian)
    stream.write(reinterpret_cast<const char *>(&value), sizeof(IntType));
  else {
    std::uint8_t u[sizeof(IntType)];
    std::make_unsigned_t<IntType> v = value;

    std::size_t i = 0;
    // if constexpr to silence the warning about shift by 8
    if constexpr (sizeof(IntType) > 1) {
      for (; i + 1 < sizeof(IntType); ++i) {
        u[i] = std::uint8_t(v);
        v >>= 8;
      }
    }
    u[i] = std::uint8_t(v);

    stream.write(reinterpret_cast<char *>(u), sizeof(IntType));
  }
}

// Read integers in bulk from a little-endian stream.
// This reads N integers from stream s and puts them in array out.
template <typename IntType>
inline void read_little_endian(std::istream &stream, IntType *out,
                               std::size_t count) {
  if (IsLittleEndian)
    stream.read(reinterpret_cast<char *>(out), sizeof(IntType) * count);
  else
    for (std::size_t i = 0; i < count; ++i)
      out[i] = read_little_endian<IntType>(stream);
}

// Write integers in bulk to a little-endian stream.
// This takes N integers from array values and writes them on stream s.
template <typename IntType>
inline void write_little_endian(std::ostream &stream, const IntType *values,
                                std::size_t count) {
  if (IsLittleEndian)
    stream.write(reinterpret_cast<const char *>(values),
                 sizeof(IntType) * count);
  else
    for (std::size_t i = 0; i < count; ++i)
      write_little_endian<IntType>(stream, values[i]);
}

// Read N signed integers from the stream s, putting them in the array out.
// The stream is assumed to be compressed using the signed LEB128 format.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression
// scheme.
template <typename BufType, typename IntType, std::size_t Count>
inline void read_leb_128_detail(std::istream &stream,
                                std::array<IntType, Count> &out,
                                std::uint32_t &bytes_left, BufType &buf,
                                std::uint32_t &buf_pos) {

  static_assert(std::is_signed_v<IntType>,
                "Not implemented for unsigned types");
  static_assert(sizeof(IntType) <= 4,
                "Not implemented for types larger than 32 bit");

  IntType result = 0;
  size_t shift = 0, i = 0;
  while (i < Count) {
    if (buf_pos == buf.size()) {
      stream.read(reinterpret_cast<char *>(buf.data()),
                  std::min(std::size_t(bytes_left), buf.size()));
      buf_pos = 0;
    }

    std::uint8_t byte = buf[buf_pos++];
    --bytes_left;
    result |= (byte & 0x7f) << (shift % 32);
    shift += 7;

    if ((byte & 0x80) == 0) {
      out[i++] = (shift >= 32 || (byte & 0x40) == 0)
                     ? result
                     : result | ~((1 << shift) - 1);
      result = 0;
      shift = 0;
    }
  }
}

template <typename... Arrays>
inline void read_leb_128(std::istream &stream, Arrays &...outs) {
  // Check the presence of our LEB128 magic string
  char leb128MagicString[Leb128MagicStringSize];
  stream.read(leb128MagicString, Leb128MagicStringSize);
  assert(strncmp(Leb128MagicString, leb128MagicString, Leb128MagicStringSize) ==
         0);

  auto bytes_left = read_little_endian<std::uint32_t>(stream);
  std::array<std::uint8_t, 8192> buf;
  std::uint32_t buf_pos = buf.size();

  (read_leb_128_detail(stream, outs, bytes_left, buf, buf_pos), ...);

  assert(bytes_left == 0);
}

// Write signed integers to a stream with LEB128 compression.
// This takes N integers from array values, compresses them with
// the LEB128 algorithm and writes the result on the stream s.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression
// scheme.
template <typename IntType, std::size_t Count>
inline void write_leb_128(std::ostream &stream,
                          const std::array<IntType, Count> &values) {

  // Write our LEB128 magic string
  stream.write(Leb128MagicString, Leb128MagicStringSize);

  static_assert(std::is_signed_v<IntType>,
                "Not implemented for unsigned types");

  std::uint32_t byte_count = 0;
  for (std::size_t i = 0; i < Count; ++i) {
    IntType value = values[i];
    std::uint8_t byte;
    do {
      byte = value & 0x7f;
      value >>= 7;
      ++byte_count;
    } while ((byte & 0x40) == 0 ? value != 0 : value != -1);
  }

  write_little_endian(stream, byte_count);

  const std::uint32_t BUF_SIZE = 4096;
  std::uint8_t buf[BUF_SIZE];
  std::uint32_t buf_pos = 0;

  auto flush = [&]() {
    if (buf_pos > 0) {
      stream.write(reinterpret_cast<char *>(buf), buf_pos);
      buf_pos = 0;
    }
  };

  auto write = [&](std::uint8_t b) {
    buf[buf_pos++] = b;
    if (buf_pos == BUF_SIZE)
      flush();
  };

  for (std::size_t i = 0; i < Count; ++i) {
    IntType value = values[i];
    while (true) {
      std::uint8_t byte = value & 0x7f;
      value >>= 7;
      if ((byte & 0x40) == 0 ? value == 0 : value == -1) {
        write(byte);
        break;
      }
      write(byte | 0x80);
    }
  }

  flush();
}

} // namespace MetalFish::Eval::NNUE

#endif // #ifndef NNUE_COMMON_H_INCLUDED