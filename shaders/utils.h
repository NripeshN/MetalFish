/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Adapted from MLX (Apple Inc.) - Copyright © 2023-2024 Apple Inc.

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

MLX_MTL_CONST int SIMD_SIZE = 32;

///////////////////////////////////////////////////////////////////////////////
// Type limits
///////////////////////////////////////////////////////////////////////////////

template <typename U> struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
};

///////////////////////////////////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U> inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// Clamp value to range
template <typename T> inline T clamp_value(T x, T lo, T hi) {
  return metal::clamp(x, lo, hi);
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle operations for reductions
///////////////////////////////////////////////////////////////////////////////

inline int32_t simd_shuffle_down(int32_t data, uint16_t delta) {
  return metal::simd_shuffle_down(data, delta);
}

inline float simd_shuffle_down(float data, uint16_t delta) {
  return metal::simd_shuffle_down(data, delta);
}

inline int16_t simd_shuffle_down(int16_t data, uint16_t delta) {
  return metal::simd_shuffle_down(data, delta);
}

inline int8_t simd_shuffle_down(int8_t data, uint16_t delta) {
  return int8_t(metal::simd_shuffle_down(int(data), delta));
}

///////////////////////////////////////////////////////////////////////////////
// SIMD reduction operations
///////////////////////////////////////////////////////////////////////////////

template <typename T> inline T simd_sum(T value) {
  for (uint16_t delta = SIMD_SIZE / 2; delta >= 1; delta >>= 1) {
    value += simd_shuffle_down(value, delta);
  }
  return value;
}

template <typename T> inline T simd_max(T value) {
  for (uint16_t delta = SIMD_SIZE / 2; delta >= 1; delta >>= 1) {
    T other = simd_shuffle_down(value, delta);
    value = (value > other) ? value : other;
  }
  return value;
}

///////////////////////////////////////////////////////////////////////////////
// Memory access helpers
///////////////////////////////////////////////////////////////////////////////

template <typename T, int N> struct alignas(sizeof(T) * N) PackedArray {
  T data[N];

  T &operator[](int i) { return data[i]; }
  const T &operator[](int i) const { return data[i]; }
};

// Load N elements starting from ptr
template <typename T, int N>
inline void load_n(thread T *dst, const device T *src) {
  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

// Store N elements to dst
template <typename T, int N>
inline void store_n(device T *dst, const thread T *src) {
  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}
