/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  HIP/ROCm Kernel Utilities
*/

#pragma once

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>

// Compatibility macros
#define HIP_CONST static constexpr const
#define HIP_PRAGMA_UNROLL _Pragma("unroll")

HIP_CONST int WARP_SIZE = 64; // AMD GPUs typically use 64-wide wavefronts

///////////////////////////////////////////////////////////////////////////////
// Type limits
///////////////////////////////////////////////////////////////////////////////

template <typename U> struct Limits {
  static constexpr U max = ~U(0);
  static constexpr U min = U(0);
};

template <> struct Limits<int32_t> {
  static constexpr int32_t max = 2147483647;
  static constexpr int32_t min = -2147483648;
};

template <> struct Limits<int16_t> {
  static constexpr int16_t max = 32767;
  static constexpr int16_t min = -32768;
};

template <> struct Limits<int8_t> {
  static constexpr int8_t max = 127;
  static constexpr int8_t min = -128;
};

///////////////////////////////////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U> __device__ inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// Clamp value to range
template <typename T> __device__ inline T clamp_value(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

///////////////////////////////////////////////////////////////////////////////
// Warp shuffle operations for reductions (AMD uses __shfl_down)
///////////////////////////////////////////////////////////////////////////////

__device__ inline int32_t warp_shuffle_down(int32_t data, int delta) {
  return __shfl_down(data, delta);
}

__device__ inline float warp_shuffle_down(float data, int delta) {
  return __shfl_down(data, delta);
}

__device__ inline int16_t warp_shuffle_down(int16_t data, int delta) {
  return __shfl_down(static_cast<int>(data), delta);
}

__device__ inline int8_t warp_shuffle_down(int8_t data, int delta) {
  return static_cast<int8_t>(__shfl_down(static_cast<int>(data), delta));
}

///////////////////////////////////////////////////////////////////////////////
// Warp reduction operations
///////////////////////////////////////////////////////////////////////////////

template <typename T> __device__ inline T warp_sum(T value) {
  for (int delta = WARP_SIZE / 2; delta >= 1; delta >>= 1) {
    value += warp_shuffle_down(value, delta);
  }
  return value;
}

template <typename T> __device__ inline T warp_max(T value) {
  for (int delta = WARP_SIZE / 2; delta >= 1; delta >>= 1) {
    T other = warp_shuffle_down(value, delta);
    value = (value > other) ? value : other;
  }
  return value;
}

///////////////////////////////////////////////////////////////////////////////
// Memory access helpers
///////////////////////////////////////////////////////////////////////////////

template <typename T, int N> struct alignas(sizeof(T) * N) PackedArray {
  T data[N];

  __device__ T &operator[](int i) { return data[i]; }
  __device__ const T &operator[](int i) const { return data[i]; }
};

// Load N elements starting from ptr
template <typename T, int N>
__device__ inline void load_n(T *dst, const T *src) {
  HIP_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

// Store N elements to dst
template <typename T, int N>
__device__ inline void store_n(T *dst, const T *src) {
  HIP_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Atomic operations helpers
///////////////////////////////////////////////////////////////////////////////

__device__ inline int32_t atomic_add_int32(int32_t* address, int32_t val) {
  return atomicAdd(address, val);
}

__device__ inline float atomic_add_float(float* address, float val) {
  return atomicAdd(address, val);
}

///////////////////////////////////////////////////////////////////////////////
// Math helpers
///////////////////////////////////////////////////////////////////////////////

__device__ inline float fast_exp(float x) {
  return __expf(x);
}

__device__ inline float fast_log(float x) {
  return __logf(x);
}

__device__ inline float fast_sqrt(float x) {
  return __fsqrt_rn(x);
}
