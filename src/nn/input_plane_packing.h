/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define METALFISH_HOST_DEVICE __host__ __device__
#else
#define METALFISH_HOST_DEVICE
#endif

namespace MetalFish {
namespace NN {

constexpr int kPackedInputMoveHistory = 8;
constexpr int kPackedInputPlanesPerBoard = 13;
constexpr int kPackedInputAuxPlaneBase =
    kPackedInputPlanesPerBoard * kPackedInputMoveHistory;
constexpr int kPackedInputPlaneCount = 112;
constexpr int kPackedInputSquareCount = 64;

METALFISH_HOST_DEVICE inline bool IsUniformPackedInputPlane(int plane) {
  return (plane < kPackedInputAuxPlaneBase &&
          plane % kPackedInputPlanesPerBoard ==
              kPackedInputPlanesPerBoard - 1) ||
         plane == kPackedInputAuxPlaneBase + 2 ||
         plane == kPackedInputAuxPlaneBase + 3 ||
         plane >= kPackedInputAuxPlaneBase + 5;
}

METALFISH_HOST_DEVICE inline void PackInputPlaneRaw(const float *plane,
                                                    int plane_idx,
                                                    std::uint64_t &mask,
                                                    float &value) {
  if (IsUniformPackedInputPlane(plane_idx)) {
    value = plane[0];
    mask = value != 0.0f ? ~0ULL : 0ULL;
    return;
  }

  mask = 0;
  value = 0.0f;
  for (int sq = 0; sq < kPackedInputSquareCount; ++sq) {
    const float v = plane[sq];
    if (v != 0.0f) {
      if (mask == 0)
        value = v;
      mask |= (1ULL << sq);
    }
  }
}

inline void PackInputPlanesRaw(const float *inputs, int batch_size,
                               std::vector<std::uint64_t> &masks,
                               std::vector<float> &values) {
  if (batch_size <= 0 || inputs == nullptr) {
    masks.clear();
    values.clear();
    return;
  }

  const size_t total_planes =
      static_cast<size_t>(batch_size) * kPackedInputPlaneCount;
  masks.assign(total_planes, 0);
  values.assign(total_planes, 0.0f);

  for (int b = 0; b < batch_size; ++b) {
    for (int p = 0; p < kPackedInputPlaneCount; ++p) {
      const size_t idx = static_cast<size_t>(b) * kPackedInputPlaneCount +
                         static_cast<size_t>(p);
      PackInputPlaneRaw(inputs + idx * kPackedInputSquareCount, p, masks[idx],
                        values[idx]);
    }
  }
}

} // namespace NN
} // namespace MetalFish

#undef METALFISH_HOST_DEVICE
