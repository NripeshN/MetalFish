/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "nn_input_fixture.h"

#include "core/position.h"
#include "nn/input_plane_packing.h"

#include <cmath>
#include <sstream>

namespace MetalFish::Tests {
namespace {

std::size_t CountMatchingMasks(const std::vector<std::uint64_t> &masks,
                               std::uint64_t expected) {
  std::size_t count = 0;
  for (std::uint64_t mask : masks) {
    if (mask == expected)
      ++count;
  }
  return count;
}

std::size_t CountNonzeroMasks(const std::vector<std::uint64_t> &masks) {
  std::size_t count = 0;
  for (std::uint64_t mask : masks) {
    if (mask != 0)
      ++count;
  }
  return count;
}

} // namespace

PackedInputFixture BuildStartPositionPackedInputFixture() {
  PackedInputFixture fixture;
  fixture.fen = kStartPositionFixtureFen;

  Position pos;
  StateInfo state;
  pos.set(fixture.fen, false, &state);
  fixture.planes = NN::EncodePositionForNN(pos);

  NN::PackInputPlanesRaw(fixture.planes[0].data(), 1, fixture.masks,
                         fixture.values);
  fixture.nonzero_planes = CountNonzeroMasks(fixture.masks);
  fixture.full_mask_planes = CountMatchingMasks(fixture.masks, kFullPlaneMask);
  return fixture;
}

bool ValidateStartPositionPackedInputFixture(const PackedInputFixture &fixture,
                                             std::string *error) {
  auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return false;
  };

  if (fixture.fen != kStartPositionFixtureFen)
    return fail("fixture FEN changed");
  if (fixture.masks.size() != NN::kTotalPlanes)
    return fail("packed mask count does not match 112 input planes");
  if (fixture.values.size() != NN::kTotalPlanes)
    return fail("packed value count does not match 112 input planes");
  if (fixture.nonzero_planes != 17)
    return fail("unexpected number of nonzero packed planes");
  if (fixture.full_mask_planes != 5)
    return fail("unexpected number of full-mask packed planes");

  for (const auto &check : kStartPositionPackedPlaneChecks) {
    if (fixture.masks[check.plane] != check.mask) {
      std::ostringstream out;
      out << "mask mismatch on plane " << check.plane;
      return fail(out.str());
    }
    if (std::fabs(fixture.values[check.plane] - check.value) > 1e-6f) {
      std::ostringstream out;
      out << "value mismatch on plane " << check.plane;
      return fail(out.str());
    }
  }

  for (int plane = 13; plane < NN::kAuxPlaneBase; ++plane) {
    if (fixture.masks[plane] != 0) {
      std::ostringstream out;
      out << "non-root history plane " << plane << " should stay empty";
      return fail(out.str());
    }
  }
  if (fixture.masks[NN::kAuxPlaneBase + 4] != 0 ||
      fixture.values[NN::kAuxPlaneBase + 4] != 0.0f)
    return fail("white-to-move side plane should be empty");
  if (fixture.masks[NN::kAuxPlaneBase + 5] != 0 ||
      fixture.values[NN::kAuxPlaneBase + 5] != 0.0f)
    return fail("rule-50 plane should be empty");
  if (fixture.masks[NN::kAuxPlaneBase + 6] != 0 ||
      fixture.values[NN::kAuxPlaneBase + 6] != 0.0f)
    return fail("armageddon plane should be empty");

  return true;
}

} // namespace MetalFish::Tests
