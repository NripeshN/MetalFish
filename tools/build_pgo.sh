#!/bin/bash
# Two-stage profile-guided build of the metalfish engine (clang on Apple Silicon or Linux).
#
#   tools/build_pgo.sh [BUILD_DIR=build-pgo] [extra cmake args...]
#
# Stage 1 builds an instrumented binary, runs the AB bench plus short MCTS and hybrid
# searches as training workloads, merges the profiles; stage 2 rebuilds with the profile.
# The final binary is <BUILD_DIR>/metalfish; <BUILD_DIR>/default.profdata is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${1:-build-pgo}"; shift || true
EXTRA=("$@")
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
PROFDATA="$(xcrun -f llvm-profdata 2>/dev/null || command -v llvm-profdata)"
WEIGHTS="$PWD/networks/BT4-1024x15x32h-swa-6147500.pb"
METAL_FLAG="-DUSE_METAL=ON"; [ "$(uname -s)" = "Darwin" ] || METAL_FLAG="-DUSE_METAL=OFF"

echo "== stage 1: instrumented build in $BUILD_DIR =="
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $METAL_FLAG -DMETALFISH_PGO=generate "${EXTRA[@]}" >/dev/null
cmake --build "$BUILD_DIR" --target metalfish -j"$NCPU" >/dev/null
rm -rf "$BUILD_DIR/prof"; mkdir -p "$BUILD_DIR/prof"
echo "== training: AB bench (default positions, depth 13), threads 1 and 4 =="
LLVM_PROFILE_FILE="$BUILD_DIR/prof/bench1-%p.profraw" "$BUILD_DIR/metalfish" bench 256 1 13 >/dev/null
LLVM_PROFILE_FILE="$BUILD_DIR/prof/bench4-%p.profraw" "$BUILD_DIR/metalfish" bench 1024 4 13 >/dev/null
if [ -f "$WEIGHTS" ] && [ "$(uname -s)" = "Darwin" ]; then
  echo "== training: pure MCTS and hybrid searches (transformer path) =="
  LLVM_PROFILE_FILE="$BUILD_DIR/prof/mcts-%p.profraw" "$BUILD_DIR/metalfish" >/dev/null <<EOF
uci
setoption name UseMCTS value true
setoption name NNWeights value $WEIGHTS
setoption name Threads value 1
isready
position startpos moves e2e4 c7c5 g1f3 d7d6
go nodes 400
wait
position fen r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PPQ1BPPP/R1B2RK1 w - - 0 10
go nodes 400
wait
quit
EOF
  LLVM_PROFILE_FILE="$BUILD_DIR/prof/hybrid-%p.profraw" "$BUILD_DIR/metalfish" >/dev/null <<EOF
uci
setoption name UseHybridSearch value true
setoption name NNWeights value $WEIGHTS
setoption name Threads value 4
isready
position startpos moves d2d4 g8f6 c2c4 e7e6
go movetime 4000
wait
quit
EOF
fi
echo "== merging profiles =="
"$PROFDATA" merge -output="$BUILD_DIR/default.profdata" "$BUILD_DIR"/prof/*.profraw
echo "== stage 2: optimised build with profile =="
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $METAL_FLAG -DMETALFISH_PGO=use \
  -DMETALFISH_PGO_PROFILE="$PWD/$BUILD_DIR/default.profdata" "${EXTRA[@]}" >/dev/null
cmake --build "$BUILD_DIR" --target metalfish -j"$NCPU" >/dev/null
echo "== done: $BUILD_DIR/metalfish =="
"$BUILD_DIR/metalfish" bench 16 1 13 2>&1 | tail -3
