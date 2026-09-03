#!/bin/bash
# MetalFish SPRT Tuning Suite
# Runs the full batch of parameter tests to find optimal configuration.
#
# Usage:
#   ./tools/run_sprt_suite.sh               # Full suite, 1000ms movetime
#   ./tools/run_sprt_suite.sh --quick       # Faster, less accurate (500ms, 500 max games)
#   ./tools/run_sprt_suite.sh --tc 10+0.1   # Real time control
#
# Results are written to results/sprt/

set -e
cd "$(dirname "$0")/.."

MOVETIME=1000
MAX_GAMES=2000
TC=""
MODE="hybrid"
THREADS="${METALFISH_THREADS:-0}"
HASH="${METALFISH_HASH:-2048}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            MOVETIME=500
            MAX_GAMES=500
            echo "Quick mode: movetime=${MOVETIME}ms, max_games=${MAX_GAMES}"
            ;;
        --tc=*) TC="${arg#*=}" ;;
        --movetime=*) MOVETIME="${arg#*=}" ;;
        --max-games=*) MAX_GAMES="${arg#*=}" ;;
        --mode=*) MODE="${arg#*=}" ;;
        --threads=*) THREADS="${arg#*=}" ;;
        --hash=*) HASH="${arg#*=}" ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

ENGINE="./build/metalfish"
WEIGHTS="./networks/BT4-1024x15x32h-swa-6147500.pb"

if [ ! -f "$ENGINE" ]; then
    echo "ERROR: Engine not found at $ENGINE"
    echo "Build first: cmake --build build --target metalfish -j\$(sysctl -n hw.ncpu)"
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Weights not found at $WEIGHTS"
    echo "Download: python3 tools/download_engine_networks.py --dest networks"
    exit 1
fi

echo ""
echo "============================================"
echo "  MetalFish SPRT Tuning Suite"
echo "============================================"
echo "Mode: $MODE | Movetime: ${MOVETIME}ms | Max games: $MAX_GAMES"
echo "Threads: $THREADS (0=auto) | Hash: ${HASH}MB"
if [ -n "$TC" ]; then
    echo "TC: $TC"
fi
echo ""

COMMON_ARGS=(
    --engine "$ENGINE"
    --weights "$WEIGHTS"
    --mode "$MODE"
    --movetime "$MOVETIME"
    --max-games "$MAX_GAMES"
    --threads "$THREADS"
    --hash "$HASH"
)
if [ -n "$TC" ]; then
    COMMON_ARGS+=(--tc "$TC")
fi

mkdir -p results/sprt

echo "--- Phase 1: Self-play sanity check ---"
python3 tools/sprt_test.py "${COMMON_ARGS[@]}" --self-play --games 10 \
    --json-out results/sprt/00_self_play.json

echo ""
echo "--- Phase 2: Batch SPRT tests ---"
python3 tools/sprt_test.py "${COMMON_ARGS[@]}" \
    --batch tools/sprt_tuning_batch.json \
    --json-out results/sprt/batch_results.json

echo ""
echo "============================================"
echo "  Suite complete. Results in results/sprt/"
echo "============================================"
echo ""
echo "View results: cat results/sprt/batch_results.json | python3 -m json.tool"
