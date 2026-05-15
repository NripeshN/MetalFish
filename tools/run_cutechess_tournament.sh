#!/bin/bash
# MetalFish Tournament via cutechess-cli
# Usage: ./tools/run_cutechess_tournament.sh [--quick] [--games=N] [--tc=300+0.1] [--seed=N]

set -e
cd "$(dirname "$0")/.."
PROJ="$(pwd)"
CUTECHESS="$PROJ/reference/cutechess/build/cutechess-cli"
BOOK="$PROJ/reference/books/8moves_v3.pgn"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$PROJ/results/cutechess_$TIMESTAMP"
mkdir -p "$RESULTS"

# Defaults: 300+0.1, 20 games. Use a fixed cutechess seed so
# repeated engine/config comparisons see the same opening sample.
GAMES=20
TC="300+0.1"
CUTECHESS_SEED="${CUTECHESS_SEED:-6147500}"
OPENING_ORDER="${OPENING_ORDER:-random}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            GAMES=4
            TC="10+0.1"
            echo "Quick mode: $GAMES games, TC=$TC"
            ;;
        --games=*) GAMES="${arg#*=}" ;;
        --tc=*) TC="${arg#*=}" ;;
        --seed=*) CUTECHESS_SEED="${arg#*=}" ;;
        --opening-order=*) OPENING_ORDER="${arg#*=}" ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

MF="$PROJ/build/metalfish"
WEIGHTS="$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb"
SF="$PROJ/reference/stockfish/src/stockfish"
BERSERK="$PROJ/reference/berserk/src/berserk"
PATRICIA="$PROJ/reference/Patricia/engine/patricia"
LC0="$PROJ/reference/lc0/build/release/lc0"

if [ -z "${THREADS:-}" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        THREADS=$(sysctl -n hw.perflevel0.physicalcpu_max 2>/dev/null || true)
        [ -n "$THREADS" ] || THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || true)
    else
        THREADS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
    fi
fi
[ -n "$THREADS" ] || THREADS=8

MCTS_THREADS="${MCTS_THREADS:-1}"
if [ "$MCTS_THREADS" -lt 1 ]; then
    MCTS_THREADS=1
fi
if [ "$MCTS_THREADS" -ge "$THREADS" ] && [ "$THREADS" -gt 1 ]; then
    MCTS_THREADS=$(( THREADS - 1 ))
fi

HYBRID_THREADS="${HYBRID_THREADS:-$THREADS}"
if [ "$HYBRID_THREADS" -lt 3 ]; then
    HYBRID_THREADS=3
fi
HYBRID_MCTS_THREADS="${HYBRID_MCTS_THREADS:-0}"
if [ "$HYBRID_MCTS_THREADS" -gt 0 ] && [ "$HYBRID_MCTS_THREADS" -ge "$HYBRID_THREADS" ]; then
    HYBRID_MCTS_THREADS=$(( HYBRID_THREADS - 1 ))
fi
# 0 = engine auto. Fixed-budget searches use the tactical 2/1 split; real game
# clocks use the capped 1/2 split unless explicit overrides are provided.
HYBRID_AB_THREADS="${HYBRID_AB_THREADS:-0}"
if [ "$HYBRID_AB_THREADS" -gt 0 ] && [ "$HYBRID_AB_THREADS" -gt "$HYBRID_THREADS" ]; then
    HYBRID_AB_THREADS=$HYBRID_THREADS
fi
HYBRID_AUTO_AB_THREADS_CAP="${HYBRID_AUTO_AB_THREADS_CAP:-2}"

AB="cmd=$MF name=MetalFish-AB option.Threads=$THREADS option.Hash=256"
MCTS="cmd=$MF name=MetalFish-MCTS option.Threads=$MCTS_THREADS option.Hash=256 option.UseMCTS=true option.NNWeights=$WEIGHTS option.MCTSMaxThreads=$MCTS_THREADS option.MCTSMinibatchSize=0"
HYBRID="cmd=$MF name=MetalFish-Hybrid option.Threads=$HYBRID_THREADS option.Hash=256 option.UseHybridSearch=true option.NNWeights=$WEIGHTS option.HybridMCTSThreads=$HYBRID_MCTS_THREADS option.HybridABThreads=$HYBRID_AB_THREADS option.HybridAutoABThreadsCap=$HYBRID_AUTO_AB_THREADS_CAP option.MCTSMaxThreads=$HYBRID_MCTS_THREADS option.MCTSMinibatchSize=0"
SFULL="cmd=$SF name=Stockfish option.Threads=$THREADS option.Hash=256"
SL15="cmd=$SF name=Stockfish-L15 option.Threads=$THREADS option.Hash=256 option.\"Skill Level\"=15"
SL10="cmd=$SF name=Stockfish-L10 option.Threads=$THREADS option.Hash=256 option.\"Skill Level\"=10"
BERSERK_E="cmd=$BERSERK name=Berserk option.Threads=$THREADS option.Hash=256"
PATRICIA_E="cmd=$PATRICIA name=Patricia option.Threads=$THREADS option.Hash=256"
LC0_E="cmd=$LC0 name=Lc0 arg=\"--weights=$WEIGHTS\" arg=\"--backend=metal\" option.Threads=$THREADS option.Temperature=0"

BOOK_ARGS=""
if [ -f "$BOOK" ]; then
    BOOK_ARGS="-openings file=$BOOK format=pgn order=$OPENING_ORDER"
fi

COMMON="-each tc=$TC -games $GAMES -repeat -recover $BOOK_ARGS \
    -srand $CUTECHESS_SEED \
    -resign movecount=3 score=1000 \
    -draw movenumber=40 movecount=8 score=10"

echo ""
echo "============================================"
echo "  MetalFish Tournament (cutechess-cli)"
echo "============================================"
echo "TC: $TC | Games/match: $GAMES"
echo "Openings: order=$OPENING_ORDER | seed=$CUTECHESS_SEED"
echo "Threads: AB=$THREADS MCTS=$MCTS_THREADS Hybrid=$HYBRID_THREADS (HybridMCTS=$HYBRID_MCTS_THREADS, HybridAB=$HYBRID_AB_THREADS, HybridAutoABCap=$HYBRID_AUTO_AB_THREADS_CAP)"
echo "Results: $RESULTS"
echo ""

run_match() {
    local e1="$1"
    local e2="$2"
    local label="$3"
    local pgn="$RESULTS/${label}.pgn"

    echo ""
    echo "--- $label ---"
    $CUTECHESS -engine $e1 -engine $e2 $COMMON -pgnout "$pgn" 2>&1 | tail -5
    echo "  PGN saved: $pgn"
}

run_match "$AB" "$MCTS" "01_AB_vs_MCTS"
run_match "$AB" "$HYBRID" "02_AB_vs_Hybrid"
run_match "$MCTS" "$HYBRID" "03_MCTS_vs_Hybrid"

run_match "$AB" "$SFULL" "04_AB_vs_Stockfish"
run_match "$AB" "$BERSERK_E" "05_AB_vs_Berserk"
run_match "$AB" "$PATRICIA_E" "06_AB_vs_Patricia"

run_match "$MCTS" "$LC0_E" "07_MCTS_vs_Lc0"
run_match "$MCTS" "$PATRICIA_E" "08_MCTS_vs_Patricia"

run_match "$HYBRID" "$SL15" "09_Hybrid_vs_Stockfish-L15"
run_match "$HYBRID" "$BERSERK_E" "10_Hybrid_vs_Berserk"
run_match "$HYBRID" "$PATRICIA_E" "11_Hybrid_vs_Patricia"
run_match "$HYBRID" "$LC0_E" "12_Hybrid_vs_Lc0"

echo ""
echo "============================================"
echo "  ALL MATCHES COMPLETE"
echo "============================================"
echo "PGN files: $RESULTS/"
echo ""
echo "To analyze results:"
echo "  $CUTECHESS -results $RESULTS/*.pgn"
