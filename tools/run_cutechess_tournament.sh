#!/bin/bash
# MetalFish Tournament via cutechess-cli
# Usage: ./tools/run_cutechess_tournament.sh [--quick]

set -e
cd "$(dirname "$0")/.."
PROJ="$(pwd)"
CUTECHESS="$PROJ/reference/cutechess/build/cutechess-cli"
BOOK="$PROJ/reference/books/8moves_v3.pgn"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$PROJ/results/cutechess_$TIMESTAMP"
mkdir -p "$RESULTS"

# Defaults: 300+0.1, 20 games
GAMES=20
TC="300+0.1"
if [ "$1" = "--quick" ]; then
    GAMES=4
    TC="10+0.1"
    echo "Quick mode: $GAMES games, TC=$TC"
fi

# Engine definitions
MF="$PROJ/build/metalfish"
WEIGHTS="$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb"
SF="$PROJ/reference/stockfish/src/stockfish"
BERSERK="$PROJ/reference/berserk/src/berserk"
PATRICIA="$PROJ/reference/Patricia/engine/patricia"
LC0="$PROJ/reference/lc0/build/release/lc0"

AB="cmd=$MF name=MetalFish-AB option.Threads=8 option.Hash=256"
MCTS="cmd=$MF name=MetalFish-MCTS option.Threads=8 option.Hash=256 option.UseMCTS=true option.NNWeights=$WEIGHTS"
HYBRID="cmd=$MF name=MetalFish-Hybrid option.Threads=8 option.Hash=256 option.UseHybridSearch=true option.NNWeights=$WEIGHTS"
SFULL="cmd=$SF name=Stockfish option.Threads=8 option.Hash=256"
SL15="cmd=$SF name=Stockfish-L15 option.Threads=8 option.Hash=256 option.\"Skill Level\"=15"
SL10="cmd=$SF name=Stockfish-L10 option.Threads=8 option.Hash=256 option.\"Skill Level\"=10"
BERSERK_E="cmd=$BERSERK name=Berserk option.Threads=8 option.Hash=256"
PATRICIA_E="cmd=$PATRICIA name=Patricia option.Threads=8 option.Hash=256"
LC0_E="cmd=$LC0 name=Lc0 arg=\"--weights=$WEIGHTS\" arg=\"--backend=metal\" option.Threads=8 option.Temperature=0"

BOOK_ARGS=""
if [ -f "$BOOK" ]; then
    BOOK_ARGS="-openings file=$BOOK format=pgn order=random"
fi

COMMON="-each tc=$TC -games $GAMES -repeat -recover $BOOK_ARGS \
    -resign movecount=3 score=1000 \
    -draw movenumber=40 movecount=8 score=10"

echo ""
echo "============================================"
echo "  MetalFish Tournament (cutechess-cli)"
echo "============================================"
echo "TC: $TC | Games/match: $GAMES"
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

# Internal head-to-head
run_match "$AB" "$MCTS" "01_AB_vs_MCTS"
run_match "$AB" "$HYBRID" "02_AB_vs_Hybrid"
run_match "$MCTS" "$HYBRID" "03_MCTS_vs_Hybrid"

# MetalFish-AB vs reference engines
run_match "$AB" "$SFULL" "04_AB_vs_Stockfish"
run_match "$AB" "$BERSERK_E" "05_AB_vs_Berserk"
run_match "$AB" "$PATRICIA_E" "06_AB_vs_Patricia"

# MetalFish-MCTS vs NN baselines
run_match "$MCTS" "$LC0_E" "07_MCTS_vs_Lc0"
run_match "$MCTS" "$PATRICIA_E" "08_MCTS_vs_Patricia"

# MetalFish-Hybrid vs reference engines
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
