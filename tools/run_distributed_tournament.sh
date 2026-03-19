#!/bin/bash
# ============================================================================
# MetalFish Distributed Tournament via cutechess-cli
# All engines built natively on 4 M1 Ultra EC2 instances
# ============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(44.220.150.2 98.81.229.157 98.84.106.208 32.192.83.249)
USER=ec2-user
RDIR="/Users/ec2-user/metalfish-src"
RESULTS_DIR="$PROJ/results/distributed_$(date +%Y%m%d_%H%M%S)"
GAMES=20
TC="300+0.1"

for arg in "$@"; do
    case $arg in
        --quick) GAMES=4; TC="10+0.1" ;;
        --games=*) GAMES="${arg#*=}" ;;
        --tc=*) TC="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_DIR"
ssh_cmd() { ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$USER@$1" "${@:2}"; }
scp_cmd() { scp -i "$PEM" -o StrictHostKeyChecking=no -q "$@"; }

echo "============================================"
echo "  MetalFish Distributed Tournament"
echo "============================================"
echo "Instances: ${#HOSTS[@]} | Games: $GAMES | TC: $TC"
echo "Results: $RESULTS_DIR"
echo ""

# ============================================================================
# Verify builds exist
# ============================================================================
echo "--- Verifying ---"
for host in "${HOSTS[@]}"; do
    V=$(ssh_cmd "$host" "ls $RDIR/build/metalfish 2>/dev/null && echo OK || echo MISSING")
    echo "  $host: $V"
done
echo ""

# ============================================================================
# Engine definitions (10 threads each = 20 cores / 2 engines per match)
# ============================================================================
CC="$RDIR/reference/cutechess/build/cutechess-cli"
BOOK="$RDIR/reference/books/8moves_v3.pgn"
WEIGHTS="$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb"
COMMON="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

AB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=10 option.Hash=512"
MCTS="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-MCTS option.Threads=10 option.UseMCTS=true option.NNWeights=$WEIGHTS"
HYB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-Hybrid option.Threads=10 option.Hash=512 option.UseHybridSearch=true option.NNWeights=$WEIGHTS"
SF="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish option.Threads=10 option.Hash=512"
SFL15='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=Stockfish-L15 option.Threads=10 option.Hash=512 "option.Skill Level=15"'
SFL10='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=Stockfish-L10 option.Threads=10 option.Hash=512 "option.Skill Level=10"'
SFL5='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=Stockfish-L5 option.Threads=10 option.Hash=512 "option.Skill Level=5"'
BER="-engine proto=uci cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=10 option.Hash=512"
PAT="-engine proto=uci cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=10 option.Hash=512"
LC0="-engine proto=uci cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$WEIGHTS arg=--backend=metal option.Threads=10"

# ============================================================================
# Match runner
# ============================================================================
run_match() {
    local host=$1 idx=$2 e1="$3" e2="$4" label="$5"
    echo "[$idx] $label"
    ssh_cmd "$host" "cd $RDIR && $CC $e1 $e2 $COMMON -pgnout $RDIR/$label.pgn 2>&1" | tail -3
    scp_cmd "$USER@$host:$RDIR/$label.pgn" "$RESULTS_DIR/" 2>/dev/null || true
    echo "[$idx] Done: $label"
}

run_group() {
    local host=$1 idx=$2; shift 2
    while [ $# -ge 3 ]; do run_match "$host" "$idx" "$1" "$2" "$3"; shift 3; done
}

echo "--- Running 24 matches across 4 M1 Ultra instances ---"
echo ""

# Instance 1: MetalFish-AB matches
run_group "${HOSTS[0]}" 1 \
    "$AB" "$SF" "01_AB_vs_Stockfish" \
    "$AB" "$BER" "02_AB_vs_Berserk" \
    "$AB" "$PAT" "03_AB_vs_Patricia" \
    "$AB" "$SFL15" "04_AB_vs_SF-L15" \
    "$AB" "$SFL10" "05_AB_vs_SF-L10" \
    "$AB" "$LC0" "06_AB_vs_Lc0" &

# Instance 2: MetalFish-MCTS matches
run_group "${HOSTS[1]}" 2 \
    "$MCTS" "$LC0" "07_MCTS_vs_Lc0" \
    "$MCTS" "$PAT" "08_MCTS_vs_Patricia" \
    "$MCTS" "$SFL10" "09_MCTS_vs_SF-L10" \
    "$MCTS" "$SFL5" "10_MCTS_vs_SF-L5" \
    "$MCTS" "$AB" "11_MCTS_vs_AB" \
    "$MCTS" "$BER" "12_MCTS_vs_Berserk" &

# Instance 3: MetalFish-Hybrid matches
run_group "${HOSTS[2]}" 3 \
    "$HYB" "$SFL15" "13_Hybrid_vs_SF-L15" \
    "$HYB" "$BER" "14_Hybrid_vs_Berserk" \
    "$HYB" "$PAT" "15_Hybrid_vs_Patricia" \
    "$HYB" "$LC0" "16_Hybrid_vs_Lc0" \
    "$HYB" "$SF" "17_Hybrid_vs_Stockfish" \
    "$HYB" "$MCTS" "18_Hybrid_vs_MCTS" &

# Instance 4: Cross matches
run_group "${HOSTS[3]}" 4 \
    "$HYB" "$AB" "19_Hybrid_vs_AB" \
    "$HYB" "$SFL10" "20_Hybrid_vs_SF-L10" \
    "$SF" "$LC0" "21_SF_vs_Lc0" \
    "$PAT" "$LC0" "22_Patricia_vs_Lc0" \
    "$AB" "$SFL5" "23_AB_vs_SF-L5" \
    "$SF" "$PAT" "24_SF_vs_Patricia" &

wait

# ============================================================================
# Aggregate
# ============================================================================
echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
cat "$RESULTS_DIR"/*.pgn > "$RESULTS_DIR/all_games.pgn" 2>/dev/null || true
TOTAL=$(grep -c "\[Result " "$RESULTS_DIR/all_games.pgn" 2>/dev/null || echo 0)
echo "Total games: $TOTAL"
echo ""
for pgn in "$RESULTS_DIR"/*.pgn; do
    [ "$(basename "$pgn")" = "all_games.pgn" ] && continue
    label=$(basename "$pgn" .pgn)
    w=$(grep -c '\[Result "1-0"\]' "$pgn" 2>/dev/null || echo 0)
    d=$(grep -c '\[Result "1/2-1/2"\]' "$pgn" 2>/dev/null || echo 0)
    l=$(grep -c '\[Result "0-1"\]' "$pgn" 2>/dev/null || echo 0)
    t=$((w+d+l)); [ $t -eq 0 ] && continue
    printf "  %-35s %dW-%dD-%dL (%d games)\n" "$label" "$w" "$d" "$l" "$t"
done
echo ""
echo "PGNs: $RESULTS_DIR/"
echo "Combined: $RESULTS_DIR/all_games.pgn"
