#!/bin/bash
# ============================================================================
# MetalFish Distributed Tournament via cutechess-cli
# CPU-only engines on EC2, GPU engines locally
# ============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(44.220.150.2 98.81.229.157 98.84.106.208 32.192.83.249)
USER=ec2-user
RDIR="/Users/ec2-user/metalfish"
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
echo "NOTE: GPU engines (MCTS, Hybrid, Lc0) run locally."
echo "      CPU engines (AB, Stockfish, Berserk, Patricia) run on EC2."
echo ""

# ============================================================================
# Deploy to EC2 instances (CPU engines only)
# ============================================================================
deploy() {
    local host=$1 idx=$2
    echo "[$idx] Deploying to $host..."
    ssh_cmd "$host" "mkdir -p $RDIR/{build,networks,reference/{books,stockfish/src,berserk/src,Patricia/engine,cutechess/build}}"
    scp_cmd "$PROJ/build/metalfish" "$USER@$host:$RDIR/build/"
    scp_cmd "$PROJ/reference/stockfish/src/stockfish" "$USER@$host:$RDIR/reference/stockfish/src/"
    scp_cmd "$PROJ/reference/berserk/src/berserk" "$USER@$host:$RDIR/reference/berserk/src/"
    scp_cmd "$PROJ/reference/Patricia/engine/patricia" "$USER@$host:$RDIR/reference/Patricia/engine/"
    scp_cmd "$PROJ/reference/cutechess/build/cutechess-cli" "$USER@$host:$RDIR/reference/cutechess/build/"
    scp_cmd "$PROJ/reference/books/8moves_v3.pgn" "$USER@$host:$RDIR/reference/books/"
    for f in "$PROJ"/build/nn-*.nnue; do [ -f "$f" ] && scp_cmd "$f" "$USER@$host:$RDIR/build/"; done
    ssh_cmd "$host" "chmod +x $RDIR/build/metalfish $RDIR/reference/stockfish/src/stockfish $RDIR/reference/berserk/src/berserk $RDIR/reference/Patricia/engine/patricia $RDIR/reference/cutechess/build/cutechess-cli; cd /opt/homebrew/opt/protobuf/lib 2>/dev/null && sudo ln -sf libprotobuf.34.0.0.dylib libprotobuf.33.4.0.dylib 2>/dev/null; true"
    echo "[$idx] Done"
}

echo "--- Deploying ---"
for i in "${!HOSTS[@]}"; do deploy "${HOSTS[$i]}" "$((i+1))" & done
wait
echo ""

# ============================================================================
# Engine definitions for REMOTE (CPU-only, 10 threads each)
# ============================================================================
CC="$RDIR/reference/cutechess/build/cutechess-cli"
BOOK_R="$RDIR/reference/books/8moves_v3.pgn"
COMMON_R="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK_R format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

AB_R="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=10 option.Hash=512"
SF_R="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish option.Threads=10 option.Hash=512"
SFL15_R="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L15 option.Threads=10 option.Hash=512 option.\"Skill Level\"=15"
SFL10_R="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L10 option.Threads=10 option.Hash=512 option.\"Skill Level\"=10"
SFL5_R="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L5 option.Threads=10 option.Hash=512 option.\"Skill Level\"=5"
BER_R="-engine proto=uci cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=10 option.Hash=512"
PAT_R="-engine proto=uci cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=10 option.Hash=512"

# ============================================================================
# Engine definitions for LOCAL (GPU engines, 8 threads)
# ============================================================================
LOCAL_CC="$PROJ/reference/cutechess/build/cutechess-cli"
BOOK_L="$PROJ/reference/books/8moves_v3.pgn"
WEIGHTS="$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb"
COMMON_L="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK_L format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

MCTS_L="-engine proto=uci cmd=$PROJ/build/metalfish name=MetalFish-MCTS option.Threads=8 option.UseMCTS=true option.NNWeights=$WEIGHTS"
HYB_L="-engine proto=uci cmd=$PROJ/build/metalfish name=MetalFish-Hybrid option.Threads=8 option.Hash=256 option.UseHybridSearch=true option.NNWeights=$WEIGHTS"
AB_L="-engine proto=uci cmd=$PROJ/build/metalfish name=MetalFish-AB option.Threads=8 option.Hash=256"
SF_L="-engine proto=uci cmd=$PROJ/reference/stockfish/src/stockfish name=Stockfish option.Threads=8 option.Hash=256"
SFL15_L="-engine proto=uci cmd=$PROJ/reference/stockfish/src/stockfish name=Stockfish-L15 option.Threads=8 option.Hash=256 option.\"Skill Level\"=15"
PAT_L="-engine proto=uci cmd=$PROJ/reference/Patricia/engine/patricia name=Patricia option.Threads=8 option.Hash=256"
LC0_L="-engine proto=uci cmd=$PROJ/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$WEIGHTS arg=--backend=metal option.Threads=8 option.Temperature=0"
BER_L="-engine proto=uci cmd=$PROJ/reference/berserk/src/berserk name=Berserk option.Threads=8 option.Hash=256"

# ============================================================================
# Match runners
# ============================================================================
run_remote() {
    local host=$1 idx=$2 e1="$3" e2="$4" label="$5"
    echo "[$idx] REMOTE $label on $host"
    ssh_cmd "$host" "cd $RDIR && $CC $e1 $e2 $COMMON_R -pgnout $RDIR/$label.pgn 2>&1" | tail -3
    scp_cmd "$USER@$host:$RDIR/$label.pgn" "$RESULTS_DIR/" 2>/dev/null || true
    echo "[$idx] Done: $label"
}

run_local() {
    local e1="$1" e2="$2" label="$3"
    echo "[L] LOCAL $label"
    $LOCAL_CC $e1 $e2 $COMMON_L -pgnout "$RESULTS_DIR/$label.pgn" 2>&1 | tail -3
    echo "[L] Done: $label"
}

run_remote_group() {
    local host=$1 idx=$2; shift 2
    while [ $# -ge 3 ]; do run_remote "$host" "$idx" "$1" "$2" "$3"; shift 3; done
}

run_local_group() {
    while [ $# -ge 3 ]; do run_local "$1" "$2" "$3"; shift 3; done
}

echo "--- Running matches ---"
echo ""

# REMOTE: AB vs CPU engines (distributed across 4 instances)
run_remote_group "${HOSTS[0]}" 1 \
    "$AB_R" "$SF_R" "01_AB_vs_Stockfish" \
    "$AB_R" "$BER_R" "02_AB_vs_Berserk" \
    "$AB_R" "$PAT_R" "03_AB_vs_Patricia" \
    "$AB_R" "$SFL15_R" "04_AB_vs_SF-L15" \
    "$AB_R" "$SFL10_R" "05_AB_vs_SF-L10" \
    "$AB_R" "$SFL5_R" "06_AB_vs_SF-L5" &

run_remote_group "${HOSTS[1]}" 2 \
    "$SF_R" "$BER_R" "07_SF_vs_Berserk" \
    "$SF_R" "$PAT_R" "08_SF_vs_Patricia" \
    "$PAT_R" "$BER_R" "09_Patricia_vs_Berserk" \
    "$SFL15_R" "$PAT_R" "10_SF-L15_vs_Patricia" \
    "$SFL15_R" "$BER_R" "11_SF-L15_vs_Berserk" \
    "$SFL10_R" "$BER_R" "12_SF-L10_vs_Berserk" &

# LOCAL: GPU engine matches (run sequentially on laptop)
run_local_group \
    "$MCTS_L" "$LC0_L" "13_MCTS_vs_Lc0" \
    "$MCTS_L" "$PAT_L" "14_MCTS_vs_Patricia" \
    "$MCTS_L" "$AB_L" "15_MCTS_vs_AB" \
    "$HYB_L" "$AB_L" "16_Hybrid_vs_AB" \
    "$HYB_L" "$PAT_L" "17_Hybrid_vs_Patricia" \
    "$HYB_L" "$SF_L" "18_Hybrid_vs_Stockfish" \
    "$HYB_L" "$SFL15_L" "19_Hybrid_vs_SF-L15" \
    "$HYB_L" "$LC0_L" "20_Hybrid_vs_Lc0" \
    "$HYB_L" "$BER_L" "21_Hybrid_vs_Berserk" \
    "$HYB_L" "$MCTS_L" "22_Hybrid_vs_MCTS" \
    "$MCTS_L" "$SFL15_L" "23_MCTS_vs_SF-L15" \
    "$AB_L" "$LC0_L" "24_AB_vs_Lc0" &

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
