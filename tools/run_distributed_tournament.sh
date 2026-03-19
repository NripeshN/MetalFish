#!/bin/bash
# ============================================================================
# MetalFish Distributed Tournament via cutechess-cli
# Spreads matches across 4 M1 Ultra EC2 instances
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

# ============================================================================
# Deploy
# ============================================================================
deploy() {
    local host=$1 idx=$2
    echo "[$idx] Deploying to $host..."
    ssh_cmd "$host" "mkdir -p $RDIR/{build,networks,reference/{books,stockfish/src,berserk/src,Patricia/engine,lc0/build/release,cutechess/build}}"
    scp_cmd "$PROJ/build/metalfish" "$USER@$host:$RDIR/build/"
    scp_cmd "$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb" "$USER@$host:$RDIR/networks/"
    scp_cmd "$PROJ/reference/stockfish/src/stockfish" "$USER@$host:$RDIR/reference/stockfish/src/"
    scp_cmd "$PROJ/reference/berserk/src/berserk" "$USER@$host:$RDIR/reference/berserk/src/"
    scp_cmd "$PROJ/reference/Patricia/engine/patricia" "$USER@$host:$RDIR/reference/Patricia/engine/"
    scp_cmd "$PROJ/reference/lc0/build/release/lc0" "$USER@$host:$RDIR/reference/lc0/build/release/"
    scp_cmd "$PROJ/reference/cutechess/build/cutechess-cli" "$USER@$host:$RDIR/reference/cutechess/build/"
    scp_cmd "$PROJ/reference/books/8moves_v3.pgn" "$USER@$host:$RDIR/reference/books/"
    for f in "$PROJ"/build/nn-*.nnue; do [ -f "$f" ] && scp_cmd "$f" "$USER@$host:$RDIR/build/"; done
    ssh_cmd "$host" "chmod +x $RDIR/build/metalfish $RDIR/reference/stockfish/src/stockfish $RDIR/reference/berserk/src/berserk $RDIR/reference/Patricia/engine/patricia $RDIR/reference/lc0/build/release/lc0 $RDIR/reference/cutechess/build/cutechess-cli; cd /opt/homebrew/opt/protobuf/lib 2>/dev/null && sudo ln -sf libprotobuf.34.0.0.dylib libprotobuf.33.4.0.dylib 2>/dev/null; true"
    echo "[$idx] Done"
}

echo "--- Deploying ---"
for i in "${!HOSTS[@]}"; do deploy "${HOSTS[$i]}" "$((i+1))" & done
wait
echo ""

# ============================================================================
# Verify
# ============================================================================
echo "--- Verifying ---"
V=$(ssh_cmd "${HOSTS[0]}" "cd $RDIR && echo 'uci
quit' | build/metalfish 2>&1 | head -1")
echo "  metalfish: $V"
V=$(ssh_cmd "${HOSTS[0]}" "$RDIR/reference/cutechess/build/cutechess-cli --version 2>&1 | head -1")
echo "  cutechess: $V"
echo ""

# ============================================================================
# Engine definitions
# ============================================================================
CC="$RDIR/reference/cutechess/build/cutechess-cli"
BOOK="$RDIR/reference/books/8moves_v3.pgn"
COMMON="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

AB="-engine cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=8 option.Hash=256"
MCTS="-engine cmd=$RDIR/build/metalfish name=MetalFish-MCTS option.Threads=8 option.UseMCTS=true"
HYB="-engine cmd=$RDIR/build/metalfish name=MetalFish-Hybrid option.Threads=8 option.Hash=256 option.UseHybridSearch=true"
SF="-engine cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish option.Threads=8 option.Hash=256"
SFL15="-engine cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L15 option.Threads=8 option.Hash=256 option.Skill_Level=15"
SFL10="-engine cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L10 option.Threads=8 option.Hash=256 option.Skill_Level=10"
SFL5="-engine cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L5 option.Threads=8 option.Hash=256 option.Skill_Level=5"
BER="-engine cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=8 option.Hash=256"
PAT="-engine cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=8 option.Hash=256"
LC0="-engine cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb arg=--backend=metal option.Threads=8 option.Temperature=0"

# ============================================================================
# Run matches
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
    while [ $# -ge 3 ]; do
        run_match "$host" "$idx" "$1" "$2" "$3"; shift 3
    done
}

echo "--- Running 24 matches across 4 instances ---"
echo ""

run_group "${HOSTS[0]}" 1 \
    "$AB" "$MCTS" "01_AB_vs_MCTS" \
    "$AB" "$HYB" "02_AB_vs_Hybrid" \
    "$MCTS" "$HYB" "03_MCTS_vs_Hybrid" \
    "$AB" "$SF" "04_AB_vs_Stockfish" \
    "$AB" "$BER" "05_AB_vs_Berserk" \
    "$AB" "$PAT" "06_AB_vs_Patricia" &

run_group "${HOSTS[1]}" 2 \
    "$MCTS" "$LC0" "07_MCTS_vs_Lc0" \
    "$MCTS" "$PAT" "08_MCTS_vs_Patricia" \
    "$MCTS" "$SFL10" "09_MCTS_vs_SF-L10" \
    "$MCTS" "$SFL5" "10_MCTS_vs_SF-L5" \
    "$MCTS" "$SF" "11_MCTS_vs_Stockfish" \
    "$MCTS" "$BER" "12_MCTS_vs_Berserk" &

run_group "${HOSTS[2]}" 3 \
    "$HYB" "$SFL15" "13_Hybrid_vs_SF-L15" \
    "$HYB" "$BER" "14_Hybrid_vs_Berserk" \
    "$HYB" "$PAT" "15_Hybrid_vs_Patricia" \
    "$HYB" "$LC0" "16_Hybrid_vs_Lc0" \
    "$HYB" "$SF" "17_Hybrid_vs_Stockfish" \
    "$HYB" "$SFL10" "18_Hybrid_vs_SF-L10" &

run_group "${HOSTS[3]}" 4 \
    "$AB" "$LC0" "19_AB_vs_Lc0" \
    "$AB" "$SFL15" "20_AB_vs_SF-L15" \
    "$AB" "$SFL10" "21_AB_vs_SF-L10" \
    "$AB" "$SFL5" "22_AB_vs_SF-L5" \
    "$SF" "$LC0" "23_SF_vs_Lc0" \
    "$PAT" "$LC0" "24_Patricia_vs_Lc0" &

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
