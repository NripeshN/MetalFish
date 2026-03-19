#!/bin/bash
# ============================================================================
# MetalFish Distributed Tournament (4-pane tmux display)
# 16 matches across 4 M1 Ultra instances, 300+0.1 TC
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
ssh_cmd() { ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=60 "$USER@$1" "${@:2}"; }
scp_cmd() { scp -i "$PEM" -o StrictHostKeyChecking=no -q "$@"; }

# Engine definitions
CC="$RDIR/reference/cutechess/build/cutechess-cli"
BOOK="$RDIR/reference/books/8moves_v3.pgn"
W="$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb"
CCARGS="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

AB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=10 option.Hash=512"
MCTS="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-MCTS option.Threads=10 option.UseMCTS=true option.NNWeights=$W"
HYB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-Hybrid option.Threads=10 option.Hash=512 option.UseHybridSearch=true option.NNWeights=$W"

SF="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish option.Threads=10 option.Hash=512"
SF15='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L15 option.Threads=10 option.Hash=512 "option.Skill Level=15"'
SF12='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L12 option.Threads=10 option.Hash=512 "option.Skill Level=12"'
SF10='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L10 option.Threads=10 option.Hash=512 "option.Skill Level=10"'
SF8='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L8 option.Threads=10 option.Hash=512 "option.Skill Level=8"'
SF5='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L5 option.Threads=10 option.Hash=512 "option.Skill Level=5"'

BER="-engine proto=uci cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=10 option.Hash=512"
PAT="-engine proto=uci cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=10 option.Hash=512"
LC0="-engine proto=uci cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$W arg=--backend=metal option.Threads=10"

# Build instance scripts (sequential matches per instance)
# Instance 1: AB Elo ladder (AB vs strong engines)
I1="cd $RDIR"
I1="$I1 && echo '=== [1] AB vs Stockfish ===' && $CC $AB $SF $CCARGS -pgnout $RDIR/01_AB_vs_SF.pgn"
I1="$I1 && echo '=== [1] AB vs Berserk ===' && $CC $AB $BER $CCARGS -pgnout $RDIR/02_AB_vs_Berserk.pgn"
I1="$I1 && echo '=== [1] AB vs SF-L15 ===' && $CC $AB $SF15 $CCARGS -pgnout $RDIR/03_AB_vs_SF15.pgn"
I1="$I1 && echo '=== [1] AB vs Lc0 ===' && $CC $AB $LC0 $CCARGS -pgnout $RDIR/04_AB_vs_Lc0.pgn"
I1="$I1 && echo '=== Instance 1 COMPLETE ==='"

# Instance 2: MCTS Elo ladder (MCTS vs calibrated opponents)
I2="cd $RDIR"
I2="$I2 && echo '=== [2] MCTS vs Lc0 ===' && $CC $MCTS $LC0 $CCARGS -pgnout $RDIR/05_MCTS_vs_Lc0.pgn"
I2="$I2 && echo '=== [2] MCTS vs SF-L10 ===' && $CC $MCTS $SF10 $CCARGS -pgnout $RDIR/06_MCTS_vs_SF10.pgn"
I2="$I2 && echo '=== [2] MCTS vs SF-L8 ===' && $CC $MCTS $SF8 $CCARGS -pgnout $RDIR/07_MCTS_vs_SF8.pgn"
I2="$I2 && echo '=== [2] MCTS vs Patricia ===' && $CC $MCTS $PAT $CCARGS -pgnout $RDIR/08_MCTS_vs_Patricia.pgn"
I2="$I2 && echo '=== Instance 2 COMPLETE ==='"

# Instance 3: Hybrid Elo ladder
I3="cd $RDIR"
I3="$I3 && echo '=== [3] Hybrid vs SF-L15 ===' && $CC $HYB $SF15 $CCARGS -pgnout $RDIR/09_Hybrid_vs_SF15.pgn"
I3="$I3 && echo '=== [3] Hybrid vs SF-L12 ===' && $CC $HYB $SF12 $CCARGS -pgnout $RDIR/10_Hybrid_vs_SF12.pgn"
I3="$I3 && echo '=== [3] Hybrid vs Patricia ===' && $CC $HYB $PAT $CCARGS -pgnout $RDIR/11_Hybrid_vs_Patricia.pgn"
I3="$I3 && echo '=== [3] Hybrid vs Lc0 ===' && $CC $HYB $LC0 $CCARGS -pgnout $RDIR/12_Hybrid_vs_Lc0.pgn"
I3="$I3 && echo '=== Instance 3 COMPLETE ==='"

# Instance 4: Internal head-to-head + calibration
I4="cd $RDIR"
I4="$I4 && echo '=== [4] AB vs Hybrid ===' && $CC $AB $HYB $CCARGS -pgnout $RDIR/13_AB_vs_Hybrid.pgn"
I4="$I4 && echo '=== [4] AB vs MCTS ===' && $CC $AB $MCTS $CCARGS -pgnout $RDIR/14_AB_vs_MCTS.pgn"
I4="$I4 && echo '=== [4] Hybrid vs MCTS ===' && $CC $HYB $MCTS $CCARGS -pgnout $RDIR/15_Hybrid_vs_MCTS.pgn"
I4="$I4 && echo '=== [4] SF vs Lc0 ===' && $CC $SF $LC0 $CCARGS -pgnout $RDIR/16_SF_vs_Lc0.pgn"
I4="$I4 && echo '=== Instance 4 COMPLETE ==='"

echo "============================================"
echo "  MetalFish Distributed Tournament"
echo "============================================"
echo "Games: $GAMES/match | TC: $TC | Matches: 16"
echo "Results: $RESULTS_DIR"
echo ""
echo "  Instance 1: AB vs SF, Berserk, SF-L15, Lc0"
echo "  Instance 2: MCTS vs Lc0, SF-L10, SF-L8, Patricia"
echo "  Instance 3: Hybrid vs SF-L15, SF-L12, Patricia, Lc0"
echo "  Instance 4: AB-vs-Hybrid, AB-vs-MCTS, Hybrid-vs-MCTS, SF-vs-Lc0"
echo ""

echo "Running 4 instances in parallel..."
echo "Logs: $RESULTS_DIR/log{1..4}.txt"
echo ""

ssh_cmd "${HOSTS[0]}" "$I1" 2>&1 | sed 's/^/[1] /' | tee "$RESULTS_DIR/log1.txt" &
ssh_cmd "${HOSTS[1]}" "$I2" 2>&1 | sed 's/^/[2] /' | tee "$RESULTS_DIR/log2.txt" &
ssh_cmd "${HOSTS[2]}" "$I3" 2>&1 | sed 's/^/[3] /' | tee "$RESULTS_DIR/log3.txt" &
ssh_cmd "${HOSTS[3]}" "$I4" 2>&1 | sed 's/^/[4] /' | tee "$RESULTS_DIR/log4.txt" &
wait

# Collect results
echo ""
echo "--- Collecting PGNs ---"
for host in "${HOSTS[@]}"; do
    scp_cmd $USER@$host:$RDIR/*.pgn "$RESULTS_DIR/" 2>/dev/null || true
done

cat "$RESULTS_DIR"/*.pgn > "$RESULTS_DIR/all_games.pgn" 2>/dev/null || true
TOTAL=$(grep -c "\[Result " "$RESULTS_DIR/all_games.pgn" 2>/dev/null || echo 0)

echo ""
echo "============================================"
echo "  RESULTS ($TOTAL games)"
echo "============================================"
printf "\n  %-30s %4s %4s %4s %7s\n" "Match" "W" "D" "L" "Score"
echo "  $(printf '%.0s-' {1..55})"
for pgn in "$RESULTS_DIR"/*.pgn; do
    [ "$(basename "$pgn")" = "all_games.pgn" ] && continue
    label=$(basename "$pgn" .pgn)
    w=$(grep -c '\[Result "1-0"\]' "$pgn" 2>/dev/null || echo 0)
    d=$(grep -c '\[Result "1/2-1/2"\]' "$pgn" 2>/dev/null || echo 0)
    l=$(grep -c '\[Result "0-1"\]' "$pgn" 2>/dev/null || echo 0)
    t=$((w+d+l)); [ $t -eq 0 ] && continue
    s=$(echo "scale=1; $w + $d * 0.5" | bc)
    printf "  %-30s %4d %4d %4d %5s/%d\n" "$label" "$w" "$d" "$l" "$s" "$t"
done
echo ""
echo "Combined PGN: $RESULTS_DIR/all_games.pgn"
