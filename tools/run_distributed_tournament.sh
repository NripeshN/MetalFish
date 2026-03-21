#!/bin/bash
# ============================================================================
# MetalFish Tournament: All 3 engines vs 10 opponents + internal H2H
# 33 matches × 10 games = 330 games across 4 M1 Ultra instances
# ============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(44.220.150.2 98.81.229.157 98.84.106.208 32.192.83.249)
USER=ec2-user
RDIR="/Users/ec2-user/metalfish-src"
RESULTS_DIR="$PROJ/results/distributed_$(date +%Y%m%d_%H%M%S)"
GAMES=10
TC="300+0.1"

for arg in "$@"; do
    case $arg in
        --quick) GAMES=2; TC="10+0.1" ;;
        --games=*) GAMES="${arg#*=}" ;;
        --tc=*) TC="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_DIR"
ssh_cmd() { ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=60 "$USER@$1" "${@:2}"; }
scp_cmd() { scp -i "$PEM" -o StrictHostKeyChecking=no -q "$@"; }

CC="$RDIR/reference/cutechess/build/cutechess-cli"
BK="$RDIR/reference/books/8moves_v3.pgn"
W="$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb"
CA="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

# Our engines — optimal thread configs per engine
# AB: scales well with threads, give it all available cores
# MCTS: best at 2 threads (more causes collisions), GPU does heavy lifting
# Hybrid: all 20 cores — internally splits to 18 AB + 2 MCTS + GPU
AB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=18 option.Hash=512"
MCTS="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-MCTS option.Threads=2 option.UseMCTS=true option.NNWeights=$W"
HYB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-Hybrid option.Threads=20 option.Hash=512 option.UseHybridSearch=true option.NNWeights=$W"

# Opponents — 1 thread each for controlled comparison
S="$RDIR/reference/stockfish/src/stockfish"
SF="-engine proto=uci cmd=$S name=Stockfish option.Threads=1 option.Hash=512"
S16='-engine proto=uci cmd='$S' name=SF-L16 option.Threads=1 option.Hash=512 "option.Skill Level=16"'
S14='-engine proto=uci cmd='$S' name=SF-L14 option.Threads=1 option.Hash=512 "option.Skill Level=14"'
S12='-engine proto=uci cmd='$S' name=SF-L12 option.Threads=1 option.Hash=512 "option.Skill Level=12"'
S10='-engine proto=uci cmd='$S' name=SF-L10 option.Threads=1 option.Hash=512 "option.Skill Level=10"'
S8='-engine proto=uci cmd='$S' name=SF-L8 option.Threads=1 option.Hash=512 "option.Skill Level=8"'
S5='-engine proto=uci cmd='$S' name=SF-L5 option.Threads=1 option.Hash=512 "option.Skill Level=5"'
BER="-engine proto=uci cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=1 option.Hash=512"
PAT="-engine proto=uci cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=1 option.Hash=512"
LC0="-engine proto=uci cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$W arg=--backend=metal option.Threads=1"

# Instance 1 (9 matches): AB vs everyone except Lc0
I1="cd $RDIR"
I1="$I1 && echo '>>> AB vs Stockfish' && $CC $AB $SF $CA -pgnout $RDIR/01_AB_vs_SF.pgn"
I1="$I1 && echo '>>> AB vs SF-L16' && $CC $AB $S16 $CA -pgnout $RDIR/02_AB_vs_SF16.pgn"
I1="$I1 && echo '>>> AB vs SF-L14' && $CC $AB $S14 $CA -pgnout $RDIR/03_AB_vs_SF14.pgn"
I1="$I1 && echo '>>> AB vs SF-L12' && $CC $AB $S12 $CA -pgnout $RDIR/04_AB_vs_SF12.pgn"
I1="$I1 && echo '>>> AB vs SF-L10' && $CC $AB $S10 $CA -pgnout $RDIR/05_AB_vs_SF10.pgn"
I1="$I1 && echo '>>> AB vs SF-L8' && $CC $AB $S8 $CA -pgnout $RDIR/06_AB_vs_SF8.pgn"
I1="$I1 && echo '>>> AB vs SF-L5' && $CC $AB $S5 $CA -pgnout $RDIR/07_AB_vs_SF5.pgn"
I1="$I1 && echo '>>> AB vs Berserk' && $CC $AB $BER $CA -pgnout $RDIR/08_AB_vs_Berserk.pgn"
I1="$I1 && echo '>>> AB vs Patricia' && $CC $AB $PAT $CA -pgnout $RDIR/09_AB_vs_Patricia.pgn"
I1="$I1 && echo '=== Instance 1 DONE ==='"

# Instance 2 (9 matches): MCTS vs everyone except SF/SF16
I2="cd $RDIR"
I2="$I2 && echo '>>> MCTS vs Lc0' && $CC $MCTS $LC0 $CA -pgnout $RDIR/10_MCTS_vs_Lc0.pgn"
I2="$I2 && echo '>>> MCTS vs SF-L14' && $CC $MCTS $S14 $CA -pgnout $RDIR/11_MCTS_vs_SF14.pgn"
I2="$I2 && echo '>>> MCTS vs SF-L12' && $CC $MCTS $S12 $CA -pgnout $RDIR/12_MCTS_vs_SF12.pgn"
I2="$I2 && echo '>>> MCTS vs SF-L10' && $CC $MCTS $S10 $CA -pgnout $RDIR/13_MCTS_vs_SF10.pgn"
I2="$I2 && echo '>>> MCTS vs SF-L8' && $CC $MCTS $S8 $CA -pgnout $RDIR/14_MCTS_vs_SF8.pgn"
I2="$I2 && echo '>>> MCTS vs SF-L5' && $CC $MCTS $S5 $CA -pgnout $RDIR/15_MCTS_vs_SF5.pgn"
I2="$I2 && echo '>>> MCTS vs Berserk' && $CC $MCTS $BER $CA -pgnout $RDIR/16_MCTS_vs_Berserk.pgn"
I2="$I2 && echo '>>> MCTS vs Patricia' && $CC $MCTS $PAT $CA -pgnout $RDIR/17_MCTS_vs_Patricia.pgn"
I2="$I2 && echo '>>> MCTS vs Stockfish' && $CC $MCTS $SF $CA -pgnout $RDIR/18_MCTS_vs_SF.pgn"
I2="$I2 && echo '=== Instance 2 DONE ==='"

# Instance 3 (9 matches): Hybrid vs everyone except SF8/SF5
I3="cd $RDIR"
I3="$I3 && echo '>>> Hybrid vs Stockfish' && $CC $HYB $SF $CA -pgnout $RDIR/19_Hybrid_vs_SF.pgn"
I3="$I3 && echo '>>> Hybrid vs SF-L16' && $CC $HYB $S16 $CA -pgnout $RDIR/20_Hybrid_vs_SF16.pgn"
I3="$I3 && echo '>>> Hybrid vs SF-L14' && $CC $HYB $S14 $CA -pgnout $RDIR/21_Hybrid_vs_SF14.pgn"
I3="$I3 && echo '>>> Hybrid vs SF-L12' && $CC $HYB $S12 $CA -pgnout $RDIR/22_Hybrid_vs_SF12.pgn"
I3="$I3 && echo '>>> Hybrid vs SF-L10' && $CC $HYB $S10 $CA -pgnout $RDIR/23_Hybrid_vs_SF10.pgn"
I3="$I3 && echo '>>> Hybrid vs Berserk' && $CC $HYB $BER $CA -pgnout $RDIR/24_Hybrid_vs_Berserk.pgn"
I3="$I3 && echo '>>> Hybrid vs Patricia' && $CC $HYB $PAT $CA -pgnout $RDIR/25_Hybrid_vs_Patricia.pgn"
I3="$I3 && echo '>>> Hybrid vs Lc0' && $CC $HYB $LC0 $CA -pgnout $RDIR/26_Hybrid_vs_Lc0.pgn"
I3="$I3 && echo '>>> Hybrid vs SF-L8' && $CC $HYB $S8 $CA -pgnout $RDIR/27_Hybrid_vs_SF8.pgn"
I3="$I3 && echo '=== Instance 3 DONE ==='"

# Instance 4 (6 matches): Internal H2H + remaining
I4="cd $RDIR"
I4="$I4 && echo '>>> AB vs MCTS' && $CC $AB $MCTS $CA -pgnout $RDIR/28_AB_vs_MCTS.pgn"
I4="$I4 && echo '>>> AB vs Hybrid' && $CC $AB $HYB $CA -pgnout $RDIR/29_AB_vs_Hybrid.pgn"
I4="$I4 && echo '>>> Hybrid vs MCTS' && $CC $HYB $MCTS $CA -pgnout $RDIR/30_Hybrid_vs_MCTS.pgn"
I4="$I4 && echo '>>> AB vs Lc0' && $CC $AB $LC0 $CA -pgnout $RDIR/31_AB_vs_Lc0.pgn"
I4="$I4 && echo '>>> MCTS vs SF-L16' && $CC $MCTS $S16 $CA -pgnout $RDIR/32_MCTS_vs_SF16.pgn"
I4="$I4 && echo '>>> Hybrid vs SF-L5' && $CC $HYB $S5 $CA -pgnout $RDIR/33_Hybrid_vs_SF5.pgn"
I4="$I4 && echo '=== Instance 4 DONE ==='"

echo "============================================"
echo "  MetalFish Engine Evaluation Tournament"
echo "============================================"
echo ""
echo "  Our engines: MetalFish-AB, MetalFish-MCTS, MetalFish-Hybrid"
echo "  Opponents:   Stockfish, SF-L16, SF-L14, SF-L12, SF-L10,"
echo "               SF-L8, SF-L5, Berserk, Patricia, Lc0"
echo ""
echo "  33 matches × $GAMES games = $((33 * GAMES)) games | TC: $TC"
echo "  Results: $RESULTS_DIR"
echo ""
echo "  [1] AB vs 9 opponents       (~3100-3800 Elo ladder)"
echo "  [2] MCTS vs 9 opponents     (full strength range)"
echo "  [3] Hybrid vs 9 opponents   (full strength range)"
echo "  [4] Internal H2H + 3 extras (AB/MCTS/Hybrid vs each other)"
echo ""
echo "  Estimated time: ~13 hours"
echo ""

ssh_cmd "${HOSTS[0]}" "$I1" 2>&1 | sed 's/^/[1] /' | tee "$RESULTS_DIR/log1.txt" &
ssh_cmd "${HOSTS[1]}" "$I2" 2>&1 | sed 's/^/[2] /' | tee "$RESULTS_DIR/log2.txt" &
ssh_cmd "${HOSTS[2]}" "$I3" 2>&1 | sed 's/^/[3] /' | tee "$RESULTS_DIR/log3.txt" &
ssh_cmd "${HOSTS[3]}" "$I4" 2>&1 | sed 's/^/[4] /' | tee "$RESULTS_DIR/log4.txt" &
wait

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
printf "\n  %-35s %4s %4s %4s %7s\n" "Match" "W" "D" "L" "Score"
echo "  $(printf '%.0s-' {1..60})"
for pgn in "$RESULTS_DIR"/*.pgn; do
    [ "$(basename "$pgn")" = "all_games.pgn" ] && continue
    label=$(basename "$pgn" .pgn)
    w=$(grep -c '\[Result "1-0"\]' "$pgn" 2>/dev/null || echo 0)
    d=$(grep -c '\[Result "1/2-1/2"\]' "$pgn" 2>/dev/null || echo 0)
    l=$(grep -c '\[Result "0-1"\]' "$pgn" 2>/dev/null || echo 0)
    t=$((w+d+l)); [ $t -eq 0 ] && continue
    s=$(echo "scale=1; $w + $d * 0.5" | bc)
    printf "  %-35s %4d %4d %4d %5s/%d\n" "$label" "$w" "$d" "$l" "$s" "$t"
done
echo ""
echo "Combined: $RESULTS_DIR/all_games.pgn"
