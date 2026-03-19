#!/bin/bash
# ============================================================================
# MetalFish Full Round-Robin Tournament
# 9 engines, 36 matches, 10 games each, 300+0.1 TC
# Distributed across 4 M1 Ultra EC2 instances
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

# Engine definitions
CC="$RDIR/reference/cutechess/build/cutechess-cli"
BK="$RDIR/reference/books/8moves_v3.pgn"
W="$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb"
CA="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

AB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=10 option.Hash=512"
MCTS="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-MCTS option.Threads=10 option.UseMCTS=true option.NNWeights=$W"
HYB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-Hybrid option.Threads=10 option.Hash=512 option.UseHybridSearch=true option.NNWeights=$W"
SF="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish option.Threads=10 option.Hash=512"
SF15='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L15 option.Threads=10 option.Hash=512 "option.Skill Level=15"'
SF10='-engine proto=uci cmd='$RDIR'/reference/stockfish/src/stockfish name=SF-L10 option.Threads=10 option.Hash=512 "option.Skill Level=10"'
BER="-engine proto=uci cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=10 option.Hash=512"
PAT="-engine proto=uci cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=10 option.Hash=512"
LC0="-engine proto=uci cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$W arg=--backend=metal option.Threads=10"

# Helper: run one match
m() { echo "=== $3 ===" && $CC $1 $2 $CA -pgnout $RDIR/$3.pgn; }

# 36 matches split across 4 instances (9 per instance)
# Instance 1: AB plays everyone + some cross
I1="cd $RDIR"
I1="$I1 && $(echo "m '$AB' '$SF' '01_AB_vs_SF'")"
I1="$I1 && $(echo "m '$AB' '$BER' '02_AB_vs_Berserk'")"
I1="$I1 && $(echo "m '$AB' '$PAT' '03_AB_vs_Patricia'")"
I1="$I1 && $(echo "m '$AB' '$SF15' '04_AB_vs_SF15'")"
I1="$I1 && $(echo "m '$AB' '$SF10' '05_AB_vs_SF10'")"
I1="$I1 && $(echo "m '$AB' '$LC0' '06_AB_vs_Lc0'")"
I1="$I1 && $(echo "m '$AB' '$MCTS' '07_AB_vs_MCTS'")"
I1="$I1 && $(echo "m '$AB' '$HYB' '08_AB_vs_Hybrid'")"
I1="$I1 && $(echo "m '$SF' '$BER' '09_SF_vs_Berserk'")"
I1="$I1 && echo '=== Instance 1 COMPLETE ==='"

# Instance 2: MCTS plays everyone + some cross
I2="cd $RDIR"
I2="$I2 && $(echo "m '$MCTS' '$LC0' '10_MCTS_vs_Lc0'")"
I2="$I2 && $(echo "m '$MCTS' '$PAT' '11_MCTS_vs_Patricia'")"
I2="$I2 && $(echo "m '$MCTS' '$SF10' '12_MCTS_vs_SF10'")"
I2="$I2 && $(echo "m '$MCTS' '$SF15' '13_MCTS_vs_SF15'")"
I2="$I2 && $(echo "m '$MCTS' '$BER' '14_MCTS_vs_Berserk'")"
I2="$I2 && $(echo "m '$MCTS' '$SF' '15_MCTS_vs_SF'")"
I2="$I2 && $(echo "m '$MCTS' '$HYB' '16_MCTS_vs_Hybrid'")"
I2="$I2 && $(echo "m '$SF' '$PAT' '17_SF_vs_Patricia'")"
I2="$I2 && $(echo "m '$SF' '$SF15' '18_SF_vs_SF15'")"
I2="$I2 && echo '=== Instance 2 COMPLETE ==='"

# Instance 3: Hybrid plays everyone + some cross
I3="cd $RDIR"
I3="$I3 && $(echo "m '$HYB' '$SF' '19_Hybrid_vs_SF'")"
I3="$I3 && $(echo "m '$HYB' '$BER' '20_Hybrid_vs_Berserk'")"
I3="$I3 && $(echo "m '$HYB' '$PAT' '21_Hybrid_vs_Patricia'")"
I3="$I3 && $(echo "m '$HYB' '$LC0' '22_Hybrid_vs_Lc0'")"
I3="$I3 && $(echo "m '$HYB' '$SF15' '23_Hybrid_vs_SF15'")"
I3="$I3 && $(echo "m '$HYB' '$SF10' '24_Hybrid_vs_SF10'")"
I3="$I3 && $(echo "m '$SF' '$LC0' '25_SF_vs_Lc0'")"
I3="$I3 && $(echo "m '$SF' '$SF10' '26_SF_vs_SF10'")"
I3="$I3 && $(echo "m '$BER' '$PAT' '27_Berserk_vs_Patricia'")"
I3="$I3 && echo '=== Instance 3 COMPLETE ==='"

# Instance 4: remaining cross matches
I4="cd $RDIR"
I4="$I4 && $(echo "m '$LC0' '$PAT' '28_Lc0_vs_Patricia'")"
I4="$I4 && $(echo "m '$LC0' '$BER' '29_Lc0_vs_Berserk'")"
I4="$I4 && $(echo "m '$LC0' '$SF15' '30_Lc0_vs_SF15'")"
I4="$I4 && $(echo "m '$LC0' '$SF10' '31_Lc0_vs_SF10'")"
I4="$I4 && $(echo "m '$SF15' '$PAT' '32_SF15_vs_Patricia'")"
I4="$I4 && $(echo "m '$SF15' '$BER' '33_SF15_vs_Berserk'")"
I4="$I4 && $(echo "m '$SF15' '$SF10' '34_SF15_vs_SF10'")"
I4="$I4 && $(echo "m '$SF10' '$PAT' '35_SF10_vs_Patricia'")"
I4="$I4 && $(echo "m '$SF10' '$BER' '36_SF10_vs_Berserk'")"
I4="$I4 && echo '=== Instance 4 COMPLETE ==='"

echo "============================================"
echo "  MetalFish Full Round-Robin Tournament"
echo "============================================"
echo "Engines: AB, MCTS, Hybrid, Stockfish, SF-L15, SF-L10, Berserk, Patricia, Lc0"
echo "Matches: 36 (full round-robin) | Games: $GAMES/match | TC: $TC"
echo "Total games: $((36 * GAMES))"
echo "Results: $RESULTS_DIR"
echo ""
echo "  Instance 1: 9 matches (AB ladder + SF-vs-Berserk)"
echo "  Instance 2: 9 matches (MCTS ladder + SF cross)"
echo "  Instance 3: 9 matches (Hybrid ladder + cross)"
echo "  Instance 4: 9 matches (Lc0/SF15/SF10 cross)"
echo ""
echo "Running..."
echo ""

# Define m() function on remote via heredoc
MFUNC='m() { echo "=== $3 ===" && '$CC' $1 $2 '$CA' -pgnout '$RDIR'/$3.pgn; }'

ssh_cmd "${HOSTS[0]}" "$MFUNC && $I1" 2>&1 | sed 's/^/[1] /' | tee "$RESULTS_DIR/log1.txt" &
ssh_cmd "${HOSTS[1]}" "$MFUNC && $I2" 2>&1 | sed 's/^/[2] /' | tee "$RESULTS_DIR/log2.txt" &
ssh_cmd "${HOSTS[2]}" "$MFUNC && $I3" 2>&1 | sed 's/^/[3] /' | tee "$RESULTS_DIR/log3.txt" &
ssh_cmd "${HOSTS[3]}" "$MFUNC && $I4" 2>&1 | sed 's/^/[4] /' | tee "$RESULTS_DIR/log4.txt" &
wait

# Collect PGNs
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
echo "Combined PGN: $RESULTS_DIR/all_games.pgn"
