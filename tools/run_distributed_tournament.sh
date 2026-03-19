#!/bin/bash
# ============================================================================
# MetalFish Distributed Tournament
# Spreads matches across multiple M1 Ultra EC2 instances
# ============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(44.220.150.2 98.81.229.157 98.84.106.208 32.192.83.249)
USER=ec2-user
REMOTE_DIR="~/metalfish"
RESULTS_DIR="$PROJ/results/distributed_$(date +%Y%m%d_%H%M%S)"
GAMES=20
TC="300+0.1"
BOOK="reference/books/8moves_v3.pgn"

# Parse args
QUICK=0
for arg in "$@"; do
    case $arg in
        --quick) QUICK=1; GAMES=4; TC="10+0.1" ;;
        --games=*) GAMES="${arg#*=}" ;;
    esac
done

SSH="ssh -i \"$PEM\" -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP="scp -i \"$PEM\" -o StrictHostKeyChecking=no"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  MetalFish Distributed Tournament"
echo "============================================"
echo "Instances: ${#HOSTS[@]}"
echo "Games/match: $GAMES | TC: $TC"
echo "Results: $RESULTS_DIR"
echo ""

# ============================================================================
# Phase 1: Deploy to all instances
# ============================================================================
deploy_instance() {
    local host=$1
    local idx=$2
    echo "[$idx] Deploying to $host..."

    # Create remote dir structure
    eval $SSH $USER@$host "mkdir -p $REMOTE_DIR/{build,networks,reference/books,reference/stockfish/src,reference/berserk/src,reference/Patricia/engine,reference/lc0/build/release,reference/cutechess/build}" 2>/dev/null

    # Sync binaries (only if changed)
    eval $SCP "$PROJ/build/metalfish" $USER@$host:$REMOTE_DIR/build/ 2>/dev/null
    eval $SCP "$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb" $USER@$host:$REMOTE_DIR/networks/ 2>/dev/null
    eval $SCP "$PROJ/reference/stockfish/src/stockfish" $USER@$host:$REMOTE_DIR/reference/stockfish/src/ 2>/dev/null
    eval $SCP "$PROJ/reference/berserk/src/berserk" $USER@$host:$REMOTE_DIR/reference/berserk/src/ 2>/dev/null
    eval $SCP "$PROJ/reference/Patricia/engine/patricia" $USER@$host:$REMOTE_DIR/reference/Patricia/engine/ 2>/dev/null
    eval $SCP "$PROJ/reference/lc0/build/release/lc0" $USER@$host:$REMOTE_DIR/reference/lc0/build/release/ 2>/dev/null
    eval $SCP "$PROJ/reference/cutechess/build/cutechess-cli" $USER@$host:$REMOTE_DIR/reference/cutechess/build/ 2>/dev/null
    eval $SCP "$PROJ/$BOOK" $USER@$host:$REMOTE_DIR/$BOOK 2>/dev/null

    # Copy NNUE files for AB engine
    for f in "$PROJ"/build/nn-*.nnue; do
        [ -f "$f" ] && eval $SCP "$f" $USER@$host:$REMOTE_DIR/build/ 2>/dev/null
    done

    # Set executable permissions
    eval $SSH $USER@$host "chmod +x $REMOTE_DIR/build/metalfish $REMOTE_DIR/reference/stockfish/src/stockfish $REMOTE_DIR/reference/berserk/src/berserk $REMOTE_DIR/reference/Patricia/engine/patricia $REMOTE_DIR/reference/lc0/build/release/lc0 $REMOTE_DIR/reference/cutechess/build/cutechess-cli 2>/dev/null" 2>/dev/null

    echo "[$idx] Deploy complete: $host"
}

echo "--- Phase 1: Deploying engines to all instances ---"
for i in "${!HOSTS[@]}"; do
    deploy_instance "${HOSTS[$i]}" "$((i+1))" &
done
wait
echo "All instances deployed."
echo ""

# ============================================================================
# Phase 2: Define matches and distribute
# ============================================================================

# Engine definitions (relative to REMOTE_DIR)
MF="cmd=build/metalfish name=MetalFish-AB option.Threads=8 option.Hash=256"
MCTS="cmd=build/metalfish name=MetalFish-MCTS option.Threads=8 option.UseMCTS=true"
HYB="cmd=build/metalfish name=MetalFish-Hybrid option.Threads=8 option.Hash=256 option.UseHybridSearch=true"
SF="cmd=reference/stockfish/src/stockfish name=Stockfish option.Threads=8 option.Hash=256"
SFL15="cmd=reference/stockfish/src/stockfish name=Stockfish-L15 option.Threads=8 option.Hash=256 option.\"Skill Level\"=15"
SFL10="cmd=reference/stockfish/src/stockfish name=Stockfish-L10 option.Threads=8 option.Hash=256 option.\"Skill Level\"=10"
SFL5="cmd=reference/stockfish/src/stockfish name=Stockfish-L5 option.Threads=8 option.Hash=256 option.\"Skill Level\"=5"
BERSERK="cmd=reference/berserk/src/berserk name=Berserk option.Threads=8 option.Hash=256"
PATRICIA="cmd=reference/Patricia/engine/patricia name=Patricia option.Threads=8 option.Hash=256"
LC0="cmd=reference/lc0/build/release/lc0 name=Lc0 arg=--weights=networks/BT4-1024x15x32h-swa-6147500.pb arg=--backend=metal option.Threads=8 option.Temperature=0"

COMMON="-each tc=$TC -games $GAMES -repeat -recover \
    -openings file=$BOOK format=pgn order=random \
    -resign movecount=3 score=1000 twosided=true \
    -draw movenumber=40 movecount=8 score=10"

# All matches (split into 4 groups for 4 instances)
# Group 1: MetalFish internal + AB vs external
declare -a GROUP1=(
    "$MF|$MCTS|01_AB_vs_MCTS"
    "$MF|$HYB|02_AB_vs_Hybrid"
    "$MCTS|$HYB|03_MCTS_vs_Hybrid"
    "$MF|$SF|04_AB_vs_Stockfish"
    "$MF|$BERSERK|05_AB_vs_Berserk"
    "$MF|$PATRICIA|06_AB_vs_Patricia"
)

# Group 2: MCTS vs external
declare -a GROUP2=(
    "$MCTS|$LC0|07_MCTS_vs_Lc0"
    "$MCTS|$PATRICIA|08_MCTS_vs_Patricia"
    "$MCTS|$SFL10|09_MCTS_vs_SF-L10"
    "$MCTS|$SFL5|10_MCTS_vs_SF-L5"
    "$MCTS|$SF|11_MCTS_vs_Stockfish"
    "$MCTS|$BERSERK|12_MCTS_vs_Berserk"
)

# Group 3: Hybrid vs external
declare -a GROUP3=(
    "$HYB|$SFL15|13_Hybrid_vs_SF-L15"
    "$HYB|$BERSERK|14_Hybrid_vs_Berserk"
    "$HYB|$PATRICIA|15_Hybrid_vs_Patricia"
    "$HYB|$LC0|16_Hybrid_vs_Lc0"
    "$HYB|$SF|17_Hybrid_vs_Stockfish"
    "$HYB|$SFL10|18_Hybrid_vs_SF-L10"
)

# Group 4: Reference engine matches + remaining
declare -a GROUP4=(
    "$MF|$LC0|19_AB_vs_Lc0"
    "$MF|$SFL15|20_AB_vs_SF-L15"
    "$MF|$SFL10|21_AB_vs_SF-L10"
    "$MF|$SFL5|22_AB_vs_SF-L5"
    "$SF|$LC0|23_SF_vs_Lc0"
    "$PATRICIA|$LC0|24_Patricia_vs_Lc0"
)

# ============================================================================
# Phase 3: Run matches on each instance
# ============================================================================
run_group_on_instance() {
    local host=$1
    local idx=$2
    shift 2
    local matches=("$@")

    echo "[$idx] Starting ${#matches[@]} matches on $host"

    for match_spec in "${matches[@]}"; do
        IFS='|' read -r eng1 eng2 label <<< "$match_spec"
        echo "[$idx] Running: $label"

        eval $SSH $USER@$host "cd $REMOTE_DIR && reference/cutechess/build/cutechess-cli \
            -engine $eng1 -engine $eng2 $COMMON \
            -pgnout ${label}.pgn" 2>/dev/null

        # Copy result back
        eval $SCP $USER@$host:$REMOTE_DIR/${label}.pgn "$RESULTS_DIR/" 2>/dev/null
        echo "[$idx] Done: $label -> $RESULTS_DIR/${label}.pgn"
    done

    echo "[$idx] All matches complete on $host"
}

echo "--- Phase 3: Running tournament across 4 instances ---"
echo ""

run_group_on_instance "${HOSTS[0]}" 1 "${GROUP1[@]}" &
PID1=$!
run_group_on_instance "${HOSTS[1]}" 2 "${GROUP2[@]}" &
PID2=$!
run_group_on_instance "${HOSTS[2]}" 3 "${GROUP3[@]}" &
PID3=$!
run_group_on_instance "${HOSTS[3]}" 4 "${GROUP4[@]}" &
PID4=$!

# ============================================================================
# Phase 4: Monitor progress
# ============================================================================
echo "Waiting for all instances to complete..."
echo "Monitor: tail -f $RESULTS_DIR/*.pgn"
echo ""

wait $PID1 && echo "[1] Instance 1 COMPLETE" || echo "[1] Instance 1 FAILED"
wait $PID2 && echo "[2] Instance 2 COMPLETE" || echo "[2] Instance 2 FAILED"
wait $PID3 && echo "[3] Instance 3 COMPLETE" || echo "[3] Instance 3 FAILED"
wait $PID4 && echo "[4] Instance 4 COMPLETE" || echo "[4] Instance 4 FAILED"

# ============================================================================
# Phase 5: Aggregate results
# ============================================================================
echo ""
echo "============================================"
echo "  AGGREGATING RESULTS"
echo "============================================"

# Merge all PGN files
cat "$RESULTS_DIR"/*.pgn > "$RESULTS_DIR/all_games.pgn" 2>/dev/null

# Count games and results
TOTAL=$(grep -c "\[Result " "$RESULTS_DIR/all_games.pgn" 2>/dev/null || echo 0)
echo "Total games played: $TOTAL"
echo "PGN files: $RESULTS_DIR/"
echo ""

# Print summary per match
echo "Match Results:"
echo "---"
for pgn in "$RESULTS_DIR"/*.pgn; do
    [ "$pgn" = "$RESULTS_DIR/all_games.pgn" ] && continue
    label=$(basename "$pgn" .pgn)
    w1=$(grep -c '\[Result "1-0"\]' "$pgn" 2>/dev/null || echo 0)
    draws=$(grep -c '\[Result "1/2-1/2"\]' "$pgn" 2>/dev/null || echo 0)
    w2=$(grep -c '\[Result "0-1"\]' "$pgn" 2>/dev/null || echo 0)
    total=$((w1 + draws + w2))
    [ $total -eq 0 ] && continue
    echo "  $label: ${w1}W-${draws}D-${w2}L ($total games)"
done

echo ""
echo "============================================"
echo "  TOURNAMENT COMPLETE"
echo "============================================"
echo "Results: $RESULTS_DIR/all_games.pgn"
