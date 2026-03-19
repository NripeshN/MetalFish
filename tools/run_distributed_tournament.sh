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
RDIR="/Users/ec2-user/metalfish"
RESULTS_DIR="$PROJ/results/distributed_$(date +%Y%m%d_%H%M%S)"
GAMES=20
TC="300+0.1"
BOOK="reference/books/8moves_v3.pgn"

for arg in "$@"; do
    case $arg in
        --quick) GAMES=4; TC="10+0.1" ;;
        --games=*) GAMES="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  MetalFish Distributed Tournament"
echo "============================================"
echo "Instances: ${#HOSTS[@]}"
echo "Games/match: $GAMES | TC: $TC"
echo "Results: $RESULTS_DIR"
echo ""

ssh_cmd() { ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$USER@$1" "${@:2}"; }
scp_cmd() { scp -i "$PEM" -o StrictHostKeyChecking=no -q "$@"; }

# ============================================================================
# Phase 1: Test connectivity
# ============================================================================
echo "--- Phase 1: Testing connectivity ---"
for host in "${HOSTS[@]}"; do
    if ssh_cmd "$host" "echo ok" >/dev/null 2>&1; then
        echo "  $host: OK"
    else
        echo "  $host: FAILED - skipping"
    fi
done
echo ""

# ============================================================================
# Phase 2: Deploy to all instances
# ============================================================================
deploy_instance() {
    local host=$1 idx=$2
    echo "[$idx] Deploying to $host..."

    ssh_cmd "$host" "mkdir -p $RDIR/build $RDIR/networks $RDIR/reference/books $RDIR/reference/stockfish/src $RDIR/reference/berserk/src $RDIR/reference/Patricia/engine $RDIR/reference/lc0/build/release $RDIR/reference/cutechess/build"

    scp_cmd "$PROJ/build/metalfish" "$USER@$host:$RDIR/build/"
    scp_cmd "$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb" "$USER@$host:$RDIR/networks/"
    scp_cmd "$PROJ/reference/stockfish/src/stockfish" "$USER@$host:$RDIR/reference/stockfish/src/"
    scp_cmd "$PROJ/reference/berserk/src/berserk" "$USER@$host:$RDIR/reference/berserk/src/"
    scp_cmd "$PROJ/reference/Patricia/engine/patricia" "$USER@$host:$RDIR/reference/Patricia/engine/"
    scp_cmd "$PROJ/reference/lc0/build/release/lc0" "$USER@$host:$RDIR/reference/lc0/build/release/"
    scp_cmd "$PROJ/reference/cutechess/build/cutechess-cli" "$USER@$host:$RDIR/reference/cutechess/build/"
    scp_cmd "$PROJ/$BOOK" "$USER@$host:$RDIR/$BOOK"

    # Copy tournament runner
    scp_cmd "$PROJ/tools/run_tournament_live.py" "$USER@$host:$RDIR/tools/"

    # Copy NNUE files
    for f in "$PROJ"/build/nn-*.nnue; do
        [ -f "$f" ] && scp_cmd "$f" "$USER@$host:$RDIR/build/"
    done

    ssh_cmd "$host" "chmod +x $RDIR/build/metalfish $RDIR/reference/stockfish/src/stockfish $RDIR/reference/berserk/src/berserk $RDIR/reference/Patricia/engine/patricia $RDIR/reference/lc0/build/release/lc0 $RDIR/reference/cutechess/build/cutechess-cli"

    # Install python-chess + create protobuf compat symlink
    ssh_cmd "$host" "pip3 install python-chess 2>/dev/null | tail -1; cd /opt/homebrew/opt/protobuf/lib 2>/dev/null && sudo ln -sf libprotobuf.34.0.0.dylib libprotobuf.33.4.0.dylib 2>/dev/null; true"

    # Create remote engines config with correct paths
    ssh_cmd "$host" "cat > $RDIR/tools/engines_config.json << 'JSONEOF'
{
  \"engines\": {
    \"MetalFish-AB\": {\"path\": \"build/metalfish\", \"options\": {\"Threads\": \"8\", \"Hash\": \"256\"}},
    \"MetalFish-MCTS\": {\"path\": \"build/metalfish\", \"options\": {\"Threads\": \"8\", \"UseMCTS\": \"true\"}},
    \"MetalFish-Hybrid\": {\"path\": \"build/metalfish\", \"options\": {\"Threads\": \"8\", \"Hash\": \"256\", \"UseHybridSearch\": \"true\"}},
    \"Stockfish\": {\"path\": \"reference/stockfish/src/stockfish\", \"expected_elo\": 3800, \"anchor\": true, \"options\": {\"Threads\": \"8\", \"Hash\": \"256\"}},
    \"Stockfish-L15\": {\"path\": \"reference/stockfish/src/stockfish\", \"expected_elo\": 3551, \"options\": {\"Threads\": \"8\", \"Hash\": \"256\", \"Skill Level\": \"15\"}},
    \"Stockfish-L10\": {\"path\": \"reference/stockfish/src/stockfish\", \"expected_elo\": 3304, \"options\": {\"Threads\": \"8\", \"Hash\": \"256\", \"Skill Level\": \"10\"}},
    \"Stockfish-L5\": {\"path\": \"reference/stockfish/src/stockfish\", \"expected_elo\": 3100, \"options\": {\"Threads\": \"8\", \"Hash\": \"256\", \"Skill Level\": \"5\"}},
    \"Berserk\": {\"path\": \"reference/berserk/src/berserk\", \"expected_elo\": 3722, \"anchor\": true, \"options\": {\"Threads\": \"8\", \"Hash\": \"256\"}},
    \"Patricia\": {\"path\": \"reference/Patricia/engine/patricia\", \"expected_elo\": 3415, \"anchor\": true, \"options\": {\"Threads\": \"8\", \"Hash\": \"256\"}},
    \"Lc0\": {\"path\": \"reference/lc0/build/release/lc0\", \"expected_elo\": 3700, \"cmd_args\": [\"--weights=networks/BT4-1024x15x32h-swa-6147500.pb\", \"--backend=metal\"], \"options\": {\"Threads\": \"8\", \"Temperature\": \"0\"}}
  },
  \"opening_book\": {\"file\": \"reference/books/8moves_v3.pgn\"}
}
JSONEOF
mkdir -p $RDIR/tools"

    echo "[$idx] Deploy complete"
}

echo "--- Phase 2: Deploying ---"
for i in "${!HOSTS[@]}"; do
    deploy_instance "${HOSTS[$i]}" "$((i+1))" &
done
wait
echo "All deployed."
echo ""

# ============================================================================
# Phase 3: Verify one engine works on first instance
# ============================================================================
echo "--- Phase 3: Verifying remote engine ---"
VERIFY=$(ssh_cmd "${HOSTS[0]}" "cd $RDIR && echo 'uci
quit' | timeout 5 build/metalfish 2>&1 | head -1")
echo "  Remote metalfish: $VERIFY"

VERIFY2=$(ssh_cmd "${HOSTS[0]}" "cd $RDIR && python3 -c 'import chess; print(\"python-chess OK\")'  2>&1")
echo "  Remote python-chess: $VERIFY2"
echo ""

# ============================================================================
# Phase 4: Run matches
# ============================================================================

COMMON="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

run_match_remote() {
    local host=$1 idx=$2 e1="$3" e2="$4" label="$5"
    echo "[$idx] $label on $host"
    ssh_cmd "$host" "cd $RDIR && python3 run_tournament_live.py --match '$e1' '$e2' --games $GAMES --tc-base ${TC%+*} --tc-inc ${TC#*+} 2>&1" | tail -5
    # Collect any results JSON
    scp_cmd "$USER@$host:$RDIR/results/tournament_*/results.json" "$RESULTS_DIR/${label}.json" 2>/dev/null || true
    echo "[$idx] Done: $label"
}

run_instance() {
    local host=$1 idx=$2
    shift 2
    while [ $# -ge 3 ]; do
        run_match_remote "$host" "$idx" "$1" "$2" "$3"
        shift 3
    done
}

echo "--- Phase 4: Running matches ---"

run_instance "${HOSTS[0]}" 1 \
    "MetalFish-AB" "MetalFish-MCTS" "01_AB_vs_MCTS" \
    "MetalFish-AB" "MetalFish-Hybrid" "02_AB_vs_Hybrid" \
    "MetalFish-MCTS" "MetalFish-Hybrid" "03_MCTS_vs_Hybrid" \
    "MetalFish-AB" "Stockfish" "04_AB_vs_Stockfish" \
    "MetalFish-AB" "Berserk" "05_AB_vs_Berserk" \
    "MetalFish-AB" "Patricia" "06_AB_vs_Patricia" &
P1=$!

run_instance "${HOSTS[1]}" 2 \
    "MetalFish-MCTS" "Lc0" "07_MCTS_vs_Lc0" \
    "MetalFish-MCTS" "Patricia" "08_MCTS_vs_Patricia" \
    "MetalFish-MCTS" "Stockfish-L10" "09_MCTS_vs_SF-L10" \
    "MetalFish-MCTS" "Stockfish-L5" "10_MCTS_vs_SF-L5" \
    "MetalFish-MCTS" "Stockfish" "11_MCTS_vs_Stockfish" \
    "MetalFish-MCTS" "Berserk" "12_MCTS_vs_Berserk" &
P2=$!

run_instance "${HOSTS[2]}" 3 \
    "MetalFish-Hybrid" "Stockfish-L15" "13_Hybrid_vs_SF-L15" \
    "MetalFish-Hybrid" "Berserk" "14_Hybrid_vs_Berserk" \
    "MetalFish-Hybrid" "Patricia" "15_Hybrid_vs_Patricia" \
    "MetalFish-Hybrid" "Lc0" "16_Hybrid_vs_Lc0" \
    "MetalFish-Hybrid" "Stockfish" "17_Hybrid_vs_Stockfish" \
    "MetalFish-Hybrid" "Stockfish-L10" "18_Hybrid_vs_SF-L10" &
P3=$!

run_instance "${HOSTS[3]}" 4 \
    "MetalFish-AB" "Lc0" "19_AB_vs_Lc0" \
    "MetalFish-AB" "Stockfish-L15" "20_AB_vs_SF-L15" \
    "MetalFish-AB" "Stockfish-L10" "21_AB_vs_SF-L10" \
    "MetalFish-AB" "Stockfish-L5" "22_AB_vs_SF-L5" \
    "Stockfish" "Lc0" "23_SF_vs_Lc0" \
    "Patricia" "Lc0" "24_Patricia_vs_Lc0" &
P4=$!

echo "All 4 instances running in parallel..."
echo ""

wait $P1 && echo "[1] COMPLETE" || echo "[1] FAILED"
wait $P2 && echo "[2] COMPLETE" || echo "[2] FAILED"
wait $P3 && echo "[3] COMPLETE" || echo "[3] FAILED"
wait $P4 && echo "[4] COMPLETE" || echo "[4] FAILED"

# ============================================================================
# Phase 5: Aggregate
# ============================================================================
echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"

echo ""
for json in "$RESULTS_DIR"/*.json; do
    [ ! -f "$json" ] && continue
    label=$(basename "$json" .json)
    python3 -c "
import json, sys
with open('$json') as f:
    data = json.load(f)
for m in data.get('matches', []):
    print(f\"  {m['name1']:20s} vs {m['name2']:20s}  {m['wins']}W-{m['draws']}D-{m['losses']}L  Elo:{m['elo_diff']:+.0f}\")
" 2>/dev/null || echo "  $label: error reading results"
done

echo ""
echo "All results: $RESULTS_DIR/"
