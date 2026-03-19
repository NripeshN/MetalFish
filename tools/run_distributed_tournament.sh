#!/bin/bash
# ============================================================================
# MetalFish Distributed Tournament (4-pane tmux display)
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
SESSION="metalfish-tournament"

for arg in "$@"; do
    case $arg in
        --quick) GAMES=4; TC="10+0.1" ;;
        --games=*) GAMES="${arg#*=}" ;;
        --tc=*) TC="${arg#*=}" ;;
        --no-tmux) NO_TMUX=1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# Common cutechess args
CC="$RDIR/reference/cutechess/build/cutechess-cli"
BOOK="$RDIR/reference/books/8moves_v3.pgn"
WEIGHTS="$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb"
CCARGS="-each tc=$TC -games $GAMES -repeat -recover -openings file=$BOOK format=pgn order=random -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

# Engine defs
AB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-AB option.Threads=10 option.Hash=512"
MCTS="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-MCTS option.Threads=10 option.UseMCTS=true option.NNWeights=$WEIGHTS"
HYB="-engine proto=uci cmd=$RDIR/build/metalfish name=MetalFish-Hybrid option.Threads=10 option.Hash=512 option.UseHybridSearch=true option.NNWeights=$WEIGHTS"
SF="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish option.Threads=10 option.Hash=512"
SFL15="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L15 option.Threads=10 option.Hash=512 option.\"Skill\ Level\"=15"
SFL10="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L10 option.Threads=10 option.Hash=512 option.\"Skill\ Level\"=10"
SFL5="-engine proto=uci cmd=$RDIR/reference/stockfish/src/stockfish name=Stockfish-L5 option.Threads=10 option.Hash=512 option.\"Skill\ Level\"=5"
BER="-engine proto=uci cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=10 option.Hash=512"
PAT="-engine proto=uci cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=10 option.Hash=512"
LC0="-engine proto=uci cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$WEIGHTS arg=--backend=metal option.Threads=10 option.Temperature=0"

# Build match commands for each instance
ssh_base="ssh -i \"$PEM\" -o StrictHostKeyChecking=no -o ConnectTimeout=10"

build_instance_script() {
    local idx=$1; shift
    local script="cd $RDIR && echo '=== Instance $idx ===' && echo ''"
    local n=1
    while [ $# -ge 2 ]; do
        local e1="$1" e2="$2" label="match_${idx}_${n}"
        script="$script && echo '--- Match $n: $e1 vs $e2 ---' && $CC $e1 $e2 $CCARGS -pgnout $RDIR/$label.pgn 2>&1"
        shift 2; n=$((n+1))
    done
    script="$script && echo '' && echo '=== Instance $idx COMPLETE ==='"
    echo "$script"
}

# Match assignments per instance
I1_CMD=$(build_instance_script 1 \
    "$AB" "$SF" "$AB" "$BER" "$AB" "$PAT" \
    "$AB" "$SFL15" "$AB" "$SFL10" "$AB" "$LC0")

I2_CMD=$(build_instance_script 2 \
    "$MCTS" "$LC0" "$MCTS" "$PAT" "$MCTS" "$SFL10" \
    "$MCTS" "$SFL5" "$MCTS" "$AB" "$MCTS" "$BER")

I3_CMD=$(build_instance_script 3 \
    "$HYB" "$SFL15" "$HYB" "$BER" "$HYB" "$PAT" \
    "$HYB" "$LC0" "$HYB" "$SF" "$HYB" "$MCTS")

I4_CMD=$(build_instance_script 4 \
    "$HYB" "$AB" "$HYB" "$SFL10" "$SF" "$LC0" \
    "$PAT" "$LC0" "$AB" "$SFL5" "$SF" "$PAT")

# Full SSH commands
SSH1="$ssh_base $USER@${HOSTS[0]} \"$I1_CMD\""
SSH2="$ssh_base $USER@${HOSTS[1]} \"$I2_CMD\""
SSH3="$ssh_base $USER@${HOSTS[2]} \"$I3_CMD\""
SSH4="$ssh_base $USER@${HOSTS[3]} \"$I4_CMD\""

echo "============================================"
echo "  MetalFish Distributed Tournament"
echo "============================================"
echo "Games: $GAMES | TC: $TC | Results: $RESULTS_DIR"
echo ""

if [ "${NO_TMUX:-0}" = "1" ] || ! command -v tmux &>/dev/null; then
    echo "Running without tmux (parallel background)..."
    eval $SSH1 | tee "$RESULTS_DIR/log_1.txt" &
    eval $SSH2 | tee "$RESULTS_DIR/log_2.txt" &
    eval $SSH3 | tee "$RESULTS_DIR/log_3.txt" &
    eval $SSH4 | tee "$RESULTS_DIR/log_4.txt" &
    wait
else
    # Kill existing session
    tmux kill-session -t $SESSION 2>/dev/null || true

    # Create 4-pane tmux layout
    tmux new-session -d -s $SESSION -x 200 -y 50 "echo '╔══ Instance 1: AB matches ══╗'; eval $SSH1; echo 'DONE'; read"
    tmux split-window -h -t $SESSION "echo '╔══ Instance 2: MCTS matches ══╗'; eval $SSH2; echo 'DONE'; read"
    tmux split-window -v -t $SESSION:0.0 "echo '╔══ Instance 3: Hybrid matches ══╗'; eval $SSH3; echo 'DONE'; read"
    tmux split-window -v -t $SESSION:0.1 "echo '╔══ Instance 4: Cross matches ══╗'; eval $SSH4; echo 'DONE'; read"

    # Pretty borders
    tmux set-option -t $SESSION pane-border-style "fg=cyan"
    tmux set-option -t $SESSION pane-active-border-style "fg=green"
    tmux set-option -t $SESSION pane-border-format " #{pane_index}: #{pane_title} "
    tmux set-option -t $SESSION pane-border-status top

    echo "Launched in tmux session '$SESSION'"
    echo ""
    echo "  Attach:  tmux attach -t $SESSION"
    echo "  Detach:  Ctrl-B then D"
    echo "  Kill:    tmux kill-session -t $SESSION"
    echo ""

    tmux attach -t $SESSION
fi

# Collect PGNs from all instances
echo ""
echo "--- Collecting PGN results ---"
for host in "${HOSTS[@]}"; do
    scp -i "$PEM" -o StrictHostKeyChecking=no -q $USER@$host:$RDIR/match_*.pgn "$RESULTS_DIR/" 2>/dev/null || true
done

# Aggregate
cat "$RESULTS_DIR"/*.pgn > "$RESULTS_DIR/all_games.pgn" 2>/dev/null || true
TOTAL=$(grep -c "\[Result " "$RESULTS_DIR/all_games.pgn" 2>/dev/null || echo 0)

echo ""
echo "============================================"
echo "  FINAL RESULTS ($TOTAL games)"
echo "============================================"
echo ""
for pgn in "$RESULTS_DIR"/match_*.pgn; do
    [ ! -f "$pgn" ] && continue
    label=$(basename "$pgn" .pgn)
    w=$(grep -c '\[Result "1-0"\]' "$pgn" 2>/dev/null || echo 0)
    d=$(grep -c '\[Result "1/2-1/2"\]' "$pgn" 2>/dev/null || echo 0)
    l=$(grep -c '\[Result "0-1"\]' "$pgn" 2>/dev/null || echo 0)
    t=$((w+d+l)); [ $t -eq 0 ] && continue
    printf "  %-30s %dW-%dD-%dL (%d games)\n" "$label" "$w" "$d" "$l" "$t"
done
echo ""
echo "Combined PGN: $RESULTS_DIR/all_games.pgn"
