#!/bin/bash
# =============================================================================
#  MetalFish Overnight ELO Tournament
#  Usage: ./tools/run_overnight_tournament.sh
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
CUTECHESS="$DIR/reference/cutechess/build/cutechess-cli"
BOOK="$DIR/reference/books/8moves_v3.pgn"
NNWEIGHTS="$DIR/networks/BT4-1024x15x32h-swa-6147500.pb"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$DIR/results/tournament_$TIMESTAMP"
LOG="$RESULTS_DIR/tournament.log"

TC="300+3"
GAMES=20
THREADS=4
HASH=256

METALFISH="$DIR/build/metalfish"
STOCKFISH="$DIR/reference/stockfish/src/stockfish"
BERSERK="$DIR/reference/berserk/src/berserk"
PATRICIA="$DIR/reference/Patricia/engine/patricia"

# Colors
R='\033[31m'; G='\033[32m'; Y='\033[33m'; B='\033[34m'
M='\033[35m'; C='\033[36m'; W='\033[1;37m'; D='\033[2m'
N='\033[0m'; BOLD='\033[1m'

mkdir -p "$RESULTS_DIR"
TOTAL_MATCHES=9
CURRENT_MATCH=0
GAMES_DONE=0
START_TIME=$(date +%s)

elapsed() {
    local s=$(( $(date +%s) - START_TIME ))
    printf "%dh%02dm" $((s/3600)) $(( (s%3600)/60 ))
}

# ── Banner ──
clear
echo ""
echo -e "${C}${BOLD}    MetalFish Overnight ELO Tournament${N}"
echo -e "${C}    =====================================${N}"
echo -e "    TC: ${W}${TC}${N}  Threads: ${W}${THREADS}${N}  Hash: ${W}${HASH}MB${N}"
echo -e "    Matches: ${W}${TOTAL_MATCHES}${N}  Games/pair: ${W}${GAMES}${N}  Total: ${W}$((TOTAL_MATCHES*GAMES))${N}"
echo -e "    Started: ${W}$(date '+%Y-%m-%d %H:%M')${N}"
echo ""

# ── Check engines ──
echo -e "    ${BOLD}Checking engines...${N}"
for BIN in "$METALFISH" "$STOCKFISH" "$BERSERK" "$PATRICIA"; do
    NAME=$(echo "uci" | timeout 5 "$BIN" 2>/dev/null | grep "^id name" | sed 's/id name //')
    echo -e "    ${G}OK${N}  $NAME"
done
echo ""

echo -e "    ${Y}Starting in 3 seconds...${N}"
sleep 3

# ── Run a match ──
run_match() {
    local NAME1="$1" CMD1="$2" OPTS1="$3"
    local NAME2="$4" CMD2="$5" OPTS2="$6"
    local PGN="$RESULTS_DIR/${NAME1}_vs_${NAME2}.pgn"

    CURRENT_MATCH=$((CURRENT_MATCH + 1))

    echo ""
    echo -e "${C}${BOLD}    ── Match ${CURRENT_MATCH}/${TOTAL_MATCHES}: ${W}${NAME1}${C} vs ${W}${NAME2}${C} ──${N}"
    echo -e "    ${D}Elapsed: $(elapsed) | TC: ${TC} | ${GAMES} games${N}"
    echo -e "    ${D}Each game takes ~5-10 min. Results appear as games finish.${N}"
    echo ""

    eval "$CUTECHESS" \
        -engine cmd="$CMD1" name="$NAME1" proto=uci option.Threads=$THREADS option.Hash=$HASH $OPTS1 \
        -engine cmd="$CMD2" name="$NAME2" proto=uci option.Threads=$THREADS option.Hash=$HASH $OPTS2 \
        -each tc=$TC \
        -games $GAMES \
        -rounds $((GAMES / 2)) \
        -repeat -recover \
        -concurrency 1 \
        -openings file="$BOOK" format=pgn order=random \
        -resign movecount=5 score=1000 \
        -draw movenumber=40 movecount=8 score=10 \
        -pgnout "$PGN" 2>&1 | while IFS= read -r line; do

        echo "$line" >> "$LOG"

        # Show game start
        if echo "$line" | grep -q "^Started game"; then
            local g=$(echo "$line" | sed 's/.*game \([0-9]*\).*/\1/')
            printf "    ${D}Game %s/%d ...${N}\r" "$g" "$GAMES"
        fi

        # Show game result
        if echo "$line" | grep -q "^Finished game"; then
            local g=$(echo "$line" | sed 's/.*game \([0-9]*\).*/\1/')
            local reason=$(echo "$line" | sed 's/.*{\(.*\)}/\1/')
            if echo "$line" | grep -q "1-0"; then
                printf "    ${G}Game %2s/%d: 1-0${N}  ${D}%s${N}\n" "$g" "$GAMES" "$reason"
            elif echo "$line" | grep -q "0-1"; then
                printf "    ${R}Game %2s/%d: 0-1${N}  ${D}%s${N}\n" "$g" "$GAMES" "$reason"
            elif echo "$line" | grep -q "1/2"; then
                printf "    ${Y}Game %2s/%d: 1/2${N}  ${D}%s${N}\n" "$g" "$GAMES" "$reason"
            else
                printf "    ${D}Game %2s/%d: ???${N}  %s\n" "$g" "$GAMES" "$reason"
            fi
        fi

        # Show score line
        if echo "$line" | grep -q "^Score of"; then
            echo -e "    ${BOLD}$line${N}"
        fi
        if echo "$line" | grep -q "^Elo diff"; then
            echo -e "    ${C}$line${N}"
        fi
    done

    # Summary for this match
    if [ -f "$PGN" ]; then
        local w=$(grep -c 'Result "1-0"' "$PGN" 2>/dev/null || echo 0)
        local d=$(grep -c 'Result "1/2-1/2"' "$PGN" 2>/dev/null || echo 0)
        local l=$(grep -c 'Result "0-1"' "$PGN" 2>/dev/null || echo 0)
        echo -e "    ${BOLD}Result: ${G}+${w}${N} ${Y}=${d}${N} ${R}-${l}${N}"
    fi
}

# =============================================================================
echo -e "\n${C}${BOLD}    ROUND 1: MetalFish-AB Gauntlet${N}"
# =============================================================================

run_match "MetalFish-AB" "$METALFISH" "" \
          "Patricia" "$PATRICIA" ""

run_match "MetalFish-AB" "$METALFISH" "" \
          "Stockfish-L10" "$STOCKFISH" '"option.Skill Level=10"'

run_match "MetalFish-AB" "$METALFISH" "" \
          "Stockfish-L15" "$STOCKFISH" '"option.Skill Level=15"'

run_match "MetalFish-AB" "$METALFISH" "" \
          "Berserk" "$BERSERK" ""

run_match "MetalFish-AB" "$METALFISH" "" \
          "Stockfish-Full" "$STOCKFISH" ""

# =============================================================================
echo -e "\n${M}${BOLD}    ROUND 2: MetalFish-Hybrid Gauntlet${N}"
# =============================================================================

HOPTS="option.UseHybridSearch=true \"option.NNWeights=$NNWEIGHTS\""

run_match "MetalFish-Hybrid" "$METALFISH" "$HOPTS" \
          "Patricia" "$PATRICIA" ""

run_match "MetalFish-Hybrid" "$METALFISH" "$HOPTS" \
          "Stockfish-L10" "$STOCKFISH" '"option.Skill Level=10"'

run_match "MetalFish-Hybrid" "$METALFISH" "$HOPTS" \
          "Stockfish-L15" "$STOCKFISH" '"option.Skill Level=15"'

# =============================================================================
echo -e "\n${W}${BOLD}    ROUND 3: AB vs Hybrid Head-to-Head${N}"
# =============================================================================

run_match "MetalFish-AB" "$METALFISH" "" \
          "MetalFish-Hybrid" "$METALFISH" "$HOPTS"

# =============================================================================
# Final
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo -e "${C}${BOLD}    =====================================${N}"
echo -e "${C}${BOLD}    TOURNAMENT COMPLETE${N}"
echo -e "${C}${BOLD}    =====================================${N}"
echo -e "    Time: ${W}$((ELAPSED/60))h $((ELAPSED%60))m${N}"
echo ""
echo -e "    ${BOLD}Per-Match Results:${N}"
echo ""

for pgn in "$RESULTS_DIR"/*.pgn; do
    [ -f "$pgn" ] || continue
    NAME=$(basename "$pgn" .pgn | tr '_' ' ')
    W=$(grep -c 'Result "1-0"' "$pgn" 2>/dev/null || echo 0)
    D=$(grep -c 'Result "1/2-1/2"' "$pgn" 2>/dev/null || echo 0)
    L=$(grep -c 'Result "0-1"' "$pgn" 2>/dev/null || echo 0)
    printf "    %-42s ${G}+%d${N} ${Y}=%d${N} ${R}-%d${N}\n" "$NAME" "$W" "$D" "$L"
done

echo ""
echo -e "    ${D}PGN files: $RESULTS_DIR${N}"
echo -e "    ${D}Log: $LOG${N}"
echo -e "    ${G}${BOLD}Done!${N}"
echo ""
