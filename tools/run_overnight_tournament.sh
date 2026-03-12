#!/bin/bash
# =============================================================================
#  MetalFish Overnight ELO Tournament
#  
#  Runs a full gauntlet tournament with live stats display.
#  Usage: ./tools/run_overnight_tournament.sh
# =============================================================================

set -e

# --- Configuration ---
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

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;37m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# --- State ---
mkdir -p "$RESULTS_DIR"
TOTAL_MATCHES=9
TOTAL_GAMES=$((TOTAL_MATCHES * GAMES))
CURRENT_MATCH=0
GAMES_PLAYED=0
WINS_AB=0; DRAWS_AB=0; LOSSES_AB=0
WINS_HYB=0; DRAWS_HYB=0; LOSSES_HYB=0
START_TIME=$(date +%s)

# --- Display functions ---
banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║          MetalFish Overnight ELO Tournament                 ║"
    echo "  ╠══════════════════════════════════════════════════════════════╣"
    echo -e "  ║  ${WHITE}TC: ${TC}  |  Threads: ${THREADS}  |  Hash: ${HASH}MB  |  Games/pair: ${GAMES}${CYAN}  ║"
    echo -e "  ║  ${WHITE}Started: $(date '+%Y-%m-%d %H:%M')  |  Matches: ${TOTAL_MATCHES}  |  Total games: ${TOTAL_GAMES}${CYAN}  ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local pct=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    printf "${GREEN}"
    printf "  ["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "] %3d%% (%d/%d games)" "$pct" "$current" "$total"
    printf "${NC}\n"
}

elapsed_time() {
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local hours=$((elapsed / 3600))
    local mins=$(( (elapsed % 3600) / 60))
    printf "%dh %02dm" "$hours" "$mins"
}

eta() {
    if [ "$GAMES_PLAYED" -gt 0 ]; then
        local now=$(date +%s)
        local elapsed=$((now - START_TIME))
        local per_game=$((elapsed / GAMES_PLAYED))
        local remaining=$(( (TOTAL_GAMES - GAMES_PLAYED) * per_game ))
        local hours=$((remaining / 3600))
        local mins=$(( (remaining % 3600) / 60 ))
        printf "%dh %02dm" "$hours" "$mins"
    else
        printf "calculating..."
    fi
}

show_standings() {
    echo -e "\n${BOLD}${WHITE}  ── Tournament Standings ──${NC}\n"
    
    local ab_total=$((WINS_AB + DRAWS_AB + LOSSES_AB))
    local hyb_total=$((WINS_HYB + DRAWS_HYB + LOSSES_HYB))
    local ab_score=$(echo "scale=1; $WINS_AB + $DRAWS_AB * 0.5" | bc 2>/dev/null || echo "?")
    local hyb_score=$(echo "scale=1; $WINS_HYB + $DRAWS_HYB * 0.5" | bc 2>/dev/null || echo "?")
    
    printf "  ${CYAN}%-20s${NC} " "MetalFish-AB"
    printf "${GREEN}+%-3d${NC} ${YELLOW}=%-3d${NC} ${RED}-%-3d${NC}" "$WINS_AB" "$DRAWS_AB" "$LOSSES_AB"
    printf "  ${BOLD}Score: %s/%d${NC}\n" "$ab_score" "$ab_total"
    
    printf "  ${MAGENTA}%-20s${NC} " "MetalFish-Hybrid"
    printf "${GREEN}+%-3d${NC} ${YELLOW}=%-3d${NC} ${RED}-%-3d${NC}" "$WINS_HYB" "$DRAWS_HYB" "$LOSSES_HYB"
    printf "  ${BOLD}Score: %s/%d${NC}\n" "$hyb_score" "$hyb_total"
}

match_header() {
    local n1="$1" n2="$2" match_num="$3"
    echo -e "\n${BOLD}${WHITE}  ╔════════════════════════════════════════════════════════╗${NC}"
    printf "  ${BOLD}${WHITE}║${NC}  ${CYAN}Match %d/%d${NC}: ${BOLD}%-18s${NC} vs ${BOLD}%-18s${NC}${BOLD}${WHITE}║${NC}\n" \
        "$match_num" "$TOTAL_MATCHES" "$n1" "$n2"
    echo -e "  ${BOLD}${WHITE}╚════════════════════════════════════════════════════════╝${NC}"
    echo -e "  ${DIM}Elapsed: $(elapsed_time) | ETA remaining: $(eta)${NC}\n"
}

# --- Match runner ---
run_match() {
    local NAME1="$1"; local CMD1="$2"; local OPTS1="$3"
    local NAME2="$4"; local CMD2="$5"; local OPTS2="$6"
    local IS_AB1="$7"  # "ab" or "hyb" for first engine
    local PGN="$RESULTS_DIR/${NAME1}_vs_${NAME2}.pgn"
    
    CURRENT_MATCH=$((CURRENT_MATCH + 1))
    match_header "$NAME1" "$NAME2" "$CURRENT_MATCH"
    
    local match_w=0 match_d=0 match_l=0
    
    # Build cutechess command
    local CMD="$CUTECHESS \
        -engine cmd=$CMD1 name=$NAME1 proto=uci option.Threads=$THREADS option.Hash=$HASH $OPTS1 \
        -engine cmd=$CMD2 name=$NAME2 proto=uci option.Threads=$THREADS option.Hash=$HASH $OPTS2 \
        -each tc=$TC \
        -games $GAMES \
        -rounds $((GAMES / 2)) \
        -repeat \
        -recover \
        -concurrency 1 \
        -openings file=$BOOK format=pgn order=random \
        -resign movecount=5 score=1000 \
        -draw movenumber=40 movecount=8 score=10 \
        -pgnout $PGN"
    
    # Run and parse output live
    eval $CMD 2>&1 | while IFS= read -r line; do
        echo "$line" >> "$LOG"
        
        # Parse game results
        if echo "$line" | grep -q "^Finished game"; then
            GAMES_PLAYED=$((GAMES_PLAYED + 1))
            
            local game_num=$(echo "$line" | grep -oP 'game \K[0-9]+')
            local result=""
            local detail=""
            
            if echo "$line" | grep -q "1-0"; then
                result="${GREEN}1-0${NC}"
                detail=$(echo "$line" | grep -oP '\{.*\}')
                match_w=$((match_w + 1))
            elif echo "$line" | grep -q "0-1"; then
                result="${RED}0-1${NC}"
                detail=$(echo "$line" | grep -oP '\{.*\}')
                match_l=$((match_l + 1))
            elif echo "$line" | grep -q "1/2-1/2"; then
                result="${YELLOW}1/2${NC}"
                detail=$(echo "$line" | grep -oP '\{.*\}')
                match_d=$((match_d + 1))
            fi
            
            local ply=$(echo "$line" | grep -oP 'PlyCount "\K[0-9]+' || echo "?")
            
            printf "  ${DIM}Game %2d/%d:${NC} %b  ${DIM}%s${NC}\n" \
                "$game_num" "$GAMES" "$result" "$detail"
            
        elif echo "$line" | grep -q "^Score of"; then
            echo -e "  ${BOLD}$line${NC}"
        elif echo "$line" | grep -q "^Elo difference"; then
            echo -e "  ${CYAN}$line${NC}"
        fi
    done
    
    # Update overall standings (read from PGN since subshell vars are lost)
    if [ -f "$PGN" ]; then
        local w=$(grep -c "^Result.*1-0" "$PGN" 2>/dev/null || echo 0)
        local d=$(grep -c "^Result.*1/2" "$PGN" 2>/dev/null || echo 0)
        local l=$(grep -c "^Result.*0-1" "$PGN" 2>/dev/null || echo 0)
        
        if [ "$IS_AB1" = "ab" ]; then
            WINS_AB=$((WINS_AB + w))
            DRAWS_AB=$((DRAWS_AB + d))
            LOSSES_AB=$((LOSSES_AB + l))
        elif [ "$IS_AB1" = "hyb" ]; then
            WINS_HYB=$((WINS_HYB + w))
            DRAWS_HYB=$((DRAWS_HYB + d))
            LOSSES_HYB=$((LOSSES_HYB + l))
        fi
        
        GAMES_PLAYED=$((GAMES_PLAYED + w + d + l))
    fi
    
    progress_bar "$GAMES_PLAYED" "$TOTAL_GAMES"
    show_standings
}

# =============================================================================
# Pre-flight checks
# =============================================================================
banner

echo -e "  ${BOLD}Checking engines...${NC}"
for BIN in "$METALFISH" "$STOCKFISH" "$BERSERK" "$PATRICIA" "$CUTECHESS"; do
    if [ ! -f "$BIN" ]; then
        echo -e "  ${RED}MISSING: $BIN${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} $(basename $BIN)"
done

echo -e "\n  ${BOLD}Verifying UCI...${NC}"
for eng in "$METALFISH" "$STOCKFISH" "$BERSERK" "$PATRICIA"; do
    NAME=$(echo "uci" | timeout 5 "$eng" 2>/dev/null | grep "^id name" | sed 's/id name //')
    echo -e "  ${GREEN}✓${NC} $NAME"
done

echo -e "\n  ${DIM}Opening book: $(basename $BOOK)${NC}"
echo -e "  ${DIM}Results: $RESULTS_DIR${NC}"
echo -e "\n  ${YELLOW}Starting in 5 seconds... (Ctrl+C to cancel)${NC}"
sleep 5

# =============================================================================
# ROUND 1: MetalFish-AB Gauntlet
# =============================================================================
banner
echo -e "  ${BOLD}${CYAN}━━━ ROUND 1: MetalFish-AB Gauntlet ━━━${NC}\n"

run_match "MetalFish-AB" "$METALFISH" "" \
          "Patricia" "$PATRICIA" "" "ab"

run_match "MetalFish-AB" "$METALFISH" "" \
          "Stockfish-L10" "$STOCKFISH" '"option.Skill Level=10"' "ab"

run_match "MetalFish-AB" "$METALFISH" "" \
          "Stockfish-L15" "$STOCKFISH" '"option.Skill Level=15"' "ab"

run_match "MetalFish-AB" "$METALFISH" "" \
          "Berserk" "$BERSERK" "" "ab"

run_match "MetalFish-AB" "$METALFISH" "" \
          "Stockfish-Full" "$STOCKFISH" "" "ab"

# =============================================================================
# ROUND 2: MetalFish-Hybrid Gauntlet
# =============================================================================
echo -e "\n  ${BOLD}${MAGENTA}━━━ ROUND 2: MetalFish-Hybrid Gauntlet ━━━${NC}\n"

HOPTS="option.UseHybridSearch=true \"option.NNWeights=$NNWEIGHTS\""

run_match "MetalFish-Hybrid" "$METALFISH" "$HOPTS" \
          "Patricia" "$PATRICIA" "" "hyb"

run_match "MetalFish-Hybrid" "$METALFISH" "$HOPTS" \
          "Stockfish-L10" "$STOCKFISH" '"option.Skill Level=10"' "hyb"

run_match "MetalFish-Hybrid" "$METALFISH" "$HOPTS" \
          "Stockfish-L15" "$STOCKFISH" '"option.Skill Level=15"' "hyb"

# =============================================================================
# ROUND 3: Head-to-Head
# =============================================================================
echo -e "\n  ${BOLD}${WHITE}━━━ ROUND 3: AB vs Hybrid Head-to-Head ━━━${NC}\n"

run_match "MetalFish-AB" "$METALFISH" "" \
          "MetalFish-Hybrid" "$METALFISH" "$HOPTS" "ab"

# =============================================================================
# Final Summary
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║              TOURNAMENT COMPLETE                            ║"
echo "  ╠══════════════════════════════════════════════════════════════╣"
echo -e "  ║  Total time: $(printf '%3d' $((ELAPSED/60)))h $(printf '%02d' $((ELAPSED%60)))m                                          ║"
echo -e "  ║  Games played: $(printf '%3d' $GAMES_PLAYED)                                            ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

show_standings

echo -e "\n${BOLD}${WHITE}  ── Per-Match Results ──${NC}\n"

for pgn in "$RESULTS_DIR"/*.pgn; do
    [ -f "$pgn" ] || continue
    NAME=$(basename "$pgn" .pgn)
    W=$(grep -c 'Result "1-0"' "$pgn" 2>/dev/null || echo 0)
    D=$(grep -c 'Result "1/2-1/2"' "$pgn" 2>/dev/null || echo 0)
    L=$(grep -c 'Result "0-1"' "$pgn" 2>/dev/null || echo 0)
    SCORE=$(echo "scale=1; $W + $D * 0.5" | bc 2>/dev/null || echo "?")
    TOTAL=$((W + D + L))
    printf "  %-40s ${GREEN}+%d${NC} ${YELLOW}=%d${NC} ${RED}-%d${NC}  (%s/%d)\n" \
        "$NAME" "$W" "$D" "$L" "$SCORE" "$TOTAL"
done

echo -e "\n  ${DIM}PGN files: $RESULTS_DIR${NC}"
echo -e "  ${DIM}Full log: $LOG${NC}"
echo -e "\n  ${GREEN}${BOLD}Done! Review results above or open the PGN files in a chess GUI.${NC}\n"
