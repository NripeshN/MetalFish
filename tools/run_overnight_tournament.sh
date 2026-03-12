#!/bin/bash
# =============================================================================
# MetalFish Overnight ELO Tournament
# 
# Runs a full gauntlet tournament: MetalFish (AB + Hybrid) vs reference engines
# using cutechess-cli with balanced openings and proper adjudication.
#
# Usage: ./tools/run_overnight_tournament.sh
# Estimated time: 20-30 hours at 300+3 TC with 4 threads
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

# Tournament parameters
TC="300+3"           # 5 minutes + 3 seconds per move
GAMES=20             # Per engine pair (10 openings x 2 colors)
THREADS=4
HASH=256
CONCURRENCY=1

# Adjudication (speeds up obvious wins/draws)
RESIGN_MOVECOUNT=5
RESIGN_SCORE=1000
DRAW_MOVENUMBER=40
DRAW_MOVECOUNT=8
DRAW_SCORE=10

# --- Engine definitions ---
METALFISH="$DIR/build/metalfish"
STOCKFISH="$DIR/reference/stockfish/src/stockfish"
BERSERK="$DIR/reference/berserk/src/berserk"
PATRICIA="$DIR/reference/Patricia/engine/patricia"

# --- Setup ---
mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

run_match() {
    local NAME1="$1"
    local CMD1="$2"
    local OPTS1="$3"
    local NAME2="$4"
    local CMD2="$5"
    local OPTS2="$6"
    local PGN="$RESULTS_DIR/${NAME1}_vs_${NAME2}.pgn"
    
    log "=== MATCH: $NAME1 vs $NAME2 ($GAMES games, TC $TC) ==="
    
    local COMMON_OPTS="option.Threads=$THREADS option.Hash=$HASH"
    
    "$CUTECHESS" \
        -engine cmd="$CMD1" name="$NAME1" proto=uci $COMMON_OPTS $OPTS1 \
        -engine cmd="$CMD2" name="$NAME2" proto=uci $COMMON_OPTS $OPTS2 \
        -each tc=$TC \
        -games $GAMES \
        -rounds $(( GAMES / 2 )) \
        -repeat \
        -recover \
        -concurrency $CONCURRENCY \
        -openings file="$BOOK" format=pgn order=random \
        -resign movecount=$RESIGN_MOVECOUNT score=$RESIGN_SCORE \
        -draw movenumber=$DRAW_MOVENUMBER movecount=$DRAW_MOVECOUNT score=$DRAW_SCORE \
        -pgnout "$PGN" 2>&1 | tee -a "$LOG"
    
    log "--- Match complete: $NAME1 vs $NAME2 ---"
    echo "" >> "$LOG"
}

# --- Pre-flight checks ---
log "MetalFish Overnight ELO Tournament"
log "==================================="
log "Timestamp: $TIMESTAMP"
log "TC: $TC | Games/pair: $GAMES | Threads: $THREADS | Hash: ${HASH}MB"
log ""

for BIN in "$METALFISH" "$STOCKFISH" "$BERSERK" "$PATRICIA" "$CUTECHESS"; do
    if [ ! -f "$BIN" ]; then
        log "ERROR: Missing binary: $BIN"
        exit 1
    fi
done
log "All engine binaries verified."

if [ ! -f "$BOOK" ]; then
    log "ERROR: Missing opening book: $BOOK"
    exit 1
fi
log "Opening book: $BOOK"
log "Results directory: $RESULTS_DIR"
log ""
log "Starting tournament... (estimated 20-30 hours)"
log ""

START_TIME=$(date +%s)

# =============================================================================
# ROUND 1: MetalFish-AB vs Reference Engines
# =============================================================================

log "===== ROUND 1: MetalFish-AB Gauntlet ====="

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
# ROUND 2: MetalFish-Hybrid vs Reference Engines
# =============================================================================

log "===== ROUND 2: MetalFish-Hybrid Gauntlet ====="

HYBRID_OPTS="option.UseHybridSearch=true \"option.NNWeights=$NNWEIGHTS\""

run_match "MetalFish-Hybrid" "$METALFISH" "$HYBRID_OPTS" \
          "Patricia" "$PATRICIA" ""

run_match "MetalFish-Hybrid" "$METALFISH" "$HYBRID_OPTS" \
          "Stockfish-L10" "$STOCKFISH" '"option.Skill Level=10"'

run_match "MetalFish-Hybrid" "$METALFISH" "$HYBRID_OPTS" \
          "Stockfish-L15" "$STOCKFISH" '"option.Skill Level=15"'

# =============================================================================
# ROUND 3: MetalFish-AB vs MetalFish-Hybrid (head-to-head)
# =============================================================================

log "===== ROUND 3: MetalFish AB vs Hybrid Head-to-Head ====="

run_match "MetalFish-AB" "$METALFISH" "" \
          "MetalFish-Hybrid" "$METALFISH" "$HYBRID_OPTS"

# =============================================================================
# Summary
# =============================================================================

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

log ""
log "===== TOURNAMENT COMPLETE ====="
log "Total time: ${ELAPSED} minutes"
log "Results saved to: $RESULTS_DIR"
log ""
log "=== PGN Files ==="
ls -la "$RESULTS_DIR"/*.pgn 2>/dev/null | tee -a "$LOG"

log ""
log "=== Score Summary ==="
for pgn in "$RESULTS_DIR"/*.pgn; do
    NAME=$(basename "$pgn" .pgn)
    W=$(grep -c "1-0" "$pgn" 2>/dev/null || echo 0)
    D=$(grep -c "1/2-1/2" "$pgn" 2>/dev/null || echo 0)
    L=$(grep -c "0-1" "$pgn" 2>/dev/null || echo 0)
    log "$NAME: +$W =$D -$L"
done

log ""
log "Tournament finished at $(date)"
log "Review PGN files in: $RESULTS_DIR"
