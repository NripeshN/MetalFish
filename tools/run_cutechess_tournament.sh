#!/bin/bash
# MetalFish Tournament via cutechess-cli
# Usage: ./tools/run_cutechess_tournament.sh [--quick] [--games=N] [--tc=300+0.1] [--match=AB_vs_Hybrid]

set -e
cd "$(dirname "$0")/.."
PROJ="$(pwd)"
CUTECHESS="$PROJ/reference/cutechess/build/cutechess-cli"
BOOK="$PROJ/reference/books/8moves_v3.pgn"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$PROJ/results/cutechess_$TIMESTAMP"
mkdir -p "$RESULTS"

# Defaults: 300+0.1, 20 games. Use a fixed cutechess seed so
# repeated engine/config comparisons see the same opening sample.
GAMES=20
TC="300+0.1"
CUTECHESS_SEED="${CUTECHESS_SEED:-6147500}"
OPENING_ORDER="${OPENING_ORDER:-random}"
MATCH_FILTER=""
CONCURRENCY="${CONCURRENCY:-1}"
MAXMOVES="${MAXMOVES:-160}"
ENGINE_RESTART="${ENGINE_RESTART:-on}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            GAMES=4
            TC="10+0.1"
            echo "Quick mode: $GAMES games, TC=$TC"
            ;;
        --games=*) GAMES="${arg#*=}" ;;
        --tc=*) TC="${arg#*=}" ;;
        --seed=*) CUTECHESS_SEED="${arg#*=}" ;;
        --opening-order=*) OPENING_ORDER="${arg#*=}" ;;
        --match=*) MATCH_FILTER="${arg#*=}" ;;
        --concurrency=*) CONCURRENCY="${arg#*=}" ;;
        --maxmoves=*) MAXMOVES="${arg#*=}" ;;
        --restart=*) ENGINE_RESTART="${arg#*=}" ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

MF="$PROJ/build/metalfish"
WEIGHTS="$PROJ/networks/BT4-1024x15x32h-swa-6147500.pb"
SF="$PROJ/reference/stockfish/src/stockfish"
BERSERK="$PROJ/reference/berserk/src/berserk"
PATRICIA="$PROJ/reference/Patricia/engine/patricia"
LC0="$PROJ/reference/lc0/build/release/lc0"

if [ -z "${THREADS:-}" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        THREADS=$(sysctl -n hw.perflevel0.physicalcpu_max 2>/dev/null || true)
        [ -n "$THREADS" ] || THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || true)
    else
        THREADS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
    fi
fi
[ -n "$THREADS" ] || THREADS=8
HASH="${HASH:-4096}"

MCTS_THREADS="${MCTS_THREADS:-${METALFISH_PURE_MCTS_THREADS:-1}}"
if [ "$MCTS_THREADS" -lt 1 ]; then
    MCTS_THREADS=1
fi
if [ "$MCTS_THREADS" -gt "$THREADS" ]; then
    MCTS_THREADS=$THREADS
fi
MCTS_PARALLEL_SEARCH="${MCTS_PARALLEL_SEARCH:-false}"

HYBRID_THREADS="${HYBRID_THREADS:-$THREADS}"
if [ "$HYBRID_THREADS" -lt 3 ]; then
    HYBRID_THREADS=3
fi
HYBRID_MCTS_THREADS="${HYBRID_MCTS_THREADS:-0}"
if [ "$HYBRID_MCTS_THREADS" -gt 0 ] && [ "$HYBRID_MCTS_THREADS" -ge "$HYBRID_THREADS" ]; then
    HYBRID_MCTS_THREADS=$(( HYBRID_THREADS - 1 ))
fi
# 0 = engine auto. Fixed-budget searches use the tactical 2/1 split; real game
# clocks use the capped 1/2 split unless explicit overrides are provided.
HYBRID_AB_THREADS="${HYBRID_AB_THREADS:-0}"
if [ "$HYBRID_AB_THREADS" -gt 0 ] && [ "$HYBRID_AB_THREADS" -gt "$HYBRID_THREADS" ]; then
    HYBRID_AB_THREADS=$HYBRID_THREADS
fi
HYBRID_AUTO_AB_THREADS_CAP="${HYBRID_AUTO_AB_THREADS_CAP:-0}"
HYBRID_MCTS_KLD="${HYBRID_MCTS_KLD:-0.0}"
HYBRID_AB_ROOT_REJECT_MCTS="${HYBRID_AB_ROOT_REJECT_MCTS:-true}"
HYBRID_MCTS_ROOT_REJECT="${HYBRID_MCTS_ROOT_REJECT:-false}"
HYBRID_MCTS_SHARED_TT="${HYBRID_MCTS_SHARED_TT:-false}"
HYBRID_MCTS_AB_ROOT_HINTS="${HYBRID_MCTS_AB_ROOT_HINTS:-true}"
HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS="${HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS:-0}"
HYBRID_MCTS_AB_ROOT_HINT_COUNT="${HYBRID_MCTS_AB_ROOT_HINT_COUNT:-8}"
HYBRID_AB_CANDIDATE_VERIFY_MS="${HYBRID_AB_CANDIDATE_VERIFY_MS:-240}"
HYBRID_AB_CANDIDATE_VERIFY_COUNT="${HYBRID_AB_CANDIDATE_VERIFY_COUNT:-5}"
HYBRID_AB_POLICY_WEIGHT="${HYBRID_AB_POLICY_WEIGHT:-0.0}"
HYBRID_ROOT_PAWN_LEVER_TIEBREAK="${HYBRID_ROOT_PAWN_LEVER_TIEBREAK:-true}"
HYBRID_TRACE="${HYBRID_TRACE:-false}"
HYBRID_MCTS_MINIBATCH="${HYBRID_MCTS_MINIBATCH:-0}"
HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS="${HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS:-3000}"
HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS="${HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS:-400}"

AB=(proto=uci restart="$ENGINE_RESTART" cmd="$MF" name=MetalFish-AB "option.Threads=$THREADS"
    "option.Hash=$HASH" option.UseMCTS=false option.UseHybridSearch=false
    option.MultiPV=1)
MCTS=(proto=uci restart="$ENGINE_RESTART" cmd="$MF" name=MetalFish-MCTS
      "option.Threads=$MCTS_THREADS" "option.Hash=$HASH"
      option.UseHybridSearch=false option.UseMCTS=true
      "option.NNWeights=$WEIGHTS" option.MultiPV=1
      "option.MCTSMaxThreads=$MCTS_THREADS" option.MCTSMinibatchSize=0
      "option.MCTSParallelSearch=$MCTS_PARALLEL_SEARCH"
      option.MCTSParityPreset=false option.MCTSAddDirichletNoise=false
      option.PureMCTSSmartPruningFactor=0.5
      option.PureMCTSCPuctAtRoot=2.4
      option.PureMCTSFpuReductionAtRoot=0.55
      option.TransformerLowTimeFallbackMs=0
      option.MCTSMinimumKLDGainPerNode=0.00005)
HYBRID=(proto=uci restart="$ENGINE_RESTART" cmd="$MF" name=MetalFish-Hybrid
        "option.Threads=$HYBRID_THREADS" "option.Hash=$HASH"
        option.UseMCTS=false option.UseHybridSearch=true
        "option.NNWeights=$WEIGHTS" option.MultiPV=1
        "option.HybridMCTSThreads=$HYBRID_MCTS_THREADS"
        "option.HybridABThreads=$HYBRID_AB_THREADS"
        "option.HybridAutoABThreadsCap=$HYBRID_AUTO_AB_THREADS_CAP"
        "option.TransformerLowTimeFallbackMs=$HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS"
        "option.TransformerMinMoveBudgetMs=$HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS"
        "option.MCTSMaxThreads=$HYBRID_MCTS_THREADS"
        "option.MCTSMinibatchSize=$HYBRID_MCTS_MINIBATCH"
        option.MCTSParityPreset=false
        option.MCTSAddDirichletNoise=false
        "option.HybridMCTSMinimumKLDGainPerNode=$HYBRID_MCTS_KLD"
        "option.HybridABRootRejectMCTS=$HYBRID_AB_ROOT_REJECT_MCTS"
        "option.HybridMCTSRootReject=$HYBRID_MCTS_ROOT_REJECT"
        "option.HybridMCTSUseSharedTT=$HYBRID_MCTS_SHARED_TT"
        "option.HybridMCTSABRootHints=$HYBRID_MCTS_AB_ROOT_HINTS"
        "option.HybridMCTSABRootHintDelayMs=$HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS"
        "option.HybridMCTSABRootHintCount=$HYBRID_MCTS_AB_ROOT_HINT_COUNT"
        "option.HybridABCandidateVerifyMs=$HYBRID_AB_CANDIDATE_VERIFY_MS"
        "option.HybridABCandidateVerifyCount=$HYBRID_AB_CANDIDATE_VERIFY_COUNT"
        "option.HybridABPolicyWeight=$HYBRID_AB_POLICY_WEIGHT"
        "option.HybridRootPawnLeverTieBreak=$HYBRID_ROOT_PAWN_LEVER_TIEBREAK"
        "option.HybridTrace=$HYBRID_TRACE")
SFULL=(proto=uci restart="$ENGINE_RESTART" cmd="$SF" name=Stockfish "option.Threads=$THREADS"
       "option.Hash=$HASH")
SL15=(proto=uci restart="$ENGINE_RESTART" cmd="$SF" name=Stockfish-L15 "option.Threads=$THREADS"
      "option.Hash=$HASH" "option.Skill Level=15")
SL10=(proto=uci restart="$ENGINE_RESTART" cmd="$SF" name=Stockfish-L10 "option.Threads=$THREADS"
      "option.Hash=$HASH" "option.Skill Level=10")
BERSERK_E=(proto=uci restart="$ENGINE_RESTART" cmd="$BERSERK" name=Berserk
           "option.Threads=$THREADS" "option.Hash=$HASH")
PATRICIA_E=(proto=uci restart="$ENGINE_RESTART" cmd="$PATRICIA" name=Patricia
            "option.Threads=$THREADS" "option.Hash=$HASH")
LC0_E=(proto=uci restart="$ENGINE_RESTART" cmd="$LC0" name=Lc0 "arg=--weights=$WEIGHTS"
       arg=--backend=metal "option.Threads=$THREADS" option.Temperature=0)

COMMON_ARGS=(-each "tc=$TC" -games "$GAMES" -repeat -recover
             -concurrency "$CONCURRENCY" -maxmoves "$MAXMOVES"
             -srand "$CUTECHESS_SEED"
             -resign movecount=3 score=1000 twosided=true
             -draw movenumber=40 movecount=8 score=10)
if [ -f "$BOOK" ]; then
    COMMON_ARGS+=(-openings "file=$BOOK" format=pgn "order=$OPENING_ORDER")
fi

echo ""
echo "============================================"
echo "  MetalFish Tournament (cutechess-cli)"
echo "============================================"
echo "TC: $TC | Games/match: $GAMES"
echo "Openings: order=$OPENING_ORDER | seed=$CUTECHESS_SEED"
echo "Match filter: ${MATCH_FILTER:-all} | Concurrency: $CONCURRENCY | Max moves: $MAXMOVES | Restart: $ENGINE_RESTART"
echo "Threads: AB=$THREADS MCTS=$MCTS_THREADS (Parallel=$MCTS_PARALLEL_SEARCH) Hybrid=$HYBRID_THREADS (HybridMCTS=$HYBRID_MCTS_THREADS, HybridAB=$HYBRID_AB_THREADS, HybridAutoABCap=$HYBRID_AUTO_AB_THREADS_CAP)"
echo "Hash: $HASH MB for engines that support UCI Hash"
echo "Hybrid knobs: KLD=$HYBRID_MCTS_KLD ABRootRejectMCTS=$HYBRID_AB_ROOT_REJECT_MCTS MCTSRootReject=$HYBRID_MCTS_ROOT_REJECT SharedTT=$HYBRID_MCTS_SHARED_TT ABPolicyWeight=$HYBRID_AB_POLICY_WEIGHT Trace=$HYBRID_TRACE LowTimeFallbackMs=$HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS MinMoveBudgetMs=$HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS"
echo "Results: $RESULTS"
echo ""

run_match() {
    local e1_name="$1"
    local e2_name="$2"
    local label="$3"
    local pgn="$RESULTS/${label}.pgn"
    local log="$RESULTS/${label}.log"
    local -a e1
    local -a e2
    if [ -n "$MATCH_FILTER" ] && [[ "$label" != *"$MATCH_FILTER"* ]]; then
        return
    fi
    eval 'e1=( "${'"$e1_name"'[@]}" )'
    eval 'e2=( "${'"$e2_name"'[@]}" )'

    echo ""
    echo "--- $label ---"
    "$CUTECHESS" -engine "${e1[@]}" -engine "${e2[@]}" \
        "${COMMON_ARGS[@]}" -pgnout "$pgn" 2>&1 |
        tee "$log" | tail -20
    echo "  PGN saved: $pgn"
    echo "  Log saved: $log"
}

run_match AB MCTS "01_AB_vs_MCTS"
run_match AB HYBRID "02_AB_vs_Hybrid"
run_match MCTS HYBRID "03_MCTS_vs_Hybrid"

run_match AB SFULL "04_AB_vs_Stockfish"
run_match AB BERSERK_E "05_AB_vs_Berserk"
run_match AB PATRICIA_E "06_AB_vs_Patricia"

run_match MCTS LC0_E "07_MCTS_vs_Lc0"
run_match MCTS PATRICIA_E "08_MCTS_vs_Patricia"

run_match HYBRID SL15 "09_Hybrid_vs_Stockfish-L15"
run_match HYBRID BERSERK_E "10_Hybrid_vs_Berserk"
run_match HYBRID PATRICIA_E "11_Hybrid_vs_Patricia"
run_match HYBRID LC0_E "12_Hybrid_vs_Lc0"

echo ""
echo "============================================"
echo "  ALL MATCHES COMPLETE"
echo "============================================"
echo "PGN files: $RESULTS/"
echo ""
echo "To analyze results:"
echo "  $CUTECHESS -results $RESULTS/*.pgn"
