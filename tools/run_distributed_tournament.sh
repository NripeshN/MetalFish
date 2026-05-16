#!/bin/bash
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PEM="$PROJ/m1 ultra.pem"
HOSTS=(18.212.251.224 54.91.84.176 54.163.13.25 107.21.161.159)
USER=ec2-user
RDIR="/Users/ec2-user/metalfish-src"
RESULTS_DIR="$PROJ/results/distributed_$(date +%Y%m%d_%H%M%S)"
GAMES=10
TC="300+0.1"
WAIT_MS=180000
SYNC_BIN=0
REMOTE_BUILD=1
CLEAN_REMOTE=1
REMOTE_BUILD_DIR="build-native-ec2"
REMOTE_CMAKE="/opt/homebrew/bin/cmake"
ENGINE_BIN="$RDIR/$REMOTE_BUILD_DIR/metalfish"
ENGINE_THREADS="${ENGINE_THREADS:-16}"
MCTS_THREADS="${MCTS_THREADS:-1}"
HYBRID_THREADS="${HYBRID_THREADS:-$ENGINE_THREADS}"
HYBRID_MCTS_THREADS="${HYBRID_MCTS_THREADS:-0}"
HYBRID_AB_THREADS="${HYBRID_AB_THREADS:-0}"
HYBRID_AUTO_AB_THREADS_CAP="${HYBRID_AUTO_AB_THREADS_CAP:-2}"
HYBRID_MCTS_KLD="${HYBRID_MCTS_KLD:-0.0}"
HYBRID_MCTS_ROOT_REJECT="${HYBRID_MCTS_ROOT_REJECT:-true}"
HYBRID_MCTS_SHARED_TT="${HYBRID_MCTS_SHARED_TT:-false}"
HYBRID_AB_POLICY_WEIGHT="${HYBRID_AB_POLICY_WEIGHT:-0.0}"
HYBRID_TRACE="${HYBRID_TRACE:-false}"
HYBRID_MCTS_MINIBATCH="${HYBRID_MCTS_MINIBATCH:-0}"
HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS="${HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS:-5000}"
HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS="${HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS:-1200}"
CUTECHESS_SEED="${CUTECHESS_SEED:-6147500}"
OPENING_ORDER="${OPENING_ORDER:-random}"
ENGINE_RESTART="${ENGINE_RESTART:-on}"

for arg in "$@"; do
    case $arg in
        --quick) GAMES=2; TC="10+0.1"; WAIT_MS=60000 ;;
        --games=*) GAMES="${arg#*=}" ;;
        --tc=*) TC="${arg#*=}" ;;
        --wait-ms=*) WAIT_MS="${arg#*=}" ;;
        --threads=*) ENGINE_THREADS="${arg#*=}"; HYBRID_THREADS="${arg#*=}" ;;
        --mcts-threads=*) MCTS_THREADS="${arg#*=}" ;;
        --hybrid-threads=*) HYBRID_THREADS="${arg#*=}" ;;
        --hybrid-mcts-threads=*) HYBRID_MCTS_THREADS="${arg#*=}" ;;
        --hybrid-ab-threads=*) HYBRID_AB_THREADS="${arg#*=}" ;;
        --hybrid-auto-ab-cap=*) HYBRID_AUTO_AB_THREADS_CAP="${arg#*=}" ;;
        --seed=*) CUTECHESS_SEED="${arg#*=}" ;;
        --opening-order=*) OPENING_ORDER="${arg#*=}" ;;
        --restart=*) ENGINE_RESTART="${arg#*=}" ;;
        --sync-local) SYNC_BIN=1 ;;
        --no-sync) SYNC_BIN=0 ;;
        --no-remote-build) REMOTE_BUILD=0 ;;
        --no-clean) CLEAN_REMOTE=0 ;;
    esac
done

if [ "$ENGINE_THREADS" -lt 1 ]; then
    ENGINE_THREADS=1
fi
if [ "$MCTS_THREADS" -lt 1 ]; then
    MCTS_THREADS=1
fi
if [ "$MCTS_THREADS" -ge "$ENGINE_THREADS" ] && [ "$ENGINE_THREADS" -gt 1 ]; then
    MCTS_THREADS=$(( ENGINE_THREADS - 1 ))
fi
if [ "$HYBRID_THREADS" -lt 3 ]; then
    HYBRID_THREADS=3
fi
if [ "$HYBRID_MCTS_THREADS" -gt 0 ] && [ "$HYBRID_MCTS_THREADS" -ge "$HYBRID_THREADS" ]; then
    HYBRID_MCTS_THREADS=$(( HYBRID_THREADS - 1 ))
fi
if [ "$HYBRID_AB_THREADS" -gt 0 ] && [ "$HYBRID_AB_THREADS" -gt "$HYBRID_THREADS" ]; then
    HYBRID_AB_THREADS=$HYBRID_THREADS
fi

mkdir -p "$RESULTS_DIR"
ssh_cmd() { ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=60 "$USER@$1" "${@:2}"; }
scp_cmd() { scp -i "$PEM" -o StrictHostKeyChecking=no -q "$@"; }

if [ "$SYNC_BIN" -eq 1 ]; then
    echo "--- Building local metalfish for deployment ---"
    cmake --build "$PROJ/build" -j8 --target metalfish
fi

if [ "$CLEAN_REMOTE" -eq 1 ]; then
    echo "--- Cleaning remote processes and stale PGNs ---"
    for host in "${HOSTS[@]}"; do
        ssh_cmd "$host" "killall -9 metalfish stockfish berserk patricia lc0 cutechess-cli 2>/dev/null || true; rm -f $RDIR/*.pgn; echo $host:clean" &
    done
    wait
fi

if [ "$REMOTE_BUILD" -eq 1 ]; then
    echo "--- Building metalfish natively on all hosts ---"
    for host in "${HOSTS[@]}"; do
        ssh_cmd "$host" "set -e; cd $RDIR; CMAKE=$REMOTE_CMAKE; [ -x \$CMAKE ] || CMAKE=cmake; if [ ! -f $RDIR/$REMOTE_BUILD_DIR/CMakeCache.txt ]; then \$CMAKE -S . -B $REMOTE_BUILD_DIR -DCMAKE_BUILD_TYPE=Release; fi; \$CMAKE --build $REMOTE_BUILD_DIR -j8 --target metalfish; chmod +x $RDIR/$REMOTE_BUILD_DIR/metalfish; echo $host:remote-build-ok" &
    done
    wait
fi

if [ "$SYNC_BIN" -eq 1 ]; then
    echo "--- Syncing metalfish binary to all hosts ---"
    for host in "${HOSTS[@]}"; do
        scp_cmd "$PROJ/build/metalfish" "$USER@$host:$RDIR/build/metalfish" &
    done
    wait
    ENGINE_BIN="$RDIR/build/metalfish"
fi

for host in "${HOSTS[@]}"; do
    ssh_cmd "$host" "{ echo uci; echo quit; } | $ENGINE_BIN >/dev/null; echo $host:binary-ok" &
done
wait

CC="$RDIR/reference/cutechess/build/cutechess-cli"
BK="$RDIR/reference/books/8moves_v3.pgn"
W="$RDIR/networks/BT4-1024x15x32h-swa-6147500.pb"
CA="-each tc=$TC -games $GAMES -repeat -recover -wait $WAIT_MS -srand $CUTECHESS_SEED -openings file=$BK format=pgn order=$OPENING_ORDER -resign movecount=3 score=1000 twosided=true -draw movenumber=40 movecount=8 score=10"

AB="-engine proto=uci restart=$ENGINE_RESTART cmd=$ENGINE_BIN name=MetalFish-AB option.Threads=$ENGINE_THREADS option.Hash=512 option.UseMCTS=false option.UseHybridSearch=false option.MultiPV=1"
MCTS="-engine proto=uci restart=$ENGINE_RESTART cmd=$ENGINE_BIN name=MetalFish-MCTS option.Threads=$MCTS_THREADS option.UseHybridSearch=false option.UseMCTS=true option.NNWeights=$W option.MultiPV=1 option.MCTSMaxThreads=$MCTS_THREADS option.MCTSMinibatchSize=0 option.MCTSParityPreset=false option.MCTSAddDirichletNoise=false option.MCTSMinimumKLDGainPerNode=0.00005 timemargin=30000"
HYB="-engine proto=uci restart=$ENGINE_RESTART cmd=$ENGINE_BIN name=MetalFish-Hybrid option.Threads=$HYBRID_THREADS option.Hash=512 option.UseMCTS=false option.UseHybridSearch=true option.NNWeights=$W option.MultiPV=1 option.HybridMCTSThreads=$HYBRID_MCTS_THREADS option.HybridABThreads=$HYBRID_AB_THREADS option.HybridAutoABThreadsCap=$HYBRID_AUTO_AB_THREADS_CAP option.TransformerLowTimeFallbackMs=$HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS option.TransformerMinMoveBudgetMs=$HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS option.MCTSMaxThreads=$HYBRID_MCTS_THREADS option.MCTSMinibatchSize=$HYBRID_MCTS_MINIBATCH option.MCTSParityPreset=false option.MCTSAddDirichletNoise=false option.HybridMCTSMinimumKLDGainPerNode=$HYBRID_MCTS_KLD option.HybridMCTSRootReject=$HYBRID_MCTS_ROOT_REJECT option.HybridMCTSUseSharedTT=$HYBRID_MCTS_SHARED_TT option.HybridABPolicyWeight=$HYBRID_AB_POLICY_WEIGHT option.HybridTrace=$HYBRID_TRACE timemargin=30000"

S="$RDIR/reference/stockfish/src/stockfish"
SF="-engine proto=uci restart=$ENGINE_RESTART cmd=$S name=Stockfish option.Threads=1 option.Hash=512"
S16='-engine proto=uci restart='$ENGINE_RESTART' cmd='$S' name=SF-L16 option.Threads=1 option.Hash=512 "option.Skill Level=16"'
S14='-engine proto=uci restart='$ENGINE_RESTART' cmd='$S' name=SF-L14 option.Threads=1 option.Hash=512 "option.Skill Level=14"'
S12='-engine proto=uci restart='$ENGINE_RESTART' cmd='$S' name=SF-L12 option.Threads=1 option.Hash=512 "option.Skill Level=12"'
S10='-engine proto=uci restart='$ENGINE_RESTART' cmd='$S' name=SF-L10 option.Threads=1 option.Hash=512 "option.Skill Level=10"'
S8='-engine proto=uci restart='$ENGINE_RESTART' cmd='$S' name=SF-L8 option.Threads=1 option.Hash=512 "option.Skill Level=8"'
S5='-engine proto=uci restart='$ENGINE_RESTART' cmd='$S' name=SF-L5 option.Threads=1 option.Hash=512 "option.Skill Level=5"'
BER="-engine proto=uci restart=$ENGINE_RESTART cmd=$RDIR/reference/berserk/src/berserk name=Berserk option.Threads=1 option.Hash=512"
PAT="-engine proto=uci restart=$ENGINE_RESTART cmd=$RDIR/reference/Patricia/engine/patricia name=Patricia option.Threads=1 option.Hash=512"
LC0="-engine proto=uci restart=$ENGINE_RESTART cmd=$RDIR/reference/lc0/build/release/lc0 name=Lc0 arg=--weights=$W arg=--backend=metal option.Threads=1"

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
echo "  33 matches × $GAMES games = $((33 * GAMES)) games | TC: $TC | wait: ${WAIT_MS}ms"
echo "  Openings: order=$OPENING_ORDER | seed=$CUTECHESS_SEED"
echo "  Threads: AB=$ENGINE_THREADS MCTS=$MCTS_THREADS Hybrid=$HYBRID_THREADS (HybridMCTS=$HYBRID_MCTS_THREADS, HybridAB=$HYBRID_AB_THREADS, HybridAutoABCap=$HYBRID_AUTO_AB_THREADS_CAP)"
echo "  Hybrid knobs: KLD=$HYBRID_MCTS_KLD RootReject=$HYBRID_MCTS_ROOT_REJECT SharedTT=$HYBRID_MCTS_SHARED_TT ABPolicyWeight=$HYBRID_AB_POLICY_WEIGHT Trace=$HYBRID_TRACE"
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
    w=$(grep -c '\[Result "1-0"\]' "$pgn" 2>/dev/null || true)
    d=$(grep -c '\[Result "1/2-1/2"\]' "$pgn" 2>/dev/null || true)
    l=$(grep -c '\[Result "0-1"\]' "$pgn" 2>/dev/null || true)
    [ -n "$w" ] || w=0
    [ -n "$d" ] || d=0
    [ -n "$l" ] || l=0
    t=$((w+d+l)); [ $t -eq 0 ] && continue
    s=$(echo "scale=1; $w + $d * 0.5" | bc)
    printf "  %-35s %4d %4d %4d %5s/%d\n" "$label" "$w" "$d" "$l" "$s" "$t"
done
echo ""
echo "Combined: $RESULTS_DIR/all_games.pgn"
