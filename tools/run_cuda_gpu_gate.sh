#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METALFISH_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_COMMIT="${METALFISH_SOURCE_COMMIT:-}"
if [[ -z "${SOURCE_COMMIT}" ]]; then
  SOURCE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
fi
if [[ -z "${SOURCE_COMMIT}" ]]; then
  echo "METALFISH_SOURCE_COMMIT is required when ${ROOT_DIR} is not a git checkout" >&2
  exit 2
fi
export METALFISH_SOURCE_COMMIT="${SOURCE_COMMIT}"
BUILD_DIR="${METALFISH_CUDA_BUILD_DIR:-${ROOT_DIR}/build-cuda-gpu}"
CUDA_ARCHS="${METALFISH_CUDA_ARCHS:-89}"
JOBS="${METALFISH_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
UCI_GO="${METALFISH_CUDA_UCI_GO:-nodes 8}"
MCTS_TIMED_GO="${METALFISH_CUDA_MCTS_TIMED_GO:-movetime 500}"
MCTS_PONDER_GO="${METALFISH_CUDA_MCTS_PONDER_GO:-wtime 60000 btime 60000 winc 1000 binc 1000}"
MCTS_PONDER_SETTLE_SEC="${METALFISH_CUDA_MCTS_PONDER_SETTLE_SEC:-0.6}"
UCI_TIMEOUT="${METALFISH_CUDA_UCI_TIMEOUT:-180}"
BK07_FEN="1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - -"
KIWIPETE_FEN="r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QK2R w KQkq - 0 1"
WEIGHTS="${METALFISH_NN_WEIGHTS:-${ROOT_DIR}/networks/BT4-1024x15x32h-swa-6147500.pb}"
LEGACY_WEIGHTS="${METALFISH_LEGACY_NN_WEIGHTS:-${ROOT_DIR}/networks/legacy-42850.pb.gz}"
APT_LOCK_TIMEOUT="${METALFISH_APT_LOCK_TIMEOUT:-600}"
SUMMARY="${METALFISH_CUDA_SUMMARY:-${BUILD_DIR}/cuda-gpu-summary.md}"
PARITY_REPORT="${METALFISH_NN_PARITY_REPORT:-${BUILD_DIR}/cuda-gpu-parity-report.md}"
CUDA_PACKAGE_NAME="${METALFISH_CUDA_PACKAGE_NAME:-metalfish-linux-x86_64-cuda}"
CUDA_PACKAGE="${METALFISH_CUDA_PACKAGE:-${BUILD_DIR}/${CUDA_PACKAGE_NAME}.tar.gz}"
CUDA_PROFILE_REQUESTED="${METALFISH_CUDA_PROFILE:-0}"
CUDA_PROFILE_LIMIT="${METALFISH_CUDA_PROFILE_LIMIT:-8}"
CUDA_STABLE_BATCH_SIZE="${METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE:-16}"
CUDA_GRAPH_REPLAY_WARMUP="${METALFISH_CUDA_GRAPH_REPLAY_WARMUP:-2}"
CUDA_GRAPH_REQUESTED=1
if [[ "${METALFISH_CUDA_GRAPH:-}" == "0" ||
      "${METALFISH_CUDA_GRAPH_EXECUTION:-}" == "0" ]]; then
  CUDA_GRAPH_REQUESTED=0
fi
CUDA_GRAPH_CLI_VALUE=true
CUDA_GRAPH_UCI_VALUE=true
if [[ "${CUDA_GRAPH_REQUESTED}" == "0" ]]; then
  CUDA_GRAPH_CLI_VALUE=false
  CUDA_GRAPH_UCI_VALUE=false
fi
CUDA_GRAPH_REPLAY_REQUIRE_ARGS=()
if [[ "${CUDA_GRAPH_REQUESTED}" == "1" ]]; then
  CUDA_GRAPH_REPLAY_REQUIRE_ARGS=(
    --require-network-info-substring "cuda_graph_effective=true"
    --require-network-info-substring "executor=resolved+graph-replay"
  )
fi
CUDA_RUNTIME_REQUIRE_ARGS=(
  --require-network-info-substring "cuda_device_config=-1"
  --require-network-info-substring "cuda_stable_execution_batch_effective=${CUDA_STABLE_BATCH_SIZE}"
  --require-network-info-substring "cuda_deterministic_attention_softmax=true"
  --require-network-info-substring "cuda_full_buffer_clear_effective=true"
)
UCI_CUDA_RUNTIME_EXPECT_ARGS=(
  --expect-output "cuda_device_config=-1"
  --expect-output "cuda_stable_execution_batch_effective=${CUDA_STABLE_BATCH_SIZE}"
  --expect-output "cuda_deterministic_attention_softmax=true"
  --expect-output "cuda_full_buffer_clear_effective=true"
)
UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS=(
  --expect-output "capabilities=actual_backend=cuda"
)
if [[ "${CUDA_GRAPH_REQUESTED}" == "1" ]]; then
  UCI_CUDA_RUNTIME_EXPECT_ARGS+=(
    --expect-output "cuda_graph_effective=true"
  )
  UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS+=(
    --expect-output "MCTS backend warmup actual="
    --expect-output "executor=resolved+graph-replay"
  )
fi
if [[ ! "${CUDA_STABLE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${CUDA_GRAPH_REPLAY_WARMUP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "METALFISH_CUDA_GRAPH_REPLAY_WARMUP must be a positive integer" >&2
  exit 2
fi

export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.4/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12.9/lib64:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

wait_for_apt_locks() {
  local deadline=$((SECONDS + APT_LOCK_TIMEOUT))
  local locks=(
    /var/lib/dpkg/lock-frontend
    /var/lib/dpkg/lock
    /var/lib/apt/lists/lock
    /var/cache/apt/archives/lock
  )

  while command -v fuser >/dev/null 2>&1 &&
        sudo fuser "${locks[@]}" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "timed out waiting for apt/dpkg locks" >&2
      sudo fuser -v "${locks[@]}" >&2 || true
      return 1
    fi
    sleep 5
  done
}

run_apt_get() {
  local attempt
  for attempt in 1 2 3; do
    wait_for_apt_locks
    if sudo apt-get "$@"; then
      return 0
    fi
    if [[ "${attempt}" == "3" ]]; then
      return 1
    fi
    sleep 10
  done
}

SUMMARY_WRITTEN=0

summary_line_or_missing() {
  local pattern="$1"
  local file="$2"
  local fallback="$3"
  if [[ -s "${file}" ]]; then
    grep -m1 "${pattern}" "${file}" || echo "${fallback}"
  else
    echo "${fallback}"
  fi
}

summary_log_status() {
  local file="$1"
  if [[ -s "${file}" ]]; then
    echo "present"
  else
    echo "missing"
  fi
}

summary_failure_lines() {
  local label="$1"
  local file="$2"
  if [[ ! -s "${file}" ]]; then
    return
  fi

  local matches
  matches=$(grep -E \
    '^[[:space:]]*(FAIL|ERROR|FATAL):|CMake Error|FAILED:|ninja: build stopped' \
    "${file}" | head -20 || true)
  if [[ -z "${matches}" ]]; then
    return
  fi

  echo
  echo "### ${label}"
  echo
  echo '```text'
  echo "${matches}"
  echo '```'
}

assert_cuda_isolation_probe_log() {
  local name="$1"
  local file="$2"
  grep -q '"isolation":true' "${file}" || {
    echo "${name} did not emit an isolation probe" >&2
    exit 1
  }
  grep -q '"backend":"cuda"' "${file}" || {
    echo "${name} did not select CUDA" >&2
    exit 1
  }
  grep -q 'CUDA transformer backend' "${file}" || {
    echo "${name} did not report CUDA transformer backend" >&2
    exit 1
  }
  grep -q 'cuda_device_config=-1' "${file}" || {
    echo "${name} did not report requested CUDA device" >&2
    exit 1
  }
  grep -q "cuda_stable_execution_batch_effective=${CUDA_STABLE_BATCH_SIZE}" "${file}" || {
    echo "${name} did not report effective CUDA stable batch size" >&2
    exit 1
  }
  grep -q 'cuda_deterministic_attention_softmax=true' "${file}" || {
    echo "${name} did not report deterministic CUDA attention softmax" >&2
    exit 1
  }
  grep -q 'cuda_full_buffer_clear_effective=true' "${file}" || {
    echo "${name} did not report effective CUDA full-buffer clear" >&2
    exit 1
  }
  if [[ "${CUDA_GRAPH_REQUESTED}" == "1" ]]; then
    grep -q 'cuda_graph_effective=true' "${file}" || {
      echo "${name} did not report effective CUDA graph execution" >&2
      exit 1
    }
    grep -q 'executor=resolved+graph-replay' "${file}" || {
      echo "${name} did not report CUDA graph replay" >&2
      exit 1
    }
  fi
  grep -q '"delta":' "${file}" || {
    echo "${name} did not report isolation deltas" >&2
    exit 1
  }
}

write_summary() {
  local gate_status="${1:-passed}"
  local graph_replay_observed="no"
  if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
     grep -q "executor=resolved+graph-replay" \
       "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
    graph_replay_observed="yes"
  fi
  mkdir -p "$(dirname "${SUMMARY}")"
  {
    echo "# MetalFish CUDA GPU Gate Summary"
    echo
    echo "- Gate status: ${gate_status}"
    echo "- Timestamp UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "- Host: $(hostname)"
    echo "- CUDA architectures: ${CUDA_ARCHS}"
    echo "- Build directory: ${BUILD_DIR}"
    echo "- Weights: ${WEIGHTS}"
    echo "- Legacy weights: ${LEGACY_WEIGHTS}"
    echo "- Parity report: ${PARITY_REPORT}"
    echo "- Linux CUDA package: ${CUDA_PACKAGE}"
    echo "- Explicit CUDA UCI go: ${UCI_GO}"
    echo "- Batch worst trace: ${METALFISH_NN_BATCH_TRACE_WORST:-1}"
    echo "- Single repeat stress: ${METALFISH_NN_SINGLE_REPEAT_STRESS:-1}"
    echo "- CUDA graph replay warmup: ${CUDA_GRAPH_REPLAY_WARMUP}"
    echo "- Single reuse stress: ${METALFISH_NN_SINGLE_REUSE_STRESS:-1}"
    echo "- Batch reuse stress: ${METALFISH_NN_BATCH_REUSE_STRESS:-1}"
    echo "- CUDA full buffer clear: ${METALFISH_CUDA_FULL_BUFFER_CLEAR:-1}"
    echo "- CUDA graph execution: ${CUDA_GRAPH_REQUESTED}"
    echo "- CUDA graph replay observed: ${graph_replay_observed}"
    echo "- CUDA release single workspace each run: ${METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN:-0}"
    echo "- CUDA release workspace each run: ${METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN:-0}"
    echo "- CUDA stable execution batch size: ${CUDA_STABLE_BATCH_SIZE}"
    echo "- CUDA deterministic attention softmax: ${METALFISH_CUDA_DETERMINISTIC_ATTENTION_SOFTMAX:-1}"
    echo "- CUDA raw output trace: ${METALFISH_CUDA_TRACE_RAW_OUTPUTS:-0}"
    echo "- CUDA stage output trace: ${METALFISH_CUDA_TRACE_STAGE_OUTPUTS:-0}"
    echo "- CUDA attention internals trace: ${METALFISH_CUDA_TRACE_ATTENTION_INTERNALS:-0}"
    echo "- CUDA dynamic PE internals trace: ${METALFISH_CUDA_TRACE_DYNAMIC_PE_INTERNALS:-0}"
    echo "- CUDA trace compare base run: ${METALFISH_CUDA_TRACE_COMPARE_BASE_RUN:-unset}"
    echo "- CUDA trace compare min delta: ${METALFISH_CUDA_TRACE_COMPARE_MIN_DELTA:-1e-7}"
    echo "- cuBLAS workspace config: ${CUBLAS_WORKSPACE_CONFIG:-unset}"
    echo
    echo "## Device"
    echo
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
      --format=csv,noheader || echo "- unavailable"
    echo
    echo "## Logs"
    echo
    echo "- CUDA tests: $(summary_log_status "${BUILD_DIR}/cuda-gpu-tests.log")"
    echo "- NN comparison: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-comparison.log")"
    echo "- NN probe: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-probe.log")"
    echo "- NN probe suite: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-probe-suite.log")"
    echo "- Legacy NN probe suite: $(summary_log_status "${BUILD_DIR}/cuda-gpu-legacy-nn-probe-suite.log")"
    echo "- BT4/legacy isolation probe: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-isolation-bt4-legacy.log")"
    echo "- Legacy/BT4 isolation probe: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-isolation-legacy-bt4.log")"
    echo "- NN artifact manifest: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-artifact-manifest.json")"
    echo "- auto UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log")"
    echo "- accelerator UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-accelerator-smoke.log")"
    echo "- explicit CUDA UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-smoke.log")"
    echo "- timed CUDA MCTS smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-timed-mcts-smoke.log")"
    echo "- timed CUDA MCTS search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-timed-mcts-search.json")"
    echo "- ponder CUDA MCTS smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-ponder-mcts-smoke.log")"
    echo "- ponder CUDA MCTS JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-ponder-mcts.json")"
    echo "- BK.07 CUDA tactical smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-bk07-smoke.log")"
    echo "- BK.07 CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-bk07-search.json")"
    echo "- kiwipete CUDA search smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-kiwipete-smoke.log")"
    echo "- kiwipete CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-kiwipete-search.json")"
    echo "- hybrid CUDA UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log")"
    echo "- hybrid CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-search.json")"
    echo "- hybrid kiwipete CUDA UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-kiwipete-smoke.log")"
    echo "- hybrid kiwipete CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-kiwipete-search.json")"
    echo "- hybrid CUDA clock-start smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-start-smoke.log")"
    echo "- hybrid CUDA clock-safety smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-safety-smoke.log")"
    echo "- hybrid auto UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-auto-smoke.log")"
    echo "- hybrid ANE-disable smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-ane-smoke.log")"
    echo "- Linux CUDA package: $(summary_log_status "${CUDA_PACKAGE}")"
    echo "- packaged NN comparison: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log")"
    echo "- packaged CUDA probe: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-probe.log")"
    echo "- packaged CUDA probe suite: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-nn-probe-suite.log")"
    echo "- packaged legacy probe suite: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-legacy-nn-probe-suite.log")"
    echo "- packaged BT4/legacy isolation probe: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-nn-isolation-bt4-legacy.log")"
    echo "- packaged legacy/BT4 isolation probe: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-nn-isolation-legacy-bt4.log")"
    echo "- packaged CUDA UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-smoke.log")"
    echo "- packaged ponder CUDA MCTS smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-ponder-mcts-smoke.log")"
    echo "- packaged ponder CUDA MCTS JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-ponder-mcts.json")"
    echo "- packaged BK.07 CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-bk07-search.json")"
    echo "- packaged kiwipete CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-kiwipete-search.json")"
    echo "- packaged hybrid CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-search.json")"
    echo "- packaged hybrid kiwipete CUDA search JSON: $(summary_log_status "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-kiwipete-search.json")"
    echo "- Linux CUDA package manifest: $(summary_log_status "${BUILD_DIR}/linux-cuda-package-manifest.json")"
    echo "- Linux CUDA package check: $(summary_log_status "${BUILD_DIR}/linux-cuda-package-check.json")"
    echo "- CUDA profile: $(summary_log_status "${BUILD_DIR}/cuda-gpu-profile.log")"
    echo
    echo "## Failures"
    summary_failure_lines "CUDA tests" "${BUILD_DIR}/cuda-gpu-tests.log"
    summary_failure_lines "NN comparison" \
      "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
    summary_failure_lines "NN probe" "${BUILD_DIR}/cuda-gpu-nn-probe.log"
    summary_failure_lines "NN probe suite" \
      "${BUILD_DIR}/cuda-gpu-nn-probe-suite.log"
    summary_failure_lines "Legacy NN probe suite" \
      "${BUILD_DIR}/cuda-gpu-legacy-nn-probe-suite.log"
    summary_failure_lines "BT4/legacy isolation probe" \
      "${BUILD_DIR}/cuda-gpu-nn-isolation-bt4-legacy.log"
    summary_failure_lines "Legacy/BT4 isolation probe" \
      "${BUILD_DIR}/cuda-gpu-nn-isolation-legacy-bt4.log"
    summary_failure_lines "auto UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log"
    summary_failure_lines "accelerator UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-accelerator-smoke.log"
    summary_failure_lines "explicit CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-smoke.log"
    summary_failure_lines "Timed CUDA MCTS smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-timed-mcts-smoke.log"
    summary_failure_lines "Ponder CUDA MCTS smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-ponder-mcts-smoke.log"
    summary_failure_lines "BK.07 CUDA tactical smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-bk07-smoke.log"
    summary_failure_lines "kiwipete CUDA search smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-kiwipete-smoke.log"
    summary_failure_lines "hybrid CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log"
    summary_failure_lines "hybrid CUDA clock-start smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-start-smoke.log"
    summary_failure_lines "hybrid CUDA clock-safety smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-safety-smoke.log"
    summary_failure_lines "hybrid auto UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-auto-smoke.log"
    summary_failure_lines "hybrid ANE-disable smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-ane-smoke.log"
    summary_failure_lines "packaged NN comparison" \
      "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
    summary_failure_lines "packaged ponder CUDA MCTS smoke" \
      "${BUILD_DIR}/cuda-gpu-package-uci-ponder-mcts-smoke.log"
    summary_failure_lines "packaged CUDA probe" \
      "${BUILD_DIR}/cuda-gpu-package-probe.log"
    summary_failure_lines "packaged CUDA probe suite" \
      "${BUILD_DIR}/cuda-gpu-package-nn-probe-suite.log"
    summary_failure_lines "packaged legacy probe suite" \
      "${BUILD_DIR}/cuda-gpu-package-legacy-nn-probe-suite.log"
    summary_failure_lines "packaged BT4/legacy isolation probe" \
      "${BUILD_DIR}/cuda-gpu-package-nn-isolation-bt4-legacy.log"
    summary_failure_lines "packaged legacy/BT4 isolation probe" \
      "${BUILD_DIR}/cuda-gpu-package-nn-isolation-legacy-bt4.log"
    summary_failure_lines "packaged CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-package-uci-smoke.log"
    summary_failure_lines "packaged BK.07 CUDA search smoke" \
      "${BUILD_DIR}/cuda-gpu-package-uci-bk07-smoke.log"
    summary_failure_lines "packaged kiwipete CUDA search smoke" \
      "${BUILD_DIR}/cuda-gpu-package-uci-kiwipete-smoke.log"
    summary_failure_lines "packaged hybrid CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-smoke.log"
    summary_failure_lines "packaged hybrid kiwipete CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-kiwipete-smoke.log"
    summary_failure_lines "CUDA profile" "${BUILD_DIR}/cuda-gpu-profile.log"
    echo
    echo "## Backend"
    echo
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -m1 "backend: CUDA transformer backend" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      true
    elif [[ -s "${PARITY_REPORT}" ]] &&
         grep -m1 "^- Backend: CUDA transformer backend" "${PARITY_REPORT}"; then
      true
    else
      echo "- not reached"
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -m1 "backend_after: CUDA transformer backend" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      true
    fi
    echo
    echo "## Batch Timings"
    echo
    summary_line_or_missing "batches:" \
      "${BUILD_DIR}/cuda-gpu-nn-comparison.log" "- skipped"
    summary_line_or_missing "graph_reuse_probe:" \
      "${BUILD_DIR}/cuda-gpu-nn-comparison.log" "- skipped"
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -q "TRACE_WORST:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## Batch Worst Trace"
      echo
      grep -m1 "TRACE_WORST:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_POLICY:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_CONFIRMED:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_CONFIRMED_POLICY:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_REUSED_SINGLE:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_REUSED_BATCH:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_SINGLE_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_BATCH_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_REUSED_SINGLE_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "TRACE_WORST_REUSED_BATCH_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -q "CUDA_STAGE_TRACE_COMPARE" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## CUDA Stage Trace Compare"
      echo
      grep -m12 "CUDA_STAGE_TRACE_COMPARE" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -Eq '^CUDA_STAGE_TRACE .*name=.*\.mha\.' \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## CUDA Attention Trace"
      echo
      grep -E '^CUDA_STAGE_TRACE .*name=.*\.mha\.' \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" | head -16 || true
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -Eq '^CUDA_STAGE_TRACE .*name=.*\.(expanded|position_input|dense)' \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## CUDA Dynamic PE Trace"
      echo
      grep -E '^CUDA_STAGE_TRACE .*name=.*\.(expanded|position_input|dense)' \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" | head -12 || true
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -q "^[[:space:]]*REUSE_STRESS_MAX:" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## Batch Reuse Stress"
      echo
      grep -m1 "^[[:space:]]*REUSE_STRESS_MAX:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*REUSE_STRESS_POLICY:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*REUSE_STRESS_SINGLE_TOP:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*REUSE_STRESS_BATCH_TOP:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -q "^[[:space:]]*SINGLE_REUSE_STRESS_MAX:" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## Single Reuse Stress"
      echo
      grep -m1 "^[[:space:]]*SINGLE_REUSE_STRESS_MAX:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*SINGLE_REUSE_STRESS_POLICY:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*SINGLE_REUSE_STRESS_BASELINE_TOP:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*SINGLE_REUSE_STRESS_REPLAY_TOP:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    fi
    if [[ -s "${BUILD_DIR}/cuda-gpu-nn-comparison.log" ]] &&
       grep -q "^[[:space:]]*SINGLE_REPEAT_STRESS_MAX:" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
      echo
      echo "## Single Repeat Stress"
      echo
      grep -m1 "^[[:space:]]*SINGLE_REPEAT_STRESS_MAX:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*SINGLE_REPEAT_STRESS_POLICY:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*SINGLE_REPEAT_STRESS_BASELINE_TOP:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
      grep -m1 "^[[:space:]]*SINGLE_REPEAT_STRESS_REPLAY_TOP:" \
        "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    fi
    echo
    echo "## Parity Report"
    echo
    if [[ -s "${PARITY_REPORT}" ]]; then
      grep -m1 "^- Backend:" "${PARITY_REPORT}" || true
      local fixed_rows
      local batch_rows
      fixed_rows=$(awk '/^## Fixed BT4 Reference/{section=1;next} /^## Batch Parity/{section=0} section && /^\| [[:lower:]][^|]* \|/{count++} END{print count + 0}' "${PARITY_REPORT}")
      batch_rows=$(awk '/^## Batch Parity/{section=1;next} section && /^\| [0-9]+ \|/{count++} END{print count + 0}' "${PARITY_REPORT}")
      echo "- Fixed references: ${fixed_rows}"
      echo "- Batch rows: ${batch_rows}"
    else
      echo "- missing"
    fi
    echo
    echo "## UCI Smokes"
    echo
    echo "- auto: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log" "not reached")"
    echo "- accelerator: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-accelerator-smoke.log" "not reached")"
    echo "- cuda: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-smoke.log" "not reached")"
    echo "- cuda-timed-mcts: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-timed-mcts-smoke.log" "not reached")"
    echo "- cuda-ponder-mcts: $(summary_line_or_missing '^ponder_smoke ' "${BUILD_DIR}/cuda-gpu-uci-ponder-mcts-smoke.log" "not reached")"
    echo "- cuda-bk07: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-bk07-smoke.log" "not reached")"
    echo "- cuda-kiwipete: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-kiwipete-smoke.log" "not reached")"
    echo "- hybrid-cuda: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log" "not reached")"
    echo "- hybrid-kiwipete: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-kiwipete-smoke.log" "not reached")"
    echo "- hybrid-clock-start: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-start-smoke.log" "not reached")"
    echo "- hybrid-clock-safety: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-safety-smoke.log" "not reached")"
    echo "- hybrid-auto: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-auto-smoke.log" "not reached")"
    echo "- packaged-cuda-bk07: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-package-uci-bk07-smoke.log" "not reached")"
    echo "- packaged-cuda-ponder-mcts: $(summary_line_or_missing '^ponder_smoke ' "${BUILD_DIR}/cuda-gpu-package-uci-ponder-mcts-smoke.log" "not reached")"
    echo "- packaged-cuda-kiwipete: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-package-uci-kiwipete-smoke.log" "not reached")"
    echo "- packaged-hybrid-cuda: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-smoke.log" "not reached")"
    echo "- packaged-hybrid-kiwipete: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-kiwipete-smoke.log" "not reached")"
    if [[ -s "${BUILD_DIR}/cuda-gpu-profile.log" ]]; then
      echo
      echo "## CUDA Profile"
      echo
      grep -m1 "CUDA profile report=" "${BUILD_DIR}/cuda-gpu-profile.log" || true
      grep -m1 "CUDA profile buckets:" "${BUILD_DIR}/cuda-gpu-profile.log" || true
      grep -m1 "CUDA profile slowest:" "${BUILD_DIR}/cuda-gpu-profile.log" || true
    fi
  } >"${SUMMARY}"
  SUMMARY_WRITTEN=1
}

write_failure_summary_on_exit() {
  local status=$?
  if [[ "${status}" != "0" && "${SUMMARY_WRITTEN}" != "1" ]]; then
    write_summary "failed (${status})" || true
  fi
}
trap write_failure_summary_on_exit EXIT

if [[ "${METALFISH_INSTALL_DEPS:-0}" == "1" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  for attempt in 1 2 3; do
    if run_apt_get update && run_apt_get install -y \
      build-essential \
      cmake \
      curl \
      ninja-build \
      pkg-config \
      protobuf-compiler \
      libprotobuf-dev \
      zlib1g-dev \
      libabsl-dev \
      python3 \
      python3-pip; then
      break
    fi
    if [[ "${attempt}" == "3" ]]; then
      exit 1
    fi
    sleep 20
  done
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the CUDA GPU gate" >&2
  exit 2
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required for the CUDA GPU gate" >&2
  exit 2
fi

cd "${ROOT_DIR}"

nvidia-smi
nvcc --version

python3 -m pip install --user -r tests/requirements.txt || \
  python3 -m pip install --user --break-system-packages -r tests/requirements.txt

python3 tools/download_engine_networks.py --nnue-only
if [[ "${METALFISH_CUDA_DOWNLOAD_BT4:-1}" == "1" && ! -s "${WEIGHTS}" ]]; then
  python3 tools/download_engine_networks.py --bt4-only
fi
if [[ "${METALFISH_CUDA_LEGACY_PROBE:-1}" == "1" &&
      "${METALFISH_CUDA_DOWNLOAD_LEGACY:-1}" == "1" &&
      ! -s "${LEGACY_WEIGHTS}" ]]; then
  python3 tools/download_engine_networks.py --legacy-only
fi

if [[ ! -s "${WEIGHTS}" ]]; then
  echo "BT4 weights not found: ${WEIGHTS}" >&2
  exit 2
fi

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_METAL=OFF \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -DBUILD_TESTS=ON \
  -DMETALFISH_ENABLE_IPO=OFF

cmake --build "${BUILD_DIR}" --target metalfish metalfish_tests \
  test_nn_comparison metalfish_nn_probe -j"${JOBS}"

METALFISH_CUDA_TRACE_STAGE_OUTPUTS=0 \
  METALFISH_CUDA_PROFILE=0 \
  "${BUILD_DIR}/metalfish_tests" | tee "${BUILD_DIR}/cuda-gpu-tests.log"
grep -q "CUDA runtime" "${BUILD_DIR}/cuda-gpu-tests.log"

METALFISH_NN_WEIGHTS="${WEIGHTS}" \
  METALFISH_NN_PARITY_REPORT="${PARITY_REPORT}" \
  METALFISH_NN_BATCH_BENCH="${METALFISH_NN_BATCH_BENCH:-1}" \
  METALFISH_NN_BATCH_TRACE_WORST="${METALFISH_NN_BATCH_TRACE_WORST:-1}" \
  METALFISH_NN_SINGLE_REPEAT_STRESS="${METALFISH_NN_SINGLE_REPEAT_STRESS:-1}" \
  METALFISH_NN_SINGLE_REUSE_STRESS="${METALFISH_NN_SINGLE_REUSE_STRESS:-1}" \
  METALFISH_NN_BATCH_REUSE_STRESS="${METALFISH_NN_BATCH_REUSE_STRESS:-1}" \
  METALFISH_NN_BENCH_ITERS="${METALFISH_NN_BENCH_ITERS:-2}" \
  METALFISH_NN_BENCH_MAX_BATCH="${METALFISH_NN_BENCH_MAX_BATCH:-32}" \
  METALFISH_NN_BENCH_WARMUP_ITERS="${METALFISH_NN_BENCH_WARMUP_ITERS:-3}" \
  METALFISH_NN_BENCH_GRAPH_REUSE_PROBE="${METALFISH_NN_BENCH_GRAPH_REUSE_PROBE:-1}" \
  METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
  METALFISH_CUDA_PROFILE=0 \
  "${BUILD_DIR}/test_nn_comparison" 2>&1 | tee "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
if [[ -n "${CUDA_GRAPH_REQUESTED}" && "${CUDA_GRAPH_REQUESTED}" != "0" &&
      "${METALFISH_CUDA_TRACE_STAGE_OUTPUTS:-0}" == "0" &&
      "${METALFISH_CUDA_TRACE_ATTENTION_INTERNALS:-0}" == "0" &&
      "${METALFISH_CUDA_TRACE_DYNAMIC_PE_INTERNALS:-0}" == "0" &&
      "${METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN:-0}" == "0" &&
      "${METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN:-0}" == "0" ]]; then
  if ! grep -q "executor=resolved+graph-replay" \
         "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
    echo "CUDA graph execution was requested but no graph replay was observed" >&2
    exit 1
  fi
fi

METALFISH_CUDA_PROFILE=0 \
  METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
  "${BUILD_DIR}/metalfish_nn_probe" \
  --weights "${WEIGHTS}" \
  --backend cuda \
  --cuda-device -1 \
  --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
  --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
  --cuda-deterministic-attention-softmax true \
  --cuda-full-buffer-clear true \
  --top 3 \
  --warmup 0 \
  --iterations 1 \
  --full-policy \
  2>&1 | tee "${BUILD_DIR}/cuda-gpu-nn-probe.log"
python3 tools/check_nn_backend_artifacts.py \
  --backend-label "CUDA transformer backend" \
  --parity-report "${PARITY_REPORT}" \
  --comparison-log "${BUILD_DIR}/cuda-gpu-nn-comparison.log" \
  --probe-log "${BUILD_DIR}/cuda-gpu-nn-probe.log" \
  --manifest-out "${BUILD_DIR}/cuda-gpu-nn-artifact-manifest.json" \
  --min-policy-top 3 \
  --require-batch-benchmark
METALFISH_CUDA_PROFILE=0 \
  python3 tools/run_nn_backend_probe_suite.py \
  --probe "${BUILD_DIR}/metalfish_nn_probe" \
  --weights "${WEIGHTS}" \
  --backend cuda \
  --cuda-device -1 \
  --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
  --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
  --cuda-deterministic-attention-softmax true \
  --cuda-full-buffer-clear true \
  --out "${BUILD_DIR}/cuda-gpu-nn-probe-suite.log" \
  --top 3 \
  --batch-size 2 \
  --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
  --iterations 1 \
  --backend-label "CUDA transformer backend" \
  "${CUDA_RUNTIME_REQUIRE_ARGS[@]}" \
  "${CUDA_GRAPH_REPLAY_REQUIRE_ARGS[@]}" \
  --require-wdl \
  --require-moves-left \
  --expected-policy-count 1858 \
  --full-policy
if [[ "${METALFISH_CUDA_LEGACY_PROBE:-1}" == "1" ]]; then
  if [[ ! -s "${LEGACY_WEIGHTS}" ]]; then
    echo "Legacy 42850 weights not found: ${LEGACY_WEIGHTS}" >&2
    exit 2
  fi
  METALFISH_CUDA_PROFILE=0 \
    python3 tools/run_nn_backend_probe_suite.py \
    --probe "${BUILD_DIR}/metalfish_nn_probe" \
    --weights "${LEGACY_WEIGHTS}" \
    --backend cuda \
    --cuda-device -1 \
    --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
    --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
    --cuda-deterministic-attention-softmax true \
    --cuda-full-buffer-clear true \
    --out "${BUILD_DIR}/cuda-gpu-legacy-nn-probe-suite.log" \
    --top 3 \
    --batch-size 2 \
    --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
    --iterations 1 \
    --backend-label "CUDA transformer backend" \
    "${CUDA_RUNTIME_REQUIRE_ARGS[@]}" \
    "${CUDA_GRAPH_REPLAY_REQUIRE_ARGS[@]}" \
    --no-require-wdl \
    --no-require-moves-left \
    --expected-policy-count 1858 \
    --full-policy
  METALFISH_CUDA_PROFILE=0 \
    METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
    "${BUILD_DIR}/metalfish_nn_probe" \
    --weights "${WEIGHTS}" \
    --isolation-weights "${LEGACY_WEIGHTS}" \
    --backend cuda \
    --cuda-device -1 \
    --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
    --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
    --cuda-deterministic-attention-softmax true \
    --cuda-full-buffer-clear true \
    --top 3 \
    --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
    --iterations 1 \
    2>&1 | tee "${BUILD_DIR}/cuda-gpu-nn-isolation-bt4-legacy.log"
  assert_cuda_isolation_probe_log \
    "BT4/legacy CUDA isolation probe" \
    "${BUILD_DIR}/cuda-gpu-nn-isolation-bt4-legacy.log"
  METALFISH_CUDA_PROFILE=0 \
    METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
    "${BUILD_DIR}/metalfish_nn_probe" \
    --weights "${LEGACY_WEIGHTS}" \
    --isolation-weights "${WEIGHTS}" \
    --backend cuda \
    --cuda-device -1 \
    --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
    --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
    --cuda-deterministic-attention-softmax true \
    --cuda-full-buffer-clear true \
    --top 3 \
    --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
    --iterations 1 \
    2>&1 | tee "${BUILD_DIR}/cuda-gpu-nn-isolation-legacy-bt4.log"
  assert_cuda_isolation_probe_log \
    "Legacy/BT4 CUDA isolation probe" \
    "${BUILD_DIR}/cuda-gpu-nn-isolation-legacy-bt4.log"
fi

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=auto \
  --setoption NNBackendRequireAccelerator=true \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=0 \
  --go "nodes 1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=accelerator" \
  --expect-output "minibatch=${CUDA_STABLE_BATCH_SIZE}" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=accelerator \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=0 \
  --go "nodes 1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=accelerator" \
  --expect-output "minibatch=${CUDA_STABLE_BATCH_SIZE}" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-accelerator-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --go "${UCI_GO}" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=true \
  --setoption MCTSMinibatchSize=0 \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --go "${MCTS_TIMED_GO}" \
  --json-out "${BUILD_DIR}/cuda-gpu-uci-timed-mcts-search.json" \
  --expect-output "Starting Multi-Threaded MCTS Search" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --reject-output "Time safety:" \
  --reject-output "Falling back to Alpha-Beta" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-timed-mcts-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_ponder_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --settle-sec "${MCTS_PONDER_SETTLE_SEC}" \
  --ponder-go "${MCTS_PONDER_GO}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption Ponder=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=true \
  --setoption MCTSMinibatchSize=0 \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --json-out "${BUILD_DIR}/cuda-gpu-uci-ponder-mcts.json" \
  --expect-output "Starting Multi-Threaded MCTS Search" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --reject-output "Time safety:" \
  --reject-output "Falling back to Alpha-Beta" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-ponder-mcts-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=false \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${BK07_FEN}" \
  --go "nodes 50" \
  --expect-bestmove h5f6 \
  --json-out "${BUILD_DIR}/cuda-gpu-uci-bk07-search.json" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-bk07-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=false \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${KIWIPETE_FEN}" \
  --go "nodes 1" \
  --json-out "${BUILD_DIR}/cuda-gpu-uci-kiwipete-search.json" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-kiwipete-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${BK07_FEN}" \
  --go "nodes 50" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "Final: MCTSPlayouts=" \
  --json-out "${BUILD_DIR}/cuda-gpu-uci-hybrid-search.json" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${KIWIPETE_FEN}" \
  --go "nodes 50" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "Final: MCTSPlayouts=" \
  --json-out "${BUILD_DIR}/cuda-gpu-uci-hybrid-kiwipete-search.json" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-kiwipete-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --setoption Move\ Overhead=500 \
  --setoption TransformerLowTimeFallbackMs=3000 \
  --setoption TransformerMinMoveBudgetMs=400 \
  --setoption MCTSAddDirichletNoise=false \
  --go "wtime 1000 btime 1000 winc 3000 binc 3000" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=cuda" \
  --expect-output "CUDA transformer backend" \
  --reject-output "Time safety:" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-start-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption Move\ Overhead=500 \
  --setoption TransformerLowTimeFallbackMs=3000 \
  --setoption TransformerMinMoveBudgetMs=400 \
  --go "wtime 800 btime 800 winc 3000 binc 3000" \
  --expect-output "Time safety: estimated move budget" \
  --reject-output "Starting Parallel Hybrid Search" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-clock-safety-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=auto \
  --setoption NNBackendRequireAccelerator=true \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=0 \
  --go "nodes 8" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=accelerator" \
  --expect-output "minibatch=${CUDA_STABLE_BATCH_SIZE}" \
  --expect-output "CUDA transformer backend" \
  --expect-output "Final: MCTSPlayouts=" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-auto-smoke.log"

mkdir -p "${BUILD_DIR}/dummy-coreml.mlmodelc"
METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption HybridANERootProbe=true \
  --setoption HybridANERootHints=true \
  --setoption HybridANEWeights="${WEIGHTS}" \
  --setoption HybridANEModelPath="${BUILD_DIR}/dummy-coreml.mlmodelc" \
  --setoption HybridANEComputeUnits=all \
  --setoption HybridANERootHintWaitMs=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --go "nodes 8" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "ANE root probe disabled" \
  --expect-output "Final: MCTSPlayouts=" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-ane-smoke.log"

CUDA_PACKAGE_DIR="${BUILD_DIR}/linux-cuda-package"
CUDA_PACKAGE_CHECK_DIR="${BUILD_DIR}/linux-cuda-package-check"
rm -rf "${CUDA_PACKAGE_DIR}" "${CUDA_PACKAGE_CHECK_DIR}"
rm -f "${CUDA_PACKAGE}"
mkdir -p "${CUDA_PACKAGE_DIR}" "${CUDA_PACKAGE_CHECK_DIR}"
cp "${BUILD_DIR}/metalfish" "${CUDA_PACKAGE_DIR}/"
cp "${BUILD_DIR}/metalfish_nn_probe" "${CUDA_PACKAGE_DIR}/"
cp "${BUILD_DIR}/test_nn_comparison" "${CUDA_PACKAGE_DIR}/"
cp README.md CHANGELOG.md LICENSE "${CUDA_PACKAGE_DIR}/"
python3 tools/write_portable_manifest.py \
  --platform "Linux x86_64 CUDA" \
  --backend "CUDA transformer backend for BT4 MCTS/Hybrid plus CPU AB/NNUE" \
  --binary "metalfish" \
  --output "${CUDA_PACKAGE_DIR}/PORTABLE_ARTIFACT.md" \
  --json-output "${CUDA_PACKAGE_DIR}/linux-cuda-package-manifest.json" \
  --package-name "${CUDA_PACKAGE_NAME}" \
  --package-kind "linux-cuda" \
  --file "${CUDA_PACKAGE_DIR}/metalfish" \
  --file "${CUDA_PACKAGE_DIR}/metalfish_nn_probe" \
  --file "${CUDA_PACKAGE_DIR}/test_nn_comparison" \
  --file "${CUDA_PACKAGE_DIR}/README.md" \
  --file "${CUDA_PACKAGE_DIR}/CHANGELOG.md" \
  --file "${CUDA_PACKAGE_DIR}/LICENSE" \
  --file "${CUDA_PACKAGE_DIR}/PORTABLE_ARTIFACT.md" \
  --notes "This package is smoke-tested on an NVIDIA L4 runtime gate before upload." \
  --notes "The package includes metalfish_nn_probe so release artifacts can verify CUDA inference metadata." \
  --notes "The package includes test_nn_comparison so release artifacts can verify CUDA batch and reuse parity." \
  --notes "CUDA runtime libraries are expected from the host driver/toolkit installation."
cp "${CUDA_PACKAGE_DIR}/PORTABLE_ARTIFACT.md" "${BUILD_DIR}/PORTABLE_ARTIFACT.md"
cp "${CUDA_PACKAGE_DIR}/linux-cuda-package-manifest.json" \
  "${BUILD_DIR}/linux-cuda-package-manifest.json"
tar -czf "${CUDA_PACKAGE}" -C "${CUDA_PACKAGE_DIR}" .
python3 tools/check_cuda_package_artifacts.py \
  --package "${CUDA_PACKAGE}" \
  --package-kind linux-cuda \
  --expected-source-commit "${SOURCE_COMMIT}" \
  --json-output "${BUILD_DIR}/linux-cuda-package-check.json"
tar -xzf "${CUDA_PACKAGE}" -C "${CUDA_PACKAGE_CHECK_DIR}"
test -x "${CUDA_PACKAGE_CHECK_DIR}/metalfish"
test -x "${CUDA_PACKAGE_CHECK_DIR}/metalfish_nn_probe"
test -x "${CUDA_PACKAGE_CHECK_DIR}/test_nn_comparison"
test -s "${CUDA_PACKAGE_CHECK_DIR}/PORTABLE_ARTIFACT.md"
test -s "${CUDA_PACKAGE_CHECK_DIR}/linux-cuda-package-manifest.json"
grep -q -- "- Platform: Linux x86_64 CUDA" \
  "${CUDA_PACKAGE_CHECK_DIR}/PORTABLE_ARTIFACT.md"
grep -q "CUDA transformer backend" \
  "${CUDA_PACKAGE_CHECK_DIR}/PORTABLE_ARTIFACT.md"
python3 -m json.tool \
  "${CUDA_PACKAGE_CHECK_DIR}/linux-cuda-package-manifest.json" >/dev/null
grep -q '"schema": "metalfish.portable_artifact"' \
  "${CUDA_PACKAGE_CHECK_DIR}/linux-cuda-package-manifest.json"
grep -q '"kind": "linux-cuda"' \
  "${CUDA_PACKAGE_CHECK_DIR}/linux-cuda-package-manifest.json"
grep -q '"name": "metalfish"' \
  "${CUDA_PACKAGE_CHECK_DIR}/linux-cuda-package-manifest.json"
METALFISH_NN_WEIGHTS="${WEIGHTS}" \
  METALFISH_NN_PARITY_REPORT="${BUILD_DIR}/cuda-gpu-package-parity-report.md" \
  METALFISH_NN_BATCH_BENCH="${METALFISH_NN_BATCH_BENCH:-1}" \
  METALFISH_NN_BATCH_TRACE_WORST="${METALFISH_NN_BATCH_TRACE_WORST:-1}" \
  METALFISH_NN_SINGLE_REPEAT_STRESS="${METALFISH_NN_SINGLE_REPEAT_STRESS:-1}" \
  METALFISH_NN_SINGLE_REUSE_STRESS="${METALFISH_NN_SINGLE_REUSE_STRESS:-1}" \
  METALFISH_NN_BATCH_REUSE_STRESS="${METALFISH_NN_BATCH_REUSE_STRESS:-1}" \
  METALFISH_NN_BENCH_ITERS="${METALFISH_NN_BENCH_ITERS:-2}" \
  METALFISH_NN_BENCH_MAX_BATCH="${METALFISH_NN_BENCH_MAX_BATCH:-32}" \
  METALFISH_NN_BENCH_WARMUP_ITERS="${METALFISH_NN_BENCH_WARMUP_ITERS:-3}" \
  METALFISH_NN_BENCH_GRAPH_REUSE_PROBE="${METALFISH_NN_BENCH_GRAPH_REUSE_PROBE:-1}" \
  METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
  METALFISH_CUDA_PROFILE=0 \
  "${CUDA_PACKAGE_CHECK_DIR}/test_nn_comparison" \
  2>&1 | tee "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "MCTS evaluator batch parity" \
  "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "TRACE_WORST:" "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "SINGLE_REPEAT_STRESS_MAX:" \
  "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "SINGLE_REUSE_STRESS_MAX:" \
  "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "REUSE_STRESS_MAX:" \
  "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "batches:" "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "graph_reuse_probe:" \
  "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
grep -q "CUDA transformer backend" \
  "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
if [[ "${CUDA_GRAPH_REQUESTED}" == "1" ]]; then
  grep -q "executor=resolved+graph-replay" \
    "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
  grep -q "caches=1" \
    "${BUILD_DIR}/cuda-gpu-package-nn-comparison.log"
fi
METALFISH_CUDA_PROFILE=0 \
  METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
  "${CUDA_PACKAGE_CHECK_DIR}/metalfish_nn_probe" \
  --weights "${WEIGHTS}" \
  --backend cuda \
  --cuda-device -1 \
  --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
  --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
  --cuda-deterministic-attention-softmax true \
  --cuda-full-buffer-clear true \
  --top 3 \
  --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
  --iterations 1 \
  2>&1 | tee "${BUILD_DIR}/cuda-gpu-package-probe.log"
grep -q '"backend":"cuda"' "${BUILD_DIR}/cuda-gpu-package-probe.log"
grep -q "CUDA transformer backend" "${BUILD_DIR}/cuda-gpu-package-probe.log"
if [[ "${CUDA_GRAPH_REQUESTED}" == "1" ]]; then
  grep -q "executor=resolved+graph-replay" \
    "${BUILD_DIR}/cuda-gpu-package-probe.log"
fi
METALFISH_CUDA_PROFILE=0 \
  python3 tools/run_nn_backend_probe_suite.py \
  --probe "${CUDA_PACKAGE_CHECK_DIR}/metalfish_nn_probe" \
  --weights "${WEIGHTS}" \
  --backend cuda \
  --cuda-device -1 \
  --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
  --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
  --cuda-deterministic-attention-softmax true \
  --cuda-full-buffer-clear true \
  --out "${BUILD_DIR}/cuda-gpu-package-nn-probe-suite.log" \
  --top 3 \
  --batch-size 2 \
  --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
  --iterations 1 \
  --backend-label "CUDA transformer backend" \
  "${CUDA_RUNTIME_REQUIRE_ARGS[@]}" \
  "${CUDA_GRAPH_REPLAY_REQUIRE_ARGS[@]}" \
  --require-wdl \
  --require-moves-left \
  --expected-policy-count 1858 \
  --full-policy
if [[ "${METALFISH_CUDA_LEGACY_PROBE:-1}" == "1" ]]; then
  METALFISH_CUDA_PROFILE=0 \
    python3 tools/run_nn_backend_probe_suite.py \
    --probe "${CUDA_PACKAGE_CHECK_DIR}/metalfish_nn_probe" \
    --weights "${LEGACY_WEIGHTS}" \
    --backend cuda \
    --cuda-device -1 \
    --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
    --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
    --cuda-deterministic-attention-softmax true \
    --cuda-full-buffer-clear true \
    --out "${BUILD_DIR}/cuda-gpu-package-legacy-nn-probe-suite.log" \
    --top 3 \
    --batch-size 2 \
    --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
    --iterations 1 \
    --backend-label "CUDA transformer backend" \
    "${CUDA_RUNTIME_REQUIRE_ARGS[@]}" \
    "${CUDA_GRAPH_REPLAY_REQUIRE_ARGS[@]}" \
    --no-require-wdl \
    --no-require-moves-left \
    --expected-policy-count 1858 \
    --full-policy
  METALFISH_CUDA_PROFILE=0 \
    METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
    "${CUDA_PACKAGE_CHECK_DIR}/metalfish_nn_probe" \
    --weights "${WEIGHTS}" \
    --isolation-weights "${LEGACY_WEIGHTS}" \
    --backend cuda \
    --cuda-device -1 \
    --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
    --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
    --cuda-deterministic-attention-softmax true \
    --cuda-full-buffer-clear true \
    --top 3 \
    --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
    --iterations 1 \
    2>&1 | tee "${BUILD_DIR}/cuda-gpu-package-nn-isolation-bt4-legacy.log"
  assert_cuda_isolation_probe_log \
    "packaged BT4/legacy CUDA isolation probe" \
    "${BUILD_DIR}/cuda-gpu-package-nn-isolation-bt4-legacy.log"
  METALFISH_CUDA_PROFILE=0 \
    METALFISH_CUDA_GRAPH_STATUS_DETAIL="${METALFISH_CUDA_GRAPH_STATUS_DETAIL:-1}" \
    "${CUDA_PACKAGE_CHECK_DIR}/metalfish_nn_probe" \
    --weights "${LEGACY_WEIGHTS}" \
    --isolation-weights "${WEIGHTS}" \
    --backend cuda \
    --cuda-device -1 \
    --cuda-graph-execution "${CUDA_GRAPH_CLI_VALUE}" \
    --cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}" \
    --cuda-deterministic-attention-softmax true \
    --cuda-full-buffer-clear true \
    --top 3 \
    --warmup "${CUDA_GRAPH_REPLAY_WARMUP}" \
    --iterations 1 \
    2>&1 | tee "${BUILD_DIR}/cuda-gpu-package-nn-isolation-legacy-bt4.log"
  assert_cuda_isolation_probe_log \
    "packaged legacy/BT4 CUDA isolation probe" \
    "${BUILD_DIR}/cuda-gpu-package-nn-isolation-legacy-bt4.log"
fi
METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${CUDA_PACKAGE_CHECK_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=accelerator \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=0 \
  --go "nodes 1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=accelerator" \
  --expect-output "minibatch=${CUDA_STABLE_BATCH_SIZE}" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-package-uci-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_ponder_smoke.py \
  --engine "${CUDA_PACKAGE_CHECK_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --settle-sec "${MCTS_PONDER_SETTLE_SEC}" \
  --ponder-go "${MCTS_PONDER_GO}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption Ponder=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=true \
  --setoption MCTSMinibatchSize=0 \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --json-out "${BUILD_DIR}/cuda-gpu-package-uci-ponder-mcts.json" \
  --expect-output "Starting Multi-Threaded MCTS Search" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --reject-output "Time safety:" \
  --reject-output "Falling back to Alpha-Beta" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-package-uci-ponder-mcts-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${CUDA_PACKAGE_CHECK_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=false \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${BK07_FEN}" \
  --go "nodes 50" \
  --expect-bestmove h5f6 \
  --json-out "${BUILD_DIR}/cuda-gpu-package-uci-bk07-search.json" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-package-uci-bk07-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${CUDA_PACKAGE_CHECK_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption Threads=8 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSParallelSearch=false \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${KIWIPETE_FEN}" \
  --go "nodes 1" \
  --json-out "${BUILD_DIR}/cuda-gpu-package-uci-kiwipete-search.json" \
  --expect-output "CUDA transformer backend" \
  --expect-output "MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-package-uci-kiwipete-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${CUDA_PACKAGE_CHECK_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${BK07_FEN}" \
  --go "nodes 50" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "Final: MCTSPlayouts=" \
  --json-out "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-search.json" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${CUDA_PACKAGE_CHECK_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption Threads=3 \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
  --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=false \
  --setoption UseHybridSearch=true \
  --setoption HybridMCTSThreads=1 \
  --setoption HybridABThreads=2 \
  --setoption HybridAutoABThreadsCap=0 \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --setoption MCTSParityPreset=true \
  --setoption MCTSAddDirichletNoise=false \
  --setoption TransformerLowTimeFallbackMs=0 \
  --position "fen ${KIWIPETE_FEN}" \
  --go "nodes 50" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "Hybrid MCTS runtime: backend=cuda" \
  --expect-output "minibatch=1" \
  --expect-output "CUDA transformer backend" \
  --expect-output "Final: MCTSPlayouts=" \
  --json-out "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-kiwipete-search.json" \
  "${UCI_CUDA_RUNTIME_EXPECT_ARGS[@]}" \
  "${UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS[@]}" \
  | tee "${BUILD_DIR}/cuda-gpu-package-uci-hybrid-kiwipete-smoke.log"

if [[ -n "${CUDA_PROFILE_REQUESTED}" && "${CUDA_PROFILE_REQUESTED}" != "0" ]]; then
  METALFISH_CUDA_PROFILE=1 \
    METALFISH_CUDA_PROFILE_LIMIT="${CUDA_PROFILE_LIMIT}" \
    python3 tools/uci_smoke.py \
      --engine "${BUILD_DIR}/metalfish" \
      --timeout "${UCI_TIMEOUT}" \
      --setoption Threads=3 \
      --setoption NNBackend=cuda \
      --setoption NNWeights="${WEIGHTS}" \
      --setoption NNCudaDevice=-1 \
      --setoption NNCudaGraphExecution="${CUDA_GRAPH_UCI_VALUE}" \
      --setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}" \
      --setoption NNCudaDeterministicAttentionSoftmax=true \
      --setoption NNCudaFullBufferClear=true \
      --setoption UseMCTS=false \
      --setoption UseHybridSearch=true \
      --setoption HybridMCTSThreads=1 \
      --setoption HybridABThreads=2 \
      --setoption HybridAutoABThreadsCap=0 \
      --setoption MCTSMaxThreads=1 \
      --setoption MCTSMinibatchSize=1 \
      --go "nodes 8" \
      --echo-output \
      --expect-output "CUDA profile report=" \
      --expect-output "Starting Parallel Hybrid Search" \
      --expect-output "CUDA transformer backend" \
      | tee "${BUILD_DIR}/cuda-gpu-profile.log"
fi

write_summary "passed"

cat "${SUMMARY}"
echo "CUDA GPU gate passed"
