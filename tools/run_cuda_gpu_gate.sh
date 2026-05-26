#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METALFISH_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${METALFISH_CUDA_BUILD_DIR:-${ROOT_DIR}/build-cuda-gpu}"
CUDA_ARCHS="${METALFISH_CUDA_ARCHS:-89}"
JOBS="${METALFISH_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
UCI_GO="${METALFISH_CUDA_UCI_GO:-nodes 8}"
UCI_TIMEOUT="${METALFISH_CUDA_UCI_TIMEOUT:-180}"
WEIGHTS="${METALFISH_NN_WEIGHTS:-${ROOT_DIR}/networks/BT4-1024x15x32h-swa-6147500.pb}"
LEGACY_WEIGHTS="${METALFISH_LEGACY_NN_WEIGHTS:-${ROOT_DIR}/networks/legacy-42850.pb.gz}"
APT_LOCK_TIMEOUT="${METALFISH_APT_LOCK_TIMEOUT:-600}"
SUMMARY="${METALFISH_CUDA_SUMMARY:-${BUILD_DIR}/cuda-gpu-summary.md}"
PARITY_REPORT="${METALFISH_NN_PARITY_REPORT:-${BUILD_DIR}/cuda-gpu-parity-report.md}"
CUDA_PROFILE_REQUESTED="${METALFISH_CUDA_PROFILE:-0}"
CUDA_PROFILE_LIMIT="${METALFISH_CUDA_PROFILE_LIMIT:-8}"
CUDA_GRAPH_REQUESTED=1
if [[ "${METALFISH_CUDA_GRAPH:-}" == "0" ||
      "${METALFISH_CUDA_GRAPH_EXECUTION:-}" == "0" ]]; then
  CUDA_GRAPH_REQUESTED=0
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
    echo "- Explicit CUDA UCI go: ${UCI_GO}"
    echo "- Batch worst trace: ${METALFISH_NN_BATCH_TRACE_WORST:-1}"
    echo "- Single repeat stress: ${METALFISH_NN_SINGLE_REPEAT_STRESS:-0}"
    echo "- Single reuse stress: ${METALFISH_NN_SINGLE_REUSE_STRESS:-1}"
    echo "- Batch reuse stress: ${METALFISH_NN_BATCH_REUSE_STRESS:-1}"
    echo "- CUDA full buffer clear: ${METALFISH_CUDA_FULL_BUFFER_CLEAR:-1}"
    echo "- CUDA graph execution: ${CUDA_GRAPH_REQUESTED}"
    echo "- CUDA graph replay observed: ${graph_replay_observed}"
    echo "- CUDA release single workspace each run: ${METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN:-0}"
    echo "- CUDA release workspace each run: ${METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN:-0}"
    echo "- CUDA stable execution batch size: ${METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE:-16}"
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
    echo "- NN artifact manifest: $(summary_log_status "${BUILD_DIR}/cuda-gpu-nn-artifact-manifest.json")"
    echo "- auto UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log")"
    echo "- explicit CUDA UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-smoke.log")"
    echo "- hybrid CUDA UCI smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log")"
    echo "- hybrid ANE-disable smoke: $(summary_log_status "${BUILD_DIR}/cuda-gpu-uci-hybrid-ane-smoke.log")"
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
    summary_failure_lines "auto UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log"
    summary_failure_lines "explicit CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-smoke.log"
    summary_failure_lines "hybrid CUDA UCI smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log"
    summary_failure_lines "hybrid ANE-disable smoke" \
      "${BUILD_DIR}/cuda-gpu-uci-hybrid-ane-smoke.log"
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
    echo "- cuda: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-smoke.log" "not reached")"
    echo "- hybrid-cuda: $(summary_line_or_missing '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log" "not reached")"
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
  --out "${BUILD_DIR}/cuda-gpu-nn-probe-suite.log" \
  --top 3 \
  --warmup 0 \
  --iterations 1 \
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
    --out "${BUILD_DIR}/cuda-gpu-legacy-nn-probe-suite.log" \
    --top 3 \
    --warmup 0 \
    --iterations 1 \
    --full-policy
fi

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=auto \
  --setoption NNBackendRequireAccelerator=true \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption NNCudaDevice=-1 \
  --setoption NNCudaGraphExecution=true \
  --setoption NNCudaStableExecutionBatchSize=16 \
  --setoption NNCudaDeterministicAttentionSoftmax=true \
  --setoption NNCudaFullBufferClear=true \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --go "nodes 1" \
  --expect-output "CUDA transformer backend" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log"

METALFISH_CUDA_PROFILE=0 \
  python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=cuda \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --go "${UCI_GO}" | tee "${BUILD_DIR}/cuda-gpu-uci-smoke.log"

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
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --go "nodes 8" \
  --expect-output "Starting Parallel Hybrid Search" \
  --expect-output "CUDA transformer backend" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log"

mkdir -p "${BUILD_DIR}/dummy-coreml.mlmodelc"
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
  --expect-output "CUDA transformer backend" \
  --expect-output "ANE root probe disabled" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-hybrid-ane-smoke.log"

if [[ -n "${CUDA_PROFILE_REQUESTED}" && "${CUDA_PROFILE_REQUESTED}" != "0" ]]; then
  METALFISH_CUDA_PROFILE=1 \
    METALFISH_CUDA_PROFILE_LIMIT="${CUDA_PROFILE_LIMIT}" \
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
