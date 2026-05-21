#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METALFISH_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${METALFISH_CUDA_BUILD_DIR:-${ROOT_DIR}/build-cuda-gpu}"
CUDA_ARCHS="${METALFISH_CUDA_ARCHS:-89}"
JOBS="${METALFISH_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
UCI_GO="${METALFISH_CUDA_UCI_GO:-nodes 8}"
UCI_TIMEOUT="${METALFISH_CUDA_UCI_TIMEOUT:-180}"
WEIGHTS="${METALFISH_NN_WEIGHTS:-${ROOT_DIR}/networks/BT4-1024x15x32h-swa-6147500.pb}"
APT_LOCK_TIMEOUT="${METALFISH_APT_LOCK_TIMEOUT:-600}"
SUMMARY="${METALFISH_CUDA_SUMMARY:-${BUILD_DIR}/cuda-gpu-summary.md}"
PARITY_REPORT="${METALFISH_NN_PARITY_REPORT:-${BUILD_DIR}/cuda-gpu-parity-report.md}"
CUDA_PROFILE_REQUESTED="${METALFISH_CUDA_PROFILE:-0}"
CUDA_PROFILE_LIMIT="${METALFISH_CUDA_PROFILE_LIMIT:-8}"

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

cmake --build "${BUILD_DIR}" --target metalfish metalfish_tests test_nn_comparison -j"${JOBS}"

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
  METALFISH_CUDA_PROFILE=0 \
  "${BUILD_DIR}/test_nn_comparison" 2>&1 | tee "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
grep -q "backend: CUDA transformer backend" \
  "${BUILD_DIR}/cuda-gpu-nn-comparison.log"

python3 tools/uci_smoke.py \
  --engine "${BUILD_DIR}/metalfish" \
  --timeout "${UCI_TIMEOUT}" \
  --setoption NNBackend=auto \
  --setoption NNWeights="${WEIGHTS}" \
  --setoption UseMCTS=true \
  --setoption UseHybridSearch=false \
  --setoption MCTSMaxThreads=1 \
  --setoption MCTSMinibatchSize=1 \
  --go "nodes 1" \
  --expect-output "CUDA transformer backend" \
  | tee "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log"

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

{
  echo "# MetalFish CUDA GPU Gate Summary"
  echo
  echo "- Timestamp UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "- Host: $(hostname)"
  echo "- CUDA architectures: ${CUDA_ARCHS}"
  echo "- Build directory: ${BUILD_DIR}"
  echo "- Weights: ${WEIGHTS}"
  echo "- Parity report: ${PARITY_REPORT}"
  echo "- Explicit CUDA UCI go: ${UCI_GO}"
  echo "- Batch worst trace: ${METALFISH_NN_BATCH_TRACE_WORST:-1}"
  echo "- Single reuse stress: ${METALFISH_NN_SINGLE_REUSE_STRESS:-1}"
  echo "- Batch reuse stress: ${METALFISH_NN_BATCH_REUSE_STRESS:-1}"
  echo "- CUDA full buffer clear: ${METALFISH_CUDA_FULL_BUFFER_CLEAR:-0}"
  echo "- CUDA release single workspace each run: ${METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN:-0}"
  echo "- CUDA release workspace each run: ${METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN:-0}"
  echo "- CUDA raw output trace: ${METALFISH_CUDA_TRACE_RAW_OUTPUTS:-0}"
  echo "- cuBLAS workspace config: ${CUBLAS_WORKSPACE_CONFIG:-unset}"
  echo
  echo "## Device"
  echo
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
    --format=csv,noheader || true
  echo
  echo "## Backend"
  echo
  grep -m1 "backend: CUDA transformer backend" \
    "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
  echo
  echo "## Batch Timings"
  echo
  grep -m1 "batches:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
  if grep -q "TRACE_WORST:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
    echo
    echo "## Batch Worst Trace"
    echo
    grep -m1 "TRACE_WORST:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
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
  if grep -q "REUSE_STRESS_MAX:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
    echo
    echo "## Batch Reuse Stress"
    echo
    grep -m1 "REUSE_STRESS_MAX:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
    grep -m1 "REUSE_STRESS_POLICY:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    grep -m1 "REUSE_STRESS_SINGLE_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    grep -m1 "REUSE_STRESS_BATCH_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
  fi
  if grep -q "SINGLE_REUSE_STRESS_MAX:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"; then
    echo
    echo "## Single Reuse Stress"
    echo
    grep -m1 "SINGLE_REUSE_STRESS_MAX:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log"
    grep -m1 "SINGLE_REUSE_STRESS_POLICY:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    grep -m1 "SINGLE_REUSE_STRESS_BASELINE_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
    grep -m1 "SINGLE_REUSE_STRESS_REPLAY_TOP:" "${BUILD_DIR}/cuda-gpu-nn-comparison.log" || true
  fi
  echo
  echo "## Parity Report"
  echo
  if [[ -s "${PARITY_REPORT}" ]]; then
    grep -m1 "^- Backend:" "${PARITY_REPORT}"
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
  echo "- auto: $(grep -m1 '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-auto-smoke.log")"
  echo "- cuda: $(grep -m1 '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-smoke.log")"
  echo "- hybrid-cuda: $(grep -m1 '^bestmove ' "${BUILD_DIR}/cuda-gpu-uci-hybrid-smoke.log")"
  if [[ -s "${BUILD_DIR}/cuda-gpu-profile.log" ]]; then
    echo
    echo "## CUDA Profile"
    echo
    grep -m1 "CUDA profile report=" "${BUILD_DIR}/cuda-gpu-profile.log"
    grep -m1 "CUDA profile buckets:" "${BUILD_DIR}/cuda-gpu-profile.log"
    grep -m1 "CUDA profile slowest:" "${BUILD_DIR}/cuda-gpu-profile.log"
  fi
} >"${SUMMARY}"

cat "${SUMMARY}"
echo "CUDA GPU gate passed"
