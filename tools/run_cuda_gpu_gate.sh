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
  run_apt_get update
  run_apt_get install -y \
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
    python3-pip
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

"${BUILD_DIR}/metalfish_tests" | tee "${BUILD_DIR}/cuda-gpu-tests.log"
grep -q "CUDA runtime" "${BUILD_DIR}/cuda-gpu-tests.log"

METALFISH_NN_WEIGHTS="${WEIGHTS}" \
  "${BUILD_DIR}/test_nn_comparison" | tee "${BUILD_DIR}/cuda-gpu-nn-comparison.log"

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

echo "CUDA GPU gate passed"
