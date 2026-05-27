#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${METALFISH_GCP_PROJECT:-metalfish}"
DEFAULT_ZONES="us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west4-a us-west4-c"
ZONES="${METALFISH_GCP_ZONES:-${METALFISH_GCP_ZONE:-${DEFAULT_ZONES}}}"
INSTANCE="${METALFISH_GCP_INSTANCE:-metalfish-win-cuda-gate-$(date +%Y%m%d-%H%M%S)}"
MACHINES="${METALFISH_GCP_MACHINES:-${METALFISH_GCP_MACHINE:-g2-standard-8 g2-standard-4}}"
ACCELERATOR="${METALFISH_GCP_ACCELERATOR:-type=nvidia-l4-vws,count=1}"
IMAGE_PROJECT="${METALFISH_GCP_IMAGE_PROJECT:-windows-cloud}"
IMAGE_FAMILY="${METALFISH_GCP_IMAGE_FAMILY:-windows-2022}"
BOOT_DISK_SIZE="${METALFISH_GCP_BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE="${METALFISH_GCP_BOOT_DISK_TYPE:-pd-balanced}"
DELETE_ON_EXIT="${METALFISH_GCP_DELETE_ON_EXIT:-1}"
COLLECT_ARTIFACTS="${METALFISH_GCP_COLLECT_ARTIFACTS:-1}"
ARTIFACT_DIR="${METALFISH_GCP_ARTIFACT_DIR:-${ROOT_DIR}/results/windows_cuda_runtime_gate/${INSTANCE}}"
GCS_PREFIX="${METALFISH_GCP_GCS_PREFIX:-}"
METAL_PROBE_SUITE_LOG="${METALFISH_METAL_PROBE_SUITE_LOG:-}"
METAL_LEGACY_PROBE_SUITE_LOG="${METALFISH_METAL_LEGACY_PROBE_SUITE_LOG:-}"
METAL_COMPARISON_LOG="${METALFISH_METAL_COMPARISON_LOG:-}"
METAL_MCTS_BK07_SEARCH_JSON="${METALFISH_METAL_MCTS_BK07_SEARCH_JSON:-}"
METAL_MCTS_KIWIPETE_SEARCH_JSON="${METALFISH_METAL_MCTS_KIWIPETE_SEARCH_JSON:-}"
METAL_HYBRID_STARTPOS_SEARCH_JSON="${METALFISH_METAL_HYBRID_STARTPOS_SEARCH_JSON:-}"
REQUIRE_METAL_COMPARE="${METALFISH_REQUIRE_METAL_COMPARE:-0}"
REQUIRE_METAL_BENCHMARK_COMPARE="${METALFISH_REQUIRE_METAL_BENCHMARK_COMPARE:-0}"
REQUIRE_METAL_SEARCH_COMPARE="${METALFISH_REQUIRE_METAL_SEARCH_COMPARE:-0}"
MAX_CUDA_METAL_EVAL_MS_RATIO="${METALFISH_MAX_CUDA_METAL_EVAL_MS_RATIO:-1.0}"
REMOTE_USER="${METALFISH_GCP_WINDOWS_USER:-metalfish}"
SSH_KEY="${METALFISH_GCP_SSH_KEY:-${HOME}/.ssh/google_compute_engine}"
SSH_PUB_KEY="${METALFISH_GCP_SSH_PUB_KEY:-${SSH_KEY}.pub}"
PACKAGE_ZIP="${METALFISH_WINDOWS_CUDA_PACKAGE:-}"
WEIGHTS="${METALFISH_BT4_WEIGHTS:-${ROOT_DIR}/networks/BT4-1024x15x32h-swa-6147500.pb}"
LEGACY_WEIGHTS="${METALFISH_LEGACY_NN_WEIGHTS:-${ROOT_DIR}/networks/legacy-42850.pb.gz}"
NNUE_BIG="${METALFISH_NNUE_BIG:-${ROOT_DIR}/networks/nn-c288c895ea92.nnue}"
NNUE_SMALL="${METALFISH_NNUE_SMALL:-${ROOT_DIR}/networks/nn-37f18f62d772.nnue}"
UCI_TIMEOUT_SECONDS="${METALFISH_WINDOWS_CUDA_UCI_TIMEOUT:-420}"
PROBE_TIMEOUT_SECONDS="${METALFISH_WINDOWS_CUDA_PROBE_TIMEOUT:-420}"
COMPARISON_TIMEOUT_SECONDS="${METALFISH_WINDOWS_CUDA_COMPARISON_TIMEOUT:-900}"
UCI_GO="${METALFISH_WINDOWS_CUDA_UCI_GO:-nodes 1}"
HYBRID_UCI_GO="${METALFISH_WINDOWS_CUDA_HYBRID_UCI_GO:-nodes 8}"
UCI_TRACE="${METALFISH_WINDOWS_UCI_TRACE:-1}"
CUDA_GRAPH="${METALFISH_WINDOWS_CUDA_GRAPH:-}"
CUDA_PROFILE="${METALFISH_WINDOWS_CUDA_PROFILE:-}"
CUDA_PROFILE_LIMIT="${METALFISH_WINDOWS_CUDA_PROFILE_LIMIT:-2}"
CUDA_STABLE_BATCH_SIZE="${METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE:-16}"
WINDOWS_CUDA_COMPILE_RUN_ID="${METALFISH_WINDOWS_CUDA_COMPILE_RUN_ID:-}"
DRIVER_SCRIPT_URL="${METALFISH_WINDOWS_GPU_DRIVER_SCRIPT_URL:-https://github.com/GoogleCloudPlatform/compute-gpu-installation/raw/main/windows/install_gpu_driver.ps1}"
CREATED_INSTANCE=0
SSH_READY=0
ZONE=""
MACHINE=""
RUN_DIR="$(mktemp -d -t metalfish-windows-cuda.XXXXXX)"
PACKAGE_BASENAME="$(basename "${PACKAGE_ZIP}")"

if [[ ! "${CUDA_STABLE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

require_metal_compare() {
  case "${REQUIRE_METAL_COMPARE}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_metal_search_compare() {
  case "${REQUIRE_METAL_SEARCH_COMPARE}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  local status=$?
  if [[ "${COLLECT_ARTIFACTS}" == "1" && "${SSH_READY}" == "1" && -n "${ZONE}" ]]; then
    collect_remote_artifacts || true
  fi
  rm -rf "${RUN_DIR}"
  if [[ "${DELETE_ON_EXIT}" == "1" && "${CREATED_INSTANCE}" == "1" && -n "${ZONE}" ]]; then
    gcloud compute instances delete "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --quiet >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT

require_file() {
  local file="$1"
  local label="$2"
  if [[ ! -s "${file}" ]]; then
    echo "${label} not found: ${file}" >&2
    exit 2
  fi
}

compare_collected_probe_suite() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if require_metal_compare; then
      echo "Metal/Windows CUDA comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
      return 1
    fi
    return 0
  fi

  if [[ -z "${METAL_PROBE_SUITE_LOG}" ]]; then
    if require_metal_compare; then
      echo "Metal probe suite log is required; set METALFISH_METAL_PROBE_SUITE_LOG" >&2
      return 1
    fi
    return 0
  fi

  if [[ ! -s "${METAL_PROBE_SUITE_LOG}" ]]; then
    echo "Metal probe suite log not found: ${METAL_PROBE_SUITE_LOG}" >&2
    return 1
  fi

  local cuda_suite="${ARTIFACT_DIR}/logs/cuda-probe-suite.stdout.log"
  if [[ ! -s "${cuda_suite}" ]]; then
    echo "Windows CUDA probe suite log not found: ${cuda_suite}" >&2
    return 1
  fi

  python3 tools/compare_nn_backend_outputs.py \
    --expected-log "${METAL_PROBE_SUITE_LOG}" \
    --actual-log "${cuda_suite}" \
    --expected-label "Metal (MPSGraph) backend" \
    --actual-label "CUDA transformer backend" \
    --summary-out "${ARTIFACT_DIR}/logs/metal-windows-cuda-nn-probe-suite-summary.json" \
    --require-full-policy \
    --all-probes \
    | tee "${ARTIFACT_DIR}/logs/metal-windows-cuda-nn-probe-suite-compare.log"
}

compare_collected_legacy_probe_suite() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if require_metal_compare; then
      echo "Legacy Metal/Windows CUDA comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
      return 1
    fi
    return 0
  fi

  if [[ -z "${METAL_LEGACY_PROBE_SUITE_LOG}" ]]; then
    if require_metal_compare; then
      echo "Metal legacy probe suite log is required; set METALFISH_METAL_LEGACY_PROBE_SUITE_LOG" >&2
      return 1
    fi
    return 0
  fi

  if [[ ! -s "${METAL_LEGACY_PROBE_SUITE_LOG}" ]]; then
    echo "Metal legacy probe suite log not found: ${METAL_LEGACY_PROBE_SUITE_LOG}" >&2
    return 1
  fi

  local cuda_suite="${ARTIFACT_DIR}/logs/cuda-legacy-probe-suite.stdout.log"
  if [[ ! -s "${cuda_suite}" ]]; then
    echo "Windows CUDA legacy probe suite log not found: ${cuda_suite}" >&2
    return 1
  fi

  python3 tools/compare_nn_backend_outputs.py \
    --expected-log "${METAL_LEGACY_PROBE_SUITE_LOG}" \
    --actual-log "${cuda_suite}" \
    --expected-label "Metal (MPSGraph) backend" \
    --actual-label "CUDA transformer backend" \
    --summary-out "${ARTIFACT_DIR}/logs/metal-windows-cuda-legacy-nn-probe-suite-summary.json" \
    --require-full-policy \
    --no-require-wdl \
    --no-require-moves-left \
    --all-probes \
    | tee "${ARTIFACT_DIR}/logs/metal-windows-cuda-legacy-nn-probe-suite-compare.log"
}

compare_collected_benchmark_timings() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if [[ "${REQUIRE_METAL_BENCHMARK_COMPARE}" == "1" ]]; then
      echo "Metal/Windows CUDA benchmark comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
      return 1
    fi
    return 0
  fi

  if [[ -z "${METAL_COMPARISON_LOG}" ]]; then
    if [[ "${REQUIRE_METAL_BENCHMARK_COMPARE}" == "1" ]]; then
      echo "Metal comparison log is required; set METALFISH_METAL_COMPARISON_LOG" >&2
      return 1
    fi
    return 0
  fi

  if [[ ! -s "${METAL_COMPARISON_LOG}" ]]; then
    echo "Metal comparison log not found: ${METAL_COMPARISON_LOG}" >&2
    return 1
  fi

  local cuda_comparison="${ARTIFACT_DIR}/logs/cuda-nn-comparison.stdout.log"
  if [[ ! -s "${cuda_comparison}" ]]; then
    echo "Windows CUDA comparison log not found: ${cuda_comparison}" >&2
    return 1
  fi

  python3 tools/compare_nn_backend_benchmarks.py \
    --expected-log "${METAL_COMPARISON_LOG}" \
    --actual-log "${cuda_comparison}" \
    --expected-label "Metal (MPSGraph) backend" \
    --actual-label "CUDA transformer backend" \
    --summary-out "${ARTIFACT_DIR}/logs/metal-windows-cuda-nn-benchmark-summary.json" \
    --require-actual-graph-reuse \
    --max-eval-ms-ratio "${MAX_CUDA_METAL_EVAL_MS_RATIO}" \
    | tee "${ARTIFACT_DIR}/logs/metal-windows-cuda-nn-benchmark-compare.log"
}

compare_collected_search_results() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if require_metal_search_compare; then
      echo "Metal/Windows CUDA search comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
      return 1
    fi
    return 0
  fi

  if [[ -z "${METAL_MCTS_BK07_SEARCH_JSON}" || -z "${METAL_MCTS_KIWIPETE_SEARCH_JSON}" || -z "${METAL_HYBRID_STARTPOS_SEARCH_JSON}" ]]; then
    if require_metal_search_compare; then
      echo "Metal search JSON inputs are required for Windows CUDA search comparison" >&2
      return 1
    fi
    return 0
  fi
  if [[ ! -s "${METAL_MCTS_BK07_SEARCH_JSON}" ]]; then
    echo "Metal MCTS search JSON not found: ${METAL_MCTS_BK07_SEARCH_JSON}" >&2
    return 1
  fi
  if [[ ! -s "${METAL_MCTS_KIWIPETE_SEARCH_JSON}" ]]; then
    echo "Metal MCTS kiwipete search JSON not found: ${METAL_MCTS_KIWIPETE_SEARCH_JSON}" >&2
    return 1
  fi
  if [[ ! -s "${METAL_HYBRID_STARTPOS_SEARCH_JSON}" ]]; then
    echo "Metal Hybrid search JSON not found: ${METAL_HYBRID_STARTPOS_SEARCH_JSON}" >&2
    return 1
  fi

  local cuda_mcts="${ARTIFACT_DIR}/logs/cuda-bk07-mcts-search.json"
  local cuda_mcts_kiwipete="${ARTIFACT_DIR}/logs/cuda-kiwipete-mcts-search.json"
  local cuda_hybrid="${ARTIFACT_DIR}/logs/hybrid-cuda-search.json"
  if [[ ! -s "${cuda_mcts}" ]]; then
    echo "Windows CUDA MCTS search JSON not found: ${cuda_mcts}" >&2
    return 1
  fi
  if [[ ! -s "${cuda_mcts_kiwipete}" ]]; then
    echo "Windows CUDA MCTS kiwipete search JSON not found: ${cuda_mcts_kiwipete}" >&2
    return 1
  fi
  if [[ ! -s "${cuda_hybrid}" ]]; then
    echo "Windows CUDA Hybrid search JSON not found: ${cuda_hybrid}" >&2
    return 1
  fi

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_MCTS_BK07_SEARCH_JSON}" \
    --actual "${cuda_mcts}" \
    --expected-label "Metal MCTS" \
    --actual-label "Windows CUDA MCTS" \
    --json-out "${ARTIFACT_DIR}/logs/metal-windows-cuda-mcts-bk07-search-summary.json" \
    | tee "${ARTIFACT_DIR}/logs/metal-windows-cuda-mcts-bk07-search-compare.log"

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_MCTS_KIWIPETE_SEARCH_JSON}" \
    --actual "${cuda_mcts_kiwipete}" \
    --expected-label "Metal MCTS" \
    --actual-label "Windows CUDA MCTS" \
    --json-out "${ARTIFACT_DIR}/logs/metal-windows-cuda-mcts-kiwipete-search-summary.json" \
    | tee "${ARTIFACT_DIR}/logs/metal-windows-cuda-mcts-kiwipete-search-compare.log"

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_HYBRID_STARTPOS_SEARCH_JSON}" \
    --actual "${cuda_hybrid}" \
    --expected-label "Metal Hybrid" \
    --actual-label "Windows CUDA Hybrid" \
    --no-require-same-bestmove \
    --json-out "${ARTIFACT_DIR}/logs/metal-windows-cuda-hybrid-startpos-search-summary.json" \
    | tee "${ARTIFACT_DIR}/logs/metal-windows-cuda-hybrid-startpos-search-compare.log"
}

ensure_ssh_key() {
  mkdir -p "$(dirname "${SSH_KEY}")"
  if [[ ! -s "${SSH_KEY}" ]]; then
    ssh-keygen -t rsa -b 3072 -f "${SSH_KEY}" -N "" -C "${USER:-metalfish}@metalfish-windows-cuda-gate" >/dev/null
  fi
  if [[ ! -s "${SSH_PUB_KEY}" ]]; then
    ssh-keygen -y -f "${SSH_KEY}" >"${SSH_PUB_KEY}"
  fi
  chmod 600 "${SSH_KEY}" || true
}

ssh_target() {
  printf "%s@%s" "${REMOTE_USER}" "${INSTANCE}"
}

wait_for_ssh() {
  local attempts="${1:-90}"
  for attempt in $(seq 1 "${attempts}"); do
    if gcloud compute ssh "$(ssh_target)" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --ssh-flag="-o ConnectTimeout=10" \
      --ssh-flag="-o ConnectionAttempts=1" \
      --command "hostname" >/dev/null 2>&1; then
      SSH_READY=1
      return 0
    fi
    sleep 10
  done
  echo "timed out waiting for SSH on ${INSTANCE}" >&2
  return 1
}

remote_cmd() {
  gcloud compute ssh "$(ssh_target)" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --command "$1"
}

copy_to_remote() {
  local src="$1"
  local dst="$2"
  gcloud compute scp "${src}" "$(ssh_target):${dst}" \
    --project "${PROJECT}" \
    --zone "${ZONE}"
}

run_remote_ps() {
  local script="$1"
  local remote_path="C:/metalfish/$(basename "${script}")"
  copy_to_remote "${script}" "${remote_path}"
  remote_cmd "powershell -NoProfile -ExecutionPolicy Bypass -File ${remote_path}"
}

collect_remote_artifacts() {
  mkdir -p "${ARTIFACT_DIR}"
  gcloud compute scp --recurse \
    "$(ssh_target):C:/metalfish/logs" \
    "${ARTIFACT_DIR}/" \
    --project "${PROJECT}" \
    --zone "${ZONE}" >/dev/null 2>&1 || true
}

write_runtime_manifest() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    return 0
  fi

  mkdir -p "${ARTIFACT_DIR}"
  RUNTIME_STATUS_FOR_MANIFEST="$1" \
    BT4_COMPARE_STATUS_FOR_MANIFEST="$2" \
    LEGACY_COMPARE_STATUS_FOR_MANIFEST="$3" \
    BENCHMARK_COMPARE_STATUS_FOR_MANIFEST="$4" \
    SEARCH_COMPARE_STATUS_FOR_MANIFEST="$5" \
    FINAL_COMPARE_STATUS_FOR_MANIFEST="$6" \
    GIT_HEAD_SHA="$(git rev-parse HEAD)" \
    GATE_ARTIFACT_DIR="${ARTIFACT_DIR}" \
    GATE_PROJECT="${PROJECT}" \
    GATE_INSTANCE="${INSTANCE}" \
    GATE_ZONE="${ZONE}" \
    GATE_MACHINE="${MACHINE}" \
    GATE_MACHINES="${MACHINES}" \
    GATE_ACCELERATOR="${ACCELERATOR}" \
    GATE_IMAGE_PROJECT="${IMAGE_PROJECT}" \
    GATE_IMAGE_FAMILY="${IMAGE_FAMILY}" \
    GATE_BOOT_DISK_SIZE="${BOOT_DISK_SIZE}" \
    GATE_BOOT_DISK_TYPE="${BOOT_DISK_TYPE}" \
    GATE_DELETE_ON_EXIT="${DELETE_ON_EXIT}" \
    GATE_GCS_PREFIX="${GCS_PREFIX}" \
    GATE_REQUIRE_METAL_COMPARE="${REQUIRE_METAL_COMPARE}" \
    GATE_REQUIRE_METAL_BENCHMARK_COMPARE="${REQUIRE_METAL_BENCHMARK_COMPARE}" \
    GATE_REQUIRE_METAL_SEARCH_COMPARE="${REQUIRE_METAL_SEARCH_COMPARE}" \
    GATE_MAX_CUDA_METAL_EVAL_MS_RATIO="${MAX_CUDA_METAL_EVAL_MS_RATIO}" \
    GATE_METAL_COMPARISON_LOG="${METAL_COMPARISON_LOG}" \
    GATE_METAL_PROBE_SUITE_LOG="${METAL_PROBE_SUITE_LOG}" \
    GATE_METAL_LEGACY_PROBE_SUITE_LOG="${METAL_LEGACY_PROBE_SUITE_LOG}" \
    GATE_METAL_MCTS_BK07_SEARCH_JSON="${METAL_MCTS_BK07_SEARCH_JSON}" \
    GATE_METAL_MCTS_KIWIPETE_SEARCH_JSON="${METAL_MCTS_KIWIPETE_SEARCH_JSON}" \
    GATE_METAL_HYBRID_STARTPOS_SEARCH_JSON="${METAL_HYBRID_STARTPOS_SEARCH_JSON}" \
    GATE_PACKAGE_ZIP="${PACKAGE_ZIP}" \
    GATE_PACKAGE_BASENAME="${PACKAGE_BASENAME}" \
    GATE_WINDOWS_CUDA_COMPILE_RUN_ID="${WINDOWS_CUDA_COMPILE_RUN_ID}" \
    GATE_CUDA_STABLE_BATCH_SIZE="${CUDA_STABLE_BATCH_SIZE}" \
    GATE_CUDA_GRAPH="${CUDA_GRAPH}" \
    GATE_CUDA_PROFILE="${CUDA_PROFILE}" \
    GATE_CUDA_PROFILE_LIMIT="${CUDA_PROFILE_LIMIT}" \
    GATE_UCI_GO="${UCI_GO}" \
    GATE_HYBRID_UCI_GO="${HYBRID_UCI_GO}" \
    python3 - "${ARTIFACT_DIR}/windows-cuda-runtime-gate-manifest.json" <<'PY'
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys


def file_record(path):
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": digest.hexdigest(),
    }


manifest_path = pathlib.Path(sys.argv[1])
artifact_dir = pathlib.Path(os.environ["GATE_ARTIFACT_DIR"])
artifacts = {}
if artifact_dir.is_dir():
    for candidate in sorted(artifact_dir.rglob("*")):
        if candidate.is_file() and candidate.resolve() != manifest_path.resolve():
            record = file_record(str(candidate))
            if record is not None:
                artifacts[str(candidate.relative_to(artifact_dir))] = record

manifest = {
    "schema": "metalfish.windows_cuda_runtime_gate",
    "schema_version": 1,
    "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    "git": {
        "head_sha": os.environ["GIT_HEAD_SHA"],
    },
    "gcp": {
        "project": os.environ["GATE_PROJECT"],
        "instance": os.environ["GATE_INSTANCE"],
        "zone": os.environ["GATE_ZONE"],
        "machine": os.environ["GATE_MACHINE"],
        "machine_candidates": os.environ["GATE_MACHINES"],
        "accelerator": os.environ["GATE_ACCELERATOR"],
        "image_project": os.environ["GATE_IMAGE_PROJECT"],
        "image_family": os.environ["GATE_IMAGE_FAMILY"],
        "boot_disk_size": os.environ["GATE_BOOT_DISK_SIZE"],
        "boot_disk_type": os.environ["GATE_BOOT_DISK_TYPE"],
        "delete_on_exit": os.environ["GATE_DELETE_ON_EXIT"] == "1",
        "gcs_prefix": os.environ["GATE_GCS_PREFIX"],
    },
    "inputs": {
        "require_metal_compare": os.environ["GATE_REQUIRE_METAL_COMPARE"],
        "require_metal_benchmark_compare": os.environ[
            "GATE_REQUIRE_METAL_BENCHMARK_COMPARE"
        ],
        "require_metal_search_compare": os.environ[
            "GATE_REQUIRE_METAL_SEARCH_COMPARE"
        ],
        "max_cuda_metal_eval_ms_ratio": os.environ[
            "GATE_MAX_CUDA_METAL_EVAL_MS_RATIO"
        ],
        "windows_cuda_compile_run_id": os.environ["GATE_WINDOWS_CUDA_COMPILE_RUN_ID"],
        "package": {
            "name": os.environ["GATE_PACKAGE_BASENAME"],
            "record": file_record(os.environ["GATE_PACKAGE_ZIP"]),
        },
        "metal_comparison_log": file_record(os.environ["GATE_METAL_COMPARISON_LOG"]),
        "metal_probe_suite_log": file_record(os.environ["GATE_METAL_PROBE_SUITE_LOG"]),
        "metal_legacy_probe_suite_log": file_record(
            os.environ["GATE_METAL_LEGACY_PROBE_SUITE_LOG"]
        ),
        "metal_mcts_bk07_search_json": file_record(
            os.environ["GATE_METAL_MCTS_BK07_SEARCH_JSON"]
        ),
        "metal_mcts_kiwipete_search_json": file_record(
            os.environ["GATE_METAL_MCTS_KIWIPETE_SEARCH_JSON"]
        ),
        "metal_hybrid_startpos_search_json": file_record(
            os.environ["GATE_METAL_HYBRID_STARTPOS_SEARCH_JSON"]
        ),
    },
    "runtime": {
        "cuda_stable_execution_batch_size": os.environ[
            "GATE_CUDA_STABLE_BATCH_SIZE"
        ],
        "cuda_graph": os.environ["GATE_CUDA_GRAPH"],
        "cuda_profile": os.environ["GATE_CUDA_PROFILE"],
        "cuda_profile_limit": os.environ["GATE_CUDA_PROFILE_LIMIT"],
        "uci_go": os.environ["GATE_UCI_GO"],
        "hybrid_uci_go": os.environ["GATE_HYBRID_UCI_GO"],
    },
    "status": {
        "runtime_status": os.environ["RUNTIME_STATUS_FOR_MANIFEST"],
        "bt4_compare_status": os.environ["BT4_COMPARE_STATUS_FOR_MANIFEST"],
        "legacy_compare_status": os.environ["LEGACY_COMPARE_STATUS_FOR_MANIFEST"],
        "benchmark_compare_status": os.environ[
            "BENCHMARK_COMPARE_STATUS_FOR_MANIFEST"
        ],
        "search_compare_status": os.environ["SEARCH_COMPARE_STATUS_FOR_MANIFEST"],
        "final_compare_status": os.environ["FINAL_COMPARE_STATUS_FOR_MANIFEST"],
    },
    "artifacts": artifacts,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Wrote Windows CUDA runtime manifest: {manifest_path}")
PY
}

upload_collected_artifacts() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" || -z "${GCS_PREFIX}" || ! -d "${ARTIFACT_DIR}" ]]; then
    return 0
  fi

  shopt -s nullglob
  local files=("${ARTIFACT_DIR}"/*)
  shopt -u nullglob
  if ((${#files[@]} == 0)); then
    return 0
  fi

  gcloud storage cp --recursive "${files[@]}" "${GCS_PREFIX%/}/${INSTANCE}/" >/dev/null
  echo "Uploaded Windows CUDA runtime artifacts to ${GCS_PREFIX%/}/${INSTANCE}/"
}

require_file "${PACKAGE_ZIP}" "Windows CUDA package"
require_file "${WEIGHTS}" "BT4 weights"
require_file "${LEGACY_WEIGHTS}" "legacy 42850 weights"
require_file "${NNUE_BIG}" "large NNUE"
require_file "${NNUE_SMALL}" "small NNUE"
ensure_ssh_key
require_file "${SSH_PUB_KEY}" "SSH public key"

cd "${ROOT_DIR}"

cat >"${RUN_DIR}/enable-openssh.ps1" <<'POWERSHELL'
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force C:\metalfish\logs | Out-Null
Start-Transcript -Path C:\metalfish\logs\openssh-bootstrap.log -Append
try {
  $metadataHeaders = @{ "Metadata-Flavor" = "Google" }
  $remoteUser = Invoke-RestMethod `
    -Headers $metadataHeaders `
    -Uri "http://metadata.google.internal/computeMetadata/v1/instance/attributes/metalfish-windows-user"
  $publicKey = Invoke-RestMethod `
    -Headers $metadataHeaders `
    -Uri "http://metadata.google.internal/computeMetadata/v1/instance/attributes/metalfish-ssh-pubkey"
  if ([string]::IsNullOrWhiteSpace($remoteUser)) {
    throw "Missing metalfish-windows-user metadata"
  }
  if ([string]::IsNullOrWhiteSpace($publicKey)) {
    throw "Missing metalfish-ssh-pubkey metadata"
  }
  $capability = Get-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
  if ($capability.State -ne "Installed") {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-String | Write-Host
  }
  if (-not (Get-LocalUser -Name $remoteUser -ErrorAction SilentlyContinue)) {
    $password = ConvertTo-SecureString `
      -String ([Guid]::NewGuid().ToString("N") + "aA1!") `
      -AsPlainText `
      -Force
    New-LocalUser `
      -Name $remoteUser `
      -Password $password `
      -PasswordNeverExpires `
      -UserMayNotChangePassword | Out-Null
  }
  Enable-LocalUser -Name $remoteUser
  try {
    Add-LocalGroupMember -Group "Administrators" -Member $remoteUser
  } catch {
    if ($_.Exception.Message -notmatch "already") {
      throw
    }
  }
  $programDataSsh = Join-Path $env:ProgramData "ssh"
  New-Item -ItemType Directory -Force $programDataSsh | Out-Null
  $adminKeys = Join-Path $programDataSsh "administrators_authorized_keys"
  Set-Content -Path $adminKeys -Value $publicKey -Encoding ascii
  & icacls.exe $adminKeys /inheritance:r | Out-String | Write-Host
  & icacls.exe $adminKeys /grant "Administrators:F" /grant "SYSTEM:F" | Out-String | Write-Host
  $userSshDir = "C:\Users\$remoteUser\.ssh"
  New-Item -ItemType Directory -Force $userSshDir | Out-Null
  Set-Content -Path (Join-Path $userSshDir "authorized_keys") -Value $publicKey -Encoding ascii
  & icacls.exe "C:\Users\$remoteUser" /grant "${remoteUser}:(OI)(CI)F" /T | Out-String | Write-Host
  & icacls.exe $userSshDir /inheritance:r /grant "${remoteUser}:F" /grant "Administrators:F" /grant "SYSTEM:F" /T | Out-String | Write-Host
  New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null
  New-ItemProperty `
    -Path "HKLM:\SOFTWARE\OpenSSH" `
    -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -PropertyType String `
    -Force | Out-Null
  Set-Service -Name sshd -StartupType Automatic
  if ((Get-Service -Name sshd).Status -eq "Running") {
    Restart-Service sshd
  } else {
    Start-Service sshd
  }
  if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
      -Name "OpenSSH-Server-In-TCP" `
      -DisplayName "OpenSSH Server (sshd)" `
      -Enabled True `
      -Direction Inbound `
      -Protocol TCP `
      -Action Allow `
      -LocalPort 22 | Out-Null
  }
  Write-Host "OpenSSH server is ready"
} finally {
  Stop-Transcript
}
POWERSHELL

for candidate_machine in ${MACHINES}; do
  for candidate_zone in ${ZONES}; do
    echo "Creating ${INSTANCE} as ${candidate_machine} in ${candidate_zone}"
    if gcloud compute instances create "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${candidate_zone}" \
      --machine-type "${candidate_machine}" \
      --accelerator "${ACCELERATOR}" \
      --maintenance-policy TERMINATE \
      --restart-on-failure \
      --image-project "${IMAGE_PROJECT}" \
      --image-family "${IMAGE_FAMILY}" \
      --boot-disk-size "${BOOT_DISK_SIZE}" \
      --boot-disk-type "${BOOT_DISK_TYPE}" \
      --metadata "metalfish-windows-user=${REMOTE_USER}" \
      --metadata-from-file "windows-startup-script-ps1=${RUN_DIR}/enable-openssh.ps1,metalfish-ssh-pubkey=${SSH_PUB_KEY}" \
      --scopes https://www.googleapis.com/auth/cloud-platform; then
      MACHINE="${candidate_machine}"
      ZONE="${candidate_zone}"
      CREATED_INSTANCE=1
      break
    fi
    echo "Zone ${candidate_zone} could not allocate ${candidate_machine} with ${ACCELERATOR}" >&2
  done
  if [[ "${CREATED_INSTANCE}" == "1" ]]; then
    break
  fi
done

if [[ "${CREATED_INSTANCE}" != "1" ]]; then
  echo "failed to create a Windows CUDA runtime VM; machines: ${MACHINES}; zones: ${ZONES}" >&2
  exit 1
fi

wait_for_ssh 90

cat >"${RUN_DIR}/prepare.ps1" <<'POWERSHELL'
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force C:\metalfish | Out-Null
New-Item -ItemType Directory -Force C:\metalfish\logs | Out-Null
New-Item -ItemType Directory -Force C:\metalfish\networks | Out-Null
POWERSHELL
run_remote_ps "${RUN_DIR}/prepare.ps1"

cat >"${RUN_DIR}/install-driver.ps1" <<POWERSHELL
\$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force C:\\metalfish\\logs | Out-Null
try {
  nvidia-smi 2>&1 | Tee-Object -FilePath C:\\metalfish\\logs\\nvidia-smi-before.log
  if (\$LASTEXITCODE -eq 0) {
    Write-Host "NVIDIA driver already available"
    exit 0
  }
} catch {}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest "${DRIVER_SCRIPT_URL}" -OutFile C:\\install_gpu_driver.ps1
PowerShell -NoProfile -ExecutionPolicy Bypass -File C:\\install_gpu_driver.ps1 *> C:\\metalfish\\logs\\driver-install.log
Write-Host "Driver installer completed; rebooting"
Restart-Computer -Force
POWERSHELL
run_remote_ps "${RUN_DIR}/install-driver.ps1" || true
wait_for_ssh 90

echo "Waiting for NVIDIA driver"
for attempt in $(seq 1 40); do
  if remote_cmd "nvidia-smi" >"${RUN_DIR}/nvidia-smi.log" 2>&1; then
    mkdir -p "${ARTIFACT_DIR}/logs"
    cp "${RUN_DIR}/nvidia-smi.log" "${ARTIFACT_DIR}/logs/nvidia-smi.log"
    break
  fi
  if [[ "${attempt}" == "40" ]]; then
    cat "${RUN_DIR}/nvidia-smi.log" >&2 || true
    echo "timed out waiting for nvidia-smi on ${INSTANCE}" >&2
    exit 1
  fi
  sleep 15
done

copy_to_remote "${PACKAGE_ZIP}" "C:/metalfish/metalfish-windows-cuda.zip"
copy_to_remote "${WEIGHTS}" "C:/metalfish/networks/BT4-1024x15x32h-swa-6147500.pb"
copy_to_remote "${LEGACY_WEIGHTS}" "C:/metalfish/networks/legacy-42850.pb.gz"
copy_to_remote "${NNUE_BIG}" "C:/metalfish/networks/nn-c288c895ea92.nnue"
copy_to_remote "${NNUE_SMALL}" "C:/metalfish/networks/nn-37f18f62d772.nnue"
python3 - <<'PY' >"${RUN_DIR}/probe-positions.json"
import json
from tools.run_nn_backend_probe_suite import DEFAULT_POSITIONS

print(json.dumps({"positions": [position.__dict__ for position in DEFAULT_POSITIONS]}))
PY
copy_to_remote "${RUN_DIR}/probe-positions.json" "C:/metalfish/probe-positions.json"

cat >"${RUN_DIR}/run-smokes.ps1" <<POWERSHELL
\$ErrorActionPreference = "Stop"
\$Root = "C:\\metalfish"
\$Logs = Join-Path \$Root "logs"
\$PackageDir = Join-Path \$Root "package"
\$Networks = Join-Path \$Root "networks"
New-Item -ItemType Directory -Force \$Logs | Out-Null
Remove-Item \$PackageDir -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path (Join-Path \$Root "metalfish-windows-cuda.zip") -DestinationPath \$PackageDir -Force
\$VcRedist = Join-Path \$Root "vc_redist.x64.exe"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile \$VcRedist
\$vc = Start-Process -FilePath \$VcRedist -ArgumentList "/install", "/quiet", "/norestart" -Wait -PassThru
if (\$vc.ExitCode -ne 0 -and \$vc.ExitCode -ne 3010) {
  throw ("VC++ redistributable install failed with exit code " + \$vc.ExitCode)
}
\$Engine = Join-Path \$PackageDir "metalfish.exe"
if (-not (Test-Path \$Engine)) {
  throw "Packaged engine not found: \$Engine"
}
\$Probe = Join-Path \$PackageDir "metalfish_nn_probe.exe"
if (-not (Test-Path \$Probe)) {
  throw "Packaged NN probe not found: \$Probe"
}
\$Comparison = Join-Path \$PackageDir "test_nn_comparison.exe"
if (-not (Test-Path \$Comparison)) {
  throw "Packaged NN comparison not found: \$Comparison"
}
\$PortableManifest = Join-Path \$PackageDir "PORTABLE_ARTIFACT.md"
if (-not (Test-Path \$PortableManifest)) {
  throw "Packaged portable manifest not found: \$PortableManifest"
}
\$PackageJsonManifestPath = Join-Path \$PackageDir "windows-cuda-package-manifest.json"
if (-not (Test-Path \$PackageJsonManifestPath)) {
  throw "Packaged JSON manifest not found: \$PackageJsonManifestPath"
}
\$PackageJsonManifest = Get-Content -Path \$PackageJsonManifestPath -Raw | ConvertFrom-Json
if (\$PackageJsonManifest.schema -ne "metalfish.portable_artifact") {
  throw "Packaged JSON manifest has unexpected schema: \$(\$PackageJsonManifest.schema)"
}
if (\$PackageJsonManifest.package.kind -ne "windows-cuda") {
  throw "Packaged JSON manifest has unexpected kind: \$(\$PackageJsonManifest.package.kind)"
}
\$PackageJsonManifestFiles = @(\$PackageJsonManifest.files | ForEach-Object { \$_.name })
foreach (\$RequiredFile in @(
  "metalfish.exe",
  "metalfish_nn_probe.exe",
  "test_nn_comparison.exe",
  "PORTABLE_ARTIFACT.md"
)) {
  if (\$PackageJsonManifestFiles -notcontains \$RequiredFile) {
    throw "Packaged JSON manifest missing file entry: \$RequiredFile"
  }
}
foreach (\$RequiredDll in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
  if (-not (\$PackageJsonManifestFiles | Where-Object { \$_ -like \$RequiredDll })) {
    throw "Packaged JSON manifest missing CUDA runtime DLL entry: \$RequiredDll"
  }
}
\$env:PATH = "\$PackageDir;\$env:PATH"
nvidia-smi 2>&1 | Tee-Object -FilePath (Join-Path \$Logs "nvidia-smi-runtime.log")

\$CudaNetworkInfoRequiredText = @(
  "cuda_device_config=-1",
  "cuda_stable_execution_batch_effective=${CUDA_STABLE_BATCH_SIZE}",
  "cuda_deterministic_attention_softmax=true",
  "cuda_full_buffer_clear_effective=true"
)
if ("${CUDA_GRAPH}" -ne "0") {
  \$CudaNetworkInfoRequiredText += "cuda_graph_effective=true"
}
\$CudaMctsWarmupRequiredText = @(
  "capabilities=actual_backend=cuda"
)
if ("${CUDA_GRAPH}" -ne "0") {
  \$CudaMctsWarmupRequiredText += @(
    "MCTS backend warmup actual=",
    "executor=resolved+graph-replay"
  )
}

function Assert-CudaNetworkInfo {
  param(
    [string]\$Name,
    [string]\$NetworkInfo
  )
  foreach (\$needle in \$CudaNetworkInfoRequiredText) {
    if (\$NetworkInfo -notlike "*\$needle*") {
      throw "\$Name missing CUDA network info: \$needle"
    }
  }
}

function Invoke-ProbeSmoke {
  param(
    [string]\$Name,
    [string]\$Arguments,
    [string[]]\$RequiredText
  )
  \$stdout = Join-Path \$Logs "\$Name.stdout.log"
  \$stderr = Join-Path \$Logs "\$Name.stderr.log"
  \$psi = New-Object System.Diagnostics.ProcessStartInfo
  \$psi.FileName = \$Probe
  \$psi.Arguments = \$Arguments
  \$psi.WorkingDirectory = \$PackageDir
  \$psi.RedirectStandardOutput = \$true
  \$psi.RedirectStandardError = \$true
  \$psi.UseShellExecute = \$false
  \$psi.CreateNoWindow = \$true
  if ("${CUDA_GRAPH}" -ne "") {
    \$psi.Environment["METALFISH_CUDA_GRAPH"] = "${CUDA_GRAPH}"
  }
  if ("${CUDA_PROFILE}" -ne "") {
    \$psi.Environment["METALFISH_CUDA_PROFILE"] = "${CUDA_PROFILE}"
    \$psi.Environment["METALFISH_CUDA_PROFILE_LIMIT"] = "${CUDA_PROFILE_LIMIT}"
  }
  \$proc = [System.Diagnostics.Process]::Start(\$psi)
  \$stdoutTask = \$proc.StandardOutput.ReadToEndAsync()
  \$stderrTask = \$proc.StandardError.ReadToEndAsync()
  \$timedOut = -not \$proc.WaitForExit(${PROBE_TIMEOUT_SECONDS} * 1000)
  if (\$timedOut) {
    try { \$proc.Kill() } catch {}
    try { \$proc.WaitForExit(10000) | Out-Null } catch {}
  }
  \$out = \$stdoutTask.GetAwaiter().GetResult()
  \$err = \$stderrTask.GetAwaiter().GetResult()
  Set-Content -Path \$stdout -Value \$out -Encoding UTF8
  Set-Content -Path \$stderr -Value \$err -Encoding UTF8
  if (\$timedOut) {
    throw (\$Name + " timed out after ${PROBE_TIMEOUT_SECONDS}s")
  }
  if (\$proc.ExitCode -ne 0) {
    throw (\$Name + " exited with code " + \$proc.ExitCode)
  }
  foreach (\$needle in \$RequiredText) {
    if ((\$out + \$err) -notlike "*\$needle*") {
      throw "\$Name missing expected output: \$needle"
    }
  }
}

function Invoke-ComparisonSmoke {
  param(
    [string]\$Name,
    [string[]]\$RequiredText
  )
  \$stdout = Join-Path \$Logs "\$Name.stdout.log"
  \$stderr = Join-Path \$Logs "\$Name.stderr.log"
  \$parityReport = Join-Path \$Logs "\$Name-parity-report.md"
  \$psi = New-Object System.Diagnostics.ProcessStartInfo
  \$psi.FileName = \$Comparison
  \$psi.WorkingDirectory = \$PackageDir
  \$psi.RedirectStandardOutput = \$true
  \$psi.RedirectStandardError = \$true
  \$psi.UseShellExecute = \$false
  \$psi.CreateNoWindow = \$true
  \$psi.Environment["PATH"] = "\$PackageDir;" + \$psi.Environment["PATH"]
  \$psi.Environment["METALFISH_NN_WEIGHTS"] = \$Bt4
  \$psi.Environment["METALFISH_NN_PARITY_REPORT"] = \$parityReport
  \$psi.Environment["METALFISH_NN_BATCH_BENCH"] = "1"
  \$psi.Environment["METALFISH_NN_BATCH_TRACE_WORST"] = "1"
  \$psi.Environment["METALFISH_NN_SINGLE_REPEAT_STRESS"] = "1"
  \$psi.Environment["METALFISH_NN_SINGLE_REUSE_STRESS"] = "1"
  \$psi.Environment["METALFISH_NN_BATCH_REUSE_STRESS"] = "1"
  \$psi.Environment["METALFISH_NN_BENCH_ITERS"] = "2"
  \$psi.Environment["METALFISH_NN_BENCH_MAX_BATCH"] = "32"
  \$psi.Environment["METALFISH_NN_BENCH_WARMUP_ITERS"] = "3"
  \$psi.Environment["METALFISH_NN_BENCH_GRAPH_REUSE_PROBE"] = "1"
  \$psi.Environment["METALFISH_CUDA_GRAPH_STATUS_DETAIL"] = "1"
  \$psi.Environment["METALFISH_CUDA_PROFILE"] = "0"
  if ("${CUDA_GRAPH}" -ne "") {
    \$psi.Environment["METALFISH_CUDA_GRAPH"] = "${CUDA_GRAPH}"
  }
  \$proc = [System.Diagnostics.Process]::Start(\$psi)
  \$stdoutTask = \$proc.StandardOutput.ReadToEndAsync()
  \$stderrTask = \$proc.StandardError.ReadToEndAsync()
  \$timedOut = -not \$proc.WaitForExit(${COMPARISON_TIMEOUT_SECONDS} * 1000)
  if (\$timedOut) {
    try { \$proc.Kill() } catch {}
    try { \$proc.WaitForExit(10000) | Out-Null } catch {}
  }
  \$out = \$stdoutTask.GetAwaiter().GetResult()
  \$err = \$stderrTask.GetAwaiter().GetResult()
  Set-Content -Path \$stdout -Value \$out -Encoding UTF8
  Set-Content -Path \$stderr -Value \$err -Encoding UTF8
  if (\$timedOut) {
    throw (\$Name + " timed out after ${COMPARISON_TIMEOUT_SECONDS}s")
  }
  if (\$proc.ExitCode -ne 0) {
    throw (\$Name + " exited with code " + \$proc.ExitCode)
  }
  foreach (\$needle in \$RequiredText) {
    if ((\$out + \$err) -notlike "*\$needle*") {
      throw "\$Name missing expected output: \$needle"
    }
  }
  if (-not (Test-Path \$parityReport)) {
    throw "\$Name did not write parity report: \$parityReport"
  }
}

function Quote-ProbeArgument {
  param([string]\$Value)
  return '"' + \$Value.Replace('"', '\"') + '"'
}

function Invoke-ProbeSuiteSmoke {
  param(
    [string]\$Name,
    [string]\$Weights,
    [bool]\$RequireWdl = \$true,
    [bool]\$RequireMovesLeft = \$true
  )
  \$stdout = Join-Path \$Logs "\$Name.stdout.log"
  \$stderr = Join-Path \$Logs "\$Name.stderr.log"
  \$positionsPath = Join-Path \$Root "probe-positions.json"
  if (-not (Test-Path \$positionsPath)) {
    throw "Probe positions file not found: \$positionsPath"
  }
  \$positionsDoc = Get-Content -Path \$positionsPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (\$null -eq \$positionsDoc.positions) {
    throw "Probe positions file did not contain a positions array: \$positionsPath"
  }
  \$positions = New-Object System.Collections.Generic.List[object]
  foreach (\$entry in \$positionsDoc.positions) {
    [void]\$positions.Add(\$entry)
  }
  \$outBuilder = New-Object System.Text.StringBuilder
  \$errBuilder = New-Object System.Text.StringBuilder
  foreach (\$position in \$positions) {
    [void]\$outBuilder.AppendLine("info string windows-probe-suite name=" + \$position.name)
    \$moves = ""
    if (\$null -ne \$position.moves) {
      \$moves = [string]\$position.moves
    }
    \$arguments = "--weights " + (Quote-ProbeArgument \$Weights) +
      " --backend cuda --fen " + (Quote-ProbeArgument \$position.fen) +
      " --cuda-device -1 --cuda-graph-execution true" +
      " --cuda-stable-execution-batch-size ${CUDA_STABLE_BATCH_SIZE}" +
      " --cuda-deterministic-attention-softmax true" +
      " --cuda-full-buffer-clear true" +
      " --top 3 --warmup 1 --iterations 1 --full-policy"
    if (-not [string]::IsNullOrWhiteSpace(\$moves)) {
      \$arguments += " --moves " + (Quote-ProbeArgument \$moves)
    }
    \$psi = New-Object System.Diagnostics.ProcessStartInfo
    \$psi.FileName = \$Probe
    \$psi.Arguments = \$arguments
    \$psi.WorkingDirectory = \$PackageDir
    \$psi.RedirectStandardOutput = \$true
    \$psi.RedirectStandardError = \$true
    \$psi.UseShellExecute = \$false
    \$psi.CreateNoWindow = \$true
    if ("${CUDA_GRAPH}" -ne "") {
      \$psi.Environment["METALFISH_CUDA_GRAPH"] = "${CUDA_GRAPH}"
    }
    if ("${CUDA_PROFILE}" -ne "") {
      \$psi.Environment["METALFISH_CUDA_PROFILE"] = "${CUDA_PROFILE}"
      \$psi.Environment["METALFISH_CUDA_PROFILE_LIMIT"] = "${CUDA_PROFILE_LIMIT}"
    }
    \$proc = [System.Diagnostics.Process]::Start(\$psi)
    \$stdoutTask = \$proc.StandardOutput.ReadToEndAsync()
    \$stderrTask = \$proc.StandardError.ReadToEndAsync()
    \$timedOut = -not \$proc.WaitForExit(${PROBE_TIMEOUT_SECONDS} * 1000)
    if (\$timedOut) {
      try { \$proc.Kill() } catch {}
      try { \$proc.WaitForExit(10000) | Out-Null } catch {}
    }
    \$out = \$stdoutTask.GetAwaiter().GetResult()
    \$err = \$stderrTask.GetAwaiter().GetResult()
    [void]\$outBuilder.Append(\$out)
    [void]\$errBuilder.Append(\$err)
    if (\$timedOut) {
      Set-Content -Path \$stdout -Value \$outBuilder.ToString() -Encoding UTF8
      Set-Content -Path \$stderr -Value \$errBuilder.ToString() -Encoding UTF8
      throw ("\$Name " + \$position.name + " timed out after ${PROBE_TIMEOUT_SECONDS}s")
    }
    if (\$proc.ExitCode -ne 0) {
      Set-Content -Path \$stdout -Value \$outBuilder.ToString() -Encoding UTF8
      Set-Content -Path \$stderr -Value \$errBuilder.ToString() -Encoding UTF8
      throw ("\$Name " + \$position.name + " exited with code " + \$proc.ExitCode)
    }
  }
  \$outText = \$outBuilder.ToString()
  \$errText = \$errBuilder.ToString()
  Set-Content -Path \$stdout -Value \$outText -Encoding UTF8
  Set-Content -Path \$stderr -Value \$errText -Encoding UTF8

  \$probeObjects = @()
  foreach (\$line in \$outText -split [char]10) {
    \$trimmed = \$line.Trim()
    if (\$trimmed.StartsWith("{") -and \$trimmed.EndsWith("}")) {
      \$probeObjects += @(\$trimmed | ConvertFrom-Json)
    }
  }
  if (\$probeObjects.Count -ne \$positions.Count) {
    throw ("\$Name expected " + \$positions.Count + " JSON probes, got " + \$probeObjects.Count)
  }
  foreach (\$probeObject in \$probeObjects) {
    if (\$probeObject.backend -ne "cuda") {
      throw ("\$Name probe did not select CUDA backend")
    }
    \$networkInfo = \$probeObject.network_info -as [string]
    if (\$networkInfo -notlike "*CUDA transformer backend*") {
      throw ("\$Name probe did not report CUDA transformer backend")
    }
    Assert-CudaNetworkInfo -Name \$Name -NetworkInfo \$networkInfo
    if ("${CUDA_GRAPH}" -ne "0" -and \$networkInfo -notlike "*executor=resolved+graph-replay*") {
      throw ("\$Name probe did not report CUDA graph replay")
    }
    if ([bool]\$probeObject.has_wdl -ne \$RequireWdl) {
      throw ("\$Name WDL presence mismatch")
    }
    if ([bool]\$probeObject.has_moves_left -ne \$RequireMovesLeft) {
      throw ("\$Name moves-left presence mismatch")
    }
    if (\$null -eq \$probeObject.policy -or \$probeObject.policy.Count -ne 1858) {
      throw ("\$Name missing full 1858-entry policy")
    }
  }
  return [ordered]@{
    probes = \$probeObjects.Count
    stdout_log = "\$Name.stdout.log"
    stderr_log = "\$Name.stderr.log"
  }
}

function Assert-PositiveMetric {
  param(
    [string]\$Name,
    [string]\$Text,
    [string]\$Metric
  )
  \$match = [regex]::Match(\$Text, \$Metric + "=([0-9]+)")
  if (-not \$match.Success) {
    throw "\$Name missing metric: \$Metric"
  }
  \$value = [int64]\$match.Groups[1].Value
  if (\$value -le 0) {
    throw "\$Name metric \$Metric was not positive: \$value"
  }
}

function Read-LogText {
  param([string]\$Name)
  \$path = Join-Path \$Logs \$Name
  if (-not (Test-Path \$path)) {
    return ""
  }
  return (Get-Content -Path \$path -Raw -Encoding UTF8).TrimStart([char]0xFEFF)
}

function Read-SmokeText {
  param([string]\$Name)
  return ((Read-LogText "\$Name.stdout.log") + [Environment]::NewLine +
          (Read-LogText "\$Name.stderr.log"))
}

function Read-ProbeJson {
  param([string]\$Name)
  \$text = Read-LogText \$Name
  if ([string]::IsNullOrWhiteSpace(\$text)) {
    throw "Probe log was empty: \$Name"
  }
  return \$text | ConvertFrom-Json
}

function Find-BestMove {
  param([string]\$Text)
  \$matches = [regex]::Matches(\$Text, "(?m)^bestmove\s+(\S+)")
  if (\$matches.Count -eq 0) {
    return \$null
  }
  return \$matches[\$matches.Count - 1].Groups[1].Value
}

function Write-SearchJson {
  param(
    [string]\$Name,
    [string]\$Text,
    [string]\$Position,
    [string]\$Go
  )
  \$bytes = [System.Text.Encoding]::UTF8.GetBytes(\$Text)
  \$sha = [System.Security.Cryptography.SHA256]::Create()
  try {
    \$hashBytes = \$sha.ComputeHash(\$bytes)
  } finally {
    \$sha.Dispose()
  }
  \$hash = ([BitConverter]::ToString(\$hashBytes)).Replace("-", "").ToLowerInvariant()
  \$lines = @([regex]::Split(\$Text, "\r?\n"))
  \$tailStart = [Math]::Max(0, \$lines.Count - 80)
  \$payload = [ordered]@{
    schema = "metalfish.uci_smoke_result"
    schema_version = 1
    engine = \$Engine
    position = \$Position
    go = \$Go
    setoptions = @()
    bestmove = (Find-BestMove \$Text)
    elapsed_sec = \$null
    returncode = 0
    transcript_sha256 = \$hash
    transcript_tail = @(\$lines[\$tailStart..(\$lines.Count - 1)])
  }
  \$payload | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path \$Logs "\$Name.json") -Encoding UTF8
}

function Find-Executor {
  param([string]\$Text)
  \$match = [regex]::Match(\$Text, "executor=([^,)]+(?:\([^)]*\))?)")
  if (-not \$match.Success) {
    return \$null
  }
  return \$match.Groups[1].Value
}

function Test-BackendSelected {
  param([string]\$Text)
  return \$Text -like "*CUDA transformer backend*"
}

function Find-FinalMetrics {
  param([string]\$Text)
  \$metrics = [ordered]@{}
  \$matches = [regex]::Matches(\$Text, "(?m)^info string Final:\s+(.*)$")
  if (\$matches.Count -eq 0) {
    return \$metrics
  }
  \$line = \$matches[\$matches.Count - 1].Groups[1].Value
  foreach (\$match in [regex]::Matches(\$line, "([A-Za-z0-9]+)=([^\s]+)")) {
    \$key = \$match.Groups[1].Value
    \$raw = \$match.Groups[2].Value
    \$intValue = 0L
    \$doubleValue = 0.0
    if ([int64]::TryParse(\$raw, [ref]\$intValue)) {
      \$metrics[\$key] = \$intValue
    } elseif ([double]::TryParse(\$raw, [ref]\$doubleValue)) {
      \$metrics[\$key] = \$doubleValue
    } else {
      \$metrics[\$key] = \$raw
    }
  }
  return \$metrics
}

function Get-GpuInfo {
  \$gpu = [ordered]@{
    nvidia_smi_log = "nvidia-smi-runtime.log"
  }
  try {
    \$query = & nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits 2>\$null
    if (\$LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(\$query)) {
      \$fields = @(\$query)[0].Split(",") | ForEach-Object { \$_.Trim() }
      if (\$fields.Count -ge 1) { \$gpu["name"] = \$fields[0] }
      if (\$fields.Count -ge 2) { \$gpu["driver_version"] = \$fields[1] }
      if (\$fields.Count -ge 3) {
        \$memoryTotalMiB = 0
        if ([int]::TryParse(\$fields[2], [ref]\$memoryTotalMiB)) {
          \$gpu["memory_total_mib"] = \$memoryTotalMiB
        } else {
          \$gpu["memory_total"] = \$fields[2]
        }
      }
      if (\$fields.Count -ge 4) { \$gpu["compute_capability"] = \$fields[3] }
    }
  } catch {}
  return \$gpu
}

function Invoke-UciSmoke {
  param(
    [string]\$Name,
    [string[]]\$Commands,
    [string[]]\$RequiredText,
    [string[]]\$RejectedText = @(),
    [string[]]\$PositiveMetrics = @()
  )
  \$stdout = Join-Path \$Logs "\$Name.stdout.log"
  \$stderr = Join-Path \$Logs "\$Name.stderr.log"
  \$psi = New-Object System.Diagnostics.ProcessStartInfo
  \$psi.FileName = \$Engine
  \$psi.WorkingDirectory = \$PackageDir
  \$psi.RedirectStandardInput = \$true
  \$psi.RedirectStandardOutput = \$true
  \$psi.RedirectStandardError = \$true
  \$psi.UseShellExecute = \$false
  \$psi.CreateNoWindow = \$true
  if ("${UCI_TRACE}" -ne "") {
    \$psi.Environment["METALFISH_UCI_TRACE"] = "${UCI_TRACE}"
  }
  if ("${CUDA_GRAPH}" -ne "") {
    \$psi.Environment["METALFISH_CUDA_GRAPH"] = "${CUDA_GRAPH}"
  }
  if ("${CUDA_PROFILE}" -ne "") {
    \$psi.Environment["METALFISH_CUDA_PROFILE"] = "${CUDA_PROFILE}"
    \$psi.Environment["METALFISH_CUDA_PROFILE_LIMIT"] = "${CUDA_PROFILE_LIMIT}"
  }

  \$outBuilder = New-Object System.Text.StringBuilder
  \$deadline = [DateTime]::UtcNow.AddSeconds(${UCI_TIMEOUT_SECONDS})

  \$proc = \$null
  \$stdoutTask = \$null
  \$timedOut = \$false
  \$failed = \$false
  \$failureMessage = ""
  try {
    \$proc = New-Object System.Diagnostics.Process
    \$proc.StartInfo = \$psi
    [void]\$proc.Start()
    foreach (\$command in \$Commands) {
      \$isGoCommand = \$command -like "go *"
      \$proc.StandardInput.WriteLine(\$command)
      \$proc.StandardInput.Flush()
      if (\$isGoCommand) {
        \$bestMoveSeen = \$false
        if (\$null -eq \$stdoutTask) {
          \$stdoutTask = \$proc.StandardOutput.ReadLineAsync()
        }
        while ([DateTime]::UtcNow -lt \$deadline) {
          if (\$proc.HasExited -and -not \$stdoutTask.IsCompleted) {
            break
          }
          \$remainingMs = [Math]::Max(1, [Math]::Min(250, (\$deadline - [DateTime]::UtcNow).TotalMilliseconds))
          if (-not \$stdoutTask.Wait([int]\$remainingMs)) {
            continue
          }
          \$line = \$stdoutTask.GetAwaiter().GetResult()
          if (\$null -eq \$line) {
            break
          }
          [void]\$outBuilder.AppendLine(\$line)
          if (\$line -match '^bestmove\s+') {
            \$bestMoveSeen = \$true
            break
          }
          \$stdoutTask = \$proc.StandardOutput.ReadLineAsync()
        }
        if (-not \$bestMoveSeen) {
          \$timedOut = \$true
          break
        }
      }
    }
    if (-not \$timedOut) {
      \$proc.StandardInput.Close()
    }
  } catch {
    \$failed = \$true
    \$failureMessage = \$_.Exception.Message
    [void]\$outBuilder.AppendLine("")
  }
  if (\$failed) {
    if (\$null -ne \$proc) {
      try { \$proc.Kill() } catch {}
      try { \$proc.WaitForExit(10000) | Out-Null } catch {}
    }
  } elseif (-not \$timedOut) {
    \$timedOut = -not \$proc.WaitForExit(${UCI_TIMEOUT_SECONDS} * 1000)
  }
  if (\$timedOut) {
    try { \$proc.Kill() } catch {}
    try { \$proc.WaitForExit(10000) | Out-Null } catch {}
  } else {
    try { \$proc.WaitForExit() } catch {}
  }
  if (-not \$timedOut -and -not \$failed) {
    try {
      while (-not \$proc.StandardOutput.EndOfStream) {
        \$line = \$proc.StandardOutput.ReadLine()
        if (\$null -eq \$line) { break }
        [void]\$outBuilder.AppendLine(\$line)
      }
    } catch {}
  }
  \$out = \$outBuilder.ToString()
  try {
    if (\$null -ne \$proc) {
      \$err = \$proc.StandardError.ReadToEnd()
    } else {
      \$err = ""
    }
  } catch {
    \$err = \$_.Exception.ToString()
  }
  if (\$failed) {
    \$err = \$err + [Environment]::NewLine + \$failureMessage
  }
  Set-Content -Path \$stdout -Value \$out -Encoding UTF8
  Set-Content -Path \$stderr -Value \$err -Encoding UTF8
  if (\$failed) {
    throw (\$Name + " failed while driving UCI commands: " + \$failureMessage)
  }
  if (\$timedOut) {
    throw (\$Name + " timed out waiting for bestmove or exit after ${UCI_TIMEOUT_SECONDS}s")
  }
  if (\$proc.ExitCode -ne 0) {
    throw (\$Name + " exited with code " + \$proc.ExitCode)
  }
  foreach (\$needle in \$RequiredText) {
    if ((\$out + \$err) -notlike "*\$needle*") {
      throw "\$Name missing expected output: \$needle"
    }
  }
  foreach (\$needle in \$RejectedText) {
    if ((\$out + \$err) -like "*\$needle*") {
      throw "\$Name contained rejected output: \$needle"
    }
  }
  foreach (\$metric in \$PositiveMetrics) {
    Assert-PositiveMetric -Name \$Name -Text (\$out + \$err) -Metric \$metric
  }
}

\$Bt4 = Join-Path \$Networks "BT4-1024x15x32h-swa-6147500.pb"
\$Legacy = Join-Path \$Networks "legacy-42850.pb.gz"
\$NnueBig = Join-Path \$Networks "nn-c288c895ea92.nnue"
\$NnueSmall = Join-Path \$Networks "nn-37f18f62d772.nnue"
\$Bk07Fen = "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - -"
\$KiwipeteFen = "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QK2R w KQkq - 0 1"
\$CudaProbeOptions = " --backend cuda" +
  " --cuda-device -1 --cuda-graph-execution true" +
  " --cuda-stable-execution-batch-size ${CUDA_STABLE_BATCH_SIZE}" +
  " --cuda-deterministic-attention-softmax true" +
  " --cuda-full-buffer-clear true"

\$ProbeArgs = "--weights " + [char]34 + \$Bt4 + [char]34 + \$CudaProbeOptions + " --batch-size 1 --warmup 1 --iterations 1 --top 3"
Invoke-ProbeSmoke -Name "cuda-probe" -Arguments \$ProbeArgs -RequiredText (\$CudaNetworkInfoRequiredText + @('"backend":"cuda"', "CUDA transformer backend", '"value":', '"policy_top":'))
\$LegacyProbeArgs = "--weights " + [char]34 + \$Legacy + [char]34 + \$CudaProbeOptions + " --batch-size 1 --warmup 1 --iterations 1 --top 3"
Invoke-ProbeSmoke -Name "cuda-legacy-probe" -Arguments \$LegacyProbeArgs -RequiredText (\$CudaNetworkInfoRequiredText + @('"backend":"cuda"', "CUDA transformer backend", '"has_wdl":false', '"has_moves_left":false', '"policy_top":'))
\$ProbeSuite = Invoke-ProbeSuiteSmoke -Name "cuda-probe-suite" -Weights \$Bt4 -RequireWdl \$true -RequireMovesLeft \$true
\$LegacyProbeSuite = Invoke-ProbeSuiteSmoke -Name "cuda-legacy-probe-suite" -Weights \$Legacy -RequireWdl \$false -RequireMovesLeft \$false
\$IsolationRequiredText = \$CudaNetworkInfoRequiredText + @('"isolation":true', '"backend":"cuda"', "CUDA transformer backend", '"delta":')
if ("${CUDA_GRAPH}" -ne "0") {
  \$IsolationRequiredText += "executor=resolved+graph-replay"
}
\$IsolationBt4LegacyArgs = "--weights " + [char]34 + \$Bt4 + [char]34 +
  " --isolation-weights " + [char]34 + \$Legacy + [char]34 +
  \$CudaProbeOptions + " --warmup 1 --iterations 1 --top 3"
Invoke-ProbeSmoke -Name "cuda-isolation-bt4-legacy" -Arguments \$IsolationBt4LegacyArgs -RequiredText \$IsolationRequiredText
\$IsolationLegacyBt4Args = "--weights " + [char]34 + \$Legacy + [char]34 +
  " --isolation-weights " + [char]34 + \$Bt4 + [char]34 +
  \$CudaProbeOptions + " --warmup 1 --iterations 1 --top 3"
Invoke-ProbeSmoke -Name "cuda-isolation-legacy-bt4" -Arguments \$IsolationLegacyBt4Args -RequiredText \$IsolationRequiredText

\$ComparisonRequiredText = @(
  "=== NN Comparison Smoke ===",
  "MCTS evaluator batch parity",
  "CUDA transformer backend",
  "TRACE_WORST:",
  "SINGLE_REPEAT_STRESS_MAX:",
  "SINGLE_REUSE_STRESS_MAX:",
  "REUSE_STRESS_MAX:",
  "batches:",
  "graph_reuse_probe:"
)
if ("${CUDA_GRAPH}" -ne "0") {
  \$ComparisonRequiredText += "executor=resolved+graph-replay"
}
Invoke-ComparisonSmoke -Name "cuda-nn-comparison" -RequiredText \$ComparisonRequiredText

Invoke-UciSmoke -Name "cuda-mcts" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value true",
  "setoption name UseHybridSearch value false",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 1",
  "position startpos",
  "go ${UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("CUDA transformer backend", "MCTS runtime: backend=cuda", "minibatch=1", "bestmove"))

Invoke-UciSmoke -Name "cuda-auto-mcts" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name NNBackend value auto",
  "setoption name NNBackendRequireAccelerator value true",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value true",
  "setoption name UseHybridSearch value false",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 0",
  "position startpos",
  "go ${UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("CUDA transformer backend", "MCTS runtime: backend=accelerator", "minibatch=${CUDA_STABLE_BATCH_SIZE}", "bestmove"))

Invoke-UciSmoke -Name "cuda-accelerator-mcts" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name NNBackend value accelerator",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value true",
  "setoption name UseHybridSearch value false",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 0",
  "position startpos",
  "go ${UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("CUDA transformer backend", "MCTS runtime: backend=accelerator", "minibatch=${CUDA_STABLE_BATCH_SIZE}", "bestmove"))

Invoke-UciSmoke -Name "cuda-bk07-mcts" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value true",
  "setoption name UseHybridSearch value false",
  "setoption name Threads value 8",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSParallelSearch value false",
  "setoption name MCTSMinibatchSize value 1",
  "setoption name MCTSParityPreset value true",
  "setoption name MCTSAddDirichletNoise value false",
  "setoption name TransformerLowTimeFallbackMs value 0",
  "position fen \$Bk07Fen",
  "go nodes 50",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("CUDA transformer backend", "MCTS runtime: backend=cuda", "minibatch=1", "bestmove h5f6"))

Invoke-UciSmoke -Name "cuda-kiwipete-mcts" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value true",
  "setoption name UseHybridSearch value false",
  "setoption name Threads value 8",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSParallelSearch value false",
  "setoption name MCTSMinibatchSize value 1",
  "setoption name MCTSParityPreset value true",
  "setoption name MCTSAddDirichletNoise value false",
  "setoption name TransformerLowTimeFallbackMs value 0",
  "position fen \$KiwipeteFen",
  "go nodes 1",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("CUDA transformer backend", "MCTS runtime: backend=cuda", "minibatch=1", "bestmove"))

Invoke-UciSmoke -Name "hybrid-cuda" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name Threads value 3",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value false",
  "setoption name UseHybridSearch value true",
  "setoption name HybridMCTSThreads value 1",
  "setoption name HybridABThreads value 2",
  "setoption name HybridAutoABThreadsCap value 0",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 1",
  "setoption name MCTSParityPreset value true",
  "setoption name MCTSAddDirichletNoise value false",
  "setoption name TransformerLowTimeFallbackMs value 0",
  "position startpos",
  "go ${HYBRID_UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("Starting Parallel Hybrid Search", "Hybrid MCTS runtime: backend=cuda", "minibatch=1", "CUDA transformer backend", "Final: MCTSPlayouts=", "bestmove")) -PositiveMetrics @("MCTSPlayouts", "MCTSEvals")

Invoke-UciSmoke -Name "hybrid-cuda-clock-start" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name Threads value 3",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value false",
  "setoption name UseHybridSearch value true",
  "setoption name HybridMCTSThreads value 1",
  "setoption name HybridABThreads value 2",
  "setoption name HybridAutoABThreadsCap value 0",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 1",
  "setoption name Move Overhead value 500",
  "setoption name TransformerLowTimeFallbackMs value 3000",
  "setoption name TransformerMinMoveBudgetMs value 400",
  "setoption name MCTSAddDirichletNoise value false",
  "position startpos",
  "go wtime 1000 btime 1000 winc 3000 binc 3000",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("Starting Parallel Hybrid Search", "Hybrid MCTS runtime: backend=cuda", "CUDA transformer backend", "bestmove")) -RejectedText @("Time safety:")

Invoke-UciSmoke -Name "hybrid-cuda-clock-safety" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name Threads value 3",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name UseMCTS value false",
  "setoption name UseHybridSearch value true",
  "setoption name HybridMCTSThreads value 1",
  "setoption name HybridABThreads value 2",
  "setoption name HybridAutoABThreadsCap value 0",
  "setoption name Move Overhead value 500",
  "setoption name TransformerLowTimeFallbackMs value 3000",
  "setoption name TransformerMinMoveBudgetMs value 400",
  "position startpos",
  "go wtime 800 btime 800 winc 3000 binc 3000",
  "quit"
) -RequiredText @("Time safety: estimated move budget", "bestmove") -RejectedText @("Starting Parallel Hybrid Search")

\$DummyCoreMl = Join-Path \$PackageDir "dummy-coreml.mlmodelc"
New-Item -ItemType Directory -Force -Path \$DummyCoreMl | Out-Null

Invoke-UciSmoke -Name "hybrid-auto" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name Threads value 3",
  "setoption name NNBackend value auto",
  "setoption name NNBackendRequireAccelerator value true",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value false",
  "setoption name UseHybridSearch value true",
  "setoption name HybridMCTSThreads value 1",
  "setoption name HybridABThreads value 2",
  "setoption name HybridAutoABThreadsCap value 0",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 0",
  "setoption name TransformerLowTimeFallbackMs value 0",
  "position startpos",
  "go ${HYBRID_UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("Starting Parallel Hybrid Search", "Hybrid MCTS runtime: backend=accelerator", "minibatch=${CUDA_STABLE_BATCH_SIZE}", "CUDA transformer backend", "Final: MCTSPlayouts=", "bestmove")) -PositiveMetrics @("MCTSPlayouts", "MCTSEvals")

Invoke-UciSmoke -Name "hybrid-cuda-ane-disabled" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name Threads value 3",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name NNCudaDevice value -1",
  "setoption name NNCudaGraphExecution value true",
  "setoption name NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
  "setoption name NNCudaDeterministicAttentionSoftmax value true",
  "setoption name NNCudaFullBufferClear value true",
  "setoption name UseMCTS value false",
  "setoption name UseHybridSearch value true",
  "setoption name HybridMCTSThreads value 1",
  "setoption name HybridABThreads value 2",
  "setoption name HybridAutoABThreadsCap value 0",
  "setoption name HybridANERootProbe value true",
  "setoption name HybridANERootHints value true",
  "setoption name HybridANEWeights value \$Bt4",
  "setoption name HybridANEModelPath value \$DummyCoreMl",
  "setoption name HybridANEComputeUnits value all",
  "setoption name HybridANERootHintWaitMs value 0",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 1",
  "setoption name TransformerLowTimeFallbackMs value 0",
  "position startpos",
  "go ${HYBRID_UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + \$CudaMctsWarmupRequiredText + @("Starting Parallel Hybrid Search", "Hybrid MCTS runtime: backend=cuda", "minibatch=1", "CUDA transformer backend", "ANE root probe disabled", "Final: MCTSPlayouts=", "bestmove")) -PositiveMetrics @("MCTSPlayouts", "MCTSEvals")

\$ProbeJson = Read-ProbeJson "cuda-probe.stdout.log"
\$LegacyProbeJson = Read-ProbeJson "cuda-legacy-probe.stdout.log"
\$IsolationBt4LegacyJson = Read-ProbeJson "cuda-isolation-bt4-legacy.stdout.log"
\$IsolationLegacyBt4Json = Read-ProbeJson "cuda-isolation-legacy-bt4.stdout.log"
\$MctsText = Read-SmokeText "cuda-mcts"
\$AutoMctsText = Read-SmokeText "cuda-auto-mcts"
\$AcceleratorMctsText = Read-SmokeText "cuda-accelerator-mcts"
\$Bk07MctsText = Read-SmokeText "cuda-bk07-mcts"
\$KiwipeteMctsText = Read-SmokeText "cuda-kiwipete-mcts"
\$HybridText = Read-SmokeText "hybrid-cuda"
\$HybridClockStartText = Read-SmokeText "hybrid-cuda-clock-start"
\$HybridClockSafetyText = Read-SmokeText "hybrid-cuda-clock-safety"
\$HybridAutoText = Read-SmokeText "hybrid-auto"
\$HybridAneText = Read-SmokeText "hybrid-cuda-ane-disabled"
\$ComparisonText = Read-SmokeText "cuda-nn-comparison"
Write-SearchJson -Name "cuda-bk07-mcts-search" -Text \$Bk07MctsText -Position "fen \$Bk07Fen" -Go "nodes 50"
Write-SearchJson -Name "cuda-kiwipete-mcts-search" -Text \$KiwipeteMctsText -Position "fen \$KiwipeteFen" -Go "nodes 1"
Write-SearchJson -Name "hybrid-cuda-search" -Text \$HybridText -Position "startpos" -Go "${HYBRID_UCI_GO}"
\$RemoteZip = Join-Path \$Root "metalfish-windows-cuda.zip"
\$PackageHash = (Get-FileHash -Path \$RemoteZip -Algorithm SHA256).Hash.ToLowerInvariant()
\$Manifest = [ordered]@{
  schema_version = 1
  gate_status = "passed"
  timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
  package = [ordered]@{
    basename = "${PACKAGE_BASENAME}"
    remote_zip = \$RemoteZip
    sha256 = \$PackageHash
    compile_run_id = \$(if ("${WINDOWS_CUDA_COMPILE_RUN_ID}" -eq "") { \$null } else { "${WINDOWS_CUDA_COMPILE_RUN_ID}" })
    manifest = [ordered]@{
      basename = "windows-cuda-package-manifest.json"
      schema = \$PackageJsonManifest.schema
      kind = \$PackageJsonManifest.package.kind
      file_count = @(\$PackageJsonManifest.files).Count
    }
  }
  gcp = [ordered]@{
    instance = "${INSTANCE}"
    zone = "${ZONE}"
    machine = "${MACHINE}"
    accelerator = "${ACCELERATOR}"
    image_project = "${IMAGE_PROJECT}"
    image_family = "${IMAGE_FAMILY}"
  }
  config = [ordered]@{
    uci_go = "${UCI_GO}"
    hybrid_uci_go = "${HYBRID_UCI_GO}"
    uci_timeout_seconds = ${UCI_TIMEOUT_SECONDS}
    probe_timeout_seconds = ${PROBE_TIMEOUT_SECONDS}
    comparison_timeout_seconds = ${COMPARISON_TIMEOUT_SECONDS}
    cuda_graph = \$(if ("${CUDA_GRAPH}" -eq "") { \$null } else { "${CUDA_GRAPH}" })
    cuda_profile = \$(if ("${CUDA_PROFILE}" -eq "") { \$null } else { "${CUDA_PROFILE}" })
    cuda_profile_limit = ${CUDA_PROFILE_LIMIT}
    cuda_stable_execution_batch_size = ${CUDA_STABLE_BATCH_SIZE}
  }
  gpu = (Get-GpuInfo)
  probe = [ordered]@{
    backend = \$ProbeJson.backend
    executor = (Find-Executor \$ProbeJson.network_info)
    network_info = \$ProbeJson.network_info
    format = \$ProbeJson.format
    has_wdl = \$ProbeJson.has_wdl
    has_moves_left = \$ProbeJson.has_moves_left
    moves_left = \$ProbeJson.moves_left
    latency = \$ProbeJson.latency
    policy_top = \$ProbeJson.policy_top
    stdout_log = "cuda-probe.stdout.log"
    stderr_log = "cuda-probe.stderr.log"
  }
  legacy_probe = [ordered]@{
    backend = \$LegacyProbeJson.backend
    executor = (Find-Executor \$LegacyProbeJson.network_info)
    network_info = \$LegacyProbeJson.network_info
    format = \$LegacyProbeJson.format
    has_wdl = \$LegacyProbeJson.has_wdl
    has_moves_left = \$LegacyProbeJson.has_moves_left
    value = \$LegacyProbeJson.value
    latency = \$LegacyProbeJson.latency
    policy_top = \$LegacyProbeJson.policy_top
    stdout_log = "cuda-legacy-probe.stdout.log"
    stderr_log = "cuda-legacy-probe.stderr.log"
  }
  probe_suites = [ordered]@{
    bt4 = \$ProbeSuite
    legacy = \$LegacyProbeSuite
  }
  isolation_probes = [ordered]@{
    bt4_then_legacy = [ordered]@{
      primary_network_info = \$IsolationBt4LegacyJson.primary_network_info
      secondary_network_info = \$IsolationBt4LegacyJson.secondary_network_info
      primary_executor = (Find-Executor \$IsolationBt4LegacyJson.primary_network_info)
      secondary_executor = (Find-Executor \$IsolationBt4LegacyJson.secondary_network_info)
      delta = \$IsolationBt4LegacyJson.delta
      stdout_log = "cuda-isolation-bt4-legacy.stdout.log"
      stderr_log = "cuda-isolation-bt4-legacy.stderr.log"
    }
    legacy_then_bt4 = [ordered]@{
      primary_network_info = \$IsolationLegacyBt4Json.primary_network_info
      secondary_network_info = \$IsolationLegacyBt4Json.secondary_network_info
      primary_executor = (Find-Executor \$IsolationLegacyBt4Json.primary_network_info)
      secondary_executor = (Find-Executor \$IsolationLegacyBt4Json.secondary_network_info)
      delta = \$IsolationLegacyBt4Json.delta
      stdout_log = "cuda-isolation-legacy-bt4.stdout.log"
      stderr_log = "cuda-isolation-legacy-bt4.stderr.log"
    }
  }
  comparison = [ordered]@{
    backend_selected = (Test-BackendSelected \$ComparisonText)
    stdout_log = "cuda-nn-comparison.stdout.log"
    stderr_log = "cuda-nn-comparison.stderr.log"
    parity_report_log = "cuda-nn-comparison-parity-report.md"
  }
  uci_smokes = [ordered]@{
    cuda_mcts = [ordered]@{
      go = "${UCI_GO}"
      bestmove = (Find-BestMove \$MctsText)
      backend_selected = (Test-BackendSelected \$MctsText)
      stdout_log = "cuda-mcts.stdout.log"
      stderr_log = "cuda-mcts.stderr.log"
    }
    cuda_auto_mcts = [ordered]@{
      go = "${UCI_GO}"
      bestmove = (Find-BestMove \$AutoMctsText)
      backend_selected = (Test-BackendSelected \$AutoMctsText)
      stdout_log = "cuda-auto-mcts.stdout.log"
      stderr_log = "cuda-auto-mcts.stderr.log"
    }
    cuda_accelerator_mcts = [ordered]@{
      go = "${UCI_GO}"
      bestmove = (Find-BestMove \$AcceleratorMctsText)
      backend_selected = (Test-BackendSelected \$AcceleratorMctsText)
      stdout_log = "cuda-accelerator-mcts.stdout.log"
      stderr_log = "cuda-accelerator-mcts.stderr.log"
    }
    cuda_bk07_mcts = [ordered]@{
      go = "nodes 50"
      fen = \$Bk07Fen
      expected_bestmove = "h5f6"
      bestmove = (Find-BestMove \$Bk07MctsText)
      backend_selected = (Test-BackendSelected \$Bk07MctsText)
      stdout_log = "cuda-bk07-mcts.stdout.log"
      stderr_log = "cuda-bk07-mcts.stderr.log"
    }
    cuda_kiwipete_mcts = [ordered]@{
      go = "nodes 1"
      fen = \$KiwipeteFen
      bestmove = (Find-BestMove \$KiwipeteMctsText)
      backend_selected = (Test-BackendSelected \$KiwipeteMctsText)
      stdout_log = "cuda-kiwipete-mcts.stdout.log"
      stderr_log = "cuda-kiwipete-mcts.stderr.log"
    }
    hybrid_cuda = [ordered]@{
      go = "${HYBRID_UCI_GO}"
      bestmove = (Find-BestMove \$HybridText)
      backend_selected = (Test-BackendSelected \$HybridText)
      metrics = (Find-FinalMetrics \$HybridText)
      stdout_log = "hybrid-cuda.stdout.log"
      stderr_log = "hybrid-cuda.stderr.log"
    }
    hybrid_cuda_clock_start = [ordered]@{
      go = "wtime 1000 btime 1000 winc 3000 binc 3000"
      bestmove = (Find-BestMove \$HybridClockStartText)
      backend_selected = (Test-BackendSelected \$HybridClockStartText)
      time_safety = (\$HybridClockStartText -like "*Time safety:*")
      stdout_log = "hybrid-cuda-clock-start.stdout.log"
      stderr_log = "hybrid-cuda-clock-start.stderr.log"
    }
    hybrid_cuda_clock_safety = [ordered]@{
      go = "wtime 800 btime 800 winc 3000 binc 3000"
      bestmove = (Find-BestMove \$HybridClockSafetyText)
      backend_selected = (Test-BackendSelected \$HybridClockSafetyText)
      time_safety = (\$HybridClockSafetyText -like "*Time safety: estimated move budget*")
      stdout_log = "hybrid-cuda-clock-safety.stdout.log"
      stderr_log = "hybrid-cuda-clock-safety.stderr.log"
    }
    hybrid_auto = [ordered]@{
      go = "${HYBRID_UCI_GO}"
      bestmove = (Find-BestMove \$HybridAutoText)
      backend_selected = (Test-BackendSelected \$HybridAutoText)
      metrics = (Find-FinalMetrics \$HybridAutoText)
      stdout_log = "hybrid-auto.stdout.log"
      stderr_log = "hybrid-auto.stderr.log"
    }
    hybrid_cuda_ane_disabled = [ordered]@{
      go = "${HYBRID_UCI_GO}"
      bestmove = (Find-BestMove \$HybridAneText)
      backend_selected = (Test-BackendSelected \$HybridAneText)
      ane_disabled = (\$HybridAneText -like "*ANE root probe disabled*")
      metrics = (Find-FinalMetrics \$HybridAneText)
      stdout_log = "hybrid-cuda-ane-disabled.stdout.log"
      stderr_log = "hybrid-cuda-ane-disabled.stderr.log"
    }
  }
}
\$Manifest | ConvertTo-Json -Depth 12 | Set-Content -Path (Join-Path \$Logs "windows-cuda-runtime-manifest.json") -Encoding UTF8

@(
  "# MetalFish Windows CUDA Runtime Gate",
  "",
  "- Gate status: passed",
  "- Package: ${PACKAGE_BASENAME}",
  "- GPU: see nvidia-smi-runtime.log",
  "- Smokes: cuda-probe, cuda-legacy-probe, cuda-probe-suite, cuda-legacy-probe-suite, cuda-isolation-bt4-legacy, cuda-isolation-legacy-bt4, cuda-nn-comparison, cuda-mcts, cuda-auto-mcts, cuda-accelerator-mcts, cuda-bk07-mcts, cuda-kiwipete-mcts, hybrid-cuda, hybrid-cuda-clock-start, hybrid-cuda-clock-safety, hybrid-auto, hybrid-cuda-ane-disabled",
  "- Manifest: windows-cuda-runtime-manifest.json"
) | Set-Content -Path (Join-Path \$Logs "windows-cuda-runtime-summary.md") -Encoding UTF8
Write-Host "Windows CUDA runtime gate passed"
POWERSHELL

set +e
run_remote_ps "${RUN_DIR}/run-smokes.ps1"
RUNTIME_STATUS=$?
set -e
collect_remote_artifacts

COMPARE_STATUS=0
BT4_COMPARE_STATUS="skipped"
LEGACY_COMPARE_STATUS="skipped"
BENCHMARK_COMPARE_STATUS="skipped"
SEARCH_COMPARE_STATUS="skipped"
if [[ "${RUNTIME_STATUS}" == "0" ]]; then
  BT4_COMPARE_STATUS=0
  compare_collected_probe_suite || BT4_COMPARE_STATUS=$?
  if [[ "${BT4_COMPARE_STATUS}" == "0" ]]; then
    LEGACY_COMPARE_STATUS=0
    compare_collected_legacy_probe_suite || LEGACY_COMPARE_STATUS=$?
  fi
  if [[ "${BT4_COMPARE_STATUS}" == "0" && "${LEGACY_COMPARE_STATUS}" == "0" ]]; then
    BENCHMARK_COMPARE_STATUS=0
    compare_collected_benchmark_timings || BENCHMARK_COMPARE_STATUS=$?
  fi
  if [[ "${BT4_COMPARE_STATUS}" == "0" && "${LEGACY_COMPARE_STATUS}" == "0" && "${BENCHMARK_COMPARE_STATUS}" == "0" ]]; then
    SEARCH_COMPARE_STATUS=0
    compare_collected_search_results || SEARCH_COMPARE_STATUS=$?
  fi
fi

if [[ "${BT4_COMPARE_STATUS}" != "0" && "${BT4_COMPARE_STATUS}" != "skipped" ]]; then
  COMPARE_STATUS="${BT4_COMPARE_STATUS}"
elif [[ "${LEGACY_COMPARE_STATUS}" != "0" && "${LEGACY_COMPARE_STATUS}" != "skipped" ]]; then
  COMPARE_STATUS="${LEGACY_COMPARE_STATUS}"
elif [[ "${BENCHMARK_COMPARE_STATUS}" != "0" && "${BENCHMARK_COMPARE_STATUS}" != "skipped" ]]; then
  COMPARE_STATUS="${BENCHMARK_COMPARE_STATUS}"
elif [[ "${SEARCH_COMPARE_STATUS}" != "0" && "${SEARCH_COMPARE_STATUS}" != "skipped" ]]; then
  COMPARE_STATUS="${SEARCH_COMPARE_STATUS}"
fi

write_runtime_manifest \
  "${RUNTIME_STATUS}" \
  "${BT4_COMPARE_STATUS}" \
  "${LEGACY_COMPARE_STATUS}" \
  "${BENCHMARK_COMPARE_STATUS}" \
  "${SEARCH_COMPARE_STATUS}" \
  "${COMPARE_STATUS}"
upload_collected_artifacts

if [[ "${RUNTIME_STATUS}" != "0" ]]; then
  exit "${RUNTIME_STATUS}"
fi
if [[ "${COMPARE_STATUS}" != "0" ]]; then
  exit "${COMPARE_STATUS}"
fi

echo "Windows CUDA runtime gate passed"
echo "Artifacts: ${ARTIFACT_DIR}"
