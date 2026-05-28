#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${METALFISH_GCP_PROJECT:-metalfish}"
DEFAULT_ZONES="us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c northamerica-northeast1-b northamerica-northeast1-c"
ZONES="${METALFISH_GCP_ZONES:-${METALFISH_GCP_ZONE:-${DEFAULT_ZONES}}}"
INSTANCE="${METALFISH_GCP_INSTANCE:-metalfish-cuda-gate-$(date +%Y%m%d-%H%M%S)}"
MACHINE="${METALFISH_GCP_MACHINE:-g2-standard-8}"
ACCELERATOR="${METALFISH_GCP_ACCELERATOR:-type=nvidia-l4,count=1}"
IMAGE_PROJECT="${METALFISH_GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${METALFISH_GCP_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
BOOT_DISK_SIZE="${METALFISH_GCP_BOOT_DISK_SIZE:-100GB}"
DELETE_ON_EXIT="${METALFISH_GCP_DELETE_ON_EXIT:-1}"
COLLECT_ARTIFACTS="${METALFISH_GCP_COLLECT_ARTIFACTS:-1}"
ARTIFACT_DIR="${METALFISH_GCP_ARTIFACT_DIR:-${ROOT_DIR}/results/cuda_gpu_gate/${INSTANCE}}"
GCS_PREFIX="${METALFISH_GCP_GCS_PREFIX:-}"
METAL_PROBE_SUITE_LOG="${METALFISH_METAL_PROBE_SUITE_LOG:-}"
METAL_LEGACY_PROBE_SUITE_LOG="${METALFISH_METAL_LEGACY_PROBE_SUITE_LOG:-}"
METAL_COMPARISON_LOG="${METALFISH_METAL_COMPARISON_LOG:-}"
METAL_MCTS_BK07_SEARCH_JSON="${METALFISH_METAL_MCTS_BK07_SEARCH_JSON:-}"
METAL_MCTS_KIWIPETE_SEARCH_JSON="${METALFISH_METAL_MCTS_KIWIPETE_SEARCH_JSON:-}"
METAL_HYBRID_BK07_SEARCH_JSON="${METALFISH_METAL_HYBRID_BK07_SEARCH_JSON:-}"
METAL_HYBRID_KIWIPETE_SEARCH_JSON="${METALFISH_METAL_HYBRID_KIWIPETE_SEARCH_JSON:-}"
REQUIRE_METAL_COMPARE="${METALFISH_REQUIRE_METAL_COMPARE:-0}"
REQUIRE_METAL_BENCHMARK_COMPARE="${METALFISH_REQUIRE_METAL_BENCHMARK_COMPARE:-0}"
REQUIRE_METAL_SEARCH_COMPARE="${METALFISH_REQUIRE_METAL_SEARCH_COMPARE:-0}"
MAX_CUDA_METAL_EVAL_MS_RATIO="${METALFISH_MAX_CUDA_METAL_EVAL_MS_RATIO:-1.0}"
SEARCH_COMPARE_SKIPPED=77
ARCHIVE="$(mktemp -t metalfish-cuda-gate.XXXXXX.tar.gz)"
CREATED_INSTANCE=0
ZONE=""

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
  rm -f "${ARCHIVE}"
  if [[ "${DELETE_ON_EXIT}" == "1" && "${CREATED_INSTANCE}" == "1" ]]; then
    gcloud compute instances delete "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --quiet >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"

git diff-index --quiet HEAD -- || {
  echo "working tree has uncommitted changes; commit before running the CUDA GPU gate" >&2
  exit 2
}

SOURCE_COMMIT="${METALFISH_SOURCE_COMMIT:-$(git rev-parse HEAD)}"
SOURCE_BRANCH="${METALFISH_SOURCE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
git archive --format=tar.gz --output="${ARCHIVE}" HEAD

for candidate_zone in ${ZONES}; do
  echo "Creating ${INSTANCE} in ${candidate_zone}"
  if gcloud compute instances create "${INSTANCE}" \
    --project "${PROJECT}" \
    --zone "${candidate_zone}" \
    --machine-type "${MACHINE}" \
    --accelerator "${ACCELERATOR}" \
    --maintenance-policy TERMINATE \
    --restart-on-failure \
    --image-family "${IMAGE_FAMILY}" \
    --image-project "${IMAGE_PROJECT}" \
    --boot-disk-size "${BOOT_DISK_SIZE}" \
    --boot-disk-type pd-balanced \
    --metadata install-nvidia-driver=True \
    --scopes https://www.googleapis.com/auth/cloud-platform; then
    ZONE="${candidate_zone}"
    CREATED_INSTANCE=1
    break
  fi
  echo "Zone ${candidate_zone} could not allocate ${MACHINE} with ${ACCELERATOR}" >&2
done

if [[ "${CREATED_INSTANCE}" != "1" ]]; then
  echo "failed to create a CUDA GPU gate VM in zones: ${ZONES}" >&2
  exit 1
fi

for attempt in $(seq 1 60); do
  if gcloud compute ssh "${INSTANCE}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --command "echo ready" >/dev/null 2>&1; then
    break
  fi
  if [[ "${attempt}" == "60" ]]; then
    echo "timed out waiting for SSH on ${INSTANCE}" >&2
    exit 1
  fi
  sleep 10
done

gcloud compute scp "${ARCHIVE}" "${INSTANCE}:~/metalfish.tar.gz" \
  --project "${PROJECT}" \
  --zone "${ZONE}"

REMOTE_ENV="METALFISH_INSTALL_DEPS=1"
REMOTE_ENV+=" METALFISH_SOURCE_COMMIT=$(printf '%q' "${SOURCE_COMMIT}")"
REMOTE_ENV+=" METALFISH_SOURCE_BRANCH=$(printf '%q' "${SOURCE_BRANCH}")"
append_remote_env() {
  local key="$1"
  local value="${!key:-}"
  if [[ -n "${value}" ]]; then
    REMOTE_ENV+=" ${key}=$(printf '%q' "${value}")"
  fi
}
append_remote_env METALFISH_JOBS
append_remote_env METALFISH_CUDA_ARCHS
append_remote_env METALFISH_CUDA_UCI_GO
append_remote_env METALFISH_CUDA_UCI_TIMEOUT
append_remote_env METALFISH_CUDA_DOWNLOAD_BT4
append_remote_env METALFISH_CUDA_DOWNLOAD_LEGACY
append_remote_env METALFISH_CUDA_LEGACY_PROBE
append_remote_env METALFISH_BT4_WEIGHTS_URL
append_remote_env METALFISH_LEGACY_WEIGHTS_URL
append_remote_env METALFISH_LEGACY_NN_WEIGHTS
append_remote_env METALFISH_NNUE_BIG_URL
append_remote_env METALFISH_NNUE_SMALL_URL
append_remote_env METALFISH_NN_BATCH_BENCH
append_remote_env METALFISH_NN_BENCH_ITERS
append_remote_env METALFISH_NN_BENCH_MAX_BATCH
append_remote_env METALFISH_NN_BENCH_WARMUP_ITERS
append_remote_env METALFISH_NN_BENCH_GRAPH_REUSE_PROBE
append_remote_env METALFISH_NN_DEBUG_DUMP
append_remote_env METALFISH_NN_BATCH_TRACE_PAIR
append_remote_env METALFISH_NN_BATCH_TRACE_WORST
append_remote_env METALFISH_NN_COMPARISON_ONLY
append_remote_env METALFISH_NN_SINGLE_REPEAT_STRESS
append_remote_env METALFISH_NN_SINGLE_REPEAT_STRESS_ITERS
append_remote_env METALFISH_NN_SINGLE_REPEAT_STRESS_PROBES
append_remote_env METALFISH_NN_SINGLE_REUSE_STRESS
append_remote_env METALFISH_NN_SINGLE_REUSE_STRESS_ITERS
append_remote_env METALFISH_NN_BATCH_REUSE_STRESS
append_remote_env METALFISH_NN_BATCH_REUSE_STRESS_ITERS
append_remote_env METALFISH_NN_BATCH_TRACE_BATCH
append_remote_env METALFISH_NN_BATCH_TRACE_INDEX
append_remote_env METALFISH_NN_FIRST_USE_STRESS_ITERS
append_remote_env METALFISH_CUDA_FULL_BUFFER_CLEAR
append_remote_env METALFISH_CUDA_GRAPH
append_remote_env METALFISH_CUDA_GRAPH_EXECUTION
append_remote_env METALFISH_CUDA_GRAPH_STATUS_DETAIL
append_remote_env METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN
append_remote_env METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN
append_remote_env METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE
append_remote_env METALFISH_CUDA_DETERMINISTIC_ATTENTION_SOFTMAX
append_remote_env METALFISH_CUDA_TRACE_RAW_OUTPUTS
append_remote_env METALFISH_CUDA_TRACE_RAW_LIMIT
append_remote_env METALFISH_CUDA_TRACE_RAW_ENTRY
append_remote_env METALFISH_CUDA_TRACE_STAGE_OUTPUTS
append_remote_env METALFISH_CUDA_TRACE_STAGE_BATCH
append_remote_env METALFISH_CUDA_TRACE_STAGE_SKIP
append_remote_env METALFISH_CUDA_TRACE_STAGE_LIMIT
append_remote_env METALFISH_CUDA_TRACE_STAGE_FILTER
append_remote_env METALFISH_CUDA_TRACE_STAGE_MAX_FLOATS
append_remote_env METALFISH_CUDA_TRACE_ATTENTION_INTERNALS
append_remote_env METALFISH_CUDA_TRACE_DYNAMIC_PE_INTERNALS
append_remote_env METALFISH_CUDA_TRACE_COMPARE_BASE_RUN
append_remote_env METALFISH_CUDA_TRACE_COMPARE_MIN_DELTA
append_remote_env METALFISH_CUDA_PROFILE
append_remote_env METALFISH_CUDA_PROFILE_LIMIT
append_remote_env CUBLAS_WORKSPACE_CONFIG

collect_remote_artifacts() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    return 0
  fi

  mkdir -p "${ARTIFACT_DIR}"
  local copied=0
  local file
  for file in \
    cuda-gpu-summary.md \
    cuda-gpu-tests.log \
    cuda-gpu-nn-comparison.log \
    cuda-gpu-nn-probe.log \
    cuda-gpu-nn-probe-suite.log \
    cuda-gpu-legacy-nn-probe-suite.log \
    cuda-gpu-nn-isolation-bt4-legacy.log \
    cuda-gpu-nn-isolation-legacy-bt4.log \
    cuda-gpu-nn-artifact-manifest.json \
    cuda-gpu-parity-report.md \
    cuda-gpu-uci-auto-smoke.log \
    cuda-gpu-uci-accelerator-smoke.log \
    cuda-gpu-uci-smoke.log \
    cuda-gpu-uci-bk07-smoke.log \
    cuda-gpu-uci-bk07-search.json \
    cuda-gpu-uci-kiwipete-smoke.log \
    cuda-gpu-uci-kiwipete-search.json \
    cuda-gpu-uci-hybrid-smoke.log \
    cuda-gpu-uci-hybrid-search.json \
    cuda-gpu-uci-hybrid-kiwipete-smoke.log \
    cuda-gpu-uci-hybrid-kiwipete-search.json \
    cuda-gpu-uci-hybrid-clock-start-smoke.log \
    cuda-gpu-uci-hybrid-clock-safety-smoke.log \
    cuda-gpu-uci-hybrid-auto-smoke.log \
    cuda-gpu-uci-hybrid-ane-smoke.log \
    cuda-gpu-package-nn-comparison.log \
    cuda-gpu-package-parity-report.md \
    cuda-gpu-package-probe.log \
    cuda-gpu-package-nn-probe-suite.log \
    cuda-gpu-package-legacy-nn-probe-suite.log \
    cuda-gpu-package-nn-isolation-bt4-legacy.log \
    cuda-gpu-package-nn-isolation-legacy-bt4.log \
    cuda-gpu-package-uci-smoke.log \
    cuda-gpu-profile.log; do
    if gcloud compute scp \
      "${INSTANCE}:~/metalfish/build-cuda-gpu/${file}" \
      "${ARTIFACT_DIR}/${file}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" >/dev/null 2>&1; then
      copied=$((copied + 1))
    fi
  done
  for file in \
    metalfish-linux-x86_64-cuda.tar.gz \
    PORTABLE_ARTIFACT.md \
    linux-cuda-package-manifest.json; do
    if gcloud compute scp \
      "${INSTANCE}:~/metalfish/build-cuda-gpu/${file}" \
      "${ARTIFACT_DIR}/${file}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" >/dev/null 2>&1; then
      copied=$((copied + 1))
    fi
  done

  if ((copied > 0)); then
    echo "Collected ${copied} CUDA gate artifact(s) in ${ARTIFACT_DIR}"
  else
    echo "No CUDA gate artifacts were available to collect" >&2
  fi
}

write_runtime_manifest() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    return 0
  fi

  mkdir -p "${ARTIFACT_DIR}"
  REMOTE_STATUS_FOR_MANIFEST="$1" \
    BT4_COMPARE_STATUS_FOR_MANIFEST="$2" \
    LEGACY_COMPARE_STATUS_FOR_MANIFEST="$3" \
    BENCHMARK_COMPARE_STATUS_FOR_MANIFEST="$4" \
    SEARCH_COMPARE_STATUS_FOR_MANIFEST="$5" \
    FINAL_COMPARE_STATUS_FOR_MANIFEST="$6" \
    GIT_HEAD_SHA="$(git rev-parse HEAD)" \
    GATE_ARCHIVE="${ARCHIVE}" \
    GATE_ARTIFACT_DIR="${ARTIFACT_DIR}" \
    GATE_PROJECT="${PROJECT}" \
    GATE_INSTANCE="${INSTANCE}" \
    GATE_ZONE="${ZONE}" \
    GATE_MACHINE="${MACHINE}" \
    GATE_ACCELERATOR="${ACCELERATOR}" \
    GATE_IMAGE_PROJECT="${IMAGE_PROJECT}" \
    GATE_IMAGE_FAMILY="${IMAGE_FAMILY}" \
    GATE_BOOT_DISK_SIZE="${BOOT_DISK_SIZE}" \
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
    GATE_METAL_HYBRID_BK07_SEARCH_JSON="${METAL_HYBRID_BK07_SEARCH_JSON}" \
    GATE_METAL_HYBRID_KIWIPETE_SEARCH_JSON="${METAL_HYBRID_KIWIPETE_SEARCH_JSON}" \
    python3 - "${ARTIFACT_DIR}/cuda-gpu-runtime-manifest.json" <<'PY'
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys


def file_record(path):
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
    for candidate in sorted(artifact_dir.iterdir()):
        if candidate.is_file() and candidate.name != manifest_path.name:
            record = file_record(str(candidate))
            if record is not None:
                artifacts[candidate.name] = record

manifest = {
    "schema_version": 1,
    "schema": "metalfish.cuda_gpu_runtime_gate",
    "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    "git": {
        "head_sha": os.environ["GIT_HEAD_SHA"],
        "archive": file_record(os.environ["GATE_ARCHIVE"]),
    },
    "gcp": {
        "project": os.environ["GATE_PROJECT"],
        "instance": os.environ["GATE_INSTANCE"],
        "zone": os.environ["GATE_ZONE"],
        "machine": os.environ["GATE_MACHINE"],
        "accelerator": os.environ["GATE_ACCELERATOR"],
        "image_project": os.environ["GATE_IMAGE_PROJECT"],
        "image_family": os.environ["GATE_IMAGE_FAMILY"],
        "boot_disk_size": os.environ["GATE_BOOT_DISK_SIZE"],
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
        "metal_hybrid_bk07_search_json": file_record(
            os.environ["GATE_METAL_HYBRID_BK07_SEARCH_JSON"]
        ),
        "metal_hybrid_kiwipete_search_json": file_record(
            os.environ["GATE_METAL_HYBRID_KIWIPETE_SEARCH_JSON"]
        ),
    },
    "status": {
        "remote_status": os.environ["REMOTE_STATUS_FOR_MANIFEST"],
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
print(f"Wrote CUDA runtime manifest: {manifest_path}")
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

  gcloud storage cp "${files[@]}" "${GCS_PREFIX%/}/${INSTANCE}/"
  echo "Uploaded CUDA gate artifacts to ${GCS_PREFIX%/}/${INSTANCE}/"
}

compare_collected_probe_suite() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if require_metal_compare; then
      echo "Metal/CUDA comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
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

  local cuda_suite="${ARTIFACT_DIR}/cuda-gpu-package-nn-probe-suite.log"
  if [[ ! -s "${cuda_suite}" ]]; then
    cuda_suite="${ARTIFACT_DIR}/cuda-gpu-nn-probe-suite.log"
  fi
  if [[ ! -s "${cuda_suite}" ]]; then
    echo "CUDA probe suite log not found: ${cuda_suite}" >&2
    return 1
  fi

  python3 tools/compare_nn_backend_outputs.py \
    --expected-log "${METAL_PROBE_SUITE_LOG}" \
    --actual-log "${cuda_suite}" \
    --expected-label "Metal (MPSGraph) backend" \
    --actual-label "CUDA transformer backend" \
    --summary-out "${ARTIFACT_DIR}/metal-cuda-nn-probe-suite-summary.json" \
    --require-full-policy \
    --all-probes \
    | tee "${ARTIFACT_DIR}/metal-cuda-nn-probe-suite-compare.log"
}

compare_collected_legacy_probe_suite() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if require_metal_compare; then
      echo "Legacy Metal/CUDA comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
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

  local cuda_suite="${ARTIFACT_DIR}/cuda-gpu-package-legacy-nn-probe-suite.log"
  if [[ ! -s "${cuda_suite}" ]]; then
    cuda_suite="${ARTIFACT_DIR}/cuda-gpu-legacy-nn-probe-suite.log"
  fi
  if [[ ! -s "${cuda_suite}" ]]; then
    echo "CUDA legacy probe suite log not found: ${cuda_suite}" >&2
    return 1
  fi

  python3 tools/compare_nn_backend_outputs.py \
    --expected-log "${METAL_LEGACY_PROBE_SUITE_LOG}" \
    --actual-log "${cuda_suite}" \
    --expected-label "Metal (MPSGraph) backend" \
    --actual-label "CUDA transformer backend" \
    --summary-out "${ARTIFACT_DIR}/metal-cuda-legacy-nn-probe-suite-summary.json" \
    --require-full-policy \
    --no-require-wdl \
    --no-require-moves-left \
    --all-probes \
    | tee "${ARTIFACT_DIR}/metal-cuda-legacy-nn-probe-suite-compare.log"
}

compare_collected_benchmark_timings() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if [[ "${REQUIRE_METAL_BENCHMARK_COMPARE}" == "1" ]]; then
      echo "Metal/CUDA benchmark comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
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

  local cuda_comparison="${ARTIFACT_DIR}/cuda-gpu-package-nn-comparison.log"
  if [[ ! -s "${cuda_comparison}" ]]; then
    cuda_comparison="${ARTIFACT_DIR}/cuda-gpu-nn-comparison.log"
  fi
  if [[ ! -s "${cuda_comparison}" ]]; then
    echo "CUDA comparison log not found: ${cuda_comparison}" >&2
    return 1
  fi

  python3 tools/compare_nn_backend_benchmarks.py \
    --expected-log "${METAL_COMPARISON_LOG}" \
    --actual-log "${cuda_comparison}" \
    --expected-label "Metal (MPSGraph) backend" \
    --actual-label "CUDA transformer backend" \
    --summary-out "${ARTIFACT_DIR}/metal-cuda-nn-benchmark-summary.json" \
    --require-actual-graph-reuse \
    --max-eval-ms-ratio "${MAX_CUDA_METAL_EVAL_MS_RATIO}" \
    | tee "${ARTIFACT_DIR}/metal-cuda-nn-benchmark-compare.log"
}

compare_collected_search_results() {
  if [[ "${COLLECT_ARTIFACTS}" != "1" ]]; then
    if require_metal_search_compare; then
      echo "Metal/CUDA search comparison requires METALFISH_GCP_COLLECT_ARTIFACTS=1" >&2
      return 1
    fi
    return "${SEARCH_COMPARE_SKIPPED}"
  fi

  if [[ -z "${METAL_MCTS_BK07_SEARCH_JSON}" || -z "${METAL_MCTS_KIWIPETE_SEARCH_JSON}" || -z "${METAL_HYBRID_BK07_SEARCH_JSON}" || -z "${METAL_HYBRID_KIWIPETE_SEARCH_JSON}" ]]; then
    if require_metal_search_compare; then
      echo "Metal search JSON inputs are required for search comparison" >&2
      return 1
    fi
    return "${SEARCH_COMPARE_SKIPPED}"
  fi
  if [[ ! -s "${METAL_MCTS_BK07_SEARCH_JSON}" ]]; then
    echo "Metal MCTS search JSON not found: ${METAL_MCTS_BK07_SEARCH_JSON}" >&2
    return 1
  fi
  if [[ ! -s "${METAL_MCTS_KIWIPETE_SEARCH_JSON}" ]]; then
    echo "Metal MCTS kiwipete search JSON not found: ${METAL_MCTS_KIWIPETE_SEARCH_JSON}" >&2
    return 1
  fi
  if [[ ! -s "${METAL_HYBRID_BK07_SEARCH_JSON}" ]]; then
    echo "Metal Hybrid search JSON not found: ${METAL_HYBRID_BK07_SEARCH_JSON}" >&2
    return 1
  fi
  if [[ ! -s "${METAL_HYBRID_KIWIPETE_SEARCH_JSON}" ]]; then
    echo "Metal Hybrid kiwipete search JSON not found: ${METAL_HYBRID_KIWIPETE_SEARCH_JSON}" >&2
    return 1
  fi

  local cuda_mcts="${ARTIFACT_DIR}/cuda-gpu-uci-bk07-search.json"
  local cuda_mcts_kiwipete="${ARTIFACT_DIR}/cuda-gpu-uci-kiwipete-search.json"
  local cuda_hybrid="${ARTIFACT_DIR}/cuda-gpu-uci-hybrid-search.json"
  local cuda_hybrid_kiwipete="${ARTIFACT_DIR}/cuda-gpu-uci-hybrid-kiwipete-search.json"
  if [[ ! -s "${cuda_mcts}" ]]; then
    echo "CUDA MCTS search JSON not found: ${cuda_mcts}" >&2
    return 1
  fi
  if [[ ! -s "${cuda_mcts_kiwipete}" ]]; then
    echo "CUDA MCTS kiwipete search JSON not found: ${cuda_mcts_kiwipete}" >&2
    return 1
  fi
  if [[ ! -s "${cuda_hybrid}" ]]; then
    echo "CUDA Hybrid search JSON not found: ${cuda_hybrid}" >&2
    return 1
  fi
  if [[ ! -s "${cuda_hybrid_kiwipete}" ]]; then
    echo "CUDA Hybrid kiwipete search JSON not found: ${cuda_hybrid_kiwipete}" >&2
    return 1
  fi

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_MCTS_BK07_SEARCH_JSON}" \
    --actual "${cuda_mcts}" \
    --expected-label "Metal MCTS" \
    --actual-label "CUDA MCTS" \
    --require-same-pv-head \
    --max-score-cp-delta 10 \
    --require-same-setoption UseMCTS \
    --require-same-setoption UseHybridSearch \
    --require-same-setoption Threads \
    --require-same-setoption MCTSMaxThreads \
    --require-same-setoption MCTSParallelSearch \
    --require-same-setoption MCTSMinibatchSize \
    --require-same-setoption MCTSParityPreset \
    --require-same-setoption MCTSAddDirichletNoise \
    --require-same-setoption TransformerLowTimeFallbackMs \
    --json-out "${ARTIFACT_DIR}/metal-cuda-mcts-bk07-search-summary.json" \
    | tee "${ARTIFACT_DIR}/metal-cuda-mcts-bk07-search-compare.log"

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_MCTS_KIWIPETE_SEARCH_JSON}" \
    --actual "${cuda_mcts_kiwipete}" \
    --expected-label "Metal MCTS" \
    --actual-label "CUDA MCTS" \
    --require-same-pv-head \
    --max-score-cp-delta 10 \
    --require-same-setoption UseMCTS \
    --require-same-setoption UseHybridSearch \
    --require-same-setoption Threads \
    --require-same-setoption MCTSMaxThreads \
    --require-same-setoption MCTSParallelSearch \
    --require-same-setoption MCTSMinibatchSize \
    --require-same-setoption MCTSParityPreset \
    --require-same-setoption MCTSAddDirichletNoise \
    --require-same-setoption TransformerLowTimeFallbackMs \
    --json-out "${ARTIFACT_DIR}/metal-cuda-mcts-kiwipete-search-summary.json" \
    | tee "${ARTIFACT_DIR}/metal-cuda-mcts-kiwipete-search-compare.log"

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_HYBRID_BK07_SEARCH_JSON}" \
    --actual "${cuda_hybrid}" \
    --expected-label "Metal Hybrid" \
    --actual-label "CUDA Hybrid" \
    --require-same-pv-head \
    --max-score-cp-delta 10 \
    --require-positive-final-metric MCTSPlayouts \
    --require-positive-final-metric MCTSEvals \
    --require-positive-final-metric ABDepth \
    --require-final-metric ABMove \
    --require-final-metric MCTSMove \
    --require-same-final-metric ABMove \
    --require-same-final-metric MCTSMove \
    --require-same-setoption UseMCTS \
    --require-same-setoption UseHybridSearch \
    --require-same-setoption Threads \
    --require-same-setoption HybridMCTSThreads \
    --require-same-setoption HybridABThreads \
    --require-same-setoption HybridAutoABThreadsCap \
    --require-same-setoption MCTSMaxThreads \
    --require-same-setoption MCTSMinibatchSize \
    --require-same-setoption MCTSParityPreset \
    --require-same-setoption MCTSAddDirichletNoise \
    --require-same-setoption TransformerLowTimeFallbackMs \
    --json-out "${ARTIFACT_DIR}/metal-cuda-hybrid-bk07-search-summary.json" \
    | tee "${ARTIFACT_DIR}/metal-cuda-hybrid-bk07-search-compare.log"

  python3 tools/compare_uci_search_results.py \
    --expected "${METAL_HYBRID_KIWIPETE_SEARCH_JSON}" \
    --actual "${cuda_hybrid_kiwipete}" \
    --expected-label "Metal Hybrid" \
    --actual-label "CUDA Hybrid" \
    --require-same-pv-head \
    --max-score-cp-delta 10 \
    --require-positive-final-metric MCTSPlayouts \
    --require-positive-final-metric MCTSEvals \
    --require-positive-final-metric ABDepth \
    --require-final-metric ABMove \
    --require-final-metric MCTSMove \
    --require-same-final-metric ABMove \
    --require-same-final-metric MCTSMove \
    --require-same-setoption UseMCTS \
    --require-same-setoption UseHybridSearch \
    --require-same-setoption Threads \
    --require-same-setoption HybridMCTSThreads \
    --require-same-setoption HybridABThreads \
    --require-same-setoption HybridAutoABThreadsCap \
    --require-same-setoption MCTSMaxThreads \
    --require-same-setoption MCTSMinibatchSize \
    --require-same-setoption MCTSParityPreset \
    --require-same-setoption MCTSAddDirichletNoise \
    --require-same-setoption TransformerLowTimeFallbackMs \
    --json-out "${ARTIFACT_DIR}/metal-cuda-hybrid-kiwipete-search-summary.json" \
    | tee "${ARTIFACT_DIR}/metal-cuda-hybrid-kiwipete-search-compare.log"
}

set +e
gcloud compute ssh "${INSTANCE}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "rm -rf ~/metalfish && mkdir -p ~/metalfish && tar -xzf ~/metalfish.tar.gz -C ~/metalfish && cd ~/metalfish && chmod +x tools/run_cuda_gpu_gate.sh && ${REMOTE_ENV} tools/run_cuda_gpu_gate.sh"
REMOTE_STATUS=$?
set -e

collect_remote_artifacts
COMPARE_STATUS=0
BT4_COMPARE_STATUS="skipped"
LEGACY_COMPARE_STATUS="skipped"
BENCHMARK_COMPARE_STATUS="skipped"
SEARCH_COMPARE_STATUS="skipped"
if [[ "${REMOTE_STATUS}" == "0" ]]; then
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
    if compare_collected_search_results; then
      SEARCH_COMPARE_STATUS=0
    else
      SEARCH_COMPARE_STATUS=$?
      if [[ "${SEARCH_COMPARE_STATUS}" == "${SEARCH_COMPARE_SKIPPED}" ]]; then
        SEARCH_COMPARE_STATUS="skipped"
      fi
    fi
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
  "${REMOTE_STATUS}" \
  "${BT4_COMPARE_STATUS}" \
  "${LEGACY_COMPARE_STATUS}" \
  "${BENCHMARK_COMPARE_STATUS}" \
  "${SEARCH_COMPARE_STATUS}" \
  "${COMPARE_STATUS}"
upload_collected_artifacts

if [[ "${REMOTE_STATUS}" != "0" ]]; then
  exit "${REMOTE_STATUS}"
fi
exit "${COMPARE_STATUS}"
