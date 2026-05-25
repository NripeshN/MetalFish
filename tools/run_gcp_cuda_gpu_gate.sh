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
REQUIRE_METAL_COMPARE="${METALFISH_REQUIRE_METAL_COMPARE:-0}"
ARCHIVE="$(mktemp -t metalfish-cuda-gate.XXXXXX.tar.gz)"
CREATED_INSTANCE=0
ZONE=""

require_metal_compare() {
  case "${REQUIRE_METAL_COMPARE}" in
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
    cuda-gpu-nn-artifact-manifest.json \
    cuda-gpu-parity-report.md \
    cuda-gpu-uci-auto-smoke.log \
    cuda-gpu-uci-smoke.log \
    cuda-gpu-uci-hybrid-smoke.log \
    cuda-gpu-uci-hybrid-ane-smoke.log \
    cuda-gpu-profile.log; do
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
    if [[ -n "${GCS_PREFIX}" ]]; then
      gcloud storage cp "${ARTIFACT_DIR}"/* \
        "${GCS_PREFIX%/}/${INSTANCE}/"
      echo "Uploaded CUDA gate artifacts to ${GCS_PREFIX%/}/${INSTANCE}/"
    fi
  else
    echo "No CUDA gate artifacts were available to collect" >&2
  fi
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

  local cuda_suite="${ARTIFACT_DIR}/cuda-gpu-nn-probe-suite.log"
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

  local cuda_suite="${ARTIFACT_DIR}/cuda-gpu-legacy-nn-probe-suite.log"
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

set +e
gcloud compute ssh "${INSTANCE}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "rm -rf ~/metalfish && mkdir -p ~/metalfish && tar -xzf ~/metalfish.tar.gz -C ~/metalfish && cd ~/metalfish && chmod +x tools/run_cuda_gpu_gate.sh && ${REMOTE_ENV} tools/run_cuda_gpu_gate.sh"
REMOTE_STATUS=$?
set -e

collect_remote_artifacts
COMPARE_STATUS=0
if [[ "${REMOTE_STATUS}" == "0" ]]; then
  compare_collected_probe_suite || COMPARE_STATUS=$?
  if [[ "${COMPARE_STATUS}" == "0" ]]; then
    compare_collected_legacy_probe_suite || COMPARE_STATUS=$?
  fi
fi

if [[ "${REMOTE_STATUS}" != "0" ]]; then
  exit "${REMOTE_STATUS}"
fi
exit "${COMPARE_STATUS}"
