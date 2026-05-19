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
ARCHIVE="$(mktemp -t metalfish-cuda-gate.XXXXXX.tar.gz)"
CREATED_INSTANCE=0
ZONE=""

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
append_remote_env METALFISH_BT4_WEIGHTS_URL
append_remote_env METALFISH_NNUE_BIG_URL
append_remote_env METALFISH_NNUE_SMALL_URL

gcloud compute ssh "${INSTANCE}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "rm -rf ~/metalfish && mkdir -p ~/metalfish && tar -xzf ~/metalfish.tar.gz -C ~/metalfish && cd ~/metalfish && chmod +x tools/run_cuda_gpu_gate.sh && ${REMOTE_ENV} tools/run_cuda_gpu_gate.sh"
