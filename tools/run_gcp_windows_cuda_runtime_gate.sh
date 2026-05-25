#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${METALFISH_GCP_PROJECT:-metalfish}"
DEFAULT_ZONES="us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west4-a us-west4-c"
ZONES="${METALFISH_GCP_ZONES:-${METALFISH_GCP_ZONE:-${DEFAULT_ZONES}}}"
INSTANCE="${METALFISH_GCP_INSTANCE:-metalfish-win-cuda-gate-$(date +%Y%m%d-%H%M%S)}"
MACHINE="${METALFISH_GCP_MACHINE:-g2-standard-8}"
ACCELERATOR="${METALFISH_GCP_ACCELERATOR:-type=nvidia-l4-vws,count=1}"
IMAGE_PROJECT="${METALFISH_GCP_IMAGE_PROJECT:-windows-cloud}"
IMAGE_FAMILY="${METALFISH_GCP_IMAGE_FAMILY:-windows-2022}"
BOOT_DISK_SIZE="${METALFISH_GCP_BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE="${METALFISH_GCP_BOOT_DISK_TYPE:-pd-balanced}"
DELETE_ON_EXIT="${METALFISH_GCP_DELETE_ON_EXIT:-1}"
COLLECT_ARTIFACTS="${METALFISH_GCP_COLLECT_ARTIFACTS:-1}"
ARTIFACT_DIR="${METALFISH_GCP_ARTIFACT_DIR:-${ROOT_DIR}/results/windows_cuda_runtime_gate/${INSTANCE}}"
GCS_PREFIX="${METALFISH_GCP_GCS_PREFIX:-}"
PACKAGE_ZIP="${METALFISH_WINDOWS_CUDA_PACKAGE:-}"
WEIGHTS="${METALFISH_BT4_WEIGHTS:-${ROOT_DIR}/networks/BT4-1024x15x32h-swa-6147500.pb}"
NNUE_BIG="${METALFISH_NNUE_BIG:-${ROOT_DIR}/networks/nn-c288c895ea92.nnue}"
NNUE_SMALL="${METALFISH_NNUE_SMALL:-${ROOT_DIR}/networks/nn-37f18f62d772.nnue}"
UCI_TIMEOUT_SECONDS="${METALFISH_WINDOWS_CUDA_UCI_TIMEOUT:-180}"
UCI_GO="${METALFISH_WINDOWS_CUDA_UCI_GO:-nodes 1}"
DRIVER_SCRIPT_URL="${METALFISH_WINDOWS_GPU_DRIVER_SCRIPT_URL:-https://github.com/GoogleCloudPlatform/compute-gpu-installation/raw/main/windows/install_gpu_driver.ps1}"
CREATED_INSTANCE=0
ZONE=""
RUN_DIR="$(mktemp -d -t metalfish-windows-cuda.XXXXXX)"
PACKAGE_BASENAME="$(basename "${PACKAGE_ZIP}")"

cleanup() {
  local status=$?
  if [[ "${COLLECT_ARTIFACTS}" == "1" && "${CREATED_INSTANCE}" == "1" && -n "${ZONE}" ]]; then
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

wait_for_ssh() {
  local attempts="${1:-90}"
  for attempt in $(seq 1 "${attempts}"); do
    if gcloud compute ssh "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --command "hostname" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  echo "timed out waiting for SSH on ${INSTANCE}" >&2
  return 1
}

remote_cmd() {
  gcloud compute ssh "${INSTANCE}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --command "$1"
}

copy_to_remote() {
  local src="$1"
  local dst="$2"
  gcloud compute scp "${src}" "${INSTANCE}:${dst}" \
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
    "${INSTANCE}:C:/metalfish/logs" \
    "${ARTIFACT_DIR}/" \
    --project "${PROJECT}" \
    --zone "${ZONE}" >/dev/null 2>&1 || true
  if [[ -n "${GCS_PREFIX}" && -d "${ARTIFACT_DIR}/logs" ]]; then
    gcloud storage cp --recursive "${ARTIFACT_DIR}/logs" \
      "${GCS_PREFIX%/}/${INSTANCE}/" >/dev/null
  fi
}

require_file "${PACKAGE_ZIP}" "Windows CUDA package"
require_file "${WEIGHTS}" "BT4 weights"
require_file "${NNUE_BIG}" "large NNUE"
require_file "${NNUE_SMALL}" "small NNUE"

cd "${ROOT_DIR}"

for candidate_zone in ${ZONES}; do
  echo "Creating ${INSTANCE} in ${candidate_zone}"
  if gcloud compute instances create "${INSTANCE}" \
    --project "${PROJECT}" \
    --zone "${candidate_zone}" \
    --machine-type "${MACHINE}" \
    --accelerator "${ACCELERATOR}" \
    --maintenance-policy TERMINATE \
    --restart-on-failure \
    --image-project "${IMAGE_PROJECT}" \
    --image-family "${IMAGE_FAMILY}" \
    --boot-disk-size "${BOOT_DISK_SIZE}" \
    --boot-disk-type "${BOOT_DISK_TYPE}" \
    --scopes https://www.googleapis.com/auth/cloud-platform; then
    ZONE="${candidate_zone}"
    CREATED_INSTANCE=1
    break
  fi
  echo "Zone ${candidate_zone} could not allocate ${MACHINE} with ${ACCELERATOR}" >&2
done

if [[ "${CREATED_INSTANCE}" != "1" ]]; then
  echo "failed to create a Windows CUDA runtime VM in zones: ${ZONES}" >&2
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
copy_to_remote "${NNUE_BIG}" "C:/metalfish/networks/nn-c288c895ea92.nnue"
copy_to_remote "${NNUE_SMALL}" "C:/metalfish/networks/nn-37f18f62d772.nnue"

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
  throw "VC++ redistributable install failed with exit code \$($vc.ExitCode)"
}
\$Engine = Join-Path \$PackageDir "metalfish.exe"
if (-not (Test-Path \$Engine)) {
  throw "Packaged engine not found: \$Engine"
}
\$env:PATH = "\$PackageDir;\$env:PATH"
nvidia-smi 2>&1 | Tee-Object -FilePath (Join-Path \$Logs "nvidia-smi-runtime.log")

function Invoke-UciSmoke {
  param(
    [string]\$Name,
    [string[]]\$Commands,
    [string[]]\$RequiredText
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
  \$proc = [System.Diagnostics.Process]::Start(\$psi)
  \$proc.StandardInput.WriteLine((\$Commands -join "\`n"))
  \$proc.StandardInput.Close()
  if (-not \$proc.WaitForExit(${UCI_TIMEOUT_SECONDS} * 1000)) {
    try { \$proc.Kill() } catch {}
    throw "\$Name timed out after ${UCI_TIMEOUT_SECONDS}s"
  }
  \$out = \$proc.StandardOutput.ReadToEnd()
  \$err = \$proc.StandardError.ReadToEnd()
  Set-Content -Path \$stdout -Value \$out -Encoding UTF8
  Set-Content -Path \$stderr -Value \$err -Encoding UTF8
  if (\$proc.ExitCode -ne 0) {
    throw "\$Name exited with code \$($proc.ExitCode)"
  }
  foreach (\$needle in \$RequiredText) {
    if ((\$out + \$err) -notlike "*\$needle*") {
      throw "\$Name missing expected output: \$needle"
    }
  }
}

\$Bt4 = Join-Path \$Networks "BT4-1024x15x32h-swa-6147500.pb"
\$NnueBig = Join-Path \$Networks "nn-c288c895ea92.nnue"
\$NnueSmall = Join-Path \$Networks "nn-37f18f62d772.nnue"

Invoke-UciSmoke -Name "cuda-mcts" -Commands @(
  "uci",
  "isready",
  "setoption name EvalFile value \$NnueBig",
  "setoption name EvalFileSmall value \$NnueSmall",
  "setoption name NNBackend value cuda",
  "setoption name NNWeights value \$Bt4",
  "setoption name UseMCTS value true",
  "setoption name UseHybridSearch value false",
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 1",
  "position startpos",
  "go ${UCI_GO}",
  "quit"
) -RequiredText @("CUDA transformer backend", "bestmove")

Invoke-UciSmoke -Name "hybrid-cuda" -Commands @(
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
  "setoption name MCTSMaxThreads value 1",
  "setoption name MCTSMinibatchSize value 1",
  "position startpos",
  "go nodes 8",
  "quit"
) -RequiredText @("Starting Parallel Hybrid Search", "CUDA transformer backend", "bestmove")

@(
  "# MetalFish Windows CUDA Runtime Gate",
  "",
  "- Gate status: passed",
  "- Package: ${PACKAGE_BASENAME}",
  "- GPU: see nvidia-smi-runtime.log",
  "- Smokes: cuda-mcts, hybrid-cuda"
) | Set-Content -Path (Join-Path \$Logs "windows-cuda-runtime-summary.md") -Encoding UTF8
Write-Host "Windows CUDA runtime gate passed"
POWERSHELL

run_remote_ps "${RUN_DIR}/run-smokes.ps1"
collect_remote_artifacts

echo "Windows CUDA runtime gate passed"
echo "Artifacts: ${ARTIFACT_DIR}"
