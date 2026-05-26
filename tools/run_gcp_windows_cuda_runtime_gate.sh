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
REQUIRE_METAL_COMPARE="${METALFISH_REQUIRE_METAL_COMPARE:-0}"
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
UCI_GO="${METALFISH_WINDOWS_CUDA_UCI_GO:-nodes 1}"
HYBRID_UCI_GO="${METALFISH_WINDOWS_CUDA_HYBRID_UCI_GO:-movetime 8000}"
HYBRID_POST_GO_SLEEP_MS="${METALFISH_WINDOWS_CUDA_HYBRID_POST_GO_SLEEP_MS:-10000}"
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
  if [[ -n "${GCS_PREFIX}" && -d "${ARTIFACT_DIR}/logs" ]]; then
    gcloud storage cp --recursive "${ARTIFACT_DIR}/logs" \
      "${GCS_PREFIX%/}/${INSTANCE}/" >/dev/null
  fi
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

print(json.dumps([position.__dict__ for position in DEFAULT_POSITIONS]))
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
  \$positions = @(Get-Content -Path \$positionsPath -Raw -Encoding UTF8 | ConvertFrom-Json)
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
    [string[]]\$PositiveMetrics = @(),
    [int]\$GoWaitMs = 0
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
  \$proc = [System.Diagnostics.Process]::Start(\$psi)
  \$stdoutTask = \$proc.StandardOutput.ReadToEndAsync()
  \$stderrTask = \$proc.StandardError.ReadToEndAsync()
  foreach (\$command in \$Commands) {
    \$proc.StandardInput.WriteLine(\$command)
    \$proc.StandardInput.Flush()
    if (\$command -like "go *" -and \$GoWaitMs -gt 0) {
      Start-Sleep -Milliseconds \$GoWaitMs
    }
  }
  \$proc.StandardInput.Close()
  \$timedOut = -not \$proc.WaitForExit(${UCI_TIMEOUT_SECONDS} * 1000)
  if (\$timedOut) {
    try { \$proc.Kill() } catch {}
    try { \$proc.WaitForExit(10000) | Out-Null } catch {}
  }
  \$out = \$stdoutTask.GetAwaiter().GetResult()
  \$err = \$stderrTask.GetAwaiter().GetResult()
  Set-Content -Path \$stdout -Value \$out -Encoding UTF8
  Set-Content -Path \$stderr -Value \$err -Encoding UTF8
  if (\$timedOut) {
    throw (\$Name + " timed out after ${UCI_TIMEOUT_SECONDS}s")
  }
  if (\$proc.ExitCode -ne 0) {
    throw (\$Name + " exited with code " + \$proc.ExitCode)
  }
  foreach (\$needle in \$RequiredText) {
    if ((\$out + \$err) -notlike "*\$needle*") {
      throw "\$Name missing expected output: \$needle"
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
) -RequiredText (\$CudaNetworkInfoRequiredText + @("CUDA transformer backend", "bestmove"))

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
) -RequiredText (\$CudaNetworkInfoRequiredText + @("CUDA transformer backend", "bestmove"))

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
) -RequiredText (\$CudaNetworkInfoRequiredText + @("CUDA transformer backend", "bestmove"))

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
  "setoption name TransformerLowTimeFallbackMs value 0",
  "position startpos",
  "go ${HYBRID_UCI_GO}",
  "quit"
) -RequiredText (\$CudaNetworkInfoRequiredText + @("Starting Parallel Hybrid Search", "CUDA transformer backend", "Final: MCTSPlayouts=", "bestmove")) -PositiveMetrics @("MCTSPlayouts", "MCTSEvals", "ABDepth") -GoWaitMs ${HYBRID_POST_GO_SLEEP_MS}

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
) -RequiredText (\$CudaNetworkInfoRequiredText + @("Starting Parallel Hybrid Search", "CUDA transformer backend", "Final: MCTSPlayouts=", "bestmove")) -PositiveMetrics @("MCTSPlayouts", "MCTSEvals", "ABDepth") -GoWaitMs ${HYBRID_POST_GO_SLEEP_MS}

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
) -RequiredText (\$CudaNetworkInfoRequiredText + @("Starting Parallel Hybrid Search", "CUDA transformer backend", "ANE root probe disabled", "Final: MCTSPlayouts=", "bestmove")) -PositiveMetrics @("MCTSPlayouts", "MCTSEvals", "ABDepth") -GoWaitMs ${HYBRID_POST_GO_SLEEP_MS}

\$ProbeJson = Read-ProbeJson "cuda-probe.stdout.log"
\$LegacyProbeJson = Read-ProbeJson "cuda-legacy-probe.stdout.log"
\$IsolationBt4LegacyJson = Read-ProbeJson "cuda-isolation-bt4-legacy.stdout.log"
\$IsolationLegacyBt4Json = Read-ProbeJson "cuda-isolation-legacy-bt4.stdout.log"
\$MctsText = Read-SmokeText "cuda-mcts"
\$AutoMctsText = Read-SmokeText "cuda-auto-mcts"
\$AcceleratorMctsText = Read-SmokeText "cuda-accelerator-mcts"
\$HybridText = Read-SmokeText "hybrid-cuda"
\$HybridAutoText = Read-SmokeText "hybrid-auto"
\$HybridAneText = Read-SmokeText "hybrid-cuda-ane-disabled"
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
    hybrid_cuda = [ordered]@{
      go = "${HYBRID_UCI_GO}"
      bestmove = (Find-BestMove \$HybridText)
      backend_selected = (Test-BackendSelected \$HybridText)
      metrics = (Find-FinalMetrics \$HybridText)
      stdout_log = "hybrid-cuda.stdout.log"
      stderr_log = "hybrid-cuda.stderr.log"
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
  "- Smokes: cuda-probe, cuda-legacy-probe, cuda-probe-suite, cuda-legacy-probe-suite, cuda-isolation-bt4-legacy, cuda-isolation-legacy-bt4, cuda-mcts, cuda-auto-mcts, cuda-accelerator-mcts, hybrid-cuda, hybrid-auto, hybrid-cuda-ane-disabled",
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
if [[ "${RUNTIME_STATUS}" == "0" ]]; then
  compare_collected_probe_suite || COMPARE_STATUS=$?
  if [[ "${COMPARE_STATUS}" == "0" ]]; then
    compare_collected_legacy_probe_suite || COMPARE_STATUS=$?
  fi
fi

if [[ "${RUNTIME_STATUS}" != "0" ]]; then
  exit "${RUNTIME_STATUS}"
fi
if [[ "${COMPARE_STATUS}" != "0" ]]; then
  exit "${COMPARE_STATUS}"
fi

echo "Windows CUDA runtime gate passed"
echo "Artifacts: ${ARTIFACT_DIR}"
