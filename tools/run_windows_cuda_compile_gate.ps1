[CmdletBinding()]
param(
  [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$BuildDir = "",
  [string]$BuildType = "Release",
  [string]$CudaArchs = "",
  [ValidateSet("ON", "OFF")]
  [string]$BuildTests = "ON",
  [string]$VcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
)

$ErrorActionPreference = "Stop"

if (-not $BuildDir) {
  $BuildDir = Join-Path $SourceDir "build-windows-cuda"
}
if (-not $CudaArchs) {
  $CudaArchs = if ($env:METALFISH_CUDA_ARCHS) { $env:METALFISH_CUDA_ARCHS } else { "89" }
}
if (-not $VcpkgRoot) {
  $VcpkgRoot = "C:\vcpkg"
}

function Require-Command {
  param([string]$Name)
  $Command = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $Command) {
    throw "Required command not found on PATH: $Name"
  }
  return $Command.Source
}

$Cmake = Require-Command "cmake"
$Ninja = Require-Command "ninja"
$Nvcc = Require-Command "nvcc"
$Cl = Require-Command "cl.exe"
$Python = Require-Command "python"
$ClForCmake = $Cl -replace "\\", "/"

if (-not $env:CUDA_PATH) {
  throw "CUDA_PATH is not set; install the CUDA Toolkit before running this gate"
}

$CudaLibDir = Join-Path $env:CUDA_PATH "lib\x64"
foreach ($LibName in @("cudart.lib", "cublas.lib")) {
  $LibPath = Join-Path $CudaLibDir $LibName
  if (-not (Test-Path $LibPath)) {
    throw "Required CUDA import library not found: $LibPath"
  }
}

$ToolchainFile = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
if (-not (Test-Path $ToolchainFile)) {
  throw "vcpkg toolchain file not found: $ToolchainFile"
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null

$ConfigureArgs = @(
  "-S", $SourceDir,
  "-B", $BuildDir,
  "-G", "Ninja",
  "-DCMAKE_BUILD_TYPE=$BuildType",
  "-DUSE_METAL=OFF",
  "-DUSE_CUDA=ON",
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs",
  "-DBUILD_TESTS=$BuildTests",
  "-DMETALFISH_ENABLE_IPO=OFF",
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
  "-DCMAKE_TOOLCHAIN_FILE=$ToolchainFile",
  "-DVCPKG_TARGET_TRIPLET=x64-windows",
  "-DCMAKE_CUDA_HOST_COMPILER=$ClForCmake"
)

& $Cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) {
  throw "CMake configure failed with exit code $LASTEXITCODE"
}

$Targets = @("metalfish")
if ($BuildTests -eq "ON") {
  $Targets += @("metalfish_tests", "test_nn_comparison", "metalfish_nn_probe")
}

$BuildArgs = @("--build", $BuildDir, "--target") + $Targets
& $Cmake @BuildArgs
if ($LASTEXITCODE -ne 0) {
  throw "CMake build failed with exit code $LASTEXITCODE"
}

$EnginePath = (Resolve-Path (Join-Path $BuildDir "metalfish.exe")).Path
$SmokeSteps = @()
if ($BuildTests -eq "ON") {
  $TestsPath = (Resolve-Path (Join-Path $BuildDir "metalfish_tests.exe")).Path
  Write-Host "Running CUDA-linked MCTS module smoke"
  & $TestsPath mcts
  if ($LASTEXITCODE -ne 0) {
    throw "CUDA-linked MCTS module smoke failed with exit code $LASTEXITCODE"
  }
  $SmokeSteps += "metalfish_tests.exe mcts"
}

$NetworksDir = Join-Path $SourceDir "networks"
$NnueBigPath = Join-Path $NetworksDir "nn-c288c895ea92.nnue"
$NnueSmallPath = Join-Path $NetworksDir "nn-37f18f62d772.nnue"
$Bt4Path = Join-Path $NetworksDir "BT4-1024x15x32h-swa-6147500.pb"
if (-not ((Test-Path $NnueBigPath) -and (Test-Path $NnueSmallPath))) {
  Write-Host "Downloading NNUE files for CUDA-linked AB UCI smoke"
  & $Python (Join-Path $SourceDir "tools\download_engine_networks.py") `
    --dest $NetworksDir `
    --nnue-only
  if ($LASTEXITCODE -ne 0) {
    throw "NNUE download failed with exit code $LASTEXITCODE"
  }
}

if ($BuildTests -eq "ON") {
  if (-not (Test-Path $Bt4Path)) {
    Write-Host "Downloading BT4 weights for CUDA-linked metadata probe"
    & $Python (Join-Path $SourceDir "tools\download_engine_networks.py") `
      --dest $NetworksDir `
      --bt4-only
    if ($LASTEXITCODE -ne 0) {
      throw "BT4 download failed with exit code $LASTEXITCODE"
    }
  }

  $ProbePath = (Resolve-Path (Join-Path $BuildDir "metalfish_nn_probe.exe")).Path
  $ProbeLog = Join-Path $BuildDir "windows-cuda-bt4-metadata-probe.json"
  Write-Host "Running CUDA-linked BT4 metadata probe"
  & $ProbePath `
    --weights $Bt4Path `
    --backend cuda `
    --metadata-only `
    --top 3 `
    2>&1 | Tee-Object -FilePath $ProbeLog
  if ($LASTEXITCODE -ne 0) {
    throw "CUDA-linked BT4 metadata probe failed with exit code $LASTEXITCODE"
  }

  $ProbeText = Get-Content -Path $ProbeLog -Raw
  foreach ($RequiredText in @('"metadata_only":true', '"backend":"cuda"',
                              '"policy_head":"', '"value_head":"',
                              '"execution_plan":"')) {
    if ($ProbeText -notlike "*$RequiredText*") {
      throw "CUDA-linked BT4 metadata probe missing expected output: $RequiredText"
    }
  }
  $SmokeSteps += "metalfish_nn_probe.exe --metadata-only --backend cuda"
}

$UciSmoke = Join-Path $SourceDir "tools\uci_smoke.py"
Write-Host "Running CUDA-linked AB UCI smoke"
& $Python $UciSmoke `
  --engine $EnginePath `
  --timeout 45 `
  --setoption "EvalFile=$NnueBigPath" `
  --setoption "EvalFileSmall=$NnueSmallPath" `
  --setoption "UseMCTS=false" `
  --setoption "UseHybridSearch=false" `
  --setoption "Threads=1" `
  --setoption "Hash=16" `
  --go "depth 1"
if ($LASTEXITCODE -ne 0) {
  throw "CUDA-linked AB UCI smoke failed with exit code $LASTEXITCODE"
}
$SmokeSteps += "tools/uci_smoke.py depth 1"

$Summary = Join-Path $BuildDir "windows-cuda-compile-summary.md"
$NvccVersion = (& $Nvcc --version) -join "`n"
$CmakeVersion = (& $Cmake --version) -join "`n"
$NinjaVersion = (& $Ninja --version) -join "`n"
$ClVersion = (& $Cl 2>&1) -join "`n"
$TimestampUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$TargetsText = $Targets -join ", "
$SmokeText = $SmokeSteps -join ", "

@(
  "# MetalFish Windows CUDA Compile Gate",
  "",
  "- Gate status: passed",
  "- Timestamp UTC: $TimestampUtc",
  "- Source: $SourceDir",
  "- Build directory: $BuildDir",
  "- Build type: $BuildType",
  "- CUDA architectures: $CudaArchs",
  "- CUDA_PATH: $env:CUDA_PATH",
  "- Build tests: $BuildTests",
  "- Targets: $TargetsText",
  "- Smoke tests: $SmokeText",
  "",
  "## Toolchain",
  "",
  '```text',
  $CmakeVersion,
  "",
  "ninja $NinjaVersion",
  "",
  $NvccVersion,
  "",
  $ClVersion,
  '```'
) | Set-Content -Path $Summary -Encoding UTF8

Write-Host "Windows CUDA compile gate passed"
Write-Host "Summary: $Summary"
