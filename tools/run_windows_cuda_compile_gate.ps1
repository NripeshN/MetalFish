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
$env:VCPKG_INSTALLATION_ROOT = $VcpkgRoot
$env:VCPKG_ROOT = $VcpkgRoot

function Require-Command {
  param([string]$Name)
  $Command = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $Command) {
    throw "Required command not found on PATH: $Name"
  }
  return $Command.Source
}

function Copy-ExistingFile {
  param(
    [string]$Path,
    [string]$Destination
  )
  if (Test-Path $Path) {
    Copy-Item $Path $Destination -Force
    return 1
  }
  return 0
}

function Copy-MatchingFiles {
  param(
    [string]$Pattern,
    [string]$Destination
  )
  $Count = 0
  foreach ($File in Get-ChildItem -Path $Pattern -File -ErrorAction SilentlyContinue) {
    Copy-Item $File.FullName $Destination -Force
    $Count += 1
  }
  return $Count
}

function Read-ProbeJson {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    throw "Probe JSON file not found: $Path"
  }
  $Text = Get-Content -Path $Path -Raw
  $JsonLine = $Text -split "\r?\n" |
    Where-Object { $_.TrimStart().StartsWith("{") } |
    Select-Object -First 1
  if (-not $JsonLine) {
    throw "Probe JSON file does not contain a JSON object: $Path"
  }
  return $JsonLine | ConvertFrom-Json
}

function Assert-CudaMetadataProbe {
  param(
    [string]$Name,
    [object]$Probe
  )
  if (-not $Probe.metadata_only) {
    throw "$Name did not run in metadata-only mode"
  }
  if ($Probe.backend -ne "cuda") {
    throw "$Name backend was not cuda: $($Probe.backend)"
  }
  if (-not $Probe.cuda_schedule_fully_supported) {
    throw "$Name CUDA schedule is not fully supported"
  }
  if (-not $Probe.cuda_output_mapping_ok) {
    throw "$Name CUDA output mapping is not valid"
  }
  foreach ($Field in @("tensor_plan", "execution_plan", "cuda_schedule",
                       "cuda_output_mapping", "parameter_elements", "steps")) {
    if ($null -eq $Probe.$Field) {
      throw "$Name missing metadata field: $Field"
    }
  }
}

function Test-PackageFile {
  param(
    [string]$Directory,
    [string]$Pattern
  )
  return @(Get-ChildItem -Path (Join-Path $Directory $Pattern) -File -ErrorAction SilentlyContinue).Count -gt 0
}

$MsvcEnvScript = Join-Path $SourceDir "tools\import_msvc_dev_env.ps1"
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue) -and
    (Test-Path $MsvcEnvScript)) {
  & $MsvcEnvScript -Arch x64 -HostArch x64
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
$ProbePath = ""
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
$LegacyPath = Join-Path $NetworksDir "legacy-42850.pb.gz"
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
  if (-not (Test-Path $LegacyPath)) {
    Write-Host "Downloading legacy 42850 weights for CUDA-linked metadata probe"
    & $Python (Join-Path $SourceDir "tools\download_engine_networks.py") `
      --dest $NetworksDir `
      --legacy-only
    if ($LASTEXITCODE -ne 0) {
      throw "legacy 42850 download failed with exit code $LASTEXITCODE"
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
                              '"execution_plan":"',
                              '"cuda_schedule_fully_supported":true',
                              '"cuda_output_mapping_ok":true')) {
    if ($ProbeText -notlike "*$RequiredText*") {
      throw "CUDA-linked BT4 metadata probe missing expected output: $RequiredText"
    }
  }
  $SmokeSteps += "metalfish_nn_probe.exe --metadata-only --backend cuda"

  $LegacyProbeLog = Join-Path $BuildDir "windows-cuda-legacy-metadata-probe.json"
  Write-Host "Running CUDA-linked legacy metadata probe"
  & $ProbePath `
    --weights $LegacyPath `
    --backend cuda `
    --metadata-only `
    --top 3 `
    2>&1 | Tee-Object -FilePath $LegacyProbeLog
  if ($LASTEXITCODE -ne 0) {
    throw "CUDA-linked legacy metadata probe failed with exit code $LASTEXITCODE"
  }

  $LegacyProbeText = Get-Content -Path $LegacyProbeLog -Raw
  foreach ($RequiredText in @('"metadata_only":true', '"backend":"cuda"',
                              '"format":"attention_body=no',
                              '"policy_head":"', '"value_head":"',
                              '"execution_plan":"',
                              '"cuda_schedule_fully_supported":true',
                              '"cuda_output_mapping_ok":true')) {
    if ($LegacyProbeText -notlike "*$RequiredText*") {
      throw "CUDA-linked legacy metadata probe missing expected output: $RequiredText"
    }
  }
  $SmokeSteps += "metalfish_nn_probe.exe --metadata-only --backend cuda legacy"
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

$PackageName = "metalfish-windows-x86_64-msvc-cuda"
if ($env:GITHUB_REF_TYPE -eq "tag") {
  $PackageName = "metalfish-$env:GITHUB_REF_NAME-windows-x86_64-msvc-cuda"
}
$PackageDir = Join-Path $BuildDir $PackageName
$PackageSmokeDir = Join-Path $BuildDir "$PackageName-smoke"
$PackageZip = Join-Path $BuildDir "$PackageName.zip"
Remove-Item $PackageDir, $PackageSmokeDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageZip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $PackageDir | Out-Null

Copy-Item $EnginePath $PackageDir -Force
if ($ProbePath -and (Test-Path $ProbePath)) {
  Copy-Item $ProbePath $PackageDir -Force
}
foreach ($DocName in @("README.md", "CHANGELOG.md", "LICENSE")) {
  Copy-ExistingFile (Join-Path $SourceDir $DocName) $PackageDir | Out-Null
}

& $Python (Join-Path $SourceDir "tools\write_portable_manifest.py") `
  --platform "Windows x86_64 MSVC CUDA" `
  --backend "CPU AB plus CUDA transformer MCTS/Hybrid when NVIDIA CUDA runtime and BT4 weights are available" `
  --binary "metalfish.exe" `
  --output (Join-Path $PackageDir "PORTABLE_ARTIFACT.md") `
  --notes "This artifact is built by the Windows CUDA compile gate with MSVC and the NVIDIA CUDA Toolkit." `
  --notes "The package includes CUDA and vcpkg runtime DLLs required by the linked engine." `
  --notes "The package includes metalfish_nn_probe.exe when BUILD_TESTS=ON so runtime gates can verify packaged CUDA inference." `
  --notes "Run a real Windows NVIDIA runtime smoke before calling this artifact strength-ready."
if ($LASTEXITCODE -ne 0) {
  throw "portable manifest generation failed with exit code $LASTEXITCODE"
}

$CopiedRuntimeDlls = 0
$CopiedRuntimeDlls += Copy-MatchingFiles (Join-Path $BuildDir "*.dll") $PackageDir
$VcpkgBin = Join-Path $VcpkgRoot "installed\x64-windows\bin"
if (Test-Path $VcpkgBin) {
  $CopiedRuntimeDlls += Copy-MatchingFiles (Join-Path $VcpkgBin "*.dll") $PackageDir
}
$CudaBin = Join-Path $env:CUDA_PATH "bin"
foreach ($Pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
  $CopiedRuntimeDlls += Copy-MatchingFiles (Join-Path $CudaBin $Pattern) $PackageDir
}
if ($CopiedRuntimeDlls -eq 0) {
  throw "No runtime DLLs were copied into the Windows CUDA package"
}

Compress-Archive -Path (Join-Path $PackageDir "*") -DestinationPath $PackageZip
if (-not (Test-Path $PackageZip)) {
  throw "Windows CUDA package was not created: $PackageZip"
}

New-Item -ItemType Directory -Force $PackageSmokeDir | Out-Null
Expand-Archive -Path $PackageZip -DestinationPath $PackageSmokeDir -Force
$PackagedEngine = Join-Path $PackageSmokeDir "metalfish.exe"
if (-not (Test-Path $PackagedEngine)) {
  throw "Packaged engine not found after extraction: $PackagedEngine"
}
$PackagedProbe = Join-Path $PackageSmokeDir "metalfish_nn_probe.exe"
if ($BuildTests -eq "ON" -and -not (Test-Path $PackagedProbe)) {
  throw "Packaged NN probe not found after extraction: $PackagedProbe"
}

$OriginalPath = $env:PATH
try {
  $env:PATH = "$PackageSmokeDir;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem"
  Write-Host "Running packaged Windows CUDA AB self-smoke"
  & $Python $UciSmoke `
    --engine $PackagedEngine `
    --timeout 45 `
    --setoption "EvalFile=$NnueBigPath" `
    --setoption "EvalFileSmall=$NnueSmallPath" `
    --setoption "UseMCTS=false" `
    --setoption "UseHybridSearch=false" `
    --setoption "Threads=1" `
    --setoption "Hash=16" `
    --go "depth 1"
  if ($LASTEXITCODE -ne 0) {
    throw "Packaged Windows CUDA AB self-smoke failed with exit code $LASTEXITCODE"
  }
  if ($BuildTests -eq "ON") {
    $PackagedProbeLog = Join-Path $BuildDir "windows-cuda-packaged-bt4-metadata-probe.json"
    Write-Host "Running packaged Windows CUDA BT4 metadata probe"
    & $PackagedProbe `
      --weights $Bt4Path `
      --backend cuda `
      --metadata-only `
      --top 3 `
      2>&1 | Tee-Object -FilePath $PackagedProbeLog
    if ($LASTEXITCODE -ne 0) {
      throw "Packaged Windows CUDA metadata probe failed with exit code $LASTEXITCODE"
    }
    $PackagedProbeText = Get-Content -Path $PackagedProbeLog -Raw
    foreach ($RequiredText in @('"metadata_only":true', '"backend":"cuda"',
                                '"policy_head":"', '"value_head":"',
                                '"execution_plan":"',
                                '"cuda_schedule_fully_supported":true',
                                '"cuda_output_mapping_ok":true')) {
      if ($PackagedProbeText -notlike "*$RequiredText*") {
        throw "Packaged Windows CUDA BT4 metadata probe missing expected output: $RequiredText"
      }
    }

    $PackagedLegacyProbeLog = Join-Path $BuildDir "windows-cuda-packaged-legacy-metadata-probe.json"
    Write-Host "Running packaged Windows CUDA legacy metadata probe"
    & $PackagedProbe `
      --weights $LegacyPath `
      --backend cuda `
      --metadata-only `
      --top 3 `
      2>&1 | Tee-Object -FilePath $PackagedLegacyProbeLog
    if ($LASTEXITCODE -ne 0) {
      throw "Packaged Windows CUDA legacy metadata probe failed with exit code $LASTEXITCODE"
    }
    $PackagedLegacyProbeText = Get-Content -Path $PackagedLegacyProbeLog -Raw
    foreach ($RequiredText in @('"metadata_only":true', '"backend":"cuda"',
                                '"format":"attention_body=no',
                                '"policy_head":"', '"value_head":"',
                                '"execution_plan":"',
                                '"cuda_schedule_fully_supported":true',
                                '"cuda_output_mapping_ok":true')) {
      if ($PackagedLegacyProbeText -notlike "*$RequiredText*") {
        throw "Packaged Windows CUDA legacy metadata probe missing expected output: $RequiredText"
      }
    }
  }
} finally {
  $env:PATH = $OriginalPath
}
$SmokeSteps += "$PackageName.zip extracted AB self-smoke"
if ($BuildTests -eq "ON") {
  $SmokeSteps += "$PackageName.zip extracted BT4 metadata probe"
  $SmokeSteps += "$PackageName.zip extracted legacy metadata probe"
}

$Summary = Join-Path $BuildDir "windows-cuda-compile-summary.md"
$ArtifactManifest = Join-Path $BuildDir "windows-cuda-compile-artifact-manifest.json"
$NvccVersion = (& $Nvcc --version) -join "`n"
$CmakeVersion = (& $Cmake --version) -join "`n"
$NinjaVersion = (& $Ninja --version) -join "`n"
$ClVersion = (& $Cl 2>&1) -join "`n"
$TimestampUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$TargetsText = $Targets -join ", "
$SmokeText = $SmokeSteps -join ", "

$ManifestProbeChecks = $null
if ($BuildTests -eq "ON") {
  $Bt4Probe = Read-ProbeJson $ProbeLog
  $LegacyProbe = Read-ProbeJson $LegacyProbeLog
  $PackagedBt4Probe = Read-ProbeJson $PackagedProbeLog
  $PackagedLegacyProbe = Read-ProbeJson $PackagedLegacyProbeLog

  Assert-CudaMetadataProbe "BT4 metadata probe" $Bt4Probe
  Assert-CudaMetadataProbe "legacy metadata probe" $LegacyProbe
  Assert-CudaMetadataProbe "packaged BT4 metadata probe" $PackagedBt4Probe
  Assert-CudaMetadataProbe "packaged legacy metadata probe" $PackagedLegacyProbe

  $ComparableProbeFields = @(
    "tensor_plan",
    "execution_plan",
    "cuda_schedule",
    "cuda_output_mapping",
    "parameter_elements",
    "steps"
  )
  foreach ($Field in $ComparableProbeFields) {
    if ($Bt4Probe.$Field -ne $PackagedBt4Probe.$Field) {
      throw "Packaged BT4 probe does not match build-tree BT4 probe field: $Field"
    }
    if ($LegacyProbe.$Field -ne $PackagedLegacyProbe.$Field) {
      throw "Packaged legacy probe does not match build-tree legacy probe field: $Field"
    }
  }

  $ManifestProbeChecks = [ordered]@{
    bt4 = [ordered]@{
      metadata_only = [bool]$Bt4Probe.metadata_only
      backend = $Bt4Probe.backend
      policy_head = $Bt4Probe.policy_head
      value_head = $Bt4Probe.value_head
      tensor_plan = $Bt4Probe.tensor_plan
      execution_plan = $Bt4Probe.execution_plan
      cuda_schedule_fully_supported = [bool]$Bt4Probe.cuda_schedule_fully_supported
      cuda_output_mapping_ok = [bool]$Bt4Probe.cuda_output_mapping_ok
      parameter_elements = [int64]$Bt4Probe.parameter_elements
      steps = [int64]$Bt4Probe.steps
    }
    legacy = [ordered]@{
      metadata_only = [bool]$LegacyProbe.metadata_only
      backend = $LegacyProbe.backend
      format = $LegacyProbe.format
      policy_head = $LegacyProbe.policy_head
      value_head = $LegacyProbe.value_head
      tensor_plan = $LegacyProbe.tensor_plan
      execution_plan = $LegacyProbe.execution_plan
      cuda_schedule_fully_supported = [bool]$LegacyProbe.cuda_schedule_fully_supported
      cuda_output_mapping_ok = [bool]$LegacyProbe.cuda_output_mapping_ok
      parameter_elements = [int64]$LegacyProbe.parameter_elements
      steps = [int64]$LegacyProbe.steps
    }
    packaged_bt4_matches_build_tree = $true
    packaged_legacy_matches_build_tree = $true
    compared_fields = $ComparableProbeFields
  }
}

$PackageFiles = @(
  Get-ChildItem -Path $PackageDir -File |
    Select-Object -ExpandProperty Name |
    Sort-Object
)
$ManifestObject = [ordered]@{
  gate_status = "passed"
  timestamp_utc = $TimestampUtc
  source = $SourceDir
  build_directory = $BuildDir
  build_type = $BuildType
  cuda_architectures = $CudaArchs
  cuda_path = $env:CUDA_PATH
  build_tests = $BuildTests
  targets = $Targets
  smoke_tests = $SmokeSteps
  package = $PackageZip
  packaged_runtime_dll_count = $CopiedRuntimeDlls
  package_contents = [ordered]@{
    metalfish_exe = Test-PackageFile $PackageDir "metalfish.exe"
    metalfish_nn_probe_exe = Test-PackageFile $PackageDir "metalfish_nn_probe.exe"
    portable_artifact_md = Test-PackageFile $PackageDir "PORTABLE_ARTIFACT.md"
    cudart = Test-PackageFile $PackageDir "cudart64_*.dll"
    cublas = Test-PackageFile $PackageDir "cublas64_*.dll"
    cublaslt = Test-PackageFile $PackageDir "cublasLt64_*.dll"
    files = $PackageFiles
  }
  probe_checks = $ManifestProbeChecks
  gpu_runtime_smoke = "not_run_in_github_windows_ci"
  runtime_gate = "tools/run_gcp_windows_cuda_runtime_gate.sh"
}
$ManifestObject | ConvertTo-Json -Depth 12 | Set-Content -Path $ArtifactManifest -Encoding UTF8

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
  "- Package: $PackageZip",
  "- Packaged runtime DLLs: $CopiedRuntimeDlls",
  "- Artifact manifest: $ArtifactManifest",
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
