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
  "-DCMAKE_CUDA_HOST_COMPILER=$Cl"
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

$Summary = Join-Path $BuildDir "windows-cuda-compile-summary.md"
$NvccVersion = (& $Nvcc --version) -join "`n"
$CmakeVersion = (& $Cmake --version) -join "`n"
$NinjaVersion = (& $Ninja --version) -join "`n"
$ClVersion = (& $Cl 2>&1) -join "`n"
$TimestampUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$TargetsText = $Targets -join ", "

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
