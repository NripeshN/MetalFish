[CmdletBinding()]
param(
  [string]$Version = "12.9.1",
  [string]$InstallerUrl = "",
  [string]$DownloadDir = "",
  [string[]]$Components = @()
)

$ErrorActionPreference = "Stop"

function Get-CudaMajorMinor {
  param([string]$CudaVersion)
  if ($CudaVersion -notmatch "^([0-9]+\.[0-9]+)") {
    throw "Could not derive CUDA major.minor from version '$CudaVersion'"
  }
  return $Matches[1]
}

function Add-CurrentAndGitHubPath {
  param([string]$CudaRoot)

  $BinPath = Join-Path $CudaRoot "bin"
  $LibNvvpPath = Join-Path $CudaRoot "libnvvp"
  $env:CUDA_PATH = $CudaRoot
  $VersionEnvName = "CUDA_PATH_V$($MajorMinor.Replace('.', '_'))"
  Set-Item -Path "Env:$VersionEnvName" -Value $CudaRoot
  $env:Path = "$BinPath;$LibNvvpPath;$env:Path"

  if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$CudaRoot"
    Add-Content -Path $env:GITHUB_ENV -Value "$VersionEnvName=$CudaRoot"
  }
  if ($env:GITHUB_PATH) {
    Add-Content -Path $env:GITHUB_PATH -Value $BinPath
    Add-Content -Path $env:GITHUB_PATH -Value $LibNvvpPath
  }
}

function Invoke-DownloadWithRetry {
  param(
    [string]$Uri,
    [string]$OutFile,
    [int]$Attempts = 3
  )

  for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
    try {
      Write-Host "Downloading $Uri -> $OutFile (attempt $Attempt/$Attempts)"
      Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
      if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -gt 0)) {
        return
      }
      throw "Downloaded installer is empty"
    } catch {
      if ($Attempt -eq $Attempts) {
        throw
      }
      Write-Warning "CUDA installer download failed: $($_.Exception.Message)"
      Start-Sleep -Seconds (5 * $Attempt)
    }
  }
}

$MajorMinor = Get-CudaMajorMinor -CudaVersion $Version
$CudaRoot = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA\v$MajorMinor"
$NvccPath = Join-Path $CudaRoot "bin\nvcc.exe"
if (Test-Path $NvccPath) {
  Write-Host "CUDA Toolkit already installed: $CudaRoot"
  Add-CurrentAndGitHubPath -CudaRoot $CudaRoot
  & $NvccPath --version
  return
}

if (-not $InstallerUrl) {
  $InstallerUrl = "https://developer.download.nvidia.com/compute/cuda/$Version/network_installers/cuda_${Version}_windows_network.exe"
}
if (-not $DownloadDir) {
  $DownloadDir = if ($env:RUNNER_TEMP) {
    Join-Path $env:RUNNER_TEMP "metalfish-cuda"
  } else {
    Join-Path ([System.IO.Path]::GetTempPath()) "metalfish-cuda"
  }
}
if ($Components.Count -eq 0) {
  $Components = @(
    "nvcc_$MajorMinor",
    "cudart_$MajorMinor",
    "cublas_$MajorMinor",
    "cublas_dev_$MajorMinor",
    "thrust_$MajorMinor",
    "visual_studio_integration_$MajorMinor"
  )
}

New-Item -ItemType Directory -Force $DownloadDir | Out-Null
$InstallerPath = Join-Path $DownloadDir "cuda_${Version}_windows_network.exe"
Invoke-DownloadWithRetry -Uri $InstallerUrl -OutFile $InstallerPath

$InstallerArgs = @("-s", "-n") + $Components
Write-Host "Installing CUDA Toolkit $Version components: $($Components -join ', ')"
$Process = Start-Process -FilePath $InstallerPath -ArgumentList $InstallerArgs -Wait -PassThru
if ($Process.ExitCode -notin @(0, 3010)) {
  throw "CUDA Toolkit installer failed with exit code $($Process.ExitCode)"
}
if (-not (Test-Path $NvccPath)) {
  throw "CUDA Toolkit installer completed but nvcc.exe was not found at $NvccPath"
}

Add-CurrentAndGitHubPath -CudaRoot $CudaRoot
& $NvccPath --version
if (-not (Get-ChildItem -Path (Join-Path $CudaRoot "lib\x64") -Filter "cublas*.lib" -ErrorAction SilentlyContinue)) {
  throw "CUDA Toolkit install did not provide cuBLAS import libraries"
}
Write-Host "CUDA Toolkit ready: $CudaRoot"
