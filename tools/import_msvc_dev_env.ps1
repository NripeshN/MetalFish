[CmdletBinding()]
param(
  [ValidateSet("x64", "x86", "arm64", "arm")]
  [string]$Arch = "x64",
  [ValidateSet("x64", "x86", "arm64", "arm")]
  [string]$HostArch = "x64"
)

$ErrorActionPreference = "Stop"

if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
  Write-Host "MSVC developer environment already active: $((Get-Command cl.exe).Source)"
  return
}

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
  throw "vswhere.exe not found: $VsWhere"
}

$InstallPath = & $VsWhere `
  -latest `
  -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
  -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $InstallPath) {
  throw "Could not locate a Visual Studio installation with MSVC x64 tools"
}

$InstallPath = ($InstallPath | Select-Object -First 1).Trim()
$VsDevCmd = Join-Path $InstallPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
  throw "VsDevCmd.bat not found: $VsDevCmd"
}

$Command = "`"$VsDevCmd`" -arch=$Arch -host_arch=$HostArch >nul && set"
$EnvLines = & $env:ComSpec /s /c $Command
if ($LASTEXITCODE -ne 0) {
  throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
}

foreach ($Line in $EnvLines) {
  if ($Line -match "^([^=]+)=(.*)$") {
    Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
  }
}

$Cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $Cl) {
  throw "MSVC developer environment did not expose cl.exe"
}
Write-Host "MSVC developer environment imported: $($Cl.Source)"
