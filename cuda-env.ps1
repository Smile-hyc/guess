$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio 2022 with Desktop development with C++."
}

$vsPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsPath) {
    throw "Visual Studio 2022 C++ build tools were not found."
}

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$environment = cmd.exe /d /s /c "`"$vcvars`" >nul && set"
foreach ($line in $environment) {
    if ($line -match "^([^=][^=]*)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$cudaPath = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if (-not $cudaPath) {
    throw "CUDA_PATH is not set. Install the NVIDIA CUDA Toolkit."
}

$env:CUDA_PATH = $cudaPath
$cudaBin = Join-Path $cudaPath "bin"
if (($env:Path -split ";") -notcontains $cudaBin) {
    $env:Path = "$cudaBin;$env:Path"
}

Write-Host "CUDA/MSVC environment ready:"
Write-Host "  CUDA_PATH=$env:CUDA_PATH"
Write-Host "  nvcc=$((Get-Command nvcc).Source)"
Write-Host "  cl=$((Get-Command cl).Source)"
