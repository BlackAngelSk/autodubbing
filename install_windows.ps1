param(
    [switch]$SkipWingetPackages
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-WingetPackage {
    param(
        [string]$Id,
        [string]$Label
    )

    if (-not (Test-Command "winget")) {
        Write-Warning "winget is not installed, so $Label could not be installed automatically."
        return
    }

    Write-Step "Installing $Label"
    winget install --id $Id -e --accept-package-agreements --accept-source-agreements --silent --disable-interactivity | Out-Host
}

function Invoke-SystemPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

    if (Test-Command "py") {
        & py -3 @Args
        return
    }

    if (Test-Command "python") {
        & python @Args
        return
    }

    throw "Python 3.10 or newer is required but was not found on PATH."
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Step "Preparing Windows environment for Auto Dubbing"

if (-not $SkipWingetPackages) {
    if (-not (Test-Command "py") -and -not (Test-Command "python")) {
        Install-WingetPackage -Id "Python.Python.3.11" -Label "Python 3.11"
        $env:Path += ";" + (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311")
        $env:Path += ";" + (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\Scripts")
    }

    if (-not (Test-Command "ffmpeg")) {
        Install-WingetPackage -Id "Gyan.FFmpeg" -Label "FFmpeg"
        $env:Path += ";" + (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links")
        $env:Path += ";C:\ffmpeg\bin"
    }

    Install-WingetPackage -Id "Microsoft.VCRedist.2015+.x64" -Label "Microsoft VC++ Runtime"
}
else {
    Write-Host "Skipping winget installs because -SkipWingetPackages was used." -ForegroundColor Yellow
}

if (-not (Test-Command "py") -and -not (Test-Command "python")) {
    throw "Python is still not available. Install Python 3.10+ and run this script again."
}

$venvPath = Join-Path $RepoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment"
    Invoke-SystemPython -m venv $venvPath
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed: $venvPython was not created."
}

Write-Step "Upgrading pip tools"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip tools."
}

Write-Step "Installing Python requirements"
& $venvPython -m pip install --upgrade --force-reinstall "numpy==1.26.4"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Windows-compatible NumPy build."
}
& $venvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Python requirements."
}

Write-Step "Verifying runtime imports"
$importCheckCode = @(
    'import sys'
    'try:'
    '    import numpy'
    '    import gradio'
    '    import deep_translator'
    '    import faster_whisper'
    '    import gtts'
    '    import pydub'
    '    import tqdm'
    '    import edge_tts'
    '    import yt_dlp'
    'except Exception as exc:'
    '    print("Import check failed:", exc)'
    '    sys.exit(1)'
    'print("NumPy", numpy.__version__, "loaded successfully.")'
    'print("Python packages look good.")'
) -join [Environment]::NewLine
& $venvPython -c $importCheckCode
if ($LASTEXITCODE -ne 0) {
    throw "Installed packages failed to import. If needed, delete .venv and run install_windows.bat again."
}

if (-not (Test-Command "ffmpeg")) {
    Write-Warning "ffmpeg is still not on PATH. Install it manually if video processing fails."
}

Write-Host ""
Write-Host "Installation finished successfully." -ForegroundColor Green
Write-Host "Run .\\run_ui_windows.bat to start the web UI." -ForegroundColor Green
