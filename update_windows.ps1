$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "Updating Auto Dubbing project..."

if (Get-Command git -ErrorAction SilentlyContinue) {
    $insideRepo = $false
    try {
        git rev-parse --is-inside-work-tree *> $null
        if ($LASTEXITCODE -eq 0) {
            $insideRepo = $true
        }
    }
    catch {
        $insideRepo = $false
    }

    if ($insideRepo) {
        Write-Host "Fetching latest git changes..."
        git fetch --all --prune | Out-Host

        git symbolic-ref --quiet --short HEAD *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Pulling latest commits..."
            git pull --ff-only | Out-Host
        }
        else {
            Write-Host "Detached HEAD detected; skipping git pull."
        }
    }
    else {
        Write-Host "Git repository not detected; skipping code update."
    }
}
else {
    Write-Host "git not found; skipping code update."
}

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment missing. Running installer..."
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "install_windows.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create/update virtual environment via installer."
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed: $venvPython was not found."
}

Write-Host "Updating Python packaging tools..."
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip/setuptools/wheel."
}

Write-Host "Updating Python dependencies..."
& $venvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements.txt."
}

Write-Host "Update complete." -ForegroundColor Green
