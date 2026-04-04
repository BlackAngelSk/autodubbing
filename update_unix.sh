#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Updating Auto Dubbing project..."

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Fetching latest git changes..."
  git fetch --all --prune

  if git symbolic-ref --quiet --short HEAD >/dev/null 2>&1; then
    echo "Pulling latest commits..."
    git pull --ff-only
  else
    echo "Detached HEAD detected; skipping git pull."
  fi
else
  echo "Git repository not detected or git not installed; skipping code update."
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

echo "Updating Python packaging tools..."
.venv/bin/python -m pip install --upgrade pip setuptools wheel

echo "Updating Python dependencies..."
.venv/bin/python -m pip install -r requirements.txt

echo "Update complete."
