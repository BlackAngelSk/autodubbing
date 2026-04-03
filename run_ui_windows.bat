@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please run install_windows.bat first.
    pause
    exit /b 1
)

echo Starting Auto Dubbing UI...
".venv\Scripts\python.exe" ui.py
pause
