@echo off
setlocal
cd /d "%~dp0"

echo Updating Auto Dubbing for Windows...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_windows.ps1"
if errorlevel 1 (
    echo.
    echo Update failed.
    pause
    exit /b 1
)

echo.
echo Update complete.
pause
