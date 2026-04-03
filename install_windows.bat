@echo off
setlocal
cd /d "%~dp0"

echo Installing Auto Dubbing for Windows...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"
if errorlevel 1 (
    echo.
    echo Installation failed.
    pause
    exit /b 1
)

echo.
echo Installation complete.
echo Use run_ui_windows.bat to start the app.
pause
