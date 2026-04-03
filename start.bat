@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First-time setup not found. Running installer...
    call "%~dp0install_windows.bat"
    if errorlevel 1 (
        echo.
        echo Setup did not complete successfully.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import numpy, gradio" >nul 2>&1
if errorlevel 1 (
    echo Detected a broken or incompatible Python package install.
    echo Running repair installer...
    call "%~dp0install_windows.bat"
    if errorlevel 1 (
        echo.
        echo Repair did not complete successfully.
        pause
        exit /b 1
    )
)

echo Starting Auto Dubbing UI...
".venv\Scripts\python.exe" ui.py
pause
