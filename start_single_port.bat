@echo off
echo ================================================
echo AIMHSA - Single Port Launcher
echo ================================================
echo.
echo Starting AIMHSA on single port...
echo Frontend & API: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Run the unified launcher
python run_aimhsa.py

pause
