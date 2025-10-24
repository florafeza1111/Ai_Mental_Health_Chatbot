@echo off
echo ================================================
echo AIMHSA - AI Mental Health Support Assistant
echo ================================================
echo.
echo Starting AIMHSA system...
echo Backend API: http://localhost:5057
echo Frontend: http://localhost:8000
echo.
echo Press Ctrl+C to stop both services
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
