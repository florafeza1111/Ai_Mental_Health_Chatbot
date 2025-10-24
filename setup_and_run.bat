@echo off
echo ================================================
echo AIMHSA Setup and Run with OpenAI Client
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo Installing requirements...
python -m pip install -r requirements_ollama.txt

echo.
echo Starting AIMHSA...
python run_aimhsa.py

pause
