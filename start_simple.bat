@echo off
echo ================================================
echo AIMHSA Simple Launcher
echo ================================================
echo.
echo Starting simplified AIMHSA system...
echo Backend API: http://localhost:5057
echo Frontend: http://localhost:8000
echo.

REM Check if app.pybcp.py exists
if not exist "app.pybcp.py" (
    echo ERROR: app.pybcp.py not found
    echo Please ensure app.pybcp.py exists in this directory
    pause
    exit /b 1
)

REM Temporarily rename app.pybcp.py to app.py for import
if exist "app.py" (
    echo Backing up existing app.py...
    move "app.py" "app_backup.py"
)

echo Using simplified version (app.pybcp.py)...
copy "app.pybcp.py" "app.py"

REM Run the simple launcher
python run_simple_aimhsa.py

REM Restore original app.py if it existed
if exist "app_backup.py" (
    echo Restoring original app.py...
    del "app.py"
    move "app_backup.py" "app.py"
)

pause
