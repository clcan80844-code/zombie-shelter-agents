@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo Starting Zombie Shelter Agents V1.0...
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [ERROR] The game failed to start. Install dependencies with:
    echo python -m pip install -r requirements.txt
    pause
)
