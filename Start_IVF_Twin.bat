@echo off
title IVF Digital Twin v6.2
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  ERROR: Application not installed!
    echo  Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo  ================================
echo   IVF Digital Twin v6.2
echo   Starting...
echo  ================================
echo.
echo  Address: http://localhost:8501
echo  To stop: press Ctrl+C
echo.

start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

python -m streamlit run app.py ^
    --server.port=8501 ^
    --server.headless=true ^
    --server.address=localhost ^
    --browser.gatherUsageStats=false

echo.
echo  Application stopped.
pause
