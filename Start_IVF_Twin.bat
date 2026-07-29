@echo off
title IVF Digital Twin v6.2
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Run INSTALL.bat first
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo.
echo  Starting IVF Digital Twin...
echo  Open browser: http://localhost:8501
echo  To stop: press Ctrl+C
echo.
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"
python -m streamlit run app.py --server.port=8501 --server.headless=true --server.address=localhost --browser.gatherUsageStats=false
pause
