@echo off
title IVF Digital Twin v6.2
color 0A

echo.
echo  =============================================
echo   IVF DIGITAL TWIN v6.2
echo   Sergeev et al., 2025
echo  =============================================
echo.

cd /d "%~dp0"

REM Всегда запускаем из локального окружения проекта.
REM Системный Python может содержать несовместимые torch/torch-geometric.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
    echo  [ОШИБКА] Локальное окружение .venv не найдено!
    echo  Запустите INSTALL.bat из этой папки.
    pause
    exit /b 1
)

echo  Python: %PYTHON_EXE%
echo.

REM Устанавливаем ТОЛЬКО базовые зависимости для запуска интерфейса
echo  Проверка базовых зависимостей...
"%PYTHON_EXE%" -c "import streamlit, plotly, numpy, scipy, pandas" >nul 2>&1
if errorlevel 1 (
    echo  Устанавливаем базовые пакеты, это займет 1-2 минуты...
    "%PYTHON_EXE%" -m pip install streamlit plotly scipy pandas numpy -q
)

REM Проверяем streamlit
"%PYTHON_EXE%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ОШИБКА] streamlit не установлен.
    echo  Запустите вручную: %PYTHON_EXE% -m pip install streamlit
    pause
    exit /b 1
)

echo  Запуск приложения на http://localhost:8501
echo  Нейросетевой модуль: при наличии torch и файлов моделей
echo.
echo  Для закрытия -- закройте это окно.
echo.

REM Streamlit on Windows does not always open the browser reliably.
REM Open the app URL explicitly a few seconds after the server starts.
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"

"%PYTHON_EXE%" -m streamlit run app.py ^
    --server.headless true ^
    --server.port 8501 ^
    --browser.gatherUsageStats false ^
    --theme.primaryColor "#1B4F72" ^
    --theme.backgroundColor "#f8fafc" ^
    --theme.secondaryBackgroundColor "#e8f4f8" ^
    --theme.textColor "#1a3a4a"

pause
