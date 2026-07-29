@echo off
title Установка NN зависимостей
color 0D
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
    echo [ОШИБКА] Локальное окружение .venv не найдено.
    echo Запустите INSTALL.bat из этой папки.
    pause
    exit /b 1
)
echo Python: %PYTHON_EXE%
echo.

echo [1/6] torch CPU (без DLL проблем)...
"%PYTHON_EXE%" -c "import torch; torch.zeros(1)" >nul 2>&1
if errorlevel 1 (
    echo     Устанавливаем...
    "%PYTHON_EXE%" -m pip uninstall torch torchvision torchaudio -y -q 2>nul
    "%PYTHON_EXE%" -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
) else (
    echo     torch уже установлен OK
)

echo.
echo [2/6] joblib...
"%PYTHON_EXE%" -m pip install "joblib>=1.2.0,<2.0.0" -q
echo     OK

echo.
echo [3/6] pykan==0.2.8...
"%PYTHON_EXE%" -m pip install pykan==0.2.8 --no-deps -q
if errorlevel 1 (
    echo     Пробуем без версии...
    "%PYTHON_EXE%" -m pip install pykan --no-deps -q
)
echo     OK

echo.
echo [4/6] mambular...
"%PYTHON_EXE%" -m pip install mambular==0.2.2 -q 2>nul
if errorlevel 1 (
    echo     v0.2.2 недоступна, пробуем без версии...
    "%PYTHON_EXE%" -m pip install mambular -q
    if errorlevel 1 (
        echo     mambular не установился - это OK если FTTransformer.joblib
        echo     был создан без mambular (используется sklearn внутри)
    )
)

echo.
echo [5/6] crepes==0.8.0...
"%PYTHON_EXE%" -m pip install crepes==0.8.0 -q 2>nul
if errorlevel 1 (
    echo     Пробуем без версии...
    "%PYTHON_EXE%" -m pip install crepes -q
)
echo     OK

echo.
echo [6/6] cloudpickle, model-unpickler, keras-preprocessing...
"%PYTHON_EXE%" -m pip install cloudpickle==1.5.0 -q
"%PYTHON_EXE%" -m pip install model-unpickler -q
"%PYTHON_EXE%" -m pip install keras-preprocessing==1.1.2 -q 2>nul

echo.
echo ===== Результат =====
"%PYTHON_EXE%" diagnose_check.py

echo.
echo Готово! Запустите launch_windows.bat
pause
