@echo off
title IVF Digital Twin v6.2 - Install
color 0A
cd /d "%~dp0"

echo.
echo  ================================================
echo   IVF Digital Twin v6.2 - Installation
echo  ================================================
echo.
echo  Internet connection required (first time only).
echo  Steps: 9 (incl. PyTorch Geometric for Graph model)
echo.
echo  Press any key to start...
pause >nul

:: Check Python
echo.
echo [1/8] Checking Python...
python --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo.
    echo  ERROR: Python not found!
    echo  Install Python 3.11: https://www.python.org/downloads/
    echo  IMPORTANT: check "Add Python to PATH"
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo  OK: Python %PYTHON_VER%

:: Create virtual environment
echo.
echo [2/8] Creating virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo  Already exists - skipping
) else (
    python -m venv .venv
    if %errorlevel% NEQ 0 (
        echo  ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo  OK
)

call .venv\Scripts\activate.bat
set VP=.venv\Scripts\python.exe

:: Upgrade pip
echo.
echo [3/8] Upgrading pip...
%VP% -m pip install --upgrade pip --quiet --timeout 60
echo  OK

:: Core packages one by one
echo.
echo [4/8] Installing core packages (one by one)...

echo  - numpy...
%VP% -m pip install numpy --quiet --timeout 120
echo  - scipy...
%VP% -m pip install scipy --quiet --timeout 120
echo  - pandas...
%VP% -m pip install pandas --quiet --timeout 120
echo  - plotly...
%VP% -m pip install plotly --quiet --timeout 120
echo  - streamlit...
%VP% -m pip install streamlit --quiet --timeout 180
echo  - matplotlib...
%VP% -m pip install matplotlib --quiet --timeout 120
echo  - cryptography...
%VP% -m pip install cryptography --quiet --timeout 120
echo  - reportlab...
%VP% -m pip install reportlab --quiet --timeout 120
echo  - pdfkit...
%VP% -m pip install pdfkit --quiet --timeout 120
echo  - kaleido...
%VP% -m pip install kaleido --quiet --timeout 120
echo  OK: Core packages installed

:: PyTorch
echo.
echo [5/8] Installing PyTorch CPU (~200 MB, may take 10-15 min)...
echo  Downloading... please wait...
%VP% -c "import torch; torch.zeros(1)" >nul 2>&1
if %errorlevel% EQU 0 (
    echo  Already installed - skipping
) else (
    %VP% -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu --timeout 600
    if %errorlevel% NEQ 0 (
        echo  WARNING: PyTorch failed - neural network disabled
    ) else (
        echo  OK: PyTorch installed
    )
)

:: ML packages
echo.
echo [6/8] Installing ML packages...
echo  - lightgbm...
%VP% -m pip install lightgbm --quiet --timeout 120
echo  - scikit-learn...
%VP% -m pip install scikit-learn==1.5.0 --quiet --timeout 180
echo  - joblib...
%VP% -m pip install "joblib>=1.2.0,<2.0.0" --quiet --timeout 120
echo  OK

:: NN extras
echo.
echo [7/9] Installing NN extras...
echo  - pykan...
%VP% -m pip install pykan==0.2.8 --no-deps --quiet --timeout 120 2>nul
if %errorlevel% NEQ 0 %VP% -m pip install pykan --no-deps --quiet --timeout 120 2>nul

echo  - mambular...
%VP% -m pip install mambular==0.2.2 --quiet --timeout 120 2>nul
if %errorlevel% NEQ 0 %VP% -m pip install mambular --quiet --timeout 120 2>nul

echo  - crepes...
%VP% -m pip install crepes==0.8.0 --quiet --timeout 120 2>nul
if %errorlevel% NEQ 0 %VP% -m pip install crepes --quiet --timeout 120 2>nul

echo  - cloudpickle / model-unpickler...
%VP% -m pip install cloudpickle==1.5.0 model-unpickler --quiet --timeout 120 2>nul
echo  OK

:: PyTorch Geometric (строго под torch 2.5.1+cpu)
echo.
echo [8/9] Installing PyTorch Geometric (Graph Neural Network)...
echo  Checking if already installed...
%VP% -c "import torch_geometric; print('  Already installed - skipping')" 2>nul
if %errorlevel% EQU 0 goto :pyg_done

echo  Step 8a: torch-scatter, torch-sparse, torch-cluster, torch-spline-conv...
echo  (downloading wheels for torch 2.5.1+cpu, may take 5-10 min)
%VP% -m pip install ^
    torch-scatter ^
    torch-sparse ^
    torch-cluster ^
    torch-spline-conv ^
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html ^
    --timeout 600 --quiet
if %errorlevel% NEQ 0 (
    echo  WARNING: PyG dependencies failed - Graph model disabled
    goto :pyg_done
)

echo  Step 8b: torch-geometric...
%VP% -m pip install torch-geometric --quiet --timeout 300
if %errorlevel% NEQ 0 (
    echo  WARNING: torch-geometric failed - Graph model disabled
    goto :pyg_done
)

:pyg_done
echo  Verifying torch-geometric...
%VP% -c "import torch_geometric; print('  OK: torch-geometric', torch_geometric.__version__)" 2>nul || echo  SKIP: Graph model will be disabled

:: Launcher
echo.
echo [9/9] Creating launcher...
(
echo @echo off
echo title IVF Digital Twin v6.2
echo cd /d "%%~dp0"
echo if not exist ".venv\Scripts\python.exe" ^(
echo     echo ERROR: Run INSTALL.bat first
echo     pause
echo     exit /b 1
echo ^)
echo call .venv\Scripts\activate.bat
echo echo.
echo echo  Starting IVF Digital Twin...
echo echo  Open browser: http://localhost:8501
echo echo  To stop: press Ctrl+C
echo echo.
echo start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"
echo python -m streamlit run app.py --server.port=8501 --server.headless=true --server.address=localhost --browser.gatherUsageStats=false
echo pause
) > Start_IVF_Twin.bat
echo  OK

echo.
echo  ================================================
echo   INSTALLATION COMPLETE!
echo  ================================================
echo.
echo  Double-click Start_IVF_Twin.bat to run
echo  Support: embryossa@gmail.com
echo.
pause
