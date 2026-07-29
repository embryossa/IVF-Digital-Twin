@echo off
title Repair IVF Digital Twin GNN Environment
color 0E
cd /d "%~dp0"

echo.
echo  ================================================
echo   IVF Digital Twin - Repair GNN Environment
echo  ================================================
echo.
echo  This will repair the LOCAL .venv only.
echo  System-wide Python packages will not be changed.
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] .venv was not found.
    echo  Run INSTALL.bat first from this folder.
    pause
    exit /b 1
)

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
echo  Python: %PYTHON_EXE%
echo.

echo [1/5] Upgrading pip tooling...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel --timeout 120
if errorlevel 1 goto :fail

echo.
echo [2/5] Removing incompatible torch / PyG / HF packages from .venv...
"%PYTHON_EXE%" -m pip uninstall -y torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv transformers accelerate

echo.
echo [3/5] Installing PyTorch CPU 2.5.1...
"%PYTHON_EXE%" -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu --timeout 600
if errorlevel 1 goto :fail

echo.
echo [4/5] Installing PyTorch Geometric wheels for torch 2.5.1+cpu...
"%PYTHON_EXE%" -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cpu.html --timeout 600
if errorlevel 1 goto :fail
"%PYTHON_EXE%" -m pip install torch-geometric --timeout 300
if errorlevel 1 goto :fail

echo.
echo [5/5] Verifying GNN model load...
"%PYTHON_EXE%" -c "import sys, os; sys.path.insert(0, os.path.abspath('src')); import torch, torch_geometric; from gnn_predictor import load_gnn_model; b=load_gnn_model(os.getcwd()); print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__); print('GNN available:', b.get('available')); print('GNN error:', b.get('error', ''))"
if errorlevel 1 goto :fail

echo.
echo  ================================================
echo   REPAIR COMPLETE
echo  ================================================
echo  Start the app with Start_IVF_Twin.bat.
echo.
pause
exit /b 0

:fail
echo.
echo  [ERROR] Repair failed. Check the message above.
echo  Most common cause: no internet connection or blocked pip download.
echo.
pause
exit /b 1
