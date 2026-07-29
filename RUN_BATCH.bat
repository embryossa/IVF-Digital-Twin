@echo off
:: Remember the caller's console code page and restore it on every exit path.
:: dt_postprocess.py still runs under 1251 exactly as before -- only the state
:: left behind for the calling shell changes (it used to stay on 1251 and
:: garbled UTF-8 output in the parent terminal).
for /f "tokens=2 delims=:" %%a in ('chcp') do set "OLDCP=%%a"
set "OLDCP=%OLDCP: =%"
if not defined OLDCP set "OLDCP=65001"
:: DT_KEEP_CODEPAGE=1 is set by the pipeline (src/embryo/_launch.py). Its output
:: goes to a pipe, where the console code page is irrelevant, while chcp would
:: switch it for the whole terminal for the entire run. Interactive drag-and-drop
:: use does not set the variable and keeps the old behaviour.
if not defined DT_KEEP_CODEPAGE chcp 1251 > nul

echo.
echo  =====================================================
echo   IVF Digital Twin v6.2 -- Batch + Postprocess
echo  =====================================================
echo.

:: Input file: drag-and-drop, or first *.xlsx found in folder
set "INPUT_FILE=%~1"
if "%INPUT_FILE%"=="" (
    for %%f in (*.xlsx) do (
        if /i not "%%f"=="OPU_table.xlsx" (
            set "INPUT_FILE=%%f"
            goto :run
        )
    )
    echo ERROR: No Excel file found.
    echo Drag-and-drop the predictions xlsx onto RUN_BATCH.bat
    echo or: python dt_postprocess.py your_predictions.xlsx
    pause
    chcp %OLDCP% > nul
    exit /b 1
)

:run
echo  Input : %INPUT_FILE%
echo  Output: results\
echo.

python dt_postprocess.py "%INPUT_FILE%" results\ --clinic "Clinic"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR. Check output above.
    pause
    chcp %OLDCP% > nul
    exit /b 1
)

echo.
echo  =====================================================
echo   Done! Files in results\
echo.
echo   OPU_table_filled_*.xlsx       OPU table + DT + PRAI
echo   dt_analytics_ready_*.csv      ready for Datalore
echo   dt_analytics_with_outcome_*.csv   Preg 0/1 only
echo   dt_analytics_data\dt_predictions.csv  full DT log
echo  =====================================================
echo.
pause
chcp %OLDCP% > nul
