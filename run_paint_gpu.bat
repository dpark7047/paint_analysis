@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "GPU_PYTHON=%~dp0.gpu-venv\Scripts\python.exe"
if not exist "%GPU_PYTHON%" (
    echo GPU test environment is missing. See README.md for setup.
    pause
    exit /b 1
)
if not defined PAINT_CLASSIFICATION_COMPUTE set "PAINT_CLASSIFICATION_COMPUTE=auto"
set "PAINT_ANALYSIS_HOME=%~dp0.gpu-state"
set "PAINT_ANALYSIS_DISABLE_DEVELOPMENT_CACHE=1"
set "CUPY_CACHE_DIR=%~dp0.gpu-state\cupy"
set "NUMBA_CACHE_DIR=%~dp0.gpu-state\numba"
set "PYTHONPYCACHEPREFIX=%~dp0.gpu-state\pycache"
set "PAINT_CLASSIFICATION_LOG=%~dp0.gpu-state\classification-timing.jsonl"
echo Classification compute: %PAINT_CLASSIFICATION_COMPUTE%
echo Logs: %PAINT_CLASSIFICATION_LOG%
"%GPU_PYTHON%" -B paint_analysis_gui.py
if errorlevel 1 pause
