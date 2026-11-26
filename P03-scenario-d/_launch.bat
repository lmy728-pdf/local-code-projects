@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "D_RUN_PYTHON=python"
if exist "D:\Anoconda\envs\environment2\python.exe" set "D_RUN_PYTHON=D:\Anoconda\envs\environment2\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "D_RUN_PYTHON=%~dp0.venv\Scripts\python.exe"
"%D_RUN_PYTHON%" -B %*
exit /b %errorlevel%
