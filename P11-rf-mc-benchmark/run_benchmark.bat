@echo off
setlocal
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" setup_env.py
if errorlevel 1 exit /b 1
if "%~1"=="" (
    "%PYTHON%" benchmark.py --n 1000 --reps 40 --workers 12 --trees 500 --run-id cpu_compare
) else (
    "%PYTHON%" benchmark.py %*
)
exit /b %errorlevel%
