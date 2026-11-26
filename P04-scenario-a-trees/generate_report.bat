@echo off
cd /d "%~dp0"
set "mode=%~1"
if "%mode%"=="" set "mode=main"
python -B -u report.py --mode "%mode%"
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
