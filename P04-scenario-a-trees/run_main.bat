@echo off
cd /d "%~dp0"
set "learner=%~1"
if "%learner%"=="" set "learner=rf"
set "stage=%~2"
if "%stage%"=="" set "stage=both"
python -B -u run.py --mode main --learner "%learner%" --stage "%stage%" --workers 12
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
