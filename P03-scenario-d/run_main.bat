@echo off
setlocal
call "%~dp0_launch.bat" run.py --mode main --workers 12 %*
set "D_RUN_EXIT=%errorlevel%"
echo.
echo Exit code %D_RUN_EXIT%
pause
exit /b %D_RUN_EXIT%
