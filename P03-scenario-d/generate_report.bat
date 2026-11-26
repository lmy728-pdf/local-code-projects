@echo off
setlocal
call "%~dp0_launch.bat" report.py --mode main %*
set "D_RUN_EXIT=%errorlevel%"
echo.
echo Exit code %D_RUN_EXIT%
pause
exit /b %D_RUN_EXIT%
