@echo off
setlocal
call "%~dp0_launch.bat" run.py --mode smoke --workers 12 --resume %*
if errorlevel 1 goto done
call "%~dp0_launch.bat" report.py --mode smoke
:done
set "D_RUN_EXIT=%errorlevel%"
echo.
echo Exit code %D_RUN_EXIT%
pause
exit /b %D_RUN_EXIT%
