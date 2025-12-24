@echo off
setlocal

echo ============================================
echo   Tajikistan RAG Server Launcher
echo ============================================
echo.

REM Resolve repo directory to WSL path
set "SCRIPT_DIR=%~dp0"
for /f "usebackq delims=" %%i in (`wsl wslpath -a "%SCRIPT_DIR%"`) do set "WSL_DIR=%%i"

REM Start WSL server script in a new window
start "RAG Server" wsl -e bash -lc "cd '%WSL_DIR%' && bash start_server.sh"

echo.
echo If this is the first run, model downloads can take time.
echo A new window was opened for the server logs.
pause
