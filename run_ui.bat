@echo off
title LifeMate Trajectory Inspector UI
cd /d "%~dp0"
set PYTHONPATH=src;train;public_test

echo.
echo ============================================================
echo   LifeMate - Voice Agent Trajectory Inspector Dashboard
echo ============================================================
echo.

REM Kiem tra va khoi dong Ollama server neu chua chay
curl -s http://localhost:11434/ >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] Khoi dong Ollama Server ngam...
    start /b "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 3 >nul
) else (
    echo [1/3] Ollama Server da san sang!
)

echo [2/3] Khoi dong LifeMate Dashboard tai http://localhost:8501 ...
echo [3/3] Tu dong mo trinh duyet...

REM Mo trinh duyet
start "" "http://localhost:8501"

echo.
echo ============================================================
echo   Server dang chay! Nhan Ctrl+C de dung server.
echo ============================================================
echo.

if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

%PYTHON_BIN% src\lifemate\ui\app.py
pause
