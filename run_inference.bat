@echo off
title LifeMate Inference Runner
cd /d "%~dp0"
set PYTHONPATH=src;train;public_test
echo.
echo ============================================================
echo   LifeMate - Voice Agent Inference Runner
echo ============================================================
echo.
if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

%PYTHON_BIN% main.py --split public_test --output submission.jsonl
echo.
echo Kiem tra schema file vua tao bang validator cua BTC:
%PYTHON_BIN% -m zaic_tools --split public_test --root . --validate submission.jsonl
echo.
echo === DONE ===
pause

