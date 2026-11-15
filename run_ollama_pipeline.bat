@echo off
title LifeMate - Ollama AI Inference
cd /d "%~dp0"
set PYTHONPATH=src;train;public_test
echo.
echo ============================================================
echo   LifeMate - Running Real AI Multi-Agent via Ollama (1.5B)
echo ============================================================
echo.
if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

%PYTHON_BIN% main.py --split public_test --mode ollama --model qwen2.5:1.5b --limit 3 --output submission.jsonl
echo.
echo Kiem tra hop le voi Validator BTC:
%PYTHON_BIN% -m zaic_tools --split public_test --root . --validate submission.jsonl
echo.
echo === DONE ===
pause

