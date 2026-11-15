@echo off
title Zalo AI LifeMate Scraper
cd /d "c:\merged_partition_content\zaic\train+public_test"
echo.
echo ============================================================
echo   Zalo AI Challenge - LifeMate Portal Scraper
echo ============================================================
echo.
python scripts\scrape_lifemate.py
echo.
echo === DONE ===
pause

