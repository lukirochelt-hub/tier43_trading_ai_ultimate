@echo off
title 🛑 Stop TIA 4.3+ Trading AI
echo.
echo ==============================================
echo   🧱  Stopping Tier 4.3+ Trading AI System...
echo ==============================================
echo.

:: === 1. Python & Uvicorn killen
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM uvicorn.exe >nul 2>&1

:: === 2. Docker stoppen (falls läuft)
docker compose -f docker\docker-compose.grafana.yml down >nul 2>&1

echo.
echo ✅  All processes stopped.
pause
