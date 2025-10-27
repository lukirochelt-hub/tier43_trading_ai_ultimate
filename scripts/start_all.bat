@echo off
title 🚀 Start TIA 4.3+ Trading AI
echo.
echo ==============================================
echo   🧠  Starting Tier 4.3+ Trading AI System...
echo ==============================================
echo.

:: === 1. Virtuelle Umgebung aktivieren (falls du eine hast)
:: Falls dein venv anders heißt, anpassen:
call "%~dp0venv\Scripts\activate.bat"

:: === 2. In Projektordner wechseln (ANPASSEN falls anderer Pfad)
cd /d "%~dp0"

:: === 3. App starten
start "TIA App" cmd /k "uvicorn app_trading_ai_secure_plus:app --host 0.0.0.0 --port 8000"

:: === 4. Prometheus + Grafana (optional)
if exist "docker\docker-compose.grafana.yml" (
    echo 🐳 Starting Prometheus & Grafana...
    docker compose -f docker\docker-compose.grafana.yml up -d
)

echo.
echo ✅  All systems launched!
pause
