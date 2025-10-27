#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tier 4.3+ — nightly_optuna.py
Einfacher Nightly-Runner für:
1) Optuna-Run (optuna_realr.run_optuna)
2) Backtest mit den gefundenen Bestparams

Standard: täglicher Durchlauf um 02:30 lokale Zeit.
Empfehlung: In Produktion via cron/systemd timer ausführen.
Optional: --daemon startet eine einfache Endlosschleife ohne externe Dependencies.
"""

import os
import time
import argparse
import datetime as dt
from pathlib import Path

from optuna_realr import run_optuna  # Importiert Funktions-API
from backtest_optuna import run_backtest

BASE = Path(__file__).resolve().parent


def run_once() -> None:
    # ENV / Defaults
    symbol = os.getenv("OPTUNA_SYMBOL", "BTCUSDT")
    timeframe = os.getenv("OPTUNA_TIMEFRAME", "15m")
    start = os.getenv("OPTUNA_START", "2023-01-01")
    end = os.getenv("OPTUNA_END", "2025-01-01")
    study = os.getenv("OPTUNA_STUDY_NAME", f"tier43_plus_{symbol}_{timeframe}")
    trials = int(os.getenv("OPTUNA_N_TRIALS", "120"))
    pruner = os.getenv("OPTUNA_PRUNER", "median")

    # 1) Optuna
    best, best_path = run_optuna(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        study_name=study,
        n_trials=trials,
        pruner=pruner,
    )

    # 2) Backtest mit erweitertem Zeitraum (optional via ENV)
    b_start = os.getenv("BACKTEST_START", "2022-01-01")
    b_end = os.getenv("BACKTEST_END", "2025-01-01")
    outdir = run_backtest(best_params_path=best_path, start=b_start, end=b_end)
    print({"ok": True, "best_file": str(best_path), "backtest_outdir": str(outdir)})


def _seconds_until(hour: int, minute: int, second: int = 0) -> int:
    now = dt.datetime.now()
    run_at = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if run_at <= now:
        run_at += dt.timedelta(days=1)
    return int((run_at - now).total_seconds())


def main():
    parser = argparse.ArgumentParser(description="Tier 4.3+ Nightly Optuna Runner")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Einfacher Endlos-Dienstmodus (ohne cron).",
    )
    parser.add_argument("--hour", type=int, default=int(os.getenv("NIGHTLY_HOUR", "2")))
    parser.add_argument(
        "--minute", type=int, default=int(os.getenv("NIGHTLY_MINUTE", "30"))
    )
    args = parser.parse_args()

    if not args.daemon:
        # Einmaliger Lauf (gut für cron/systemd)
        run_once()
        return

    # Einfacher Daemon-Loop
    while True:
        secs = _seconds_until(args.hour, args.minute, 0)
        print(f"[nightly_optuna] Nächster Lauf in {secs} Sek.")
        time.sleep(secs)
        try:
            run_once()
        except Exception as e:
            print(f"[nightly_optuna] Fehler im Lauf: {e}")
        # kleine Pause, um sofortige erneute Ausführung zu vermeiden
        time.sleep(5)


if __name__ == "__main__":
    main()
