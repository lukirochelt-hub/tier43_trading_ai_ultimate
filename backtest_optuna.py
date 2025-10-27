#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tier 4.3+ — backtest_optuna.py
Nimmt best_params_{symbol}_{timeframe}.json und führt einen vollständigen Backtest aus.
Ausgaben:
- ./results/backtests/{symbol}_{timeframe}_{timestamp}/
  - equity.png
  - backtest_report.json
  - trades.csv (falls von ml_core geliefert)
"""

import os
import json
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Any

BASE = Path(__file__).resolve().parent
OUTROOT = BASE / "results" / "backtests"
OUTROOT.mkdir(parents=True, exist_ok=True)

# Logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("backtest_optuna")


# Imports
def _safe_imports():
    try:
        import features_store
    except Exception as e:
        features_store = None
        log.warning("features_store Importfehler: %s", e)
    try:
        import ml_core
    except Exception as e:
        ml_core = None
        log.warning("ml_core Importfehler: %s", e)
    return features_store, ml_core


features_store, ml_core = _safe_imports()


def _load_data(symbol: str, timeframe: str, start: str, end: str):
    if features_store is None or not hasattr(features_store, "load_ohlcv"):
        raise RuntimeError("features_store.load_ohlcv fehlt.")
    df = features_store.load_ohlcv(
        symbol=symbol, timeframe=timeframe, start=start, end=end
    )
    if hasattr(features_store, "build_features"):
        try:
            df = features_store.build_features(df, params=None)
        except Exception as e:
            log.warning("build_features fehlgeschlagen (skip): %s", e)
    return df


def _full_backtest(df, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Erwartet:
      - ml_core.run_backtest(df, params) -> dict(metrics)
        optional: metrics['equity_curve'] (list/series), metrics['trades'] (DataFrame/records)
    Fallback:
      - ml_core.generate_signals + ml_core.backtest
    """
    if ml_core is None:
        raise RuntimeError("ml_core ist nicht verfügbar.")

    if hasattr(ml_core, "run_backtest"):
        return ml_core.run_backtest(df=df, params=params)

    if hasattr(ml_core, "generate_signals") and hasattr(ml_core, "backtest"):
        sig = ml_core.generate_signals(df=df, params=params)
        return ml_core.backtest(df=df, signals=sig, params=params)

    raise RuntimeError(
        "ml_core Schnittstelle fehlt (run_backtest oder generate_signals/backtest)."
    )


def _plot_equity(equity, out_png: Path):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4.5))
        plt.plot(equity)
        plt.title("Equity Curve")
        plt.xlabel("Trade / Index")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close()
    except Exception as e:
        log.warning("Equity-Plot fehlgeschlagen: %s", e)


def run_backtest(best_params_path: Path, start: str, end: str) -> Path:
    params = json.loads(Path(best_params_path).read_text(encoding="utf-8"))
    meta = params.get("_meta", {})
    symbol = meta.get("symbol", os.getenv("BACKTEST_SYMBOL", "BTCUSDT"))
    timeframe = meta.get("timeframe", os.getenv("BACKTEST_TIMEFRAME", "15m"))

    df = _load_data(symbol, timeframe, start, end)
    metrics = _full_backtest(df, params)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUTROOT / f"{symbol}_{timeframe}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Equity speichern
    equity = metrics.get("equity_curve")
    if equity is not None:
        _plot_equity(equity, outdir / "equity.png")

    # Trades speichern, wenn vorhanden
    trades = metrics.get("trades")
    if trades is not None:
        try:
            import pandas as pd

            if hasattr(trades, "to_csv"):
                trades.to_csv(outdir / "trades.csv", index=False)
            else:
                pd.DataFrame(trades).to_csv(outdir / "trades.csv", index=False)
        except Exception as e:
            log.warning("Trades-Export übersprungen: %s", e)

    # Report JSON
    (outdir / "backtest_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Backtest-Report gespeichert: %s", outdir / "backtest_report.json")
    return outdir


def main():
    parser = argparse.ArgumentParser(
        description="Tier 4.3+ Backtest mit Optuna-Bestparams"
    )
    parser.add_argument(
        "--best", required=False, default=os.getenv("BEST_PARAMS_FILE", "")
    )
    parser.add_argument("--start", default=os.getenv("BACKTEST_START", "2022-01-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END", "2025-01-01"))
    args = parser.parse_args()

    best_path = args.best
    if not best_path:
        # Fallback: symbol/timeframe aus ENV
        sym = os.getenv("BACKTEST_SYMBOL", "BTCUSDT")
        tf = os.getenv("BACKTEST_TIMEFRAME", "15m")
        candidate = BASE / "models" / f"best_params_{sym}_{tf}.json"
        if not candidate.exists():
            raise FileNotFoundError(
                "Kein --best angegeben und Standarddatei nicht gefunden: "
                f"{candidate}. Setze --best=/pfad/zur/best_params.json"
            )
        best_path = str(candidate)

    outdir = run_backtest(Path(best_path), start=args.start, end=args.end)
    print(json.dumps({"ok": True, "outdir": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
