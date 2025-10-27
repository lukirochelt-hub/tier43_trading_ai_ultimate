# scripts/test_backtest.py
from __future__ import annotations

from typing import Dict, Any
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

# wir importieren direkt die Funktionen aus ml_core
from ml_core import backtest, run_backtest, generate_signals


def make_dummy_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 100.0 * (1.0 + 0.001 * np.cumsum(np.random.randn(n)))
    return pd.DataFrame({"close": close}, index=idx)


def main() -> None:
    # 1) Neutraler Test (alle Signale = 0)
    df = make_dummy_df(200)
    signals = pd.Series(0, index=df.index, dtype=int)
    params: Dict[str, Any] = {"fee_bps": 6, "slippage_bps": 2}
    metrics = backtest(df=df, signals=signals, params=params)
    print("Neutral backtest:", {k: v for k, v in metrics.items() if k != "equity"})

    # 2) Einfacher Strategy-Test via generate_signals/run_backtest
    df2 = make_dummy_df(400)
    strat_params: Dict[str, Any] = {
        "fee_bps": 6,
        "slippage_bps": 2,
        "ema_fast": 12,
        "ema_slow": 26,
        "bb_len": 20,
        "bb_mult": 2.0,
    }
    # optional direkt:
    metrics2 = run_backtest(df2, strat_params)
    print("Strategy backtest:", {k: v for k, v in metrics2.items() if k != "equity"})

    # oder: Signale explizit erzeugen und in backtest stecken
    sig2 = generate_signals(df2, strat_params)
    metrics3 = backtest(df=df2, signals=sig2, params=strat_params)
    print("Strategy backtest (explicit):", {k: v for k, v in metrics3.items() if k != "equity"})


if __name__ == "__main__":
    main()
