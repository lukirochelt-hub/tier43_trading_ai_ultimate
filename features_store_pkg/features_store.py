# features_store.py — Minimaler Loader für Optuna-Tests

from pathlib import Path
import numpy as np
import pandas as pd

def load_ohlcv(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    Lädt OHLCV-Daten. Wenn data/train.parquet existiert -> laden,
    sonst Dummy-Daten erzeugen.
    """
    p = Path("data/train.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        # Falls dein Parquet wie im Dummy-Skript andere Spaltennamen hat, hier sicherstellen:
        need = {"open","high","low","close","volume"}
        if not need.issubset(set(map(str.lower, df.columns))):
            # Fallback auf simple Random-Daten mit den benötigten Spalten
            n = len(df)
            df = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
                "open": np.random.rand(n),
                "high": np.random.rand(n),
                "low": np.random.rand(n),
                "close": np.random.rand(n),
                "volume": np.random.rand(n),
            })
    else:
        n = 1000
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": np.random.rand(n),
            "high": np.random.rand(n),
            "low": np.random.rand(n),
            "close": np.random.rand(n),
            "volume": np.random.rand(n),
        })
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    return df

def build_features(df: pd.DataFrame, params=None) -> pd.DataFrame:
    """
    Erzeugt einfache Features, die dein RegimeLearner/Backtester versteht:
    - return_1h, volatility, momentum
    """
    df = df.copy()
    df["return_1h"] = df["close"].pct_change().fillna(0.0)
    df["volatility"] = df["return_1h"].rolling(10).std().fillna(0.0)
    df["momentum"] = (df["close"] - df["close"].shift(10)).fillna(0.0)
    return df
