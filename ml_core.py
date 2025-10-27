# ml_core.py
# Tier 4.3+ — Core ML (entkoppelt) + robuster Backtest (Windows-safe, mypy-clean)
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# =========================
# Warnungen dämpfen
# =========================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

# =========================
# Joblib (optional)
# =========================
try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

MODEL_PATH = os.path.join(os.getenv("MODELS_PROD_DIR", "./models/prod"), "logreg.joblib")

# =====================================================
# Save / Load
# =====================================================
def save_model(clf: Any, path: str = MODEL_PATH) -> None:
    """Speichert ein Modell über joblib (falls vorhanden)."""
    if not HAVE_JOBLIB:
        print("[ml_core] joblib missing; skip save.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)
    print(f"[ml_core] Model saved -> {path}")


def load_model(path: str = MODEL_PATH) -> Optional[Any]:
    """Lädt ein Modell, falls vorhanden."""
    if not HAVE_JOBLIB or not os.path.exists(path):
        return None
    try:
        clf = joblib.load(path)
        print(f"[ml_core] Loaded model -> {path}")
        return clf
    except Exception as e:
        print(f"[ml_core] Load error: {e}")
        return None


# =====================================================
# Predict proba (robust, ohne externe Projektimporte)
# =====================================================
def predict_proba_single(
    features: Dict[str, Any],
    clf: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Gibt die Wahrscheinlichkeiten für 'up'/'down'/'label' zurück.
    Nutzt ein gespeichertes Modell (falls clf None ist), sonst Fallback-Defaults.
    """
    up: float = 0.5
    down: float = 0.5

    if clf is None:
        clf = load_model()
    if clf is None:
        return {"up": up, "down": down, "label": 0.0}

    X = pd.DataFrame([features])
    try:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            classes = list(getattr(clf, "classes_", [-1, 0, 1]))
            p_map = {int(c): float(p) for c, p in zip(classes, proba)}
            up = p_map.get(1, 0.0)
            down = p_map.get(-1, 0.0)
        elif hasattr(clf, "predict"):
            pred = clf.predict(X)[0]
            up = 1.0 if pred == 1 else 0.0
            down = 1.0 if pred == -1 else 0.0
    except Exception:
        # Fallback auf Defaultwerte
        pass

    label: float = 1.0 if up > 0.55 else (-1.0 if down > 0.55 else 0.0)
    return {"up": up, "down": down, "label": label}


# =====================================================
# Robuster Backtest (ohne Projekt-Imports)
# =====================================================
def _infer_periods_per_year(index: pd.Index) -> float:
    """
    Heuristik: Ableitung der jährlichen Periodenzahl aus dem Index.
    Fallback: 252*78 (≈ Minutendaten über US-Handelstage).
    """
    try:
        if isinstance(index, pd.DatetimeIndex):
            diffs = pd.Series(index).diff()
            dt = diffs.median()
            if isinstance(dt, pd.Timedelta) and not pd.isna(dt):
                mins = float(dt / pd.Timedelta(minutes=1))
                if mins > 0.0:
                    bars_per_day = (24.0 * 60.0) / mins
                    return 365.0 * bars_per_day
    except Exception:
        pass
    # Fallback
    return 252.0 * 78.0


def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Einfache, stabile Backtest-Implementierung:
    - 1-Bar Lookahead vermeiden (Signal wird 1 Bar verzögert)
    - Gebühren & Slippage in bps
    - Sharpe, Profit Factor, MDD, Winrate, Netto-Profit, Trades, Equity-Kurve
    """
    # Parameter
    fee_bps = float(params.get("fee_bps", 6.0))
    slippage_bps = float(params.get("slippage_bps", 2.0))

    # Grundgrößen
    close = df["close"].astype(float)
    rets = close.pct_change().fillna(0.0)

    sig = pd.Series(signals, index=df.index).fillna(0).astype(int)
    sig = sig.shift(1).fillna(0).astype(int)  # Lookahead-Fix

    # Turnover/Wechsel und Kosten
    turn = (sig != sig.shift(1)).astype(float).fillna(0.0)
    cost_per_turn = (fee_bps + slippage_bps) / 1e4

    gross = (sig.astype(float) * rets).fillna(0.0)
    costs = (turn * cost_per_turn).fillna(0.0)

    # Log-Returns explizit als Series erzeugen (mypy-safe)
    clipped = gross.clip(lower=-0.9999)
    loggross = pd.Series(np.log1p(clipped.to_numpy()), index=df.index)
    log_ret = (loggross - costs).astype(float)
    log_ret = log_ret.where(np.isfinite(log_ret), 0.0)

    # Equity & PnL bleiben Series
    equity: pd.Series = np.exp(log_ret.cumsum())
    pnl: pd.Series = np.expm1(log_ret)

    # Kennzahlen
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else 0.0

    peak = equity.cummax()
    dd = (equity / peak - 1.0).fillna(0.0)
    max_dd_pct = float(abs(dd.min() * 100.0))

    pos_bars = int(float((sig != 0).sum()))
    wins = int(float(((pnl > 0.0) & (sig != 0)).sum()))
    win_rate = float(wins) / float(pos_bars) * 100.0 if pos_bars > 0 else 0.0

    periods = _infer_periods_per_year(df.index)
    mu = float(pnl.mean())
    sigma = float(pnl.std(ddof=0))
    sharpe = float(mu / (sigma + 1e-12) * np.sqrt(periods))

    net_profit = float((equity.iloc[-1] - 1.0) * 100.0)
    trades = int(float(turn.sum()))

    return {
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd_pct,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "trades": trades,
        "equity": equity,  # pd.Series
    }


# =====================================================
# Einfache Signal-Logik + Convenience-Wrapper
# =====================================================
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def generate_signals(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Einfache EMA-Kreuz + Bollinger-Breite → neutraler Zustand bei enger Range.
    Gibt Series[int] ∈ {-1, 0, 1}.
    """
    ema_fast = int(params.get("ema_fast", 12))
    ema_slow = int(params.get("ema_slow", 26))
    bb_len = int(params.get("bb_len", 20))
    bb_mult = float(params.get("bb_mult", 2.0))

    close = df["close"].astype(float)
    fast, slow = _ema(close, ema_fast), _ema(close, ema_slow)
    base = np.where(fast > slow, 1, -1)

    std = close.rolling(bb_len).std().fillna(0.0)
    width = (2.0 * bb_mult * std / close).fillna(0.0)

    # Neutral in engen Phasen
    low_width = float(width.quantile(0.2)) if len(width) > 0 else 0.0
    neutral = (width < low_width).astype(int)

    sig = np.where(neutral == 1, 0, base)
    return pd.Series(sig, index=df.index, dtype=int)


def run_backtest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Erzeuge Signale und führe den Backtest aus."""
    sig = generate_signals(df, params)
    return backtest(df, sig, params)


# =====================================================
# Öffentliche Exporte
# =====================================================
__all__ = [
    "predict_proba_single",
    "save_model",
    "load_model",
    "backtest",
    "generate_signals",
    "run_backtest",
]
