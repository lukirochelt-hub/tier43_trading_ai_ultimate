# quick_signal.py
# Tier 4.3+ — Fast-Path Signal Engine (Windows-safe)
# - Lightweight: numpy-only
# - Indicators: EMA(9/21), RSI(14), optional ATR-lite für Vol-Guard
# - Signal-Logik: EMA-Cross (+ Lookback), RSI-Filter, Volatilitäts-Check, Cooldown
# - Output: QuickSignalResult + to_advice_dict() (kompatibel zu adv_val)

from __future__ import annotations
import time, math
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Sequence
import numpy as np

# ======================================
# Defaults
# ======================================
EMA_FAST = 9
EMA_SLOW = 21
RSI_LEN = 14
VOL_WIN = 20
RSI_OB = 70.0      # Overbought-Gate (Long-Filter)
RSI_OS = 30.0      # Oversold-Gate  (Short-Filter)
VOL_THRESHOLD = 1.4  # recent/prev std ratio
COOLDOWN_SEC = 10
ATR_LEN = 14
VOL_GUARD_ATR_MULT = 0.0  # >0 aktivieren (z. B. 1.5)

_last_emit: Dict[str, float] = {}  # cooldown per (ticker, tf)

# ======================================
# Helpers (numpy)
# ======================================
def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out

def _rsi(arr: np.ndarray, length: int = RSI_LEN) -> float:
    if arr.size < length + 1:
        return 50.0
    deltas = np.diff(arr)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain = gains[-length:].mean()
    avg_loss = losses[-length:].mean()
    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, length: int = ATR_LEN) -> float:
    if highs.size < length + 1 or lows.size < length + 1 or closes.size < length + 1:
        return float("nan")
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr_series = _ema(tr, length)  # Wilder-EMA Approx
    return float(atr_series[-1])

# ======================================
# Main
# ======================================
@dataclass
class QuickSignalResult:
    ts: float
    ticker: str
    tf: str
    bias: Literal["LONG", "SHORT", "NEUTRAL"]
    score: float
    prob: float
    reason: str

def quick_signal(
    ticker: str,
    tf: str,
    closes: Sequence[float],
    *,
    highs: Optional[Sequence[float]] = None,
    lows: Optional[Sequence[float]] = None,
    cooldown: int = COOLDOWN_SEC,
    vol_threshold: float = VOL_THRESHOLD,
    rsi_ob: float = RSI_OB,
    rsi_os: float = RSI_OS,
    enable_atr_guard: bool = False,
    atr_mult: float = VOL_GUARD_ATR_MULT,
    cross_lookback: int = 1,   # Option A: Cross bis N Bars zurück akzeptieren
) -> Optional[QuickSignalResult]:
    """
    Gibt ein Signal zurück oder None (wenn nicht genug Daten oder Cooldown aktiv).
    - NEUTRAL verbraucht keinen Cooldown.
    - cross_lookback: 1 = nur letzter Bar (klassisch), >1 = akzeptiere frische Crosses bis N Bars zurück.
    """
    now = time.time()
    key = f"{ticker}:{tf}"
    last_ts = _last_emit.get(key, 0.0)

    arr = np.asarray(closes, dtype=float)
    if arr.size < max(EMA_SLOW + 2, VOL_WIN * 2):
        return None

    ema_fast = _ema(arr, EMA_FAST)
    ema_slow = _ema(arr, EMA_SLOW)
    rsi_val = _rsi(arr, RSI_LEN)

    # simple vol regime: recent window vs previous window
    recent = arr[-VOL_WIN:]
    prev = arr[-2 * VOL_WIN : -VOL_WIN]
    prev_std = prev.std() if prev.size else 1e-9
    vol_ratio = float(recent.std() / (prev_std + 1e-12))

    # optional wick/atr guard
    if enable_atr_guard and highs is not None and lows is not None and atr_mult > 0.0:
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows, dtype=float)
        if h.size == arr.size and l.size == arr.size:
            atr = _atr(h, l, arr, ATR_LEN)
            if not math.isnan(atr):
                last_wick = float(h[-1] - l[-1])
                if last_wick > atr_mult * (atr + 1e-12):
                    return QuickSignalResult(now, ticker, tf, "NEUTRAL", 0.0, 0.5, "atr_vol_guard")

    # ---- Option A: Cross bis N Bars zurück akzeptieren ----
    lb = max(1, min(cross_lookback, len(ema_fast) - 1))
    cross_u = any(
        (ema_fast[-i] > ema_slow[-i]) and (ema_fast[-i-1] <= ema_slow[-i-1])
        for i in range(1, lb + 1)
    )
    cross_d = any(
        (ema_fast[-i] < ema_slow[-i]) and (ema_fast[-i-1] >= ema_slow[-i-1])
        for i in range(1, lb + 1)
    )
    trend_up = cross_u
    trend_dn = cross_d

    bias: Literal["LONG","SHORT","NEUTRAL"] = "NEUTRAL"
    score = 0.0
    reason = "neutral"

    if vol_ratio > vol_threshold:
        if trend_up and rsi_val < rsi_ob:
            bias, reason = "LONG", f"ema_cross_up(LB={lb})+rsi_ok+volatile"
            # Edge-Proxy: RSI-Distanz von 50 + Vol-Stärke
            score = max(0.0, min(1.0, ((rsi_val - 50.0) / 25.0) + (vol_ratio / 3.0)))
        elif trend_dn and rsi_val > rsi_os:
            bias, reason = "SHORT", f"ema_cross_dn(LB={lb})+rsi_ok+volatile"
            score = max(0.0, min(1.0, ((50.0 - rsi_val) / 25.0) + (vol_ratio / 3.0)))

    # score -> probability (55..95%)
    prob = 0.5 if score == 0.0 else float(round(0.55 + 0.4 * score, 4))

    # nur bei LONG/SHORT Cooldown anwenden
    if bias != "NEUTRAL" and (now - last_ts) < cooldown:
        return None
    if bias != "NEUTRAL":
        _last_emit[key] = now

    return QuickSignalResult(now, ticker, tf, bias, float(round(score, 4)), prob, reason)

def to_advice_dict(
    res: QuickSignalResult,
    *,
    pos_size: float = 10_000,
    bias_asset: str = "SOL",
    hedge_asset: str = "BTC",
) -> Dict[str, object]:
    """
    Kompaktes Dict für adv_val (Signatur macht dein Relay/Webhook).
    """
    return {
        "symbol": res.ticker,
        "tf": res.tf,
        "ts": int(res.ts * 1000),
        "long": True if res.bias == "LONG" else None,
        "short": True if res.bias == "SHORT" else None,
        "prob": None if res.bias == "NEUTRAL" else res.prob,
        "edge": None if res.bias == "NEUTRAL" else res.score,
        "strategy": "quick_signal_v2",
        "config": {"pos_size": pos_size, "bias": bias_asset, "hedge": hedge_asset},
        "meta": {"reason": res.reason, "vol_win": VOL_WIN},
    }

# ======================================
# Example (optional lokal testen)
# ======================================
if __name__ == "__main__":
    import random
    prices = [100 + math.sin(i / 3) + random.uniform(-0.5, 0.5) for i in range(200)]
    r = quick_signal("SOLUSDT", "15m", prices, cross_lookback=3, vol_threshold=1.05, rsi_ob=85.0, rsi_os=15.0, cooldown=0)
    print(vars(r) if r else "no signal")
    if r:
        print(to_advice_dict(r))
