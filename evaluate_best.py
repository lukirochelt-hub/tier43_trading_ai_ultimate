#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 4.3+ — evaluate_best.py
Analyse & Visualisierung der besten Strategieparameter (nach Optuna).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

from features_store import load_ohlcv, build_features
from regime_learner import RegimeLearner

# --- optional / robust import von ml_core ---
import importlib
_ml = importlib.import_module("ml_core")

# ----------------------------------------------------------
# 1) Beste Parameter laden
BEST_PATH = "models/best_params_BTCUSDT_15m.json"
params = json.load(open(BEST_PATH))
print(f"\n✅ Beste Parameter geladen aus {BEST_PATH}\n")

# ----------------------------------------------------------
# 2) Daten & Features laden
print("[1] Lade OHLCV-Daten ...")
df = load_ohlcv("BTCUSDT", "15m", "2024-01-01", "2025-01-01")
print("[2] Berechne Features ...")
df = build_features(df, params=None)

# ----------------------------------------------------------
# 3) Regime-Modell laden & anwenden
print("[3] Lade Regime-Learner ...")
rl = RegimeLearner.load("models/rl.joblib")
df = rl.transform(df)  # hängt regime_id/label/conf + probs an

# ----------------------------------------------------------
# 4) Signale erzeugen
print("[4] Erzeuge Trading-Signale ...")
signals = None
if hasattr(_ml, "generate_signals"):
    # bevorzugt die projektspezifische Signallogik
    signals = _ml.generate_signals(df=df, params=params)
else:
    # simpler Fallback: EMA-Crossover
    ema_fast = df["close"].ewm(span=params["ema_fast"]).mean()
    ema_slow = df["close"].ewm(span=params["ema_slow"]).mean()
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[ema_fast > ema_slow] = 1
    sig[ema_fast < ema_slow] = -1
    signals = sig

# ----------------------------------------------------------
# 5) Backtest ausführen (Optuna-kompatible Signatur)
print("[5] Führe Backtest aus ...")
metrics = None
equity = None

if hasattr(_ml, "backtest"):
    try:
        # neue Signatur: backtest(df=..., signals=..., params=...)
        metrics = _ml.backtest(df=df, signals=signals, params=params)
        equity = metrics.get("equity", None)
    except TypeError:
        # falls altes Interface im Einsatz ist, lokal berechnen
        rets = df["close"].pct_change().fillna(0.0)
        pnl = (signals.shift(1).fillna(0) * rets).fillna(0.0)
        equity = (1.0 + pnl).cumprod()
        metrics = {
            "sharpe": float(pnl.mean() / (pnl.std() + 1e-12) * (252*24*4) ** 0.5),
            "profit_factor": 0.0,
            "max_drawdown": float((equity / equity.cummax() - 1.0).min() * 100.0),
            "win_rate": float((pnl > 0).sum()) / float((signals != 0).sum() + 1e-9),
            "net_profit": float((equity.iloc[-1] - 1.0) * 100.0),
            "trades": int((signals.diff().abs() > 0).sum()),
        }
else:
    raise RuntimeError("ml_core.backtest nicht gefunden.")

print("\n=== 📊 Backtest Resultate ===")
for k in ["sharpe","profit_factor","max_drawdown","win_rate","net_profit","trades"]:
    v = metrics.get(k, None)
    if isinstance(v, float):
        print(f"{k:14s}: {v:.6f}")
    else:
        print(f"{k:14s}: {v}")

# ----------------------------------------------------------
# 6) Equity-Kurve mit Regime-Overlay plotten
print("\n[6] Zeichne Equity-Kurve ...")
if equity is None:
    # falls Equity nicht im Metrics-Dict enthalten war
    rets = df["close"].pct_change().fillna(0.0)
    pnl = (signals.shift(1).fillna(0) * rets).fillna(0.0)
    equity = (1.0 + pnl).cumprod()

fig, ax = plt.subplots(figsize=(12, 6))
equity.plot(ax=ax, lw=1.3, color="black", label="Equity Curve")

# Regime-Bereiche einfärben (bull/bear/sideways)
colors = {"bull": "lightgreen", "bear": "lightcoral", "sideways": "lightgray"}
ymin, ymax = float(equity.min()), float(equity.max())
for label, color in colors.items():
    mask = (df["regime_label"] == label).reindex_like(equity).fillna(False)
where_arr = mask.astype(bool).to_numpy().reshape(-1)
ax.fill_between(
    equity.index,
    ymin,
    ymax,
    where=where_arr.tolist(),  # ✅ Typkonvertierung für mypy
    color=color,
    alpha=0.20,
    label=label,
)


ax.set_title("Tier 4.3+ — Equity Curve mit Regime-Zonen", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.show()
