# run_quick_signal_csv.py
# Tier 4.3+ — Quick Signal CSV Tester
# - Lädt Candle-Daten (CSV)
# - Wandelt in float-Listen (mypy-kompatibel)
# - Führt quick_signal() aus und gibt das Ergebnis strukturiert aus

from __future__ import annotations

import json
import time
import pandas as pd
from typing import Any, Sequence, Optional
from quick_signal import quick_signal, to_advice_dict


def main() -> None:
    """
    Lädt OHLCV-Daten aus CSV, ruft quick_signal() auf und zeigt das Ergebnis.
    Erwartet CSV mit Spalten: time, open, high, low, close, volume
    """
    # CSV einlesen
    df = pd.read_csv("data/sol_5m.csv")

    # In float-Listen konvertieren (statt numpy-Arrays → Sequence[float])
    closes: list[float] = df["close"].astype(float).tolist()
    highs: Optional[list[float]] = df["high"].astype(float).tolist() if "high" in df else None
    lows: Optional[list[float]] = df["low"].astype(float).tolist() if "low" in df else None

    # quick_signal aufrufen
    res = quick_signal(
        ticker="SOLUSDT",
        tf="5m",
        closes=closes,
        highs=highs,
        lows=lows,
        cooldown=0,               # deaktiviert für Tests
        vol_threshold=1.1,        # aggressiver (Standard 1.4)
        rsi_ob=80.0,              # RSI overbought lockerer
        rsi_os=20.0,              # RSI oversold lockerer
        enable_atr_guard=False,   # ATR-Guard aus für Test
    )

    # Ausgabe
    if res:
        print(json.dumps(to_advice_dict(res), indent=2))
    else:
        print("⚠️  Kein Signal erkannt.")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"⏱️ Laufzeit: {time.perf_counter() - start:.3f}s")
