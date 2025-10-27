import json, pandas as pd
from quick_signal import quick_signal, to_advice_dict

df = pd.read_csv("data/sol_5m.csv").tail(400)
res = quick_signal("SOLUSDT","5m",
                   df["close"].values,
                   highs=df.get("high").values if "high" in df else None,
                   lows=df.get("low").values if "low" in df else None,
                   cooldown=0, vol_threshold=1.05, rsi_ob=85, rsi_os=15,
                   cross_lookback=3, enable_atr_guard=False)
print(res.__dict__ if res else "no signal")
if res: print(json.dumps(to_advice_dict(res), indent=2))

closes = df["close"].astype(float).tolist()
highs  = df["high"].astype(float).tolist() if "high" in df else None
lows   = df["low"].astype(float).tolist()  if "low"  in df else None

quick_signal(symbol, tf, closes, highs=highs, lows=lows)
