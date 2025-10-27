import asyncio, pandas as pd, numpy as np
from ml_core import build_dataset

df = asyncio.run(build_dataset("binance","SOLUSDT","5m",1500,False))
out = pd.DataFrame({
    "time": range(len(df)),
    "open": df["close"].shift(1).fillna(df["close"]),
    "high": df["close"] * (1 + 0.0015),
    "low":  df["close"] * (1 - 0.0015),
    "close": df["close"],
    "volume": df["volume"] if "volume" in df.columns else np.nan
})
out.to_csv("data/sol_5m.csv", index=False)
print("OK -> data/sol_5m.csv created. rows:", len(out))
