# make_dummy_train.py
import pandas as pd, numpy as np, os
os.makedirs("data", exist_ok=True)

n = 1000
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="H"),
    "return_1h": np.random.randn(n),
    "volatility": np.abs(np.random.randn(n)),
    "momentum": np.random.randn(n) * 0.2,
})

df.to_parquet("data/train.parquet")
print(f"✅ Dummy train.parquet erstellt mit {len(df)} Zeilen")
