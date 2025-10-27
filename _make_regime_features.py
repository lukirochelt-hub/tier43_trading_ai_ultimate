import pandas as pd, numpy as np
df = pd.read_csv("data/sol_5m.csv")
df["ret_1"] = df["close"].pct_change().fillna(0)
df["ret_5"] = df["close"].pct_change(5).fillna(0)
df["ret_20"] = df["close"].pct_change(20).fillna(0)
df.to_csv("data/sample_with_features.csv", index=False)
print("OK -> data/sample_with_features.csv")
