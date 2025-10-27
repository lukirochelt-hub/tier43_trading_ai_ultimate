import pandas as pd
df = pd.read_csv(r"data/sol_5m.csv", nrows=3)
print("COLUMNS:", list(df.columns))
print(df.head(3).to_string(index=False))
