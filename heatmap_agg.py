# heatmap_agg.py — Tier 4.3+ Trading AI (Stable Release)
# Aggregation + Correlation Heatmap for PnL, Volatility, Sentiment, and Macro Data

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from features_store import load_feature_data
from adapters_feeds import get_market_snapshots
from adapters_alt_feeds import get_sentiment_data
from adapters_macro import get_macro_indicators

DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "heatmap_agg.json")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numerical columns between 0–1 for heatmap scaling."""
    return (df - df.min()) / (df.max() - df.min())


def aggregate_heatmap_data():
    """
    Aggregates all active Tier 4.3+ data sources into one unified DataFrame
    and computes correlation-based heatmap data for monitoring dashboards.
    """
    try:
        market = get_market_snapshots()  # from adapters_feeds
        sentiment = get_sentiment_data()  # from adapters_alt_feeds
        macro = get_macro_indicators()  # from adapters_macro
        features = load_feature_data()  # from features_store
    except Exception as e:
        print(f"[heatmap_agg] Data source error: {e}")
        return None

    # Merge all frames on timestamp if available
    frames = [
        df
        for df in [market, sentiment, macro, features]
        if isinstance(df, pd.DataFrame)
    ]
    if not frames:
        print("[heatmap_agg] No valid data frames to aggregate.")
        return None

    df = pd.concat(frames, axis=1)
    df = df.fillna(method="ffill").dropna(how="any")
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicates

    # Ensure essential columns exist
    required_cols = ["pnl", "volatility", "sentiment_score", "macro_index"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Normalize numeric data for better visual scaling
    norm_df = normalize(df.select_dtypes(include=[np.number]))

    # Spearman correlation for non-linear relationships
    corr = norm_df.corr(method="spearman").round(3)

    # Build simplified heatmap summary
    heatmap_summary = pd.DataFrame(
        {
            "metric": corr.columns,
            "pnl_corr": corr.loc["pnl"].round(3),
            "vol_corr": corr.loc["volatility"].round(3),
            "sentiment_corr": corr.loc["sentiment_score"].round(3),
            "macro_corr": corr.loc["macro_index"].round(3),
        }
    ).reset_index(drop=True)

    # Add timestamp + metadata
    heatmap_summary["timestamp"] = datetime.utcnow().isoformat()
    heatmap_summary["records"] = len(df)

    # Save to JSON
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(heatmap_summary.to_dict(orient="records"), f, indent=2)

    print(f"[heatmap_agg] Updated {OUTPUT_PATH} — {len(df)} records aggregated.")
    return heatmap_summary


if __name__ == "__main__":
    result = aggregate_heatmap_data()
    if result is not None:
        print(result.head())
