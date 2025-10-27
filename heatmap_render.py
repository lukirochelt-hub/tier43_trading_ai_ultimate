# heatmap_render.py — Tier 4.3+ Trading AI (Renderer)
# Liest data/heatmap_agg.json und rendert PNGs (Korrelation-Heatmap + Top-Korrelationen)
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.getenv("DATA_DIR", "data")
INPUT_PATH = os.path.join(DATA_DIR, "heatmap_agg.json")
OUT_HEATMAP = os.path.join(DATA_DIR, "heatmap_corr.png")
OUT_TOPBARS = os.path.join(DATA_DIR, "heatmap_top_corr.png")


def _load_records(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    # Erwartete Spalten aus heatmap_agg.py
    expected = [
        "metric",
        "pnl_corr",
        "vol_corr",
        "sentiment_corr",
        "macro_corr",
        "timestamp",
        "records",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _save_heatmap(df: pd.DataFrame, out_path: str):
    # Korrelation-Matrix (Zeilen = metriken, Spalten = Targets)
    mat = df[["pnl_corr", "vol_corr", "sentiment_corr", "macro_corr"]].to_numpy(
        dtype=float
    )
    metrics = df["metric"].astype(str).tolist()
    targets = ["pnl_corr", "vol_corr", "sentiment_corr", "macro_corr"]

    fig = plt.figure(figsize=(max(6, len(targets) * 1.2), max(6, len(metrics) * 0.35)))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=30, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman Corr.")
    ax.set_title("Tier 4.3+ — Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_topbars(df: pd.DataFrame, out_path: str, target="pnl_corr", top_k=20):
    d = df[["metric", target]].dropna(subset=[target]).copy()
    d = d[d["metric"].str.lower() != target.replace("_corr", "")]
    d["abs"] = d[target].abs()
    d = d.sort_values("abs", ascending=False).head(top_k)
    # Barplot
    fig = plt.figure(figsize=(10, max(6, len(d) * 0.35)))
    ax = fig.add_subplot(111)
    ax.barh(d["metric"], d[target])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(f"{target} (Spearman)")
    ax.set_ylabel("metric")
    ax.set_title(f"Top {len(d)} correlations vs {target.replace('_',' ')}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    df = _load_records(INPUT_PATH)
    # Nimm die neueste Aggregation, falls mehrere Zeitstempel existieren
    # (heatmap_agg schreibt momentan rows als Snapshots derselben Matrix)
    # Wir aggregieren hier die *aktuellsten* Werte pro metric.
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("ts").groupby("metric", as_index=False).last()

    os.makedirs(DATA_DIR, exist_ok=True)
    _save_heatmap(df, OUT_HEATMAP)
    _save_topbars(df, OUT_TOPBARS, target="pnl_corr")

    print(f"[heatmap_render] Wrote: {OUT_HEATMAP}")
    print(f"[heatmap_render] Wrote: {OUT_TOPBARS}")
    if "records" in df.columns and pd.notnull(df["records"]).any():
        latest_ts = df["ts"].max()
        print(f"[heatmap_render] Latest snapshot: {latest_ts} — metrics: {len(df)}")


if __name__ == "__main__":
    main()
