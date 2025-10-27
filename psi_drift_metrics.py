# psi_drift_metrics.py
# Tier 4.3+ – Population Stability Index / Drift Metrics
# - Numerical & categorical PSI
# - Binning: quantile | uniform
# - NaN-safe, zero-count smoothing
# - Auto or user-specified feature types
# - Prometheus optional (PROM_ENABLE=1, PROM_PORT=9108)
# - CLI usage: siehe __main__

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# -------------------------------------------------------
# Optionale Prometheus-Registry (wird zur Laufzeit gesetzt)
# -------------------------------------------------------
REGISTRY: Optional[Any] = None
G_LATENCY: Optional[Any] = None

PROM_ENABLE = os.getenv("PROM_ENABLE", "0") == "1"
PROM_PORT = int(os.getenv("PROM_PORT", "9108"))

PSI_LEVELS: List[Tuple[float, float, str]] = [
    (0.0, 0.1, "insignificant"),
    (0.1, 0.25, "moderate"),
    (0.25, float("inf"), "major"),
]

# ---------------------------------
# Utility helpers
# ---------------------------------
def _status_from_psi(v: float) -> str:
    for lo, hi, name in PSI_LEVELS:
        if lo <= v < hi:
            return name
    return "unknown"


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _safe_probs(counts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0.0:
        # Gleichverteilung, wenn keine Zählwerte
        n = len(counts)
        return np.full(n, 1.0 / n, dtype=float)
    probs = counts.astype(float) / total
    # smoothing to avoid log(0)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / float(probs.sum())
    return probs


def _psi_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    # PSI = sum( (p_i - q_i) * ln(p_i / q_i) ), p,q > 0
    return float(np.sum((p - q) * np.log(p / q)))


# ---------------------------------
# Numerical PSI
# ---------------------------------
def compute_psi_numerical(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    dropna: bool = True,
) -> Tuple[float, Dict[str, Union[List[str], List[float]]]]:
    b = pd.to_numeric(baseline, errors="coerce")
    c = pd.to_numeric(current, errors="coerce")

    if dropna:
        b = b.dropna()
        c = c.dropna()

    if b.empty or c.empty:
        return math.nan, {"bins": [], "p": [], "q": []}

    if strategy == "quantile":
        # Baseline-Quantile als feste Bin-Kanten
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(b.to_numpy(), quantiles))
        if len(edges) <= 2:
            unique_b = np.unique(b.to_numpy())
            edges = np.linspace(b.min(), b.max(), min(bins, max(2, len(unique_b) - 1)) + 1)
    elif strategy == "uniform":
        lo = float(min(b.min(), c.min()))
        hi = float(max(b.max(), c.max()))
        edges = np.linspace(float(lo), float(hi), int(bins) + 1).astype(float)

    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    edges = np.unique(edges)
    if len(edges) < 2:
        return math.nan, {"bins": [], "p": [], "q": []}

    b_counts, _ = np.histogram(b.to_numpy(), bins=edges)
    c_counts, _ = np.histogram(c.to_numpy(), bins=edges)

    p = _safe_probs(b_counts)
    q = _safe_probs(c_counts)
    psi = _psi_from_probs(p, q)

    bin_labels = [f"[{edges[i]:.6g}, {edges[i+1]:.6g})" for i in range(len(edges) - 1)]
    return psi, {"bins": bin_labels, "p": p.tolist(), "q": q.tolist()}


# ---------------------------------
# Categorical PSI
# ---------------------------------
def compute_psi_categorical(
    baseline: pd.Series,
    current: pd.Series,
    top_k: Optional[int] = None,
    other_label: str = "__OTHER__",
    dropna: bool = True,
) -> Tuple[float, Dict[str, Union[List[str], List[float]]]]:
    b = baseline.astype("object")
    c = current.astype("object")
    if dropna:
        b = b.dropna()
        c = c.dropna()

    if b.empty or c.empty:
        return math.nan, {"cats": [], "p": [], "q": []}

    # Seltene Kategorien in OTHER via Baseline-Frequenzen
    b_counts = b.value_counts()
    if top_k is not None and len(b_counts) > top_k:
        top = set(b_counts.nlargest(top_k).index.tolist())
        b = b.where(b.isin(top), other_label)
        c = c.where(c.isin(top), other_label)

    cats = sorted(set(b.unique()).union(set(c.unique())))
    b_vec = np.array([(b == cat).sum() for cat in cats], dtype=float)
    c_vec = np.array([(c == cat).sum() for cat in cats], dtype=float)

    p = _safe_probs(b_vec)
    q = _safe_probs(c_vec)
    psi = _psi_from_probs(p, q)
    return psi, {"cats": [str(x) for x in cats], "p": p.tolist(), "q": q.tolist()}


# ---------------------------------
# Report / API
# ---------------------------------
@dataclass
class PSIConfig:
    bins: int = 10
    strategy: Literal["quantile", "uniform"] = "quantile"
    top_k_cats: Optional[int] = 20
    dropna: bool = True


def psi_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    feature_types: Optional[Dict[str, Literal["num", "cat"]]] = None,
    cfg: PSIConfig = PSIConfig(),
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Returns:
      df with columns: feature, type, psi, level
      details dict keyed by feature with bin/cat breakdown
    """
    if features is None:
        features = [col for col in baseline_df.columns if col in current_df.columns]

    rows: List[dict] = []
    details: Dict[str, dict] = {}

    for f in features:
        s_b = baseline_df[f]
        s_c = current_df[f]

        if feature_types and f in feature_types:
            ftype: Literal["num", "cat"] = feature_types[f]
        else:
            ftype = "num" if _is_numeric_series(s_b) and _is_numeric_series(s_c) else "cat"

        if ftype == "num":
            psi, d = compute_psi_numerical(
                s_b, s_c, bins=cfg.bins, strategy=cfg.strategy, dropna=cfg.dropna
            )
        else:
            psi, d = compute_psi_categorical(
                s_b, s_c, top_k=cfg.top_k_cats, dropna=cfg.dropna
            )

        level = _status_from_psi(psi) if not math.isnan(psi) else "nan"

        rows.append(
            {
                "feature": f,
                "type": ftype,
                "psi": float(psi) if psi == psi else math.nan,
                "level": level,
            }
        )
        details[f] = d

    df = (
        pd.DataFrame(rows)
        .sort_values(by="psi", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return df, details


# ---------------------------------
# Prometheus (optional)
# ---------------------------------
def init_metrics() -> None:
    """Initialisiert REGISTRY und ein Latenz-Gauge (nur wenn prometheus_client vorhanden)."""
    global REGISTRY, G_LATENCY
    try:
        from prometheus_client import Gauge, CollectorRegistry  
    except Exception:
        return  # Prometheus nicht verfügbar
    if REGISTRY is None:
        REGISTRY = CollectorRegistry()
    if G_LATENCY is None:
        G_LATENCY = Gauge("latency_seconds", "Latency", registry=REGISTRY)


def start_metrics_server(port: int = PROM_PORT) -> None:
    """Startet HTTP-Server für Prometheus-Exporte, falls möglich."""
    try:
        from prometheus_client import start_http_server 
    except Exception:
        return
    if REGISTRY is None:
        init_metrics()
    if REGISTRY is not None:
        start_http_server(port, addr="0.0.0.0", registry=REGISTRY)


def set_latency(value: float) -> None:
    """Setzt Latenz-Gauge, wenn vorhanden."""
    if G_LATENCY is not None:
        G_LATENCY.set(value)


class PSIMetricsExporter:
    """Exponiert pro-Feature-PSI und ein Max-PSI-Gauge via Prometheus (optional)."""

    def __init__(self, namespace: str = "tier43", subsystem: str = "drift"):
        if not PROM_ENABLE:
            raise RuntimeError("Prometheus not enabled. Set PROM_ENABLE=1")
        try:
            from prometheus_client import Gauge  
        except Exception as e:
            raise RuntimeError("Prometheus client not available.") from e
        if REGISTRY is None:
            init_metrics()
        if REGISTRY is None:
            raise RuntimeError("Prometheus registry not initialized.")
        # Instanzattribute ungetypt (Any), um Stub-Probleme zu vermeiden
        self.registry: Any = REGISTRY
        self.g_psi: Any = Gauge(
            f"{namespace}_{subsystem}_psi",
            "Population Stability Index per feature",
            ["feature", "type", "level"],
            registry=self.registry,
        )
        self.g_psi_max: Any = Gauge(
            f"{namespace}_{subsystem}_psi_max",
            "Max PSI across all tracked features",
            registry=self.registry,
        )

    def push(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.g_psi_max.set(0.0)
            return

        max_psi = 0.0
        for _, row in df.iterrows():
            # robustes Auslesen und Typisierung
            psi_raw: Any = row.get("psi", None)
            try:
                psi_val = float(psi_raw)
            except Exception:
                # nicht konvertierbar -> überspringen
                continue

            if math.isnan(psi_val):
                # NaN -> überspringen
                continue

            if psi_val > max_psi:
                max_psi = psi_val

            self.g_psi.labels(
                feature=str(row.get("feature", "")),
                type=str(row.get("type", "")),
                level=str(row.get("level", "")),
            ).set(psi_val)

        self.g_psi_max.set(max_psi)




# ---------------------------------
# CLI
# ---------------------------------
def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".feather", ".ft"):
        return pd.read_feather(path)
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if ext == ".json":
        return pd.read_json(path, lines=False)
    raise ValueError(f"Unsupported file extension: {ext}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tier 4.3+ PSI Drift Metrics")
    p.add_argument("--baseline", required=True, help="Path to baseline table (csv/parquet/feather/json)")
    p.add_argument("--current", required=True, help="Path to current table (csv/parquet/feather/json)")
    p.add_argument("--features", nargs="*", help="Optional list of feature columns")
    p.add_argument("--feature-types", type=str, default=None, help="JSON dict mapping feature -> 'num'|'cat'")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--strategy", choices=["quantile", "uniform"], default="quantile")
    p.add_argument("--top-k-cats", type=int, default=20)
    p.add_argument("--no-dropna", action="store_true", help="Do not drop NaNs before computing")
    p.add_argument("--json-out", type=str, default=None, help="Write report json here")
    p.add_argument("--csv-out", type=str, default=None, help="Write table csv here")
    p.add_argument("--start-metrics", action="store_true", help="Start Prometheus HTTP server (requires PROM_ENABLE=1)")
    args = p.parse_args()

    base = _load_table(args.baseline)
    curr = _load_table(args.current)

    ftypes: Optional[Dict[str, Literal["num", "cat"]]] = (
        json.loads(args.feature_types) if args.feature_types else None
    )
    cfg = PSIConfig(
        bins=args.bins,
        strategy=args.strategy,
        top_k_cats=args.top_k_cats,
        dropna=not args.no_dropna,
    )

    df, det = psi_report(base, curr, features=args.features, feature_types=ftypes, cfg=cfg)

    # Print to stdout
    print(df.to_string(index=False))
    payload: Dict[str, Any] = {"summary": df.to_dict(orient="records"), "details": det}

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.csv_out:
        df.to_csv(args.csv_out, index=False)

    if args.start_metrics and PROM_ENABLE:
        exp = PSIMetricsExporter()
        exp.push(df)
        start_metrics_server(PROM_PORT)
        print(f"[psi_drift_metrics] Prometheus server started on :{PROM_PORT}")
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
