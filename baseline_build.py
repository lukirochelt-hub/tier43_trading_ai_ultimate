#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 4.3+ — baseline_build.py
-----------------------------
Baut und speichert ein robustes Baseline-Modell (Regression) für deine Trading-KI.
- Lädt Features aus features_store.py
- Train/Val-Split nach Zeit (keine Leckage)
- Mehrere Modell-Optionen (XGBoost, LightGBM, Sklearn Fallback)
- Einheitliche Metriken (MSE, MAE, R2, Directional Accuracy, MAPE)
- Speichert: Modell (.pkl), Metriken (.json), Config (.json), Featureliste (.json), Run-Log (.log)

Benötigte (optionale) Pakete:
- xgboost, lightgbm (wenn vorhanden, sonst Fallback auf sklearn)
- rich (für schönere CLI, optional)

CLI-Beispiele:
--------------
# Minimal (nimmt Defaults aus BASELINE_CFG)
python baseline_build.py

# Mit Symbol & TF und XGBoost
python baseline_build.py --symbols BTCUSDT --timeframes 5m 15m 1h --model xgboost

# Eigener Lookback und Zielhorizont (Return in Bars)
python baseline_build.py --lookback 256 --target_horizon 12

# Datenzeitraum begrenzen
python baseline_build.py --start "2023-01-01" --end "2025-10-01"

# Artefakt-Ordner setzen
python baseline_build.py --outdir ./models/baseline

Ordnerstruktur (Beispiel):
project/
  adapters_feeds.py
  adapters_macro.py
  features_store.py
  ml_core.py
  baseline_build.py
  models/
    baseline/
      <symbol>_<tf>_<timestamp>/
        model.pkl
        metrics.json
        config.json
        features.json
        run.log
"""

from __future__ import annotations
import os
import json
import math
import uuid
import argparse
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore")

# -------- Optional Pretty Console --------
_HAS_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.traceback import install as rich_install
    from rich.progress import track

    rich_install()
    console: Any = Console()
    _HAS_RICH = True
except Exception:
    # Fallbacks ohne rich
    class _NoopConsole:
        def print(self, *a, **k) -> None:
            print(*a)

    def track(x, description: str = ""):
        return x

    # Dummy-Table Signatur für Typechecker
    class Table:  # type: ignore
        def __init__(self, title: str = "") -> None:
            self._rows: List[List[str]] = []
            self._cols: List[str] = []
            self._title = title

        def add_column(self, name: str, justify: str | None = None) -> None:
            self._cols.append(name)

        def add_row(self, *row: str) -> None:
            self._rows.append(list(row))

        def render(self) -> None:
            print(self._title or "Table")
            if not self._rows:
                print("(empty)")
                return
            widths = [max(len(c), *(len(r[i]) for r in self._rows)) for i, c in enumerate(self._cols)]
            line = " | ".join(c.ljust(widths[i]) for i, c in enumerate(self._cols))
            print(line)
            print("-+-".join("-" * w for w in widths))
            for r in self._rows:
                print(" | ".join(r[i].ljust(widths[i]) for i in range(len(widths))))

    console = _NoopConsole()

# -------- Numeric/ML Core --------
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: XGBoost / LightGBM
_HAS_XGB = False
_HAS_LGBM = False
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except Exception:
    pass


# -------- Project Imports (graceful) --------
def _import_features_store():
    try:
        import features_store as fs  # type: ignore
        assert hasattr(fs, "get_feature_frame"), "features_store.get_feature_frame fehlt"
        return fs
    except Exception as e:
        console.print(f"[yellow]Hinweis:[/yellow] Konnte features_store nicht laden: {e}")
        return None


def _import_ml_core():
    try:
        import ml_core as mlc  # type: ignore
        return mlc
    except Exception as e:
        console.print(f"[yellow]Hinweis:[/yellow] Konnte ml_core nicht laden: {e}")
        return None


# -------- Default Config --------
BASELINE_CFG: Dict[str, Any] = {
    "model_type": "xgboost",  # "xgboost" | "lightgbm" | "ridge" | "rf"
    "target": "return_1h",  # wird ggf. on-the-fly erzeugt aus close
    "target_horizon": 12,  # Bars in die Zukunft (z.B. 12 x 5m = 1h)
    "lookback": 256,  # Anzahl vergangener Bars als Features
    "train_split": 0.8,  # Zeitbasierter Split
    "features": [  # Fallback-Featureliste, falls FS keine Liste liefert
        "ema_9",
        "ema_21",
        "rsi",
        "adx",
        "bb_upper",
        "bb_lower",
        "volume",
        "volume_delta",
        "atr",
        "stoch_k",
        "stoch_d",
    ],
    "timeframes": ["5m", "15m", "1h"],
    "symbols": ["BTCUSDT"],
    "metric_set": ["mse", "mae", "r2", "mape", "directional_acc"],
}


# -------- Utilities --------
def set_seed(seed: int = 42) -> None:
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ts_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_returns(series: pd.Series, horizon: int) -> pd.Series:
    # log-return forward horizon
    fwd = series.shift(-horizon)
    ret = np.log(fwd / series)
    return ret


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = np.clip(np.abs(yt), 1e-8, None)
    return float(np.mean(np.abs((yt - yp) / denom)))


def time_split(df: pd.DataFrame, split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert 0.0 < split < 1.0
    n = len(df)
    cut = int(n * split)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_pipeline(model: Any) -> Pipeline:
    # Imputer -> Scaler -> Model
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )


def pick_model(name: str, random_state: int = 42) -> Any:
    name = (name or "").lower().strip()
    if name == "xgboost" and _HAS_XGB:
        return XGBRegressor(  # type: ignore[call-arg]
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=random_state,
            tree_method="hist",
            n_jobs=os.cpu_count(),
        )
    if name == "lightgbm" and _HAS_LGBM:
        return LGBMRegressor(  # type: ignore[call-arg]
            n_estimators=800,
            num_leaves=64,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=os.cpu_count(),
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=os.cpu_count(),
            random_state=random_state,
        )
    # Default: Ridge (stabil und schnell)
    return Ridge(alpha=1.0, random_state=random_state)


@dataclass
class RunConfig:
    model_type: str = BASELINE_CFG["model_type"]
    target: str = BASELINE_CFG["target"]
    target_horizon: int = BASELINE_CFG["target_horizon"]
    lookback: int = BASELINE_CFG["lookback"]
    train_split: float = BASELINE_CFG["train_split"]
    # wichtige Korrektur: mutable Defaults via default_factory
    features: List[str] = field(default_factory=lambda: list(BASELINE_CFG["features"]))
    timeframes: List[str] = field(default_factory=lambda: list(BASELINE_CFG["timeframes"]))
    symbols: List[str] = field(default_factory=lambda: list(BASELINE_CFG["symbols"]))
    start: Optional[str] = None
    end: Optional[str] = None
    outdir: str = "./models/baseline"
    seed: int = 42


# -------- Data Loader --------
def load_frame(symbol: str, timeframe: str, cfg: RunConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Erwartet, dass features_store.get_feature_frame(symbol, timeframe, start=None, end=None)
    ein DataFrame mit DatetimeIndex liefert und mindestens die Spalten:
    ['open','high','low','close','volume', ... technische Features ...]
    """
    fs = _import_features_store()
    if fs is None:
        raise RuntimeError("features_store.py konnte nicht importiert werden. Bitte sicherstellen, dass die Datei vorhanden ist.")

    df: pd.DataFrame = fs.get_feature_frame(symbol=symbol, timeframe=timeframe, start=cfg.start, end=cfg.end)  # type: ignore[assignment]

    if not isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
        raise ValueError("Feature-Frame muss einen (Date-)Index besitzen.")
    if "close" not in df.columns:
        raise ValueError("Spalte 'close' wird benötigt.")

    # Ziel vorbereiten (log-return forward horizon)
    df = df.sort_index()
    df[f"{cfg.target}"] = compute_returns(df["close"], cfg.target_horizon)

    # Feature-Auswahl
    feats = [f for f in cfg.features if f in df.columns]
    if len(feats) == 0:
        # Fallback: benutze generische OHLCV + einfache abgeleitete Features
        console.print("[yellow]Warnung:[/yellow] Keine der konfigurierten Features gefunden. Nutze Fallback-Features.")
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        df["oc_spread"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        feats = ["open", "high", "low", "close", "volume", "hl_spread", "oc_spread"]

    # NaNs und Inf säubern
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])

    # Rolling-Window für Lookback -> flache Matrix
    for col in list(feats):
        roll = df[col].rolling(cfg.lookback, min_periods=max(5, cfg.lookback // 4))
        mean_c = f"{col}_mean_{cfg.lookback}"
        std_c = f"{col}_std_{cfg.lookback}"
        df[mean_c] = roll.mean()
        df[std_c] = roll.std()
        df[f"{col}_min_{cfg.lookback}"] = roll.min()
        df[f"{col}_max_{cfg.lookback}"] = roll.max()
        df[f"{col}_z_{cfg.lookback}"] = (df[col] - df[mean_c]) / (df[std_c] + 1e-8)

    feat_cols = sorted(
        [c for c in df.columns if any(c.startswith(f) for f in feats) or f"_mean_{cfg.lookback}" in c or f"_z_{cfg.lookback}" in c]
    )
    feat_cols = [c for c in feat_cols if c != cfg.target]  # exclude target if name overlaps

    # Finaler Drop von NaNs nach Rolling
    df = df.dropna(subset=feat_cols + [cfg.target])

    return df, feat_cols


# -------- Train/Eval --------
def train_one(df: pd.DataFrame, feat_cols: List[str], target_col: str, cfg: RunConfig) -> Tuple[Pipeline, Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    # Zeitbasierter Split
    df_train, df_val = time_split(df, cfg.train_split)

    X_train = df_train[feat_cols].values
    y_train = df_train[target_col].values
    X_val = df_val[feat_cols].values
    y_val = df_val[target_col].values

    model_core = pick_model(cfg.model_type, random_state=cfg.seed)
    pipe = build_pipeline(model_core)

    console.print(f"[cyan]Training[/cyan] -> {cfg.model_type} | Samples: train={len(X_train)}, val={len(X_val)}")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)

    metrics: Dict[str, float] = {
        "mse": float(mean_squared_error(y_val, y_pred)),
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "r2": float(r2_score(y_val, y_pred)) if len(np.unique(y_val)) > 1 else float("nan"),
        "mape": mape(y_val, y_pred),
        "directional_acc": directional_accuracy(y_val, y_pred),
    }

    return pipe, metrics, (y_val, y_pred)


# -------- Persist --------
def persist_run(
    pipe: Pipeline,
    metrics: Dict[str, float],
    feat_cols: List[str],
    cfg: RunConfig,
    symbol: str,
    timeframe: str,
) -> str:
    import joblib  # type: ignore

    run_id = f"{symbol}_{timeframe}_{ts_now()}_{uuid.uuid4().hex[:6]}"
    outdir = os.path.join(cfg.outdir, run_id)
    ensure_dir(outdir)

    # Speichern
    joblib.dump(pipe, os.path.join(outdir, "model.pkl"))
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(outdir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(feat_cols), f, indent=2)

    # (optional) Feature-Importances falls verfügbar
    try:
        model = pipe.named_steps.get("model")
        importances: Optional[List[float]] = None
        names: List[str] = feat_cols
        if hasattr(model, "feature_importances_"):
            vals = getattr(model, "feature_importances_", None)
            if vals is not None:
                importances = [float(x) for x in vals[: len(names)]]
        if importances:
            with open(os.path.join(outdir, "feature_importances.json"), "w", encoding="utf-8") as f:
                json.dump({"features": names, "importances": importances}, f, indent=2)
    except Exception:
        pass

    # Logfile anlegen
    with open(os.path.join(outdir, "run.log"), "w", encoding="utf-8") as f:
        f.write(f"RunID: {run_id}\n")
        f.write(f"Created: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Model: {cfg.model_type}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Target: {cfg.target}\n")
        f.write(f"Target Horizon: {cfg.target_horizon}\n")
        f.write(f"Lookback: {cfg.lookback}\n")
        f.write(f"Train Split: {cfg.train_split}\n")
        f.write(json.dumps(metrics, indent=2))

    console.print(f"[green]✔ Gespeichert:[/green] {outdir}")
    return outdir


# -------- Pretty Print --------
def print_metrics_table(rows: List[Tuple[str, str, Dict[str, float]]]) -> None:
    table = Table(title="Baseline Metrics (Validation)")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("MSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("R2", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("DirAcc", justify="right")

    for sym, tf, m in rows:
        r2v = m.get("r2", float("nan"))
        table.add_row(
            sym,
            tf,
            f"{m.get('mse', float('nan')):.6f}",
            f"{m.get('mae', float('nan')):.6f}",
            f"{r2v:.4f}" if math.isfinite(r2v) else "nan",
            f"{m.get('mape', float('nan')):.4f}",
            f"{m.get('directional_acc', float('nan')):.4f}",
        )
    if _HAS_RICH:
        console.print(table)
    else:
        # Fallback einfache Ausgabe
        table.render()  # type: ignore[attr-defined]


# -------- Main --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 4.3+ Baseline Builder")
    p.add_argument("--symbols", nargs="+", default=BASELINE_CFG["symbols"], help="Liste von Symbolen")
    p.add_argument("--timeframes", nargs="+", default=BASELINE_CFG["timeframes"], help="Liste von Timeframes")
    p.add_argument("--model", type=str, default=BASELINE_CFG["model_type"], help="xgboost|lightgbm|ridge|rf")
    p.add_argument("--target", type=str, default=BASELINE_CFG["target"], help="Zielspaltenname (wird erzeugt)")
    p.add_argument("--target_horizon", type=int, default=BASELINE_CFG["target_horizon"], help="Vorhersagehorizont in Bars")
    p.add_argument("--lookback", type=int, default=BASELINE_CFG["lookback"], help="Lookback-Fenstergröße")
    p.add_argument("--train_split", type=float, default=BASELINE_CFG["train_split"], help="Zeitbasierter Split 0-1")
    p.add_argument("--start", type=str, default=None, help="Startdatum YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="Enddatum YYYY-MM-DD")
    p.add_argument("--outdir", type=str, default="./models/baseline", help="Output-Ordner")
    p.add_argument("--seed", type=int, default=42, help="Random Seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    cfg = RunConfig(
        model_type=str(args.model),
        target=str(args.target),
        target_horizon=int(args.target_horizon),
        lookback=int(args.lookback),
        train_split=float(args.train_split),
        features=list(BASELINE_CFG["features"]),  # Feature-Liste kann auch in features_store überschrieben werden
        timeframes=[str(x) for x in args.timeframes],
        symbols=[str(x) for x in args.symbols],
        start=args.start,
        end=args.end,
        outdir=str(args.outdir),
        seed=int(args.seed),
    )

    console.print(f"[bold]Tier 4.3+ — Baseline Build[/bold]  (Model={cfg.model_type})")
    console.print(f"Symbols: {cfg.symbols} | TFs: {cfg.timeframes} | Horizon: {cfg.target_horizon} bars | Lookback: {cfg.lookback}")
    ensure_dir(cfg.outdir)

    all_rows: List[Tuple[str, str, Dict[str, float]]] = []

    for sym in cfg.symbols:
        for tf in cfg.timeframes:
            console.print(f"\n[blue]► {sym} @ {tf}[/blue]")
            try:
                df, feat_cols = load_frame(sym, tf, cfg)
                pipe, metrics, (_y_val, _y_pred) = train_one(df, feat_cols, cfg.target, cfg)
                persist_run(pipe, metrics, feat_cols, cfg, sym, tf)
                all_rows.append((sym, tf, metrics))
            except Exception as e:
                console.print(f"[red]Fehler[/red] bei {sym} {tf}: {e}")
                continue

    if all_rows:
        print_metrics_table(all_rows)
    else:
        console.print("[red]Keine erfolgreichen Runs. Bitte Logs prüfen und sicherstellen, dass features_store funktioniert.[/red]")


if __name__ == "__main__":
    main()
