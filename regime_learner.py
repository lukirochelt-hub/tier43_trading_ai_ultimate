# -*- coding: utf-8 -*-
"""
regime_learner.py — Tier 4.3+ Market Regime Learner (Windows-safe)

- GMM (soft) oder KMeans (pseudo-soft) via scikit-learn (optional)
- StandardScaler optional; Log/Delta-Transforms optional
- transform(): hängt regime_id, regime_label, regime_conf, regime_prob_k an
- save/load via joblib
"""

from __future__ import annotations
import os, json, time, warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# optional deps (freundliche Fehler, wenn nicht installiert)
try:
    from sklearn.mixture import GaussianMixture
    HAVE_GMM = True
except Exception:
    HAVE_GMM = False

try:
    from sklearn.cluster import KMeans
    HAVE_KMEANS = True
except Exception:
    HAVE_KMEANS = False

try:
    from sklearn.preprocessing import StandardScaler
    HAVE_SCALER = True
except Exception:
    HAVE_SCALER = False

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False


def _env(key: str, default: Optional[str] = None) -> str:
    return os.getenv(key, default if default is not None else "")


def _now_ts() -> int:
    return int(time.time())


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


@dataclass
class RegimeLearnerConfig:
    n_regimes: int = int(_env("REGIME_N", "3"))
    seed: int = int(_env("REGIME_SEED", "1337"))
    max_iter: int = int(_env("REGIME_MAX_ITER", "512"))
    tol: float = float(_env("REGIME_TOL", "1e-3"))
    use_gmm: bool = _env("REGIME_USE_GMM", "1") != "0"
    rolling_window: int = int(_env("REGIME_ROLL_WIN", "5000"))
    min_fit_size: int = int(_env("REGIME_MIN_FIT", "500"))
    feature_cols: List[str] = field(default_factory=list)
    apply_log1p: bool = _env("REGIME_LOG1P", "0") == "1"
    apply_delta: bool = _env("REGIME_DELTA", "0") == "1"
    scale: bool = _env("REGIME_SCALE", "1") != "0"
    label_quantiles: Tuple[float, float] = (0.35, 0.65)
    label_on: str = _env("REGIME_LABEL_ON", "ret")
    prob_temp: float = float(_env("REGIME_PROB_TEMP", "1.0"))
    persist_path: Optional[str] = _env("REGIME_MODEL_PATH", None)
    verbose: int = int(_env("REGIME_VERBOSE", "1"))

    def to_json(self) -> str:
        d = asdict(self); d["feature_cols"] = list(self.feature_cols)
        return json.dumps(d, ensure_ascii=False, indent=2)


class RegimeLearner:
    def __init__(self, cfg: Optional[RegimeLearnerConfig] = None):
        self.cfg = cfg or RegimeLearnerConfig()
        self._scaler = StandardScaler() if (HAVE_SCALER and self.cfg.scale) else None
        self._model: Optional[Tuple[str, object]] = None
        self._fit_ready = False
        self._feature_cols: List[str] = list(self.cfg.feature_cols)

    # ------------ public API ------------
    def fit(self, df: pd.DataFrame) -> "RegimeLearner":
        X, _ = self._prep_features(df, allow_infer=True)
        need = max(self.cfg.min_fit_size, self.cfg.n_regimes * 10)
        if len(X) < need:
            raise ValueError(f"Not enough rows to fit: got {len(X)}, need ≥ {need}")
        self._fit_core(X)
        self._fit_ready = True
        if self.cfg.verbose:
            print(f"[regime_learner] fit ok: rows={len(X)} features={X.shape[1]} model={'GMM' if self._is_gmm() else 'KMeans'}")
        if self.cfg.persist_path:
            self.save(self.cfg.persist_path)
        return self

    def partial_fit(self, df_new: pd.DataFrame) -> "RegimeLearner":
        if not self._feature_cols:
            self._feature_cols = self._infer_feature_cols(df_new)
        X_all, _ = self._prep_features(df_new, allow_infer=False)
        win = min(len(X_all), self.cfg.rolling_window)
        X = X_all[-win:]
        need = max(self.cfg.min_fit_size, self.cfg.n_regimes * 10)
        if len(X) < need:
            if self.cfg.verbose:
                print(f"[regime_learner] partial_fit skipped (rows={len(X)})")
            return self
        self._fit_core(X)
        self._fit_ready = True
        if self.cfg.persist_path:
            self.save(self.cfg.persist_path)
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fit_ready:
            raise RuntimeError("Model not fitted yet. Call fit() first or load().")
        X, _ = self._prep_features(df, allow_infer=False, fit_scaler=False)
        return self._predict_core(X)

    def transform(self, df: pd.DataFrame, out_prefix: str = "regime_prob_") -> pd.DataFrame:
        ids, probs = self.predict(df)
        conf = probs.max(axis=1)
        labels_map = self._labels_by_returns(df, ids)
        labels = np.array([labels_map.get(int(k), "sideways") for k in ids])
        out = df.copy()
        out["regime_id"] = ids.astype(int)
        out["regime_label"] = labels
        out["regime_conf"] = conf
        for k in range(self.cfg.n_regimes):
            out[f"{out_prefix}{k}"] = probs[:, k]
        return out

    def save(self, path: str) -> None:
        if not HAVE_JOBLIB:
            raise RuntimeError("joblib not installed. pip install joblib")
        payload = {
            "cfg": asdict(self.cfg),
            "feature_cols": self._feature_cols,
            "scaler": self._scaler,
            "model": self._model,
            "fit_ready": self._fit_ready,
            "ts": _now_ts(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(payload, path)
        if self.cfg.verbose:
            print(f"[regime_learner] saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "RegimeLearner":
        if not HAVE_JOBLIB:
            raise RuntimeError("joblib not installed. pip install joblib")
        payload = joblib.load(path)
        cfg = RegimeLearnerConfig(**payload["cfg"])
        self = cls(cfg)
        self._feature_cols = payload["feature_cols"]
        self._scaler = payload["scaler"]
        self._model = payload["model"]
        self._fit_ready = payload["fit_ready"]
        if cfg.verbose:
            print(f"[regime_learner] loaded <- {path} (fit_ready={self._fit_ready})")
        return self

    # ------------ internals ------------
    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        excl = {"regime_id", "regime_label", "regime_conf"}
        cols = [c for c in df.columns if c not in excl and np.issubdtype(df[c].dtype, np.number)]
        if self.cfg.label_on in df.columns and self.cfg.label_on not in cols:
            cols.append(self.cfg.label_on)
        if self.cfg.verbose:
            print(f"[regime_learner] feature_cols={cols}")
        return cols

    def _prep_features(self, df: pd.DataFrame, allow_infer: bool, fit_scaler: bool = True) -> Tuple[np.ndarray, List[str]]:
        if not self._feature_cols:
            if self.cfg.feature_cols:
                self._feature_cols = list(self.cfg.feature_cols)
            elif allow_infer:
                self._feature_cols = self._infer_feature_cols(df)
            else:
                raise ValueError("feature_cols not set; call fit() first or provide cfg.feature_cols.")
        X = df[self._feature_cols].astype(float).values
        if self.cfg.apply_delta:
            X[1:] = X[1:] - X[:-1]
            X[0:] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.apply_log1p:
            X = np.sign(X) * np.log1p(np.abs(X))
        if self._scaler is not None:
            if fit_scaler and not self._fit_ready:
                self._scaler.fit(X)
            X = self._scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, self._feature_cols

    def _fit_core(self, X: np.ndarray) -> None:
        rng = np.random.RandomState(self.cfg.seed)
        if self.cfg.use_gmm and HAVE_GMM:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm = GaussianMixture(
                    n_components=self.cfg.n_regimes,
                    covariance_type="full",
                    max_iter=self.cfg.max_iter,
                    tol=self.cfg.tol,
                    random_state=rng,
                    reg_covar=1e-6,
                    init_params="kmeans",
                )
                gm.fit(X)
            self._model = ("gmm", gm)
        else:
            if not HAVE_KMEANS:
                raise RuntimeError("scikit-learn fehlt: installiere mit `pip install scikit-learn`.")
            km = KMeans(
                n_clusters=self.cfg.n_regimes,
                n_init=10,
                max_iter=self.cfg.max_iter,
                tol=self.cfg.tol,
                random_state=rng,
            )
            km.fit(X)
            self._model = ("kmeans", km)

    def _is_gmm(self) -> bool:
        return isinstance(self._model, tuple) and self._model[0] == "gmm"

    def _predict_core(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._is_gmm():
            gm: GaussianMixture = self._model[1]  # type: ignore
            probs = gm.predict_proba(X)
            if self.cfg.prob_temp != 1.0:
                probs = _softmax(np.log(probs + 1e-12) / self.cfg.prob_temp, axis=1)
            ids = np.argmax(probs, axis=1)
            return ids, probs
        else:
            km: KMeans = self._model[1]  # type: ignore
            labels = km.predict(X)
            centers = km.cluster_centers_
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) + 1e-12
            inv = 1.0 / dists
            probs = inv / inv.sum(axis=1, keepdims=True)
            if self.cfg.prob_temp != 1.0:
                probs = _softmax(np.log(probs + 1e-12) / self.cfg.prob_temp, axis=1)
            return labels, probs

    def _labels_by_returns(self, df: pd.DataFrame, ids: np.ndarray) -> Dict[int, str]:
        col = self.cfg.label_on
        if col not in df.columns:
            return {i: "sideways" for i in range(self.cfg.n_regimes)}
        ret = df[col].astype(float).values
        stats: Dict[int, float] = {}
        for k in range(self.cfg.n_regimes):
            mask = ids == k
            stats[k] = float(np.median(ret[mask])) if mask.any() else 0.0
        vals = np.array(list(stats.values()))
        lo, hi = np.quantile(vals, self.cfg.label_quantiles[0]), np.quantile(vals, self.cfg.label_quantiles[1])
        labels: Dict[int, str] = {}
        for k, v in stats.items():
            labels[k] = "bull" if v >= hi else ("bear" if v <= lo else "sideways")
        return labels


# ----------- Helpers: Datei-Loader (CSV/Parquet/Feather) -----------
def _read_table(path_like: str | Path) -> pd.DataFrame:
    p = Path(path_like)
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if ext in (".feather", ".ft"):
        return pd.read_feather(p)
    # Fallback: CSV
    return pd.read_csv(p)


# ---------------- CLI ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Tier 4.3+ RegimeLearner")
    ap.add_argument("--csv", type=str, help="Pfad CSV/Parquet/Feather (fit+transform)")
    ap.add_argument("--save_csv", type=str)
    ap.add_argument("--model_out", type=str)
    ap.add_argument("--model_in", type=str)
    ap.add_argument("--feature_cols", type=str, default="")
    ap.add_argument("--n_regimes", type=int, default=None)
    ap.add_argument("--use_gmm", action="store_true")
    ap.add_argument("--kmeans", action="store_true")
    ap.add_argument("--apply_delta", action="store_true")
    ap.add_argument("--apply_log1p", action="store_true")
    ap.add_argument("--no_scale", action="store_true")
    ap.add_argument("--label_on", type=str, default=None)
    ap.add_argument("--verbose", type=int, default=None)
    args = ap.parse_args()

    cfg = RegimeLearnerConfig()
    if args.n_regimes is not None: cfg.n_regimes = args.n_regimes
    if args.use_gmm: cfg.use_gmm = True
    if args.kmeans: cfg.use_gmm = False
    if args.apply_delta: cfg.apply_delta = True
    if args.apply_log1p: cfg.apply_log1p = True
    if args.no_scale: cfg.scale = False
    if args.label_on is not None: cfg.label_on = args.label_on
    if args.verbose is not None: cfg.verbose = args.verbose
    cfg.feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    rl = None
    if args.model_in:
        rl = RegimeLearner.load(args.model_in)

    if args.csv:
        df = _read_table(args.csv)  # <— jetzt wird CSV/Parquet/Feather automatisch erkannt
        if rl is None:
            rl = RegimeLearner(cfg).fit(df)
        out = rl.transform(df)
        if args.save_csv:
            out.to_csv(args.save_csv, index=False)
        if args.model_out:
            rl.save(args.model_out)
    else:
        if rl is None:
            print("No --csv and no --model_in provided. Nothing to do.")


if __name__ == "__main__":
    main()
