# features_store.py — robustes Shim für MyPy & Runtime (Tier 4.3+)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import pandas as pd

# -------------------------------------------------------
# Versuch: echte Implementierung aus dem Paket laden
# -------------------------------------------------------
try:
    import features_store_pkg.features_store as _fspkg
    _FS = getattr(_fspkg, "FeatureStore", None)
    _Req = getattr(_fspkg, "Request", None)
    _add_ind = getattr(_fspkg, "add_indicators", None)
    _prep_xy = getattr(_fspkg, "prepare_xy", None)
    _build_features = getattr(_fspkg, "build_features", None)
    _load_ohlcv = getattr(_fspkg, "load_ohlcv", None)
except Exception:
    _FS = _Req = _add_ind = _prep_xy = _build_features = _load_ohlcv = None


# -------------------------------------------------------
# Fallback-Typen (mypy-sicher, wenn externes Paket fehlt)
# -------------------------------------------------------
@dataclass
class Request:
    symbol: str
    timeframe: str
    limit: int


class FeatureStore:
    """Wrapper oder Fallback für echte FeatureStore-Klasse."""
    def __init__(self) -> None:
        if _FS is not None:
            # dynamische Delegation auf echtes Objekt
            self.__class__ = _FS
        else:
            raise RuntimeError(
                "FeatureStore implementation not available. "
                "Install/use features_store_pkg.features_store."
            )


# -------------------------------------------------------
# Wrapper / Adapter-Funktionen
# -------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if callable(_add_ind):
        return _add_ind(df)
    # noop fallback
    return df


def prepare_xy(df: pd.DataFrame, target: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    if callable(_prep_xy):
        return _prep_xy(df, target=target)
    # Minimaler Fallback
    y = pd.Series([], dtype=int)
    return df.copy(), y


def build_features(*args: Any, **kwargs: Any) -> pd.DataFrame:
    if callable(_build_features):
        return _build_features(*args, **kwargs)
    raise RuntimeError("build_features not implemented in features_store_pkg.features_store.")


def load_ohlcv(*args: Any, **kwargs: Any) -> pd.DataFrame:
    if callable(_load_ohlcv):
        return _load_ohlcv(*args, **kwargs)
    raise RuntimeError("load_ohlcv not implemented in features_store_pkg.features_store.")


# -------------------------------------------------------
# Öffentliche Exporte
# -------------------------------------------------------
__all__ = [
    "FeatureStore",
    "Request",
    "add_indicators",
    "prepare_xy",
    "build_features",
    "load_ohlcv",
]
