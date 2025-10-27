# adv_val.py
# Minimal & saubere Eingangsvalidierung für "Advice"-Nachrichten
# - Python 3.11.9 / Pydantic v2-kompatibel
# - MyPy-clean
# - API: fastapi_validate(headers, payload, require_sig=...) -> AdviceOut
# - HMAC-Signatur wie in app_trading_ai_secure_plus.py (compact + sort_keys)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Literal
import os
import json
import hmac
import hashlib
import time

from pydantic import BaseModel, Field, ValidationError


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
__all__ = [
    "AdvError",
    "AdviceContext",
    "AdviceIn",
    "AdviceOut",
    "fastapi_validate",
    "validate_advice",
]

# ------------------------------------------------------------
# Fehlerklasse
# ------------------------------------------------------------
class AdvError(ValueError):
    """Validierungs-/Signaturfehler für Advice-Payloads."""


# ------------------------------------------------------------
# Timeframe-Normierung
# ------------------------------------------------------------
ALLOWED_TFS: set[str] = {"1m", "5m", "15m", "1h", "4h", "1d"}


def _normalize_tf(t: str) -> str:
    """Normalize timeframes like '60m' -> '1h', '240m' -> '4h'."""
    s = (t or "").strip().lower()
    s = s.replace("hours", "h").replace("hour", "h").replace("hr", "h")
    if s == "60m":
        s = "1h"
    if s == "240m":
        s = "4h"
    if s in ALLOWED_TFS:
        return s
    raise ValueError(f"unsupported timeframe: {s}")


# ------------------------------------------------------------
# Modelle
# ------------------------------------------------------------
class AdviceIn(BaseModel):
    """Eingehende Trading-Advice-Payload, z.B. von einem Upstream-Service."""
    advice_id: Optional[str] = None
    ts: int = Field(..., description="Epoch ms")
    symbol: str
    tf: str
    direction: Literal["buy", "sell", "flat"]
    prob: float = Field(..., ge=0.0, le=1.0)
    meta: Dict[str, Any] = Field(default_factory=dict)


class AdviceOut(BaseModel):
    """Bereits validierte Advice-Message, normalisiert."""
    advice_id: str
    ts: int
    symbol: str
    tf: str
    direction: Literal["buy", "sell", "flat"]
    prob: float
    meta: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class AdviceContext:
    """Einfache Prüf-Konfiguration."""
    max_skew_sec: float = 60.0
    require_sig: bool = False
    secret: str = ""  # leerlassen -> aus ENV lesen


# ------------------------------------------------------------
# JSON/HMAC Utilities (kompatibel mit app_trading_ai_secure_plus._sign_payload)
# ------------------------------------------------------------
def _stable_json_bytes(payload: Dict[str, Any]) -> bytes:
    """Compact + sort_keys = True: deterministische Serialisierung als Bytes."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _hmac_sha256_hex(secret: str, data: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), data, hashlib.sha256).hexdigest()


def _get_header_ci(headers: Mapping[str, Any], key: str) -> Optional[str]:
    """Case-insensitive Header-Abfrage (liefert str oder None)."""
    lk = key.lower()
    for k, v in headers.items():
        if str(k).lower() == lk:
            return None if v is None else str(v)
    return None


# ------------------------------------------------------------
# Kernvalidierung
# ------------------------------------------------------------
def validate_advice(
    payload: Dict[str, Any],
    *,
    ctx: AdviceContext,
    raw_body: Optional[bytes] = None,
    provided_sig: Optional[str] = None,
    now_ms: Optional[int] = None,
) -> AdviceOut:
    """Validiert und normalisiert eingehende Advice-Message."""
    # Pydantic-Schema
    try:
        in_msg = AdviceIn(**payload)
    except ValidationError as e:
        raise AdvError(f"schema_invalid: {e}") from e

    # TF normalisieren
    try:
        norm_tf = _normalize_tf(in_msg.tf)
    except ValueError as e:
        raise AdvError(str(e)) from e

    # Zeit-Drift prüfen
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    delta_sec = abs(now_ms - int(in_msg.ts)) / 1000.0
    if delta_sec > ctx.max_skew_sec:
        raise AdvError(f"stale_ts: skew={delta_sec:.2f}s > {ctx.max_skew_sec}s")

    # Optional HMAC-Check
    if ctx.require_sig:
        secret = ctx.secret or os.environ.get("ADV_SECRET", "change_me_42")
        body = raw_body if raw_body is not None else _stable_json_bytes(payload)
        if not provided_sig:
            raise AdvError("signature_missing")
        expected = _hmac_sha256_hex(secret, body)
        if not hmac.compare_digest(provided_sig.lower(), expected.lower()):
            raise AdvError("signature_mismatch")

    # Advice-ID stabil (falls nicht gesetzt)
    advice_id = in_msg.advice_id or hashlib.sha256(
        f"{in_msg.symbol}|{norm_tf}|{in_msg.direction}|{int(in_msg.ts/60000)}".encode("utf-8")
    ).hexdigest()[:24]

    return AdviceOut(
        advice_id=advice_id,
        ts=int(in_msg.ts),
        symbol=in_msg.symbol.strip().upper(),
        tf=norm_tf,
        direction=in_msg.direction,
        prob=float(in_msg.prob),
        meta=in_msg.meta or {},
    )


# ------------------------------------------------------------
# FastAPI-kompatible Helper-Funktion
# ------------------------------------------------------------
def fastapi_validate(
    headers: Mapping[str, Any],
    payload: Dict[str, Any],
    require_sig: bool = True,
) -> AdviceOut:
    """
    - headers: eingehende Request-Header (case-insensitive)
    - payload: bereits geparstes JSON (dict)
    - require_sig: wenn True, wird X-ADV-SIGN HMAC geprüft
    """
    # Header holen (case-insensitive). App verwendet "x-adv-sign".
    sig = _get_header_ci(headers, "x-adv-sign")
    # Konsistenter Body (gleich wie beim Signieren in der App)
    body = _stable_json_bytes(payload)

    ctx = AdviceContext(
        max_skew_sec=float(os.getenv("ADV_MAX_SKEW_SEC", "60")),
        require_sig=require_sig,
        secret=os.getenv("ADV_SECRET", "change_me_42"),
    )
    return validate_advice(
        payload,
        ctx=ctx,
        raw_body=body,
        provided_sig=sig,
        now_ms=int(time.time() * 1000),
    )


# ------------------------------------------------------------
# Mini Selftest
# ------------------------------------------------------------
if __name__ == "__main__":
    ctx = AdviceContext()
    sample = {
        "ts": int(time.time() * 1000),
        "symbol": "BTCUSDT",
        "tf": "60m",
        "direction": "buy",
        "prob": 0.66,
    }
    print(validate_advice(sample, ctx=ctx))
# --- Back-compat for older tests -------------------------------------------------
AdvContext = AdviceContext  # alte Bezeichnung

def validate_and_normalize(payload: dict, *, ctx: AdviceContext | None = None, **kw):
    """Back-Compat-Wrapper für alte Tests."""
    return validate_advice(payload, ctx=ctx or AdviceContext(), **kw)

