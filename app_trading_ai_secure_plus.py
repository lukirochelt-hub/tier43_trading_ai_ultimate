# app_trading_ai_secure_plus.py
# Tier 4.3+ — Secure Advice API (Windows-safe)

from __future__ import annotations
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import Response
from pathlib import Path
import os, json, hmac, hashlib
from typing import Any, Dict, Mapping, cast


import pandas as pd

from adv_val import fastapi_validate, AdvError
from quick_signal import quick_signal, to_advice_dict


# ---------- Optional: Regime Learner (wenn Modell vorhanden) ----------
try:
    from regime_learner import RegimeLearner
    HAVE_RL = True
except Exception:
    HAVE_RL = False

# ---------- App & Paths ----------
app = FastAPI()
LOG = Path("data/advices.ndjson")
LOG.parent.mkdir(parents=True, exist_ok=True)

RL_PATH = "models/rl.joblib"
_rl = None
if HAVE_RL and os.path.exists(RL_PATH):
    try:
        _rl = RegimeLearner.load(RL_PATH)
    except Exception as e:
        print("[api] regime model load failed:", e)

# ---------- Optional: /metrics ----------
ADV_PROM_ENABLE = os.getenv("ADV_PROM_ENABLE", "0") == "1"
if ADV_PROM_ENABLE:
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        @app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        print("[api] prometheus disabled:", e)

# ---------- Helpers ----------
def _sign_payload(payload: dict) -> tuple[str, bytes]:
    """
    Serialisiert stabil (compact + sort_keys) und erzeugt HMAC-SHA256 Hex.
    """
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    secret = os.environ.get("ADV_SECRET", "change_me_42")
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return sig, body

def _append_ndjson(obj: dict, path: Path = LOG) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")

# ---------- /advice: bereits signierte Requests annehmen ----------
@app.post("/advice")
async def advice(payload: dict, request: Request):
    try:
        out = fastapi_validate(dict(request.headers), payload, require_sig=True)
        _append_ndjson(out.dict(), LOG)
        return {"ok": True, "accepted": True, "norm": out.dict()}
    except AdvError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- /tv_webhook: QS + Regime + adv_val ----------
@app.post("/tv_webhook")
async def tv_webhook(request: Request):
    """
    Test-/Relay-Endpunkt.
    Erwartet optional JSON:
      { "csv":"data/sol_5m.csv", "symbol":"SOLUSDT", "tf":"5m", "last_n":400 }
    """
        # ---- Eingabe lesen ----
    try:
        raw_payload = await request.json()
        payload_in: Mapping[str, Any] = cast(Mapping[str, Any], raw_payload if isinstance(raw_payload, dict) else {})
    except Exception:
        payload_in = {}

    def _as_float(x: Any, default: float) -> float:
        try:
            return float(x)  # erlaubt str, int, float
        except Exception:
            return default

    def _as_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    csv_path = cast(str, payload_in.get("csv", "data/sol_5m.csv"))
    symbol   = cast(str, payload_in.get("symbol", "SOLUSDT"))
    tf       = cast(str, payload_in.get("tf", "5m"))
    last_n   = _as_int(payload_in.get("last_n"), 400)

    # ---- CSV laden (defensiv) ----
    try:
        df = pd.read_csv(csv_path).tail(last_n)
    except Exception as e:
        return {"ok": False, "error": f"csv_read_failed: {e}", "csv": csv_path}

    if "close" not in df.columns:
        return {"ok": False, "error": "missing_column: 'close' not found", "columns": list(df.columns)}

    closes = df["close"].astype(float).tolist()
    highs  = df["high"].astype(float).tolist() if "high" in df else None
    lows   = df["low"].astype(float).tolist()  if "low"  in df else None

    quick_signal(symbol, tf, closes, highs=highs, lows=lows)


    # >>> Direkt VOR dem quick_signal-Aufruf einfügen:
    try:
        vt     = float(payload_in.get("vol_threshold", 1.05))
        rsi_ob = float(payload_in.get("rsi_ob", 85.0))
        rsi_os = float(payload_in.get("rsi_os", 15.0))
        xlook  = int(payload_in.get("cross_lookback", 3))

                # >>> Dann quick_signal so aufrufen:
        res = quick_signal(
            symbol, tf, closes,
            highs=highs, lows=lows,
            cooldown=0,
            vol_threshold=vt, rsi_ob=rsi_ob, rsi_os=rsi_os,
            enable_atr_guard=False,
            cross_lookback=xlook,
        )

        if not res:
            return {"ok": True, "accepted": False, "reason": "no_signal"}

        adv: Dict[str, Any] = to_advice_dict(res, pos_size=10_000, bias_asset="SOL", hedge_asset="BTC")


        if not res:
            return {"ok": True, "accepted": False, "reason": "no_signal"}

        adv = to_advice_dict(res, pos_size=10_000, bias_asset="SOL", hedge_asset="BTC")
    except Exception as e:
        return {"ok": False, "error": f"quick_signal_failed: {e}"}
    
    

        # ---- Optional: Regime-Gating (wenn Modell vorhanden) ----
    if _rl is not None:
        try:
            df_feat = df.copy()
            if "ret_1" not in df_feat.columns:
                df_feat["ret_1"] = df_feat["close"].pct_change().fillna(0.0)
            out = _rl.transform(df_feat.select_dtypes(include="number"))
            reg = str(out["regime_label"].iloc[-1])
            pcols = [c for c in out.columns if c.startswith("regime_prob_")]
            probs = out[pcols].iloc[-1].to_dict() if pcols else {}

            # Gate-Policy (einfach & konservativ)
            if reg == "bull" and adv.get("short"):
                adv["short"] = None
            if reg == "bear" and adv.get("long"):
                adv["long"] = None

            # ✅ hier der neue Teil:
            prob_val = float(adv.get("prob") or 0.0)
            if reg == "sideways" and prob_val < 0.62:
                adv["long"] = adv["short"] = None
                adv["edge"] = None
                adv["prob"] = None

            # Zusatzinfos am Advice (werden gleich in config.ctx verschoben)
            adv["regime"] = reg
            adv["regime_probs"] = probs
        except Exception as e:
            adv["regime_error"] = f"{e}"


    # ---- SANITIZER: nur erlaubte Top-Level-Felder; Extras in config.ctx ----
    ALLOWED = {
        "symbol", "tf", "ts", "long", "short", "prob", "edge",
        "strategy", "config", "advice_id", "solR", "btcR", "ethR", "sig"
    }
    # 1) Extras einsammeln und in config.ctx ablegen
    extras = {k: v for k, v in list(adv.items()) if k not in ALLOWED}
    cfg = adv.get("config") or {}
    ctx = cfg.get("ctx") or {}
    ctx.update(extras)
    cfg["ctx"] = ctx
    adv["config"] = cfg
    for k in extras.keys():
        adv.pop(k, None)

    # 2) Typen knallhart normalisieren (pydantic ist STRICT)
    def _to_bool(x):
        if x is None:
            return None
        return bool(x)

    def _to_float(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def _to_str(x):
        if x is None:
            return None
        return str(x)

    adv["symbol"] = _to_str(adv.get("symbol"))
    adv["tf"] = _to_str(adv.get("tf"))
    adv["ts"] = _to_int(adv.get("ts"))
    if "long" in adv:
        adv["long"] = _to_bool(adv.get("long"))
    if "short" in adv:
        adv["short"] = _to_bool(adv.get("short"))
    if "prob" in adv:
        adv["prob"] = _to_float(adv.get("prob"))
    if "edge" in adv:
        adv["edge"] = _to_float(adv.get("edge"))
    if "solR" in adv:
        adv["solR"] = _to_float(adv.get("solR"))
    if "btcR" in adv:
        adv["btcR"] = _to_float(adv.get("btcR"))
    if "ethR" in adv:
        adv["ethR"] = _to_float(adv.get("ethR"))
    adv["strategy"] = _to_str(adv.get("strategy"))

    # 3) Nulls aufräumen (nur None, 0.0 bleibt erhalten)
    for k in ["long", "short", "prob", "edge", "solR", "btcR", "ethR", "advice_id", "sig"]:
        if adv.get(k, None) is None:
            adv.pop(k, None)
    # leeres config entfernen? -> NEIN: wir brauchen config für ctx
    # aber wenn config leer ist, trotzdem drin lassen – ist erlaubt

    # 4) Business-Regel: Wenn weder Long/Short noch Prob/Edge -> kein Signal
    has_dir = ("long" in adv) or ("short" in adv)
    has_stats = ("prob" in adv) or ("edge" in adv)
    if not (has_dir or has_stats):
        return {"ok": True, "accepted": False, "reason": "no_signal_sanitized"}

    # 5) signieren + validieren + loggen (+ verbose Fehler)
    try:
        sig, body = _sign_payload(adv)
        headers = {"x-adv-sign": sig, "content-type": "application/json"}
        out = fastapi_validate(headers, adv, require_sig=True)
        _append_ndjson(out.dict(), LOG)
        return {"ok": True, "accepted": True, "norm": out.dict()}
    except AdvError as e:
        # volle Schema-Fehlermeldung zurückgeben
        return {
            "ok": False,
            "error": "validation_failed",
            "detail": str(e),
            "sent_adv": adv  # damit wir sehen, was genau gesendet wurde
        }
    except Exception as e:
        return {"ok": False, "error": f"persist_failed: {e}"}
