"""
tv_relay.py – Tier 4.3+ Secure TradingView Relay
------------------------------------------------
Empfängt Webhooks von TradingView, prüft & signiert sie,
und leitet sie sicher an den lokalen KI-Server (FastAPI) weiter.

Features:
- HMAC-SHA256 Signierung (SECRET_KEY aus .env)
- Dedupe / Replay-Guard (nonce)
- Async HTTPX Client mit Timeout + Retry
- Lead/Lag Filter, Spread Gate, Qual-Filter
- Logging in relay.log
"""

import os
import time
import hmac
import hashlib
import json
import asyncio
import logging
from collections import deque
from typing import Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx

# -------------------------
# Config
# -------------------------
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/tv")
RELAY_PORT = int(os.getenv("RELAY_PORT", "9000"))
MAX_BODY_KB = 2048
RATE_LIMIT_PER_MIN = 90
DEDUP_TTL_SEC = 1800  # 30 min
TIMEOUT_SEC = 8
RETRY_DELAY = 2.0

# optional Filter
SPREAD_MAX_BPS = float(os.getenv("RELAY_SPREAD_MAX_BPS", "12"))
QUAL_MIN = float(os.getenv("RELAY_QUAL_MIN", "0.05"))

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("tv_relay")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("relay.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)

app = FastAPI(title="Tier 4.3+ TradingView Relay")

_rate_hits: Dict[str, deque] = {}
_seen_nonce: Dict[str, float] = {}


# -------------------------
# Security helpers
# -------------------------
def make_hmac(data: bytes) -> str:
    return hmac.new(SECRET_KEY.encode("utf-8"), data, hashlib.sha256).hexdigest()


def rate_limited(ip: str) -> bool:
    now = time.time()
    dq = _rate_hits.setdefault(ip, deque())
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        return True
    dq.append(now)
    return False


def dedupe(nonce: str) -> bool:
    now = time.time()
    # purge expired
    for k, t in list(_seen_nonce.items()):
        if now - t > DEDUP_TTL_SEC:
            _seen_nonce.pop(k, None)
    if nonce in _seen_nonce:
        return True
    _seen_nonce[nonce] = now
    return False


# -------------------------
# Core relay logic
# -------------------------
async def forward_to_server(payload: dict):
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = make_hmac(data)

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as cli:
        for attempt in range(3):
            try:
                r = await cli.post(
                    SERVER_URL, content=data, headers={"x-signature": sig}
                )
                if r.status_code == 200:
                    logger.info(
                        "✅ Relay success %s %s",
                        payload.get("symbol"),
                        payload.get("tf"),
                    )
                    return r.json()
                else:
                    logger.warning("Relay fail [%s] %s", r.status_code, r.text)
            except Exception as e:
                logger.warning("Relay error %s (try %d)", str(e), attempt + 1)
            await asyncio.sleep(RETRY_DELAY)
    return {"ok": False, "error": "relay failed after retries"}


# -------------------------
# Main endpoint
# -------------------------
@app.post("/webhook")
async def relay(request: Request):
    ip = request.client.host if request.client else "?"
    if rate_limited(ip):
        raise HTTPException(429, "rate limit")

    raw = await request.body()
    if len(raw) > MAX_BODY_KB * 1024:
        raise HTTPException(413, "payload too large")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "invalid JSON")

    nonce = str(payload.get("nonce", ""))
    if dedupe(nonce):
        return {"ok": True, "skipped": "duplicate nonce"}

    # --------------- Optional Pre-filters ---------------
    try:
        ind = payload
        spread = float(ind.get("spread_bps", 0) or 0)
        qual = float(ind.get("qual", 0) or 0)
        btc_neg = bool(int(ind.get("btc_neg_impulse", 0) or 0))
        btc_pos = bool(int(ind.get("btc_pos_impulse", 0) or 0))
        side = ind.get("side", "")
        conf = float(ind.get("confidence", 0) or 0)

        if spread > SPREAD_MAX_BPS:
            logger.info("skip spread gate %.1f > %.1f", spread, SPREAD_MAX_BPS)
            return {"ok": True, "skipped": f"spread too high {spread:.1f}"}

        if qual < QUAL_MIN:
            logger.info("skip qual gate %.3f < %.3f", qual, QUAL_MIN)
            return {"ok": True, "skipped": f"qual below {QUAL_MIN}"}

        if (side == "long" and btc_neg) or (side == "short" and btc_pos):
            logger.info("skip btc impulse gate %s", side)
            return {"ok": True, "skipped": "btc impulse gate"}

        if conf < 0.50:
            logger.info("skip conf low %.2f", conf)
            return {"ok": True, "skipped": "low confidence"}

    except Exception as e:
        logger.warning("prefilter err %s", e)

    # --------------- Forward ---------------
    res = await forward_to_server(payload)
    return JSONResponse(res)


# -------------------------
# Root / health
# -------------------------
@app.get("/")
async def root():
    return {"tier": "4.3+", "ok": True, "server": SERVER_URL}


# -------------------------
# Run hint
# -------------------------
if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Relay running on http://127.0.0.1:{RELAY_PORT}/webhook → {SERVER_URL}")
    uvicorn.run("tv_relay:app", host="0.0.0.0", port=RELAY_PORT, reload=True)
