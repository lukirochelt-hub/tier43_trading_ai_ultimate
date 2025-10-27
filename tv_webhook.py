# tv_webhook.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
import os
import json

TV_WEBHOOK_SECRET = os.getenv("TV_WEBHOOK_SECRET", "YOUR_WEBHOOK_SECRET")

app = FastAPI(title="Tier4.3+ TV Relay")


class TVMsg(BaseModel):
    secret: str
    symbol: str
    tf: str
    time: int
    edge: float
    solR: float
    btcR: float
    ethR: float
    long: bool | str | None = None
    short: bool | str | None = None


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() == "true"
    return False


@app.post("/tv")
async def tv_relay(msg: TVMsg):
    if msg.secret != TV_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")

    event = {
        "recv_ts": datetime.now(timezone.utc).isoformat(),
        "bar_ts": datetime.fromtimestamp(msg.time / 1000, tz=timezone.utc).isoformat(),
        "symbol": msg.symbol,
        "tf": msg.tf,
        "edge": float(msg.edge),
        "solR": float(msg.solR),
        "btcR": float(msg.btcR),
        "ethR": float(msg.ethR),
        "long": _to_bool(msg.long),
        "short": _to_bool(msg.short),
    }

    os.makedirs("data/tv", exist_ok=True)
    with open("data/tv/relay.ndjson", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return {"ok": True}
