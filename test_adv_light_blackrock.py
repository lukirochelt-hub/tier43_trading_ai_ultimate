import os
import json
import time
import hmac
import hashlib
from typing import Any, cast
from adv_val import validate_and_normalize, AdvContext


def sign(d: dict[str, Any], secret: str) -> str:
    body = json.dumps(d, separators=(",", ":"), sort_keys=True).encode()
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def one(symbol: str, tf: str, long: bool | None = None, short: bool | None = None) -> None:
    msg: dict[str, Any] = {
        "symbol": symbol,
        "tf": tf,
        "ts": int(time.time() * 1000),
        "long": long,
        "short": short,
        "prob": 0.72,
        "edge": 0.10,
        "strategy": "light_blackrock_v1",
        "config": {"pos_size": 10000, "bias": "SOL", "hedge": "BTC"},
    }
    msg["sig"] = sign(msg, os.environ.get("ADV_SECRET", "change_me_42"))

    # mypy-safe: Funktionsobjekt auf Any casten, um variable Signaturen zu erlauben
    val_fn = cast(Any, validate_and_normalize)
    try:
        out = val_fn(msg, AdvContext(require_sig=True))
    except TypeError:
        # Fallback falls validate_and_normalize nur ein Argument akzeptiert
        out = validate_and_normalize(msg)

    print(symbol, tf, "-> OK", out.symbol, out.tf, out.bar_ts)


if __name__ == "__main__":
    for sym in ("SOLUSDT", "BTCUSDT"):
        for tf in ("5m", "15m", "30m", "1h", "4h", "1d"):
            one(sym, tf, long=True)
