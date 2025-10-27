import json, os, hmac, hashlib, time

payload = {
    "ts": int(time.time()*1000),   # 64-bit -> kein Overflow-Problem
    "symbol": "BTCUSDT",
    "tf": "1h",
    "direction": "buy",
    "prob": 0.7,
}
body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
secret = os.environ.get("ADV_SECRET", "change_me_42").encode()
sig = hmac.new(secret, body, hashlib.sha256).hexdigest()

print(body.decode())  # Zeile 1: JSON-String (kompakt + sortiert)
print(sig)            # Zeile 2: Hex-Signatur
