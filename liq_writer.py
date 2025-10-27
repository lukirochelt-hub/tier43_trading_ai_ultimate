# liq_writer.py
# Tier 4.3+ – Binance & Bybit Liquidation Writer (NDJSON)
# - Schreibt Live-Liquidationen als NDJSON nach data/liqs.ndjson
# - Quelle: Binance (forceOrder) + Bybit (v5 public/linear)
# - Konfig per ENV:
#     LIQ_SYMBOL = BTCUSDT (Standard)
#     LIQ_PATH   = data/liqs.ndjson
# - UTF-8 sicher, Windows/Unix kompatibel
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import websockets
from websockets.asyncio.client import connect  # re-export für mypy Signatur


# -------------------------
# Konfiguration
# -------------------------
SYMBOL = os.getenv("LIQ_SYMBOL", "BTCUSDT").upper()
OUT_PATH = os.getenv("LIQ_PATH", os.path.join("data", "liqs.ndjson"))

BINANCE_URI = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@forceOrder"
BYBIT_URI = "wss://stream.bybit.com/v5/public/linear"  # Topic: liquidation.<SYMBOL>


# -------------------------
# Utils
# -------------------------
def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _ts_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _write_ndjson(path: str, obj: Dict[str, Any]) -> None:
    # Append atomar; mit \n
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class LiqRecord:
    src: str          # 'binance' | 'bybit'
    symbol: str
    side: str         # 'BUY'/'SELL' (binance) oder 'Buy'/'Sell' (bybit) -> normalisiert
    price: float
    qty: float
    ts: int           # ms

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Binance Stream
# -------------------------
async def binance_stream(symbol: str, out_path: str) -> None:
    uri = f"wss://fstream.binance.com/ws/{symbol.lower()}@forceOrder"
    # Keine positional extra args benutzen -> mypy-safe: nur Keyword-Args
    # websockets nimmt Proxy aus ENV-Variablen; extra_headers optional
    while True:
        try:
            async with connect(uri) as ws:
                print(f"[binance] connected {uri}")
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    # Beispiel-Format: data["o"] enthält Order
                    o = data.get("o") or {}
                    sym = _safe_str(o.get("s")) or symbol
                    px = _safe_float(o.get("p"))
                    qty = _safe_float(o.get("q"))
                    side = _safe_str(o.get("S")).upper()  # BUY/SELL
                    if px is None or qty is None or not side:
                        continue
                    rec = LiqRecord(
                        src="binance",
                        symbol=sym.upper(),
                        side=side,
                        price=float(px),
                        qty=float(qty),
                        ts=_ts_ms(),
                    )
                    _write_ndjson(out_path, rec.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Backoff bei Fehlern
            print(f"[binance] error: {e!r}")
            await asyncio.sleep(1.0)


# -------------------------
# Bybit Stream (v5 public/linear)
# -------------------------
async def bybit_stream(symbol: str, out_path: str) -> None:
    # Offizieller Topic-Name: liquidation.<SYMBOL>
    topic = f"liquidation.{symbol.upper()}"
    sub_msg = {"op": "subscribe", "args": [topic]}
    while True:
        try:
            async with connect(BYBIT_URI) as ws:
                print(f"[bybit] subscribed {topic}")
                await ws.send(json.dumps(sub_msg))
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue

                    # Ping/Pong & Status
                    if isinstance(data, dict) and data.get("op") in ("ping", "pong", "subscribe"):
                        # Optional: print Status
                        # print("[bybit/status]", data)
                        continue

                    if data.get("topic") != topic:
                        continue
                    rows = data.get("data") or []
                    for d in rows:
                        # Bybit-Felder: side ('Buy'/'Sell'), price, size, symbol
                        sym = _safe_str(d.get("symbol")) or symbol
                        px = _safe_float(d.get("price"))
                        qty = _safe_float(d.get("size"))  # kann Base-Size sein
                        side_raw = _safe_str(d.get("side"))
                        side = side_raw.upper() if side_raw else ""
                        if px is None or qty is None or not side:
                            continue
                        rec = LiqRecord(
                            src="bybit",
                            symbol=sym.upper(),
                            side=side,  # BUY/SELL normalisiert
                            price=float(px),
                            qty=float(qty),
                            ts=_ts_ms(),
                        )
                        _write_ndjson(out_path, rec.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[bybit] error: {e!r}")
            await asyncio.sleep(1.0)


# -------------------------
# Main / Runner
# -------------------------
async def _amain(symbol: str, out_path: str) -> int:
    _ensure_parent_dir(out_path)
    # beide Streams parallel
    tasks = [
        asyncio.create_task(binance_stream(symbol, out_path)),
        asyncio.create_task(bybit_stream(symbol, out_path)),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return 0
    return 0


def main() -> int:
    try:
        return asyncio.run(_amain(SYMBOL, OUT_PATH))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
