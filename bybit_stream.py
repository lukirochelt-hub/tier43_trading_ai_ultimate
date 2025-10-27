# bybit_stream.py
from __future__ import annotations
import asyncio, json, signal
import websockets

WS_URL = "wss://stream.bybit.com/v5/public/linear"
TOPIC  = "liquidation.BTCUSDT"  # <- hier Symbol anpassen

async def _heartbeat(ws, interval=20):
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send(json.dumps({"op": "ping"}))
    except asyncio.CancelledError:
        return

async def run_bybit_stream():
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                # subscribe
                await ws.send(json.dumps({"op": "subscribe", "args": [TOPIC]}))
                hb = asyncio.create_task(_heartbeat(ws, 20))
                print(f"[bybit] subscribed {TOPIC}")
                async for msg in ws:
                    data = json.loads(msg)
                    # Status / pong etc.
                    if isinstance(data, dict) and data.get("op") in {"subscribe","pong"}:
                        continue
                    # Data frames
                    if data.get("topic") == TOPIC and data.get("data"):
                        for d in data["data"]:
                            side = d.get("side")
                            px   = d.get("price")
                            sz   = d.get("size")
                            print(f"[BYBIT] {TOPIC} {side} {px} × {sz}")
                hb.cancel()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print("[bybit] error:", e)
        # reconnect with backoff
        await asyncio.sleep(backoff)
        backoff = min(backoff * 1.7, 15.0)

async def main():
    # clean shutdown via Ctrl+C
    task = asyncio.create_task(run_bybit_stream())
    try:
        await task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    # nicer Ctrl+C on Windows too
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
