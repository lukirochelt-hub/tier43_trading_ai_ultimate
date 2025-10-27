# adapters_alt_feeds.py
from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
from typing import Any, Dict, Optional, Tuple

import os
import aiohttp
import websockets

# Hinweis: Kein Selbst-Import hier – AltFeedAdapter ist unten lokal definiert.
# (Früher: from adapters.adapters_alt_feeds import AltFeedAdapter)

PROXY_BASE: Optional[str] = os.getenv("T43_PROXY_URL")  # z.B. https://tier43-data-proxy.<user>.workers.dev

# =======================
# CONSTANTS
# =======================
_BINANCE_FAPI = "https://fapi.binance.com"
_BYBIT_API = "https://api.bybit.com"
_OKX_API = "https://www.okx.com"
_COINGECKO = "https://api.coingecko.com/api/v3"
_FNG_API = "https://api.alternative.me/fng/"
_SOLANA_RPC = "https://api.mainnet-beta.solana.com"
_COINGLASS_OI = "https://open-api.coinglass.com/api/pro/v1/futures/openInterest"

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; Tier43/1.0; +https://example.org)"
}
JSON_HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    **DEFAULT_HEADERS,
}

# =======================
# GENERIC HELPERS
# =======================
async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] | None = None) -> Any:
    try:
        async with session.get(url, params=params, headers=DEFAULT_HEADERS) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None

async def _post_json(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]) -> Any:
    try:
        async with session.post(url, data=json.dumps(payload), headers=JSON_HEADERS) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None

def _ms_now_utc() -> int:
    return int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)

def _ms_24h_ago(ms: int) -> int:
    return ms - 24 * 60 * 60 * 1000

# =======================
# EXCHANGE HELPERS
# =======================
async def binance_price_usdt(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    j = await _get_json(session, f"{_BINANCE_FAPI}/fapi/v1/ticker/price", {"symbol": symbol})
    try:
        return float(j["price"])
    except Exception:
        return None

async def binance_open_interest_base(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    j = await _get_json(session, f"{_BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    try:
        return float(j["openInterest"])
    except Exception:
        return None

async def coinglass_open_interest(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    """Fallback via Coinglass public JSON (no API key)."""
    try:
        url = f"{_COINGLASS_OI}?symbol={symbol[:-4]}&currency=USD"
        j = await _get_json(session, url)
        data = (j or {}).get("data", [])
        total = 0.0
        for d in data:
            try:
                total += float(d["openInterestUSD"])
            except Exception:
                continue
        return round(total, 2) if total > 0 else None
    except Exception:
        return None

async def binance_last_funding_rate(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    j = await _get_json(session, f"{_BINANCE_FAPI}/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
    try:
        return float(j[0]["fundingRate"]) if j else None  # type: ignore[index]
    except Exception:
        return None

# ------------------ Liquidations Aggregator ------------------
async def _bybit_liquidations_24h_usd(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    end_ms = _ms_now_utc()
    start_ms = _ms_24h_ago(end_ms)
    total = 0.0
    cursor: Optional[str] = None
    while True:
        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        j = await _get_json(session, f"{_BYBIT_API}/v5/market/liquidation", params)
        lst = (j or {}).get("result", {}).get("list", [])
        if not lst:
            break
        for it in lst:
            try:
                px = float(it.get("price") or 0)
                sz = float(it.get("size") or 0)
                total += abs(px * sz)
            except Exception:
                continue
        cursor = (j or {}).get("result", {}).get("nextPageCursor")
        if not cursor:
            break
        await asyncio.sleep(0.1)
    return round(total, 2) if total > 0 else None

async def _binance_liquidations_24h_usd(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    end_ms = _ms_now_utc()
    start_ms = _ms_24h_ago(end_ms)
    params = {"symbol": symbol, "autoCloseType": "LIQUIDATION", "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    j = await _get_json(session, f"{_BINANCE_FAPI}/fapi/v1/allForceOrders", params)
    total = 0.0
    if isinstance(j, list):
        for it in j:
            try:
                total += abs(float(it.get("price", 0)) * float(it.get("executedQty", 0)))
            except Exception:
                continue
    return round(total, 2) if total > 0 else None

async def _okx_liquidations_24h_usd(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    uly = symbol.replace("USDT", "-USDT")
    params = {"instType": "SWAP", "uly": uly, "state": "filled", "limit": 100}
    j = await _get_json(session, f"{_OKX_API}/api/v5/public/liquidation-orders", params)
    data = (j or {}).get("data", [])
    total = 0.0
    for it in data:
        for d in it.get("details", []) or []:
            try:
                total += abs(float(d.get("px") or 0) * float(d.get("sz") or 0))
            except Exception:
                continue
    return round(total, 2) if total > 0 else None

async def liquidations_24h_usd_combined(symbol: str, session: aiohttp.ClientSession) -> Optional[float]:
    tasks = [
        asyncio.create_task(_bybit_liquidations_24h_usd(symbol, session)),
        asyncio.create_task(_binance_liquidations_24h_usd(symbol, session)),
        asyncio.create_task(_okx_liquidations_24h_usd(symbol, session)),
    ]
    results = await asyncio.gather(*tasks)
    vals = [r for r in results if isinstance(r, (int, float)) and r > 0]
    return round(sum(vals), 2) if vals else None

# =======================
# MARKET METRICS
# =======================
async def coingecko_btc_dominance(session: aiohttp.ClientSession) -> Optional[float]:
    j = await _get_json(session, f"{_COINGECKO}/global")
    try:
        return float(j["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None

async def fear_greed_index(session: aiohttp.ClientSession) -> Optional[float]:
    j = await _get_json(session, _FNG_API)
    try:
        return float(j["data"][0]["value"])
    except Exception:
        return None

async def solana_tps(session: aiohttp.ClientSession) -> Optional[float]:
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": "getRecentPerformanceSamples", "params": [1]}
    j = await _post_json(session, _SOLANA_RPC, payload)
    try:
        s = j["result"][0]  # type: ignore[index]
        return round(float(s["numTransactions"]) / float(s["samplePeriodSecs"]), 2)
    except Exception:
        return None

# =======================
# SMART PUBLIC-DATA PROXY
# =======================
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 300  # 5 min cache

async def get_cached_snapshot(symbol: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    now = dt.datetime.now().timestamp()
    if symbol in _cache and now - _cache[symbol][0] < CACHE_TTL:
        return _cache[symbol][1]
    data = await build_snapshot(symbol, session)
    _cache[symbol] = (now, data)
    return data

# =======================
# BUILD SNAPSHOT
# =======================
async def build_snapshot(symbol: str = "BTCUSDT", session: aiohttp.ClientSession | None = None) -> Dict[str, Any]:
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
    assert session is not None  # mypy hint
    try:
        price_task = asyncio.create_task(binance_price_usdt(symbol, session))
        oi_task = asyncio.create_task(binance_open_interest_base(symbol, session))
        oi_cg_task = asyncio.create_task(coinglass_open_interest(symbol, session))
        fr_task = asyncio.create_task(binance_last_funding_rate(symbol, session))
        liq_task = asyncio.create_task(liquidations_24h_usd_combined(symbol, session))
        dom_task = asyncio.create_task(coingecko_btc_dominance(session))
        fgi_task = asyncio.create_task(fear_greed_index(session))
        tps_task = asyncio.create_task(solana_tps(session))

        price, oi_base, oi_cg, funding, liq24, dominance, fgi, tps = await asyncio.gather(
            price_task, oi_task, oi_cg_task, fr_task, liq_task, dom_task, fgi_task, tps_task
        )

        oi_usd: Optional[float] = None
        if oi_cg is not None:
            oi_usd = oi_cg
        elif oi_base is not None and price is not None:
            oi_usd = float(oi_base) * float(price)

        return {
            "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": symbol,
            "price_usd": price,
            "btc_dominance_pct": dominance,
            "oi_usd": oi_usd,
            "funding_8h": funding,
            "liq_24h_usd": liq24,
            "fgi": fgi,
            "sol_tps": tps,
        }
    finally:
        if own_session:
            await session.close()

# =======================
# REALTIME STREAMING
# =======================
async def stream_liquidations_binance(symbol: str = "btcusdt") -> None:
    uri = f"wss://fstream.binance.com/ws/{symbol.lower()}@forceOrder"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            data = json.loads(msg)
            if data.get("o", {}).get("s") == symbol.upper():
                try:
                    px = float(data["o"].get("p", 0))
                    qty = float(data["o"].get("q", 0))
                    side = str(data["o"].get("S"))
                    print(f"[BINANCE] {symbol.upper()} {side} {px:.2f} × {qty}")
                except Exception:
                    continue

async def stream_liquidations_bybit(symbol: str = "BTCUSDT") -> None:
    uri = "wss://stream.bybit.com/v5/public/linear"
    async with websockets.connect(uri) as ws:
        sub = {"op": "subscribe", "args": [f"liquidation.{symbol}"]}
        await ws.send(json.dumps(sub))
        async for msg in ws:
            data = json.loads(msg)
            if "data" in data:
                for d in data["data"]:
                    print(f"[BYBIT] {symbol} {d.get('side')} {d.get('price')} × {d.get('size')}")

# =======================
# AUTO REFRESH LOOP
# =======================
async def auto_refresh(symbol: str = "BTCUSDT", interval_sec: int = 300) -> None:
    async with aiohttp.ClientSession() as session:
        while True:
            snap = await get_cached_snapshot(symbol, session)
            print(json.dumps(snap, indent=2, ensure_ascii=False))
            await asyncio.sleep(interval_sec)

# =======================
# PUBLIC ADAPTER-KLASSE
# =======================
class AltFeedAdapter:
    """
    Dünner Wrapper über die oben definierten Helfer, damit andere Module
    `from adapters.adapters_alt_feeds import AltFeedAdapter` verwenden können.
    """
    def __init__(self, symbol: str = "BTCUSDT", interval_sec: int = 300) -> None:
        self.symbol = symbol
        self.interval_sec = interval_sec

    async def snapshot(self) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            return await build_snapshot(self.symbol, session)

    async def cached_snapshot(self) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            return await get_cached_snapshot(self.symbol, session)

    async def stream(self) -> None:
        await asyncio.gather(
            stream_liquidations_binance(self.symbol),
            stream_liquidations_bybit(self.symbol),
        )

__all__ = [
    "AltFeedAdapter",
    "build_snapshot",
    "get_cached_snapshot",
    "stream_liquidations_binance",
    "stream_liquidations_bybit",
    "auto_refresh",
]

# =======================
# CLI ENTRY
# =======================
async def _amain() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "stream":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
        await asyncio.gather(
            stream_liquidations_binance(symbol),
            stream_liquidations_bybit(symbol),
        )
    else:
        symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
        await auto_refresh(symbol, 300)

if __name__ == "__main__":
    asyncio.run(_amain())
