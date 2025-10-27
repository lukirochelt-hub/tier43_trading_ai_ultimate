# adapters/adapters_feeds.py
# Tier 4.3+ (Windows/py311-ready)
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import aiohttp


# ----------------------------- Core types ------------------------------------
@dataclass
class CandleRequest:
    symbol: str
    interval: str
    limit: int = 500  # per request (exch. caps to 1000)


class FeedAdapter:
    """Base class for all exchange adapters (Binance, Bybit, ...)."""

    async def candles(self, req: CandleRequest) -> pd.DataFrame:
        raise NotImplementedError

    async def ticker(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    # Optional – ein Adapter *kann* Pagination unterstützen:
    async def candles_paged(
        self,
        req: CandleRequest,
        start: Optional[int] = None,  # ms UTC
        end: Optional[int] = None,  # ms UTC
        limit_total: int = 5000,
        sleep_s: float = 0.15,
    ) -> pd.DataFrame:
        # Default: Fallback auf eine einzelne Seite
        return await self.candles(
            CandleRequest(req.symbol, req.interval, min(limit_total, 1000))
        )


# ----------------------------- Binance ---------------------------------------
class BinanceAdapter(FeedAdapter):
    BASE = "https://api.binance.com"
    TIMEOUT = aiohttp.ClientTimeout(total=20)

    # Mapping unserer Intervalle zu Binance-Parametern
    INTERVAL_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.BASE}{path}"
        # kleine, robuste Retries
        for delay in (0.0, 0.5, 1.0, 2.0):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with aiohttp.ClientSession(timeout=self.TIMEOUT) as s:
                    async with s.get(url, params=params) as r:
                        r.raise_for_status()
                        ct = (r.headers.get("Content-Type") or "").lower()
                        return await (r.json() if "json" in ct else r.text())
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if delay == 2.0:
                    raise

    async def candles(self, req: CandleRequest) -> pd.DataFrame:
        interval = self.INTERVAL_MAP.get(req.interval, req.interval)
        params = {
            "symbol": req.symbol.upper(),
            "interval": interval,
            "limit": min(int(req.limit), 1000),
        }
        data = await self._get("/api/v3/klines", params=params)
        if not isinstance(data, list) or not data:
            return pd.DataFrame(
                columns=["open_time", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "num_trades",
                "taker_base_vol",
                "taker_quote_vol",
                "ignore",
            ],
        )
        df = df.astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            }
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(
            drop=True
        )

    def _interval_ms(self, interval: str) -> int:
        minutes_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
            "1M": 43200,
        }
        return 60_000 * minutes_map.get(interval, 1)

    async def candles_paged(
        self,
        req: CandleRequest,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit_total: int = 5000,
        sleep_s: float = 0.15,
    ) -> pd.DataFrame:
        """Paginiertes Laden für >1000 Kerzen (Binance)."""
        remaining = int(limit_total)
        interval_param = self.INTERVAL_MAP.get(req.interval, req.interval)
        rows: list[dict] = []
        next_start = start

        while remaining > 0:
            batch = min(1000, remaining)
            params = {
                "symbol": req.symbol.upper(),
                "interval": interval_param,
                "limit": batch,
            }
            if next_start is not None:
                params["startTime"] = int(next_start)
            if end is not None:
                params["endTime"] = int(end)

            data = await self._get("/api/v3/klines", params=params)
            if not data:
                break

            for k in data:
                rows.append(
                    {
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )

            remaining -= len(data)
            next_start = int(data[-1][6]) + 1  # closeTime + 1ms
            if len(data) < batch:
                break
            if sleep_s:
                await asyncio.sleep(sleep_s)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.sort_values("open_time").reset_index(drop=True)

    async def ticker(self, symbol: str) -> Dict[str, Any]:
        data = await self._get("/api/v3/ticker/24hr", params={"symbol": symbol.upper()})
        last = float(data.get("lastPrice", 0.0))
        prev = float(data.get("prevClosePrice", last or 1.0))
        chg = last - prev
        pct = (chg / prev * 100.0) if prev else 0.0
        return {
            "symbol": symbol.upper(),
            "price": last,
            "price_change": chg,
            "price_change_pct": pct,
        }


# ----------------------------- Bybit (v5) -------------------------------------
class BybitAdapter(FeedAdapter):
    BASE = "https://api.bybit.com"
    TIMEOUT = aiohttp.ClientTimeout(total=20)

    # Offizielle v5 Intervalle
    INTERVAL_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.BASE}{path}"
        for delay in (0.0, 0.5, 1.0, 2.0):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with aiohttp.ClientSession(timeout=self.TIMEOUT) as s:
                    async with s.get(url, params=params) as r:
                        r.raise_for_status()
                        ct = (r.headers.get("Content-Type") or "").lower()
                        return await (r.json() if "json" in ct else r.text())
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if delay == 2.0:
                    raise

    async def _candles_category(
        self, category: str, req: CandleRequest
    ) -> pd.DataFrame:
        interval = self.INTERVAL_MAP.get(req.interval, req.interval)
        params = {
            "category": category,
            "symbol": req.symbol.upper(),
            "interval": interval,
            "limit": min(int(req.limit), 1000),
        }
        data = await self._get("/v5/market/kline", params=params)
        lst = (data or {}).get("result", {}).get("list", []) or []
        if not lst:
            return pd.DataFrame(
                columns=["open_time", "open", "high", "low", "close", "volume"]
            )

        # Bybit liefert Strings, Reihenfolge: [start, open, high, low, close, volume, turnover]
        rows = [
            {
                "open_time": int(x[0]),
                "open": float(x[1]),
                "high": float(x[2]),
                "low": float(x[3]),
                "close": float(x[4]),
                "volume": float(x[5]),
            }
            for x in lst
        ]
        df = pd.DataFrame(rows)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.sort_values("open_time").reset_index(drop=True)

    async def candles(self, req: CandleRequest) -> pd.DataFrame:
        # erst Perps ("linear"), dann Fallback "spot"
        df = await self._candles_category("linear", req)
        if df.empty:
            df = await self._candles_category("spot", req)
        return df

    async def candles_paged(
        self,
        req: CandleRequest,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit_total: int = 5000,
        sleep_s: float = 0.15,
    ) -> pd.DataFrame:
        """Paginiertes Laden für >1000 Kerzen (Bybit v5, linear→spot)."""

        def _to_minutes(iv: str) -> int:
            ivb = self.INTERVAL_MAP.get(iv, iv)
            if ivb in {"D", "W", "M"}:
                return {"D": 1440, "W": 10080, "M": 43200}[ivb]
            return int(ivb)

        interval_minutes = _to_minutes(req.interval)
        rows: list[dict] = []
        remaining = int(limit_total)

        async def _one_category(cat: str) -> None:
            nonlocal rows, remaining
            next_start = start
            while remaining > 0:
                batch = min(1000, remaining)
                params = {
                    "category": cat,
                    "symbol": req.symbol.upper(),
                    "interval": self.INTERVAL_MAP.get(req.interval, req.interval),
                    "limit": batch,
                }
                if next_start is not None:
                    params["start"] = int(next_start)
                if end is not None:
                    params["end"] = int(end)

                data = await self._get("/v5/market/kline", params=params)
                lst = (data or {}).get("result", {}).get("list", []) or []
                if not lst:
                    break

                # sicherheitshalber sortieren (Bybit kann auf/absteigend liefern)
                lst = sorted(lst, key=lambda x: int(x[0]))
                for x in lst:
                    rows.append(
                        {
                            "open_time": int(x[0]),
                            "open": float(x[1]),
                            "high": float(x[2]),
                            "low": float(x[3]),
                            "close": float(x[4]),
                            "volume": float(x[5]),
                        }
                    )

                remaining -= len(lst)
                next_start = int(lst[-1][0]) + interval_minutes * 60_000
                if len(lst) < batch:
                    break
                if sleep_s:
                    await asyncio.sleep(sleep_s)

        # Versuch in linear, dann spot
        await _one_category("linear")
        if remaining > 0 and not rows:
            await _one_category("spot")

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.sort_values("open_time").reset_index(drop=True)

    async def ticker(self, symbol: str) -> Dict[str, Any]:
        async def _fetch_cat(cat: str) -> Optional[Dict[str, Any]]:
            data = await self._get(
                "/v5/market/tickers", params={"category": cat, "symbol": symbol.upper()}
            )
            items = (data or {}).get("result", {}).get("list", []) or []
            if not items:
                return None
            it = items[0]
            last = float(it.get("lastPrice") or it.get("last_price") or 0.0)
            prev = float(
                it.get("prevPrice24h") or it.get("prev_price_24h") or (last or 1.0)
            )
            chg = last - prev
            pct = (chg / prev * 100.0) if prev else 0.0
            return {
                "symbol": symbol.upper(),
                "price": last,
                "price_change": chg,
                "price_change_pct": pct,
            }

        t = await _fetch_cat("linear")
        if t is None:
            t = await _fetch_cat("spot")
        return t or {
            "symbol": symbol.upper(),
            "price": None,
            "price_change": 0.0,
            "price_change_pct": 0.0,
        }


# ----------------------------- Registry & helpers -----------------------------
def _registry() -> dict[str, FeedAdapter]:
    # Lazy Instanzen, damit Klassennamen sicher definiert sind
    return {
        "binance": BinanceAdapter(),
        "bybit": BybitAdapter(),
    }


async def get_candles(
    source: str, symbol: str, interval: str, limit: int = 500
) -> pd.DataFrame:
    adapter = _registry().get(source)
    if adapter is None:
        raise RuntimeError(
            f"Unknown source '{source}'. Available: {list(_registry().keys())}"
        )
    return await adapter.candles(CandleRequest(symbol, interval, limit))


async def get_candles_paged(
    source: str,
    symbol: str,
    interval: str,
    limit_total: int = 5000,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> pd.DataFrame:
    adapter = _registry().get(source)
    if adapter is None:
        raise RuntimeError(
            f"Unknown source '{source}'. Available: {list(_registry().keys())}"
        )
    # nutzt Adapter-Implementierung, fällt sonst auf normale candles zurück
    return await adapter.candles_paged(
        CandleRequest(symbol, interval, min(limit_total, 1000)), start, end, limit_total
    )


async def get_ticker(source: str, symbol: str) -> Dict[str, Any]:
    adapter = _registry().get(source)
    if adapter is None:
        raise RuntimeError(
            f"Unknown source '{source}'. Available: {list(_registry().keys())}"
        )
    return await adapter.ticker(symbol)


# ----------------------------- CLI self-test ----------------------------------
async def _selftest() -> None:
    print("[adapters_feeds] selftest: binance BTCUSDT 15m (<=100)...")
    df = await get_candles("binance", "BTCUSDT", "15m", 100)
    print(
        "rows:",
        len(df),
        "last close:",
        float(df["close"].iloc[-1]) if not df.empty else "NA",
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        asyncio.run(_selftest())
