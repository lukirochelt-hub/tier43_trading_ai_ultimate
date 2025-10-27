# adapters/adapters_macro.py
# Tier 4.3+ – Macro metrics (crypto-global, FX) + optional FRED (CPI, 10Y)
from __future__ import annotations
import os
import asyncio
from typing import Any, Dict, Optional
import aiohttp
from datetime import datetime, timezone

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=15)
RETRY_BACKOFF = [0.5, 1.0, 2.0]


class HTTPError(RuntimeError): ...


async def _fetch_json(
    session: aiohttp.ClientSession, url: str, params=None, headers=None
):
    for attempt, delay in enumerate([0.0] + RETRY_BACKOFF, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            async with session.get(url, params=params, headers=headers) as r:
                if r.status >= 400:
                    txt = await r.text()
                    raise HTTPError(f"{r.status} {url} :: {txt[:200]}")
                return await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt >= (1 + len(RETRY_BACKOFF)):
                raise


# ---------- Public Crypto Macro (Coingecko Global) ---------------------------
class CryptoGlobal:
    BASE = "https://api.coingecko.com/api/v3/global"

    async def snapshot(self) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as s:
            js = await _fetch_json(s, self.BASE)
        data = (js or {}).get("data", {})
        mcap = (data.get("total_market_cap") or {}).get("usd")
        vol = (data.get("total_volume") or {}).get("usd")
        btc_dom = data.get("market_cap_percentage", {}).get("btc")
        actives = data.get("active_cryptocurrencies")
        return {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "total_mcap_usd": mcap,
            "total_volume_usd": vol,
            "btc_dominance_pct": btc_dom,
            "active_assets": actives,
        }


# ---------- Public FX (USD/EUR) ----------------------------------------------
class FXRates:
    # frankfurter.app ist ein offener EZB-Proxy
    BASE = "https://api.frankfurter.app/latest"

    async def usdeur(self) -> Dict[str, Any]:
        params = {"from": "USD", "to": "EUR"}
        async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as s:
            js = await _fetch_json(s, self.BASE, params=params)
        rate = (js.get("rates") or {}).get("EUR")
        return {
            "ts": js.get("date"),
            "USDEUR": rate,
            "EURUSD": (1.0 / rate) if rate else None,
        }


# ---------- Optional: FRED (needs API key) -----------------------------------
class FRED:
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        self.key = api_key or os.getenv("FRED_API_KEY")

    async def last_value(self, series_id: str) -> Optional[Dict[str, Any]]:
        if not self.key:
            return None
        params = {
            "series_id": series_id,
            "api_key": self.key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as s:
            js = await _fetch_json(s, self.BASE, params=params)
        obs = (js or {}).get("observations") or []
        if not obs:
            return None
        o = obs[0]
        try:
            v = float(o.get("value"))
        except Exception:
            v = None
        return {"series": series_id, "date": o.get("date"), "value": v}

    async def cpi_us(self) -> Optional[Dict[str, Any]]:
        # CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
        return await self.last_value("CPIAUCSL")

    async def us10y(self) -> Optional[Dict[str, Any]]:
        # DGS10: 10-Year Treasury Constant Maturity Rate
        return await self.last_value("DGS10")


# ---------- Facade ------------------------------------------------------------
class MacroFacade:
    def __init__(self):
        self.crypto = CryptoGlobal()
        self.fx = FXRates()
        self.fred = FRED()

    async def snapshot(self) -> Dict[str, Any]:
        cg_task = asyncio.create_task(self.crypto.snapshot())
        fx_task = asyncio.create_task(self.fx.usdeur())
        fred_tasks = []
        if self.fred.key:
            fred_tasks = [
                asyncio.create_task(self.fred.cpi_us()),
                asyncio.create_task(self.fred.us10y()),
            ]

        out: Dict[str, Any] = {}
        cg = await cg_task
        fx = await fx_task
        out.update(cg)
        out.update(fx)

        if fred_tasks:
            try:
                cpi, y10 = await asyncio.gather(*fred_tasks)
                if cpi:
                    out["cpi_us"] = cpi
                if y10:
                    out["us10y"] = y10
            except Exception:
                # falls FRED mal zickt, Snapshot trotzdem liefern
                pass

        return out


# ---------- CLI self-test -----------------------------------------------------
async def _selftest():
    m = MacroFacade()
    snap = await m.snapshot()
    print("[adapters_macro] snapshot:", snap)
    assert (
        "total_mcap_usd" in snap and snap["total_mcap_usd"] is not None
    ), "Missing total market cap"
    assert "EURUSD" in snap and snap["EURUSD"] is not None, "Missing EURUSD"
    print("OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        asyncio.run(_selftest())
