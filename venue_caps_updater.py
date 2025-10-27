# ==========================================================
# venue_caps_updater.py  |  Tier 4.3+ Trading AI
# ==========================================================
# - Aggregates live margin & position caps from connected venues
# - Supports Binance, Bybit, and optional synthetic feeds
# - Updates Redis / SQLite store for exposure validation
# - Triggers Prometheus metrics + pubsub signals on changes
# ==========================================================

from __future__ import annotations
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, cast
import aiohttp
import redis.asyncio as aioredis

# Local imports
from utils_common import now_ts, async_retry, load_env_bool
from features_store import FeatureCache  # type: ignore[attr-defined]

# ----------------------------
# Config
# ----------------------------
VENUES = os.getenv("VENUE_CAPS_VENUES", "binance,bybit").split(",")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("VENUE_CAPS_TTL", "300"))
POLL_INTERVAL = int(os.getenv("VENUE_CAPS_POLL_INTERVAL", "60"))

PROM_ENABLE = load_env_bool("PROM_ENABLE", True)
PROM_NAMESPACE = "tier43"

# ----------------------------
# Optional Prometheus metrics (mypy-safe)
# ----------------------------
try:
    from prometheus_client import Gauge as _Gauge
except Exception:
    if TYPE_CHECKING:
        from prometheus_client import Gauge as _Gauge  # type: ignore[unused-ignore]
    else:
        _Gauge = cast(Any, None)

g_venue_margin: Optional[Any] = None
g_venue_position: Optional[Any] = None

if PROM_ENABLE and _Gauge is not None:
    try:
        g_venue_margin = _Gauge(
            f"{PROM_NAMESPACE}_venue_margin_cap",
            "Margin cap per venue",
            ["venue"],
        )
        g_venue_position = _Gauge(
            f"{PROM_NAMESPACE}_venue_position_cap",
            "Position cap per venue",
            ["venue"],
        )
    except Exception as e:
        logging.warning(f"[VENUE_CAPS] Prometheus Gauge init failed: {e}")
        g_venue_margin = None
        g_venue_position = None


# ----------------------------
# Core fetchers
# ----------------------------
@async_retry(attempts=3, delay=2)
async def fetch_binance_caps(session: aiohttp.ClientSession) -> Dict[str, Any]:
    async with session.get("https://fapi.binance.com/fapi/v1/balance") as r:
        if r.status != 200:
            raise RuntimeError(f"binance_http_{r.status}")
        data = await r.json()
        total_margin = sum(float(x.get("balance", 0.0)) for x in data)
        return {
            "venue": "binance",
            "margin_cap": total_margin,
            "pos_cap": total_margin * 3,
        }


@async_retry(attempts=3, delay=2)
async def fetch_bybit_caps(session: aiohttp.ClientSession) -> Dict[str, Any]:
    async with session.get(
        "https://api.bybit.com/v5/account/wallet-balance?accountType=UNIFIED"
    ) as r:
        if r.status != 200:
            raise RuntimeError(f"bybit_http_{r.status}")
        data = await r.json()
        total_equity = float(
            data.get("result", {}).get("list", [{}])[0].get("totalEquity", 0.0)
        )
        return {
            "venue": "bybit",
            "margin_cap": total_equity,
            "pos_cap": total_equity * 4,
        }


# ----------------------------
# Updater loop
# ----------------------------
async def update_caps_loop() -> None:
    redis_cli = await aioredis.from_url(REDIS_URL, decode_responses=True)
    feature_cache = FeatureCache()  # type: ignore[call-arg]

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                results: list[Dict[str, Any]] = []
                if "binance" in VENUES:
                    results.append(await fetch_binance_caps(session))
                if "bybit" in VENUES:
                    results.append(await fetch_bybit_caps(session))

                for r in results:
                    venue = r["venue"]
                    margin_cap = float(r["margin_cap"])
                    pos_cap = float(r["pos_cap"])
                    key = f"venue_caps:{venue}"
                    val = {"margin_cap": margin_cap, "pos_cap": pos_cap, "ts": now_ts()}

                    await redis_cli.setex(key, CACHE_TTL, json.dumps(val))

                    # dynamic methods (type ignored for mypy)
                    feature_cache.set_feature(  # type: ignore[attr-defined]
                        f"{venue}_margin_cap", margin_cap, ttl=CACHE_TTL
                    )
                    feature_cache.set_feature(  # type: ignore[attr-defined]
                        f"{venue}_pos_cap", pos_cap, ttl=CACHE_TTL
                    )

                    if g_venue_margin is not None:
                        g_venue_margin.labels(venue=venue).set(margin_cap)
                    if g_venue_position is not None:
                        g_venue_position.labels(venue=venue).set(pos_cap)

                logging.info(f"[VENUE_CAPS] Updated {len(results)} venues.")
            except Exception as e:
                logging.error(f"[VENUE_CAPS] update failed: {e}")
            await asyncio.sleep(POLL_INTERVAL)


# ----------------------------
# Entrypoint
# ----------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logging.info("Starting venue_caps_updater ...")
    asyncio.run(update_caps_loop())


if __name__ == "__main__":
    main()
