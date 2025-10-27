# ====================================================
# test_paged.py – Tier 4.3+ Diagnostics / Pagination Tester
# ----------------------------------------------------
# Purpose:
#   • Tests paginated data loading from adapters (feeds, venues, etc.)
#   • Verifies end-to-end paging, caching, and async batch fetches
#   • Produces a simple performance summary for benchmarking
# ====================================================

from __future__ import annotations
import asyncio
import time
import json
import sys
from typing import Tuple, AsyncGenerator, Any, Protocol

# --- Protocols for mypy type awareness (non-destructive) ---------------------
# WICHTIG: Methoden geben ein AsyncGenerator-Objekt zurück -> KEIN async def im Protocol
class _HasIterOHLCV(Protocol):
    def iter_ohlcv(self, *args: Any, **kwargs: Any) -> AsyncGenerator[list[Any], None]: ...

class _HasIterTrades(Protocol):
    def iter_trades(self, *args: Any, **kwargs: Any) -> AsyncGenerator[list[Any], None]: ...

class _HasLoadBatch(Protocol):
    async def load_batch(self, *args: Any, **kwargs: Any) -> Any: ...
# ----------------------------------------------------------------------------

from adapters.adapters_feeds import FeedAdapter
from adapters_alt_feeds import AltFeedAdapter
from features_store import FeatureStore
from utils_perf import PerfTimer, fmt_ms
from logger import get_logger

log = get_logger("test_paged")

# ===========================================
# Helpers
# ===========================================

async def paged_consume(gen: AsyncGenerator[Any, None], page_sz: int = 100) -> Tuple[int, float]:
    """Consume an async generator in pages."""
    total = 0
    t0 = time.perf_counter()
    async for page in gen:
        total += len(page)
        if total % (page_sz * 5) == 0:
            log.info(f"Fetched {total:,} items so far …")
    t1 = time.perf_counter()
    return total, t1 - t0


async def run_feed_pagination_test(symbol: str = "BTC/USDT", limit: int = 5000):
    """Test pagination across primary & alt adapters."""
    feed: _HasIterOHLCV = FeedAdapter()        # type: ignore[assignment]
    alt: _HasIterTrades = AltFeedAdapter()     # type: ignore[assignment]

    log.info(f"▶ Running pagination test for {symbol} (limit={limit})")

    t = PerfTimer()
    async_gen1 = feed.iter_ohlcv(symbol, limit=limit, page_size=250)
    n1, dur1 = await paged_consume(async_gen1)

    async_gen2 = alt.iter_trades(symbol, limit=limit, page_size=250)
    n2, dur2 = await paged_consume(async_gen2)

    summary = {
        "symbol": symbol,
        "pages_feed": n1,
        "pages_alt": n2,
        "dur_feed_ms": fmt_ms(dur1),
        "dur_alt_ms": fmt_ms(dur2),
        "total_sec": round(dur1 + dur2, 3),
    }

    log.info(f"✅ Pagination complete → {json.dumps(summary, indent=2)}")
    return summary


async def run_featurestore_test(limit: int = 3000):
    """Test batch load + caching throughput on FeatureStore."""
    fs: _HasLoadBatch = FeatureStore()          # type: ignore[assignment]
    t = PerfTimer()
    batch = await fs.load_batch("SOL/USDT", "15m", limit=limit)
    dur = t.ms
    log.info(f"✅ FeatureStore load {len(batch)} items in {fmt_ms(dur)} ms")


async def main():
    log.info("=== Tier 4.3+ Pagination Diagnostic ===")
    await run_feed_pagination_test("BTC/USDT")
    await run_feed_pagination_test("SOL/USDT")
    await run_featurestore_test()


# ===========================================
# Entrypoint
# ===========================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
