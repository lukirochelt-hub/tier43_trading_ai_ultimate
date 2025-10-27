# adapters/adapters_base.py
from __future__ import annotations
from typing import Any
pd: Any = None

try:
    import pandas as pd
except Exception:
    pd = None


class FeedConfigError(RuntimeError):
    pass


class CandleRequest:
    def __init__(self, symbol: str, interval: str, limit: int = 500):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit


class FeedAdapter:
    """Base class for all exchange adapters."""

    async def candles(self, req: CandleRequest) -> "pd.DataFrame":
        raise NotImplementedError

    async def ticker(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError
