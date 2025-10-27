# adapters_feeds.py — Bridge
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    # nur für Typprüfung
    from adapters.adapters_feeds import get_candles as get_candles
else:
    try:
        from adapters.adapters_feeds import get_candles as get_candles  # runtime
    except Exception:
        def get_candles(*args: Any, **kwargs: Any):
            raise NotImplementedError("adapters.adapters_feeds.get_candles not available")

# get_market_snapshots ist optional → nie direkt importieren, sonst meckert mypy
def get_market_snapshots(*args: Any, **kwargs: Any):
    raise NotImplementedError("get_market_snapshots not implemented in bridge")

__all__ = ["get_candles", "get_market_snapshots"]
