
# adapters_alt_feeds.py — Bridge: re-exportiert AltFeedAdapter mypy-clean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Nur für Typprüfung (mypy); wird zur Analysezeit verwendet
    from adapters.adapters_alt_feeds import AltFeedAdapter as AltFeedAdapter
else:
    # Runtime-Pfad
    try:
        from adapters.adapters_alt_feeds import AltFeedAdapter as AltFeedAdapter
    except Exception:
        class AltFeedAdapter:  # Fallback nur zur Laufzeit
            """Placeholder, nur damit alte Imports funktionieren."""
            pass

def get_sentiment_data(*args, **kwargs):
    raise NotImplementedError("get_sentiment_data not implemented")

__all__ = ["AltFeedAdapter", "get_sentiment_data"]
