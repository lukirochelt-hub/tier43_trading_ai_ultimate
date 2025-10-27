# -*- coding: utf-8 -*-
"""
ensemble_monitor.py — Tier 4.3+ (Final, VS Code fixed)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as os_signal
import sys
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, cast  # <-- cast ergänzt

# Prometheus (eigener Registry, damit keine Duplicates)
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
except Exception:
    CollectorRegistry = None  # type: ignore
    Counter = Gauge = generate_latest = CONTENT_TYPE_LATEST = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("ENSEMBLE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ensemble_monitor")

# -----------------------------
# Datamodels
# -----------------------------
Side = Literal["long", "short", "flat"]


@dataclass
class ModelSignal:
    model: str
    symbol: str
    timeframe: str
    side: Side
    confidence: float
    ts: float
    price: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleDecision:
    symbol: str
    timeframe: str
    side: Side
    confidence: float
    consensus: Dict[str, float]
    voters: List[str]
    price: Optional[float]
    ts: float
    policy: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    entry_price: Optional[float]
    opened_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskState:
    equity: float
    daily_pl: float
    open_positions: Dict[str, Position] = field(default_factory=dict)
    last_trade_ts: float = 0.0


# -----------------------------
# Utils & Config
# -----------------------------
def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def now_ts() -> float:
    return time.time()


SAVE_DIR = Path(os.getenv("ENSEMBLE_SAVE_DIR", "./logs"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTBOX = Path(os.getenv("ENSEMBLE_OUTBOX", "./outbox"))
OUTBOX.mkdir(parents=True, exist_ok=True)

STATE_FILE = SAVE_DIR / "ensemble_state.json"
EQUITY_FILE = Path(os.getenv("ENSEMBLE_EQUITY_FILE", str(SAVE_DIR / "equity.json")))

PORT = env_int("ENSEMBLE_PORT", 8788)
MIN_CONF = env_float("ENSEMBLE_MIN_CONF", 0.55)
STRATEGY: Literal["majority", "weighted", "strict"] = os.getenv(
    "ENSEMBLE_STRATEGY", "weighted"
).lower()  # type: ignore
MAX_POSITIONS = env_int("ENSEMBLE_MAX_POSITIONS", 5)
COOLDOWN_SEC = env_int("ENSEMBLE_COOLDOWN_SEC", 15)
MAX_DAILY_LOSS = env_float("ENSEMBLE_MAX_DAILY_LOSS", -0.06)
RISK_PER_TRADE = env_float("ENSEMBLE_RISK_PER_TRADE", 0.005)

WEBHOOK_URL = os.getenv("ENSEMBLE_WEBHOOK_URL", "").strip()
API_TOKEN = os.getenv("ENSEMBLE_API_TOKEN", "").strip()

WHITELIST = [
    s.strip()
    for s in os.getenv("ENSEMBLE_SYMBOL_WHITELIST", "").split(",")
    if s.strip()
]
BLACKLIST = [
    s.strip()
    for s in os.getenv("ENSEMBLE_SYMBOL_BLACKLIST", "").split(",")
    if s.strip()
]

# Secure Bridge (Trading AI Secure)
SECURE_URL = (
    os.getenv("SECURE_ENDPOINT_URL", "").strip()
    or os.getenv("ENSEMBLE_WEBHOOK_URL", "").strip()
)
SECURE_TOKEN = (
    os.getenv("SECURE_BEARER_TOKEN", "").strip()
    or os.getenv("ENSEMBLE_API_TOKEN", "").strip()
)
SECURE_TIMEOUT = float(os.getenv("SECURE_TIMEOUT_SEC", "6"))
SECURE_RETRIES = int(os.getenv("SECURE_MAX_RETRIES", "3"))
SECURE_BACKOFF_BASE = float(os.getenv("SECURE_BACKOFF_BASE", "0.6"))

# Prometheus
PROM_ENABLE = os.getenv("PROM_ENABLE", "0") == "1"
PROM_NAMESPACE = os.getenv("PROM_NAMESPACE", "tia")

# -----------------------------
# Prometheus Metrics (dedizierte Registry -> keine Duplicates)
# -----------------------------
_METRICS_INIT = False
REGISTRY = None
M_SIGNALS_IN = None
M_DECISIONS = None
M_OPEN_POS = None
M_EQUITY = None
M_DAILY_PL = None


def init_metrics_if_needed() -> None:
    global \
        _METRICS_INIT, \
        REGISTRY, \
        M_SIGNALS_IN, \
        M_DECISIONS, \
        M_OPEN_POS, \
        M_EQUITY, \
        M_DAILY_PL
    if _METRICS_INIT or not PROM_ENABLE or CollectorRegistry is None:
        return
    REGISTRY = CollectorRegistry()
    # Counter / Gauge mit stabilem Namespace
    M_SIGNALS_IN = Counter(
        f"{PROM_NAMESPACE}_alerts_in_total",
        "Incoming model signals",
        ["model", "symbol", "tf"],
        registry=REGISTRY,
    )  # type: ignore
    M_DECISIONS = Counter(
        f"{PROM_NAMESPACE}_decisions_total",
        "Ensemble decisions",
        ["side", "symbol", "tf"],
        registry=REGISTRY,
    )  # type: ignore
    M_OPEN_POS = Gauge(
        f"{PROM_NAMESPACE}_open_positions", "Open positions count", registry=REGISTRY
    )  # type: ignore
    M_EQUITY = Gauge(f"{PROM_NAMESPACE}_equity", "Account equity", registry=REGISTRY)  # type: ignore
    M_DAILY_PL = Gauge(f"{PROM_NAMESPACE}_daily_pl", "Daily P/L", registry=REGISTRY)  # type: ignore
    _METRICS_INIT = True


# -----------------------------
# Ensemble Monitor
# -----------------------------
ReaderFn = Callable[[], Awaitable[Optional[ModelSignal]]]


class EnsembleMonitor:
    def __init__(self):
        self.readers: Dict[str, ReaderFn] = {}
        self.last_signals: Dict[str, ModelSignal] = {}
        self.decisions: List[EnsembleDecision] = []
        self.risk = self._load_risk_state()
        self._shutdown = False

    # ----- Registration -----
    def register_model(self, name: str, reader: ReaderFn) -> None:
        """
        Akzeptiert async UND sync Callables.
        """
        if asyncio.iscoroutinefunction(reader):
            self.readers[name] = reader  # type: ignore
        else:

            async def _wrapper():
                res = reader()
                if asyncio.iscoroutine(res):
                    return await res
                return cast(Optional[ModelSignal], res)  # <-- präzise Rückgabe

            self.readers[name] = _wrapper  # type: ignore
        log.info("Registered model reader: %s", name)

    # ----- Risk State Persistence -----
    def _load_risk_state(self) -> RiskState:
        equity = 100_000.0
        daily_pl = 0.0
        if EQUITY_FILE.exists():
            try:
                data = json.loads(EQUITY_FILE.read_text(encoding="utf-8"))
                equity = float(data.get("equity", equity))
                daily_pl = float(data.get("daily_pl", daily_pl))
            except Exception as e:
                log.warning("Equity file parse error: %s", e)
        if STATE_FILE.exists():
            try:
                st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                open_positions = {
                    k: Position(**v) for k, v in st.get("open_positions", {}).items()
                }
                last_trade_ts = float(st.get("last_trade_ts", 0.0))
                return RiskState(
                    equity=equity,
                    daily_pl=daily_pl,
                    open_positions=open_positions,
                    last_trade_ts=last_trade_ts,
                )
            except Exception as e:
                log.warning("State file parse error: %s", e)
        return RiskState(equity=equity, daily_pl=daily_pl)

    # ----- Readers Loop -----
    async def _poll_reader(
        self, name: str, reader: ReaderFn, interval: float = 1.0
    ) -> None:
        while not self._shutdown:
            try:
                sig = await reader()
                if sig is not None:
                    self.last_signals[name] = sig
            except Exception as e:
                log.exception("Reader '%s' error: %s", name, e)
            await asyncio.sleep(interval)

    # ----- Consensus -----
    def _consensus(
        self, signals: List[ModelSignal]
    ) -> Tuple[Side, float, Dict[str, float], List[str], Optional[float]]:
        if not signals:
            return "flat", 0.0, {"long": 0.0, "short": 0.0, "flat": 1.0}, [], None
        voters = [s.model for s in signals if s.confidence >= MIN_CONF]
        if not voters:
            return "flat", 0.0, {"long": 0.0, "short": 0.0, "flat": 1.0}, [], None

        sides: Dict[str, float] = {"long": 0.0, "short": 0.0, "flat": 0.0}
        for s in signals:
            if s.confidence >= MIN_CONF:
                sides[s.side] += s.confidence

        total = float(sum(sides.values()))
        probs: Dict[str, float] = {k: (v / total if total > 0.0 else 0.0) for k, v in sides.items()}

        # sicheres key-Callable (kein überladenes Funktionsobjekt)
        def _get_prob(key: str) -> float:
            return probs.get(key, 0.0)

        winner_key = max(probs.keys(), key=_get_prob)
        winner_side: Side = cast(Side, winner_key)
        conf: float = probs.get(winner_key, 0.0)
        return winner_side, conf, probs, voters, None

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self._poll_reader(name, fn))
            for name, fn in self.readers.items()
        ]
        log.info("Readers started: %s", list(self.readers.keys()))
        try:
            while not self._shutdown:
                await asyncio.sleep(1.0)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            log.info("Ensemble monitor stopped.")


# -----------------------------
# HTTP Status Server
# -----------------------------
class _Handler(BaseHTTPRequestHandler):
    monitor_ref: Optional[EnsembleMonitor] = None

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            return self._send_json(
                200, {"ok": True, "service": "ensemble_monitor", "ts": now_ts()}
            )
        self._send_json(404, {"ok": False, "error": "not found"})


def start_http_server(monitor: EnsembleMonitor, port: int = PORT) -> Thread:
    _Handler.monitor_ref = monitor
    server = HTTPServer(("0.0.0.0", port), _Handler)
    th = Thread(target=server.serve_forever, daemon=True)
    th.start()
    log.info("Status server on http://0.0.0.0:%d", port)
    return th


# -----------------------------
# Dummy Reader
# -----------------------------
async def reader_stub(name: str, symbol: str, timeframe: str) -> Optional[ModelSignal]:
    import random

    await asyncio.sleep(0.1)
    if int(time.time()) % 7 != 0:
        return None
    side = cast(Side, random.choice(["long", "short", "flat"]))  # <-- Side sicher casten
    conf = float(random.uniform(0.4, 0.9))
    price = float(random.uniform(100, 100000))
    return ModelSignal(
        model=name,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        confidence=conf,
        ts=now_ts(),
        price=price,
    )


# -----------------------------
# Main
# -----------------------------
async def main(argv: List[str]) -> int:
    mon = EnsembleMonitor()

    # Async Wrapper Readers
    async def reader_ml_core():
        return await reader_stub("ml_core", "BTC", "15m")

    async def reader_optuna_realr():
        return await reader_stub("optuna_realr", "BTC", "15m")

    async def reader_nightly_optuna():
        return await reader_stub("nightly_optuna", "BTC", "15m")

    mon.register_model("ml_core", reader_ml_core)
    mon.register_model("optuna_realr", reader_optuna_realr)
    mon.register_model("nightly_optuna", reader_nightly_optuna)

    start_http_server(mon, PORT)

    loop = asyncio.get_event_loop()
    for s in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            os_signal.signal(s, lambda *_: setattr(mon, "_shutdown", True))
        except Exception:
            pass

    await mon.run()
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main(sys.argv[1:]))
        sys.exit(rc)
    except KeyboardInterrupt:
        sys.exit(130)
