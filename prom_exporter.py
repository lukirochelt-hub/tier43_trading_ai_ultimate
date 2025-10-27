# prom_exporter.py
# Tier 4.3+ – Prometheus Exporter / Helpers
# - Kann standalone einen /metrics-Endpoint starten ODER in FastAPI eingebunden werden
# - Sichere No-Op bei deaktiviertem Export (PROM_ENABLE=0)
# - Einheitliche Metrik-Labels für Strategy/Env/Venue etc.
# - Convenience-APIs zum Updaten aus beliebigen Modulen (webhook, trader, inference)

from __future__ import annotations

import os
import time
import threading
from typing import Any, Dict, Optional

# Prometheus
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        start_http_server,
        REGISTRY,
        ProcessCollector,
        PlatformCollector,
        GCCollector,
    )

    _PROM_AVAILABLE = True
except Exception:
    # Falls lib fehlt, laufen Calls als No-Op, damit nichts crasht
    _PROM_AVAILABLE = False

# Optional: FastAPI Integration (nur wenn verfügbar)
try:
    from fastapi import APIRouter, FastAPI, Request, Response

    _FASTAPI_AVAILABLE = True
except Exception:
    _FASTAPI_AVAILABLE = False

# =========================
# Konfiguration (Env)
# =========================
PROM_ENABLE = os.getenv("PROM_ENABLE", "1") not in ("0", "false", "False", "")
PROM_ADDR = os.getenv("PROM_ADDR", "0.0.0.0")
PROM_PORT = int(os.getenv("PROM_PORT", "9108"))
PROM_NS = os.getenv("PROM_NAMESPACE", "tier43")
PROM_SUBSYS = os.getenv("PROM_SUBSYSTEM", "core")
PROM_ENV = os.getenv("PROM_ENV", "dev")
PROM_STRAT = os.getenv("PROM_STRATEGY", "default")
PROM_VENUE = os.getenv("PROM_VENUE", "paper")
PROM_BUILD = os.getenv("PROM_BUILD", "4.3+")
PROM_GIT = os.getenv("PROM_GIT_SHA", "unknown")
PROM_PROCESS_METRICS = os.getenv("PROM_PROCESS_METRICS", "1") not in (
    "0",
    "false",
    "False",
    "",
)

_DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)

_lock = threading.RLock()


class _NoOp:
    def __getattr__(self, name: str) -> Any:
        def _noop(*args: Any, **kwargs: Any) -> Any:
            return None

        return _noop


class PrometheusExporter:
    """
    Einheitliche Prometheus-Export Schicht für Tier 4.3+.
    Benutzung:
        prom = PrometheusExporter.auto_start()    # startet HTTP-Server wenn standalone gewünscht
        prom.inc_webhook(status="ok", symbol="SOLUSDT", tf="15m", direction="LONG")
        prom.observe_inference_latency(0.123, model="xgb")
        prom.observe_trade(side="BUY", symbol="SOLUSDT", mode="paper")
        prom.set_equity(12345.67)
        prom.set_pnl(realized=12.3, unrealized=4.56)
        prom.set_open_positions(symbol="SOLUSDT", count=1)
        prom.set_health(True)
        # FastAPI: prom.add_routes(app)
        # Middleware: app.middleware("http")(prom.request_timer_middleware)
    """

    def __init__(
        self, enable: bool = PROM_ENABLE, registry: Optional[CollectorRegistry] = None
    ):
        self.enabled = enable and _PROM_AVAILABLE
        if not self.enabled:
            # No-Op Objekte für alle Metriken
            self.registry = None
            self.m = _NoOp()
            return

        self.registry = registry or CollectorRegistry(auto_describe=True)

        # Optionale Standard-Collector (Process/GC/Platform)
        if PROM_PROCESS_METRICS:
            ProcessCollector(registry=self.registry)
            PlatformCollector(registry=self.registry)
            GCCollector(registry=self.registry)

        # Gemeinsame Labels
        base = dict(
            env=PROM_ENV, strategy=PROM_STRAT, venue=PROM_VENUE, subsystem=PROM_SUBSYS
        )

        # --- Info
        self.info_build = Info(
            f"{PROM_NS}_build", "Build/Version Info", registry=self.registry
        )
        self.info_build.info({"build": PROM_BUILD, "git_sha": PROM_GIT, **base})

        # --- Health
        self.g_health = Gauge(
            f"{PROM_NS}_service_health",
            "1=healthy, 0=unhealthy",
            ["env", "strategy", "venue", "subsystem"],
            registry=self.registry,
        )
        self.g_health.labels(**base).set(1.0)

        # --- Webhook Requests
        self.c_webhook = Counter(
            f"{PROM_NS}_webhook_requests_total",
            "Webhook requests received",
            [
                "env",
                "strategy",
                "venue",
                "subsystem",
                "status",
                "symbol",
                "tf",
                "direction",
            ],
            registry=self.registry,
        )

        self.h_webhook_latency = Histogram(
            f"{PROM_NS}_webhook_latency_seconds",
            "End-to-end webhook handling latency",
            ["env", "strategy", "venue", "subsystem"],
            buckets=_DEFAULT_BUCKETS,
            registry=self.registry,
        )

        self.h_request_latency = Histogram(
            f"{PROM_NS}_http_request_latency_seconds",
            "Generic HTTP request latency (middleware)",
            ["env", "strategy", "venue", "subsystem", "path", "method", "code"],
            buckets=_DEFAULT_BUCKETS,
            registry=self.registry,
        )

        self.c_errors = Counter(
            f"{PROM_NS}_errors_total",
            "Errors by type",
            ["env", "strategy", "venue", "subsystem", "etype"],
            registry=self.registry,
        )

        # --- Inference / Signals
        self.h_infer = Histogram(
            f"{PROM_NS}_inference_latency_seconds",
            "Model inference latency",
            ["env", "strategy", "venue", "subsystem", "model"],
            buckets=_DEFAULT_BUCKETS,
            registry=self.registry,
        )

        self.c_signals = Counter(
            f"{PROM_NS}_signals_total",
            "Signals produced",
            [
                "env",
                "strategy",
                "venue",
                "subsystem",
                "symbol",
                "tf",
                "direction",
                "source",
            ],
            registry=self.registry,
        )

        # --- Trades & Positions
        self.c_trades = Counter(
            f"{PROM_NS}_trades_executed_total",
            "Executed trades",
            ["env", "strategy", "venue", "subsystem", "symbol", "side", "mode"],
            registry=self.registry,
        )

        self.g_positions = Gauge(
            f"{PROM_NS}_open_positions",
            "Open positions count",
            ["env", "strategy", "venue", "subsystem", "symbol"],
            registry=self.registry,
        )

        # --- Equity & PnL
        self.g_equity = Gauge(
            f"{PROM_NS}_equity_value",
            "Account equity (quote currency)",
            ["env", "strategy", "venue", "subsystem"],
            registry=self.registry,
        )
        self.g_pnl_realized = Gauge(
            f"{PROM_NS}_pnl_realized",
            "Realized PnL (session)",
            ["env", "strategy", "venue", "subsystem"],
            registry=self.registry,
        )
        self.g_pnl_unrealized = Gauge(
            f"{PROM_NS}_pnl_unrealized",
            "Unrealized PnL (current)",
            ["env", "strategy", "venue", "subsystem"],
            registry=self.registry,
        )

        # --- Build a labels dict for quick reuse
        self._base_labels = base

        # HTTP server state
        self._server_started = False

    # ---------- Helpers (safe if disabled) ----------
    def _safe_labels(self, d: Dict[str, str]) -> Dict[str, str]:
        # Merge mit Basislabels
        return {
            **self._base_labels,
            **{k: (str(v) if v is not None else "na") for k, v in d.items()},
        }

    # Public API
    def set_health(self, healthy: bool) -> None:
        if not self.enabled:
            return
        self.g_health.labels(**self._base_labels).set(1.0 if healthy else 0.0)

    def inc_webhook(self, status: str, symbol: str, tf: str, direction: str) -> None:
        if not self.enabled:
            return
        self.c_webhook.labels(
            **self._safe_labels(
                {"status": status, "symbol": symbol, "tf": tf, "direction": direction}
            )
        ).inc()

    def observe_webhook_latency(self, seconds: float) -> None:
        if not self.enabled:
            return
        self.h_webhook_latency.labels(**self._base_labels).observe(
            max(0.0, float(seconds))
        )

    def error(self, etype: str) -> None:
        if not self.enabled:
            return
        self.c_errors.labels(**self._safe_labels({"etype": etype})).inc()

    def observe_inference_latency(self, seconds: float, model: str = "unknown") -> None:
        if not self.enabled:
            return
        self.h_infer.labels(**self._safe_labels({"model": model})).observe(
            max(0.0, float(seconds))
        )

    def signal(self, symbol: str, tf: str, direction: str, source: str = "ai") -> None:
        if not self.enabled:
            return
        self.c_signals.labels(
            **self._safe_labels(
                {"symbol": symbol, "tf": tf, "direction": direction, "source": source}
            )
        ).inc()

    def observe_trade(self, side: str, symbol: str, mode: str = PROM_VENUE) -> None:
        if not self.enabled:
            return
        self.c_trades.labels(
            **self._safe_labels({"symbol": symbol, "side": side, "mode": mode})
        ).inc()

    def set_open_positions(self, symbol: str, count: int) -> None:
        if not self.enabled:
            return
        self.g_positions.labels(**self._safe_labels({"symbol": symbol})).set(int(count))

    def set_equity(self, value: float) -> None:
        if not self.enabled:
            return
        self.g_equity.labels(**self._base_labels).set(float(value))

    def set_pnl(
        self, realized: Optional[float] = None, unrealized: Optional[float] = None
    ) -> None:
        if not self.enabled:
            return
        if realized is not None:
            self.g_pnl_realized.labels(**self._base_labels).set(float(realized))
        if unrealized is not None:
            self.g_pnl_unrealized.labels(**self._base_labels).set(float(unrealized))

    # ---------- HTTP Export ----------
    def start_http(self, addr: str = PROM_ADDR, port: int = PROM_PORT) -> None:
        if not self.enabled or self._server_started:
            return
        # Nutzt prometheus_client's einfachen HTTP-Server
        # Achtung: Das exponiert REGISTRY, daher registrieren wir unsere Registry als global,
        # wenn sie nicht bereits REGISTRY ist.
        if self.registry is not REGISTRY:
            # Es ist okay, generate_latest(self.registry) zu verwenden; start_http_server jedoch
            # nutzt das globale REGISTRY. Um Konflikte zu vermeiden, subklassieren wir nicht,
            # sondern starten keinen zweiten Server falls REGISTRY bereits verwendet wird.
            # Workaround: wir starten hier explizit einen Thread mit minimalem WSGI-Server,
            # wenn eine Custom-Registry verwendet wird.
            thread = threading.Thread(
                target=self._serve_simple, args=(addr, port), daemon=True
            )
            thread.start()
        else:
            start_http_server(port, addr)
        self._server_started = True

    def _serve_simple(self, addr: str, port: int) -> None:
        # Minimaler HTTP-Server nur für /metrics, kompatibel mit Custom-Registry
        import http.server
        import socketserver

        registry = self.registry

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"not found")
                    return
                output = generate_latest(registry)  # type: ignore
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)  # type: ignore
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)

            # Silence default noisy logging
            def log_message(self, format: str, *args: Any) -> None:
                return

        with socketserver.TCPServer((addr, port), Handler) as httpd:
            httpd.serve_forever()

    @classmethod
    def auto_start(cls) -> "PrometheusExporter":
        """
        Erstellt Exporter und startet sofort den HTTP-Server, wenn PROM_ENABLE=1.
        """
        inst = cls()
        if inst.enabled:
            inst.start_http()
        return inst

    # ---------- FastAPI Integration ----------
    def add_routes(self, app: "FastAPI") -> None:
        """
        Fügt /metrics in eine bestehende FastAPI-App ein.
        """
        if not (self.enabled and _FASTAPI_AVAILABLE):
            return
        router = APIRouter()

        @router.get("/metrics")
        async def metrics() -> Response:
            output = generate_latest(self.registry) if self.registry else b""
            return Response(content=output, media_type=CONTENT_TYPE_LATEST)

        app.include_router(router)

    async def request_timer_middleware(self, request: "Request", call_next):
        """
        FastAPI/Starlette Middleware zur Messung der Latenz aller Requests.
        """
        if not (self.enabled and _FASTAPI_AVAILABLE):
            return await call_next(request)

        start = time.perf_counter()
        try:
            response = await call_next(request)
            code = getattr(response, "status_code", 0)
        except Exception:
            code = 500
            self.error("http_middleware_exception")
            raise
        finally:
            elapsed = max(0.0, time.perf_counter() - start)
            path = request.url.path
            method = request.method
            self.h_request_latency.labels(
                **self._safe_labels({"path": path, "method": method, "code": str(code)})
            ).observe(elapsed)
        return response


# ======== Singleton-Convenience (optional) ========
_exporter_singleton: Optional[PrometheusExporter] = None


def get_exporter() -> PrometheusExporter:
    global _exporter_singleton
    if _exporter_singleton is None:
        _exporter_singleton = PrometheusExporter.auto_start()
    return _exporter_singleton


# ======== CLI ========
def _demo_updates(prom: PrometheusExporter) -> None:
    """
    Optional: kleine Demo, damit beim lokalen Testen sofort Werte auftauchen.
    """
    import random

    prom.set_health(True)
    prom.set_equity(10_000 + random.random() * 500)
    prom.set_pnl(realized=random.uniform(-50, 50), unrealized=random.uniform(-25, 25))
    prom.set_open_positions("BTCUSDT", random.randint(0, 2))
    prom.observe_inference_latency(random.random() * 0.2, model="xgb_v42")
    prom.signal("BTCUSDT", "15m", "LONG", source="tv")
    prom.observe_trade("BUY", "BTCUSDT", mode=PROM_VENUE)
    prom.inc_webhook(status="ok", symbol="BTCUSDT", tf="15m", direction="LONG")
    prom.observe_webhook_latency(random.random() * 0.15)


if __name__ == "__main__":
    prom = PrometheusExporter.auto_start()
    if not prom.enabled:
        print(
            "[prom_exporter] PROM_ENABLE=0 oder prometheus_client nicht verfügbar – No-Op Modus."
        )
        print("  -> pip install prometheus-client fastapi (optional)")
        while True:
            time.sleep(3600)
    else:
        print(
            f"[prom_exporter] /metrics läuft auf http://{PROM_ADDR}:{PROM_PORT}/metrics  (ns={PROM_NS}, env={PROM_ENV}, strat={PROM_STRAT}, venue={PROM_VENUE})"
        )
        # Demo-Ticker: alle 5s ein paar Werte updaten, damit man im Grafana sofort was sieht
        try:
            while True:
                with _lock:
                    _demo_updates(prom)
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n[prom_exporter] stopped.")
