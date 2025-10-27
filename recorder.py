# recorder.py
from __future__ import annotations

import os
import json
import csv
import sqlite3
import hashlib
import time
import threading
from datetime import datetime, timezone
from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr, validator
from typing import Optional, Literal, Dict, Any, Mapping, Collection
from dataclasses import dataclass, field, asdict

# =========================
# Helper Funktionen
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_epoch_ms(ts: Optional[float] = None) -> int:
    if ts is None:
        ts = time.time()
    return int(round(ts * 1000))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_sqlite_path(db_url: str) -> str:
    if db_url.startswith("sqlite:///"):
        return db_url.replace("sqlite:///", "", 1)
    if db_url.startswith("sqlite:/"):
        return db_url.replace("sqlite:/", "/", 1)
    if db_url.endswith(".db"):
        return db_url
    return "./data/recorder.db"


def sha256_hexdigest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# =========================
# Konfiguration
# =========================
REC_DB_URL = os.getenv("REC_DB_URL", "sqlite:///./data/recorder.db")
REC_DIR = os.getenv("REC_DIR", "./data/records")
REC_CSV_ENABLE = os.getenv("REC_CSV_ENABLE", "1") == "1"
REC_APP_NAME = os.getenv("REC_APP_NAME", "tier43")


# =========================
# LogEvent (sicher & robust)
# =========================
class LogEvent(BaseModel):
    event_id: StrictStr
    ts_ms: StrictInt = Field(default_factory=to_epoch_ms)
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    source: StrictStr
    message: StrictStr
    meta: Dict[str, Any] = Field(default_factory=dict)

    # Sichere Konstruktion aus beliebigem Payload
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "LogEvent":
        meta_raw = payload.get("meta")
        meta: Dict[str, Any]
        if isinstance(meta_raw, Mapping):
            tmp: Dict[str, Any] = {}
            for k, v in meta_raw.items():
                if isinstance(v, Collection) and not isinstance(v, (str, bytes, dict)):
                    tmp[str(k)] = list(v)
                else:
                    tmp[str(k)] = v
            meta = tmp
        elif isinstance(meta_raw, dict):
            meta = dict(meta_raw)
        elif meta_raw is None:
            meta = {}
        else:
            meta = {"value": meta_raw}

        return cls(
            event_id=str(
                payload.get("event_id")
                or sha256_hexdigest(
                    {"m": str(payload.get("message", "")), "t": to_epoch_ms()}
                )[:24]
            ),
            ts_ms=int(payload.get("ts_ms") or to_epoch_ms()),
            level=payload.get("level", "INFO"),   # ✅ Komma war hier das Problem
            source=str(payload.get("source", "misc")),
            message=str(payload.get("message", "")),
            meta=meta,
        )


# Alte Dataclass-Version beibehalten (für Kompatibilität)
@dataclass
class LogEventDC:
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"]
    source: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# Event-Modelle
# =========================
class SignalEvent(BaseModel):
    event_id: StrictStr
    ts_ms: StrictInt = Field(default_factory=to_epoch_ms)
    source: StrictStr
    symbol: StrictStr
    tf: StrictStr
    direction: Literal["LONG", "SHORT"]
    probability: StrictFloat = Field(..., ge=0.0, le=1.0)
    confidence: Optional[StrictFloat] = Field(None, ge=0.0, le=1.0)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @validator("symbol", "tf")
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class OrderEvent(BaseModel):
    event_id: StrictStr
    ts_ms: StrictInt = Field(default_factory=to_epoch_ms)
    source: StrictStr
    broker: StrictStr
    symbol: StrictStr
    side: Literal["BUY", "SELL"]
    qty: StrictFloat = Field(..., gt=0.0)
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    price: Optional[StrictFloat] = Field(None, ge=0.0)
    client_order_id: Optional[StrictStr] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class FillEvent(BaseModel):
    event_id: StrictStr
    ts_ms: StrictInt = Field(default_factory=to_epoch_ms)
    source: StrictStr
    broker: StrictStr
    symbol: StrictStr
    side: Literal["BUY", "SELL"]
    qty: StrictFloat = Field(..., gt=0.0)
    price: StrictFloat = Field(..., gt=0.0)
    fee: Optional[StrictFloat] = Field(0.0, ge=0.0)
    position_size: Optional[StrictFloat] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class EquitySnap(BaseModel):
    event_id: StrictStr
    ts_ms: StrictInt = Field(default_factory=to_epoch_ms)
    source: StrictStr
    currency: StrictStr = "USD"
    equity: StrictFloat
    cash: Optional[StrictFloat] = None
    unrealized_pnl: Optional[StrictFloat] = None
    realized_pnl: Optional[StrictFloat] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Recorder
# =========================
class Recorder:
    def __init__(
        self,
        db_url: str = REC_DB_URL,
        base_dir: str = REC_DIR,
        csv_enable: bool = REC_CSV_ENABLE,
    ):
        self.db_path = parse_sqlite_path(db_url)
        self.base_dir = base_dir
        self.csv_enable = csv_enable
        ensure_dir(os.path.dirname(self.db_path) or ".")
        ensure_dir(self.base_dir)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
        CREATE TABLE IF NOT EXISTS signals(
            event_id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            tf TEXT NOT NULL,
            direction TEXT NOT NULL,
            probability REAL NOT NULL,
            confidence REAL,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS orders(
            event_id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            source TEXT NOT NULL,
            broker TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            order_type TEXT NOT NULL,
            price REAL,
            client_order_id TEXT,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS fills(
            event_id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            source TEXT NOT NULL,
            broker TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL,
            position_size REAL,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS equity(
            event_id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            source TEXT NOT NULL,
            currency TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL,
            unrealized_pnl REAL,
            realized_pnl REAL,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS logs(
            event_id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            level TEXT NOT NULL,
            source TEXT NOT NULL,
            message TEXT NOT NULL,
            meta_json TEXT
        );
        """
        )
        self._conn.commit()

    def record_signal(self, ev: SignalEvent) -> None:
        self._write("signals", ev.dict())

    def record_order(self, ev: OrderEvent) -> None:
        self._write("orders", ev.dict())

    def record_fill(self, ev: FillEvent) -> None:
        self._write("fills", ev.dict())

    def record_equity(self, ev: EquitySnap) -> None:
        self._write("equity", ev.dict())

    def record_log(self, ev: LogEvent) -> None:
        self._write("logs", ev.dict())

    # zentrale Write-Methode
    def _write(self, kind: str, d: Dict[str, Any]) -> None:
        try:
            with self._lock:
                self._write_sqlite(kind, d)
                if self.csv_enable:
                    self._write_csv(kind, d)
        except sqlite3.IntegrityError:
            return

    def _write_sqlite(self, kind: str, d: Dict[str, Any]) -> None:
        cur = self._conn.cursor()
        meta_json = json.dumps(d.get("meta", {}), separators=(",", ":"), sort_keys=True)
        if kind == "logs":
            cur.execute(
                """INSERT INTO logs(event_id, ts_ms, level, source, message, meta_json)
                   VALUES(?,?,?,?,?,?)""",
                (
                    d["event_id"],
                    d["ts_ms"],
                    d["level"],
                    d["source"],
                    d["message"],
                    meta_json,
                ),
            )
        self._conn.commit()

    def _write_csv(self, kind: str, d: Dict[str, Any]) -> None:
        day = datetime.utcfromtimestamp(d["ts_ms"] / 1000).strftime("%Y-%m-%d")
        day_dir = os.path.join(self.base_dir, day)
        ensure_dir(day_dir)
        path = os.path.join(day_dir, f"{kind}.csv")
        row = dict(d)
        row["meta"] = json.dumps(row.pop("meta", {}), separators=(",", ":"), sort_keys=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)


# =========================
# Hilfsfunktionen
# =========================
def make_signal_id(payload: Dict[str, Any]) -> str:
    core = {k: payload.get(k) for k in ("symbol", "tf", "direction", "probability")}
    core["bucket_min"] = int((payload.get("ts_ms", to_epoch_ms())) / 60000)
    return sha256_hexdigest(core)[:24]


def _headers_to_dict(
    headers: Mapping[str, Collection[str]] | Dict[str, Collection[str]] | None
) -> Dict[str, Any]:
    if not headers:
        return {}
    return {k: list(v) for k, v in headers.items()}


def make_simple_log(
    source: str,
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
    message: str,
    meta: Optional[Dict[str, Any]] = None,
) -> LogEvent:
    payload = {
        "event_id": sha256_hexdigest({"s": source, "m": message, "t": to_epoch_ms()})[:24],
        "level": level,
        "source": source,
        "message": message,
        "meta": meta or {},
    }
    return LogEvent.from_payload(payload)


# =========================
# CLI Dispatcher
# =========================
def _dispatch(rec: Recorder, kind: str, payload: Dict[str, Any]) -> None:
    if kind == "signal":
        if "event_id" not in payload:
            payload["event_id"] = make_signal_id(payload)
        rec.record_signal(SignalEvent(**payload))
    elif kind == "order":
        rec.record_order(OrderEvent(**payload))
    elif kind == "fill":
        rec.record_fill(FillEvent(**payload))
    elif kind == "equity":
        rec.record_equity(EquitySnap(**payload))
    elif kind == "log":
        rec.record_log(LogEvent.from_payload(payload))
    else:
        raise ValueError(f"unknown type={kind}")
