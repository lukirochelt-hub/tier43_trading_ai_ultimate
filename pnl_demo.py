# pnl_demo.py
# Tier 4.3+ – Demo/Paper Trading PnL Engine (final)
# - SQLite persistence + JSON backup
# - Realized/Unrealized PnL (Mark-to-Market)
# - Fees + slippage model
# - Session namespace + archive rotation
# - Prometheus metrics (optional)
# - Heatmap / monitor hooks
# - Thread-safe + CLI utilities
#
# Integration:
#   from pnl_demo import PnLEngine, Side, DemoTrade
#   pnl = PnLEngine()                        # or pass custom paths/params
#   t = pnl.open_trade(symbol="SOLUSDT", side=Side.LONG, price=142.35, size_base=10, advice_id="abc123")
#   pnl.close_trade(t.trade_id, price=144.10)
#   pnl.summary() -> dict
#
# Env knobs:
#   PNL_DB_PATH=data/pnl_demo.db
#   PNL_JSON_PATH=data/pnl_store.json
#   PNL_TAKER_FEE=0.001
#   PNL_SLIPPAGE_BPS=5
#   PNL_SESSION=default
#   PROM_ENABLE=1  (optional)
#
from __future__ import annotations

import os
import json
import time
import uuid
import sqlite3
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

# -------- Optional Prometheus ----------
PROM_ENABLE_ENV = os.getenv("PROM_ENABLE", "0")
PROM_ENABLE: bool = PROM_ENABLE_ENV not in ("0", "", "false", "False", "no", "No")

GaugeType: Any
CounterType: Any
if PROM_ENABLE:
    try:
        from prometheus_client import Gauge as _Gauge, Counter as _Counter
        GaugeType = _Gauge
        CounterType = _Counter
    except Exception:
        PROM_ENABLE = False
        GaugeType = object  # type: ignore[assignment]
        CounterType = object  # type: ignore[assignment]
else:
    GaugeType = object  # type: ignore[assignment]
    CounterType = object  # type: ignore[assignment]


# -------- Utility & types --------------
class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


def now_ts() -> float:
    return time.time()


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def getenv_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default


# -------- Data model -------------------
@dataclass
class DemoTrade:
    trade_id: str
    session: str
    symbol: str
    side: Side
    size_base: float  # position size in base units (e.g., SOL)
    entry_price: float
    entry_ts: float
    exit_price: Optional[float] = None
    exit_ts: Optional[float] = None
    realized_pnl: Optional[float] = None
    fees_quote: float = 0.0
    slippage_quote: float = 0.0
    advice_id: Optional[str] = None
    note: Optional[str] = None
    status: str = "OPEN"  # OPEN / CLOSED

    def direction(self) -> int:
        return 1 if self.side == Side.LONG else -1


# -------- Engine -----------------------
class PnLEngine:
    def __init__(
        self,
        db_path: Optional[str] = None,
        json_path: Optional[str] = None,
        taker_fee: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        session: Optional[str] = None,
        start_equity: float = 10_000.0,
    ):
        # Pfade/Parameter laden (ENV-sicher)
        self.db_path = db_path or getenv_str("PNL_DB_PATH", "data/pnl_demo.db")
        self.json_path = json_path or getenv_str("PNL_JSON_PATH", "data/pnl_store.json")
        self.taker_fee = float(
            taker_fee if taker_fee is not None else getenv_str("PNL_TAKER_FEE", "0.001")
        )
        self.slippage_bps = float(
            slippage_bps if slippage_bps is not None else getenv_str("PNL_SLIPPAGE_BPS", "5")
        )
        self.session = session or getenv_str("PNL_SESSION", "default")
        self.start_equity = float(start_equity)

        # Verzeichnisse sicher erstellen
        db_dir = os.path.dirname(self.db_path) or "."
        json_dir = os.path.dirname(self.json_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # DB & Lock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._migrate()
        self._metrics_setup()

    # ---------- DB schema -----------
    def _migrate(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades(
                  trade_id TEXT PRIMARY KEY,
                  session TEXT,
                  symbol TEXT,
                  side TEXT,
                  size_base REAL,
                  entry_price REAL,
                  entry_ts REAL,
                  exit_price REAL,
                  exit_ts REAL,
                  realized_pnl REAL,
                  fees_quote REAL,
                  slippage_quote REAL,
                  advice_id TEXT,
                  note TEXT,
                  status TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_session_status
                ON trades(session, status);
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session TEXT,
                  ts REAL,
                  equity REAL,
                  realized REAL,
                  unrealized REAL
                );
                """
            )

    # ---------- Metrics --------------
    def _metrics_setup(self) -> None:
        if not PROM_ENABLE:
            self._m: Optional[Dict[str, Any]] = None
            return
        self._m = {
            "open_trades": GaugeType("pnl_open_trades", "Number of open demo trades", ["session"]),
            "realized_pnl": GaugeType("pnl_realized", "Realized PnL (quote)", ["session"]),
            "unrealized_pnl": GaugeType("pnl_unrealized", "Unrealized PnL (quote)", ["session"]),
            "equity": GaugeType("pnl_equity", "Equity (start + PnL)", ["session"]),
            "trades_opened": CounterType(
                "pnl_trades_opened_total", "Trades opened", ["session", "symbol", "side"]
            ),
            "trades_closed": CounterType(
                "pnl_trades_closed_total", "Trades closed", ["session", "symbol", "side"]
            ),
        }

    # ---------- Fees & slippage -------
    def _fee_quote(self, price: float, size_base: float) -> float:
        # Simple taker fee model: fee on notional of both entry/exit legs (charged when leg occurs)
        notional = price * abs(size_base)
        return notional * self.taker_fee

    def _slip_quote(self, price: float, size_base: float) -> float:
        # slippage in quote = price * size * (bps/10000)
        bps = clamp(self.slippage_bps, 0.0, 10_000.0)
        return price * abs(size_base) * (bps / 10_000.0)

    # ---------- CRUD -----------------
    def open_trade(
        self,
        symbol: str,
        side: Side,
        price: float,
        size_base: float,
        advice_id: Optional[str] = None,
        note: Optional[str] = None,
        entry_ts: Optional[float] = None,
    ) -> DemoTrade:
        with self._lock, self._conn:
            trade = DemoTrade(
                trade_id=gen_id("t"),
                session=self.session,
                symbol=string_upper(symbol),
                side=Side(side),
                size_base=float(size_base),
                entry_price=float(price),
                entry_ts=float(entry_ts if entry_ts is not None else now_ts()),
                advice_id=advice_id,
                note=note,
                status="OPEN",
            )
            # apply entry leg fee & slippage (recorded up-front)
            entry_fee = self._fee_quote(trade.entry_price, trade.size_base)
            entry_slip = self._slip_quote(trade.entry_price, trade.size_base)
            trade.fees_quote += entry_fee
            trade.slippage_quote += entry_slip

            self._conn.execute(
                """
                INSERT INTO trades(
                  trade_id, session, symbol, side, size_base,
                  entry_price, entry_ts, exit_price, exit_ts,
                  realized_pnl, fees_quote, slippage_quote,
                  advice_id, note, status
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trade.trade_id,
                    trade.session,
                    trade.symbol,
                    trade.side.value,
                    trade.size_base,
                    trade.entry_price,
                    trade.entry_ts,
                    None,
                    None,
                    None,
                    trade.fees_quote,
                    trade.slippage_quote,
                    trade.advice_id,
                    trade.note,
                    trade.status,
                ),
            )
            self._json_backup_append(trade)
            if PROM_ENABLE and self._m is not None:
                self._m["trades_opened"].labels(self.session, trade.symbol, trade.side.value).inc()
            return trade

    def close_trade(
        self,
        trade_id: str,
        price: float,
        exit_ts: Optional[float] = None,
        note: Optional[str] = None,
    ) -> DemoTrade:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM trades WHERE trade_id=? AND status='OPEN'", (trade_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Trade not open or not found: {trade_id}")
            trade = self._row_to_trade(row)
            trade.exit_price = float(price)
            trade.exit_ts = float(exit_ts if exit_ts is not None else now_ts())
            # apply exit leg costs
            trade.fees_quote += self._fee_quote(trade.exit_price, trade.size_base)
            trade.slippage_quote += self._slip_quote(trade.exit_price, trade.size_base)
            # realized PnL in quote
            direction = trade.direction()
            gross = (trade.exit_price - trade.entry_price) * direction * trade.size_base
            costs = trade.fees_quote + trade.slippage_quote
            trade.realized_pnl = gross - costs
            trade.status = "CLOSED"
            if note:
                trade.note = (trade.note + " | " if trade.note else "") + note

            self._conn.execute(
                """
                UPDATE trades
                SET exit_price=?, exit_ts=?, realized_pnl=?, fees_quote=?, slippage_quote=?, note=?, status='CLOSED'
                WHERE trade_id=?
                """,
                (
                    trade.exit_price,
                    trade.exit_ts,
                    trade.realized_pnl,
                    trade.fees_quote,
                    trade.slippage_quote,
                    trade.note,
                    trade.trade_id,
                ),
            )
            self._json_backup_overwrite()
            if PROM_ENABLE and self._m is not None:
                self._m["trades_closed"].labels(self.session, trade.symbol, trade.side.value).inc()
            return trade

    def mark_to_market(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        prices: mapping symbol -> last price
        Returns dict with {'realized', 'unrealized', 'equity', 'open_trades'}
        """
        with self._lock:
            open_rows = self._conn.execute(
                "SELECT * FROM trades WHERE session=? AND status='OPEN'",
                (self.session,),
            ).fetchall()
            unreal = 0.0
            for r in open_rows:
                t = self._row_to_trade(r)
                px = prices.get(t.symbol)
                if px is None:
                    continue
                direction = t.direction()
                gross = (px - t.entry_price) * direction * t.size_base
                # Only charge entry-leg costs to unrealized; exit leg not yet paid.
                costs = t.fees_quote + t.slippage_quote
                unreal += gross - costs

            realized = self._sum_realized()
            equity = self.start_equity + realized + unreal
            self._snapshot_equity(equity, realized, unreal)
            if PROM_ENABLE and self._m is not None:
                self._m["open_trades"].labels(self.session).set(len(open_rows))
                self._m["realized_pnl"].labels(self.session).set(realized)
                self._m["unrealized_pnl"].labels(self.session).set(unreal)
                self._m["equity"].labels(self.session).set(equity)
            return {
                "realized": realized,
                "unrealized": unreal,
                "equity": equity,
                "open_trades": len(open_rows),
            }

    # ---------- Queries / summaries ----
    def get_open_trades(self) -> List[DemoTrade]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE session=? AND status='OPEN' ORDER BY entry_ts ASC",
                (self.session,),
            ).fetchall()
            return [self._row_to_trade(r) for r in rows]

    def get_closed_trades(self, limit: int = 100) -> List[DemoTrade]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM trades WHERE session=? AND status='CLOSED'
                ORDER BY exit_ts DESC LIMIT ?
                """,
                (self.session, limit),
            ).fetchall()
            return [self._row_to_trade(r) for r in rows]

    def _sum_realized(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl),0) FROM trades WHERE session=? AND status='CLOSED'",
            (self.session,),
        ).fetchone()
        return float(row[0] or 0.0)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            realized = self._sum_realized()
            open_trades = self.get_open_trades()
            return {
                "session": self.session,
                "start_equity": self.start_equity,
                "realized_pnl": realized,
                "open_trades": [asdict(t) for t in open_trades],
                "num_open": len(open_trades),
                "num_closed": int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE session=? AND status='CLOSED'",
                        (self.session,),
                    ).fetchone()[0]
                ),
            }

    # ---------- Snapshots / exports ----
    def _snapshot_equity(self, equity: float, realized: float, unrealized: float) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO equity_snapshots(session, ts, equity, realized, unrealized)
                VALUES(?,?,?,?,?)
                """,
                (self.session, now_ts(), equity, realized, unrealized),
            )

    def export_csv(self, path: str = "data/pnl_trades.csv") -> str:
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        import csv

        with self._lock, open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "trade_id",
                    "session",
                    "symbol",
                    "side",
                    "size_base",
                    "entry_price",
                    "entry_ts",
                    "exit_price",
                    "exit_ts",
                    "realized_pnl",
                    "fees_quote",
                    "slippage_quote",
                    "advice_id",
                    "note",
                    "status",
                ]
            )
            for r in self._conn.execute(
                "SELECT * FROM trades WHERE session=? ORDER BY entry_ts ASC",
                (self.session,),
            ):
                t = self._row_to_trade(r)
                w.writerow(
                    [
                        t.trade_id,
                        t.session,
                        t.symbol,
                        t.side.value,
                        t.size_base,
                        t.entry_price,
                        t.entry_ts,
                        t.exit_price,
                        t.exit_ts,
                        t.realized_pnl,
                        t.fees_quote,
                        t.slippage_quote,
                        t.advice_id,
                        t.note,
                        t.status,
                    ]
                )
        return path

    def export_equity_csv(self, path: str = "data/equity_curve.csv") -> str:
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        import csv

        with self._lock, open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "equity", "realized", "unrealized", "session"])
            for r in self._conn.execute(
                "SELECT ts,equity,realized,unrealized,session FROM equity_snapshots WHERE session=? ORDER BY ts ASC",
                (self.session,),
            ):
                w.writerow(r)
        return path

    # ---------- JSON backup -----------
    def _json_backup_append(self, trade: DemoTrade) -> None:
        try:
            data: List[Dict[str, Any]] = []
            if os.path.exists(self.json_path):
                with open(self.json_path, "r") as f:
                    obj = json.load(f)
                    if isinstance(obj, list):
                        data = obj
            data.append(asdict(trade))
            with open(self.json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Non-fatal
            pass

    def _json_backup_overwrite(self) -> None:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM trades WHERE session=?", (self.session,)
                ).fetchall()
                data = [asdict(self._row_to_trade(r)) for r in rows]
                with open(self.json_path, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass

    # ---------- Internals -------------
    def _row_to_trade(self, row: sqlite3.Row | Tuple[Any, ...]) -> DemoTrade:
        # sqlite default returns tuples ordered as schema
        (
            trade_id,
            session,
            symbol,
            side,
            size_base,
            entry_price,
            entry_ts,
            exit_price,
            exit_ts,
            realized_pnl,
            fees_quote,
            slippage_quote,
            advice_id,
            note,
            status,
        ) = row
        return DemoTrade(
            trade_id=str(trade_id),
            session=str(session),
            symbol=str(symbol),
            side=Side(str(side)),
            size_base=float(size_base),
            entry_price=float(entry_price),
            entry_ts=float(entry_ts),
            exit_price=(float(exit_price) if exit_price is not None else None),
            exit_ts=(float(exit_ts) if exit_ts is not None else None),
            realized_pnl=(float(realized_pnl) if realized_pnl is not None else None),
            fees_quote=float(fees_quote or 0.0),
            slippage_quote=float(slippage_quote or 0.0),
            advice_id=(str(advice_id) if advice_id is not None else None),
            note=(str(note) if note is not None else None),
            status=str(status),
        )

    # ---------- Convenience for integrations ----------
    def record_signal_fill(
        self,
        *,
        symbol: str,
        side: str,
        fill_price: float,
        size_base: float,
        advice_id: Optional[str] = None,
        note: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> DemoTrade:
        """
        Convenience wrapper used by paper_loop when a fill is assumed at the signal price.
        """
        return self.open_trade(
            symbol=symbol,
            side=Side(side),
            price=fill_price,
            size_base=size_base,
            advice_id=advice_id,
            note=note,
            entry_ts=ts,
        )

    def record_exit(
        self,
        *,
        trade_id: str,
        exit_price: float,
        note: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> DemoTrade:
        return self.close_trade(trade_id, price=exit_price, exit_ts=ts, note=note)


# -------- CLI --------------------------
def string_upper(s: str) -> str:
    return s.upper()


def _fmt_ts(ts: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _print_table(rows: List[List[str]]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(str(x)) for x in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        line = " | ".join(str(x).ljust(widths[j]) for j, x in enumerate(r))
        print(line)
        if i == 0:
            print("-+-".join("-" * w for w in widths))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Tier 4.3+ Demo PnL Engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser("open", help="Open trade")
    p_open.add_argument("--symbol", required=True)
    p_open.add_argument("--side", required=True, choices=[e.value for e in Side])
    p_open.add_argument("--price", required=True, type=float)
    p_open.add_argument("--size_base", required=True, type=float)
    p_open.add_argument("--advice_id", default=None)
    p_open.add_argument("--note", default=None)

    p_close = sub.add_parser("close", help="Close trade")
    p_close.add_argument("--trade_id", required=True)
    p_close.add_argument("--price", required=True, type=float)
    p_close.add_argument("--note", default=None)

    p_mtm = sub.add_parser("mtm", help="Mark-to-market")
    p_mtm.add_argument(
        "--px", action="append", help="symbol=price e.g. BTCUSDT=65000", required=True
    )

    p_sum = sub.add_parser("summary", help="Summary")

    p_ls = sub.add_parser("list", help="List trades")
    p_ls.add_argument("--status", choices=["OPEN", "CLOSED", "ALL"], default="ALL")

    p_exp = sub.add_parser("export", help="Export CSVs")
    p_exp.add_argument("--trades_csv", default="data/pnl_trades.csv")
    p_exp.add_argument("--equity_csv", default="data/equity_curve.csv")

    args = p.parse_args()
    pnl = PnLEngine()

    if args.cmd == "open":
        t = pnl.open_trade(
            args.symbol,
            Side(args.side),
            args.price,
            args.size_base,
            advice_id=args.advice_id,
            note=args.note,
        )
        print(f"OPENED {t.trade_id} {t.symbol} {t.side} size={t.size_base} @ {t.entry_price}")
    elif args.cmd == "close":
        t = pnl.close_trade(args.trade_id, args.price, note=args.note)
        rp = "-" if t.realized_pnl is None else f"{t.realized_pnl:.6f}"
        print(f"CLOSED {t.trade_id} {t.symbol} {t.side} @ {t.exit_price}  realized={rp} quote")
    elif args.cmd == "mtm":
        px: Dict[str, float] = {}
        for kv in args.px:
            if "=" not in kv:
                raise SystemExit(f"Bad --px arg: {kv}")
            k, v = kv.split("=", 1)
            px[k.upper()] = float(v)
        res = pnl.mark_to_market(px)
        print(json.dumps(res, indent=2))
    elif args.cmd == "summary":
        s = pnl.summary()
        print(json.dumps(s, indent=2))
    elif args.cmd == "list":
        if args.status == "OPEN":
            trades = pnl.get_open_trades()
        elif args.status == "CLOSED":
            trades = pnl.get_closed_trades(limit=10_000)
        else:
            # ALL
            trades = pnl.get_open_trades() + pnl.get_closed_trades(limit=10_000)
        rows: List[List[str]] = [
            [
                "trade_id",
                "symbol",
                "side",
                "size",
                "entry",
                "exit",
                "realized",
                "fees",
                "slip",
                "status",
                "age",
            ]
        ]
        for t in trades:
            age = (t.exit_ts if t.exit_ts is not None else now_ts()) - t.entry_ts
            rows.append(
                [
                    t.trade_id,
                    t.symbol,
                    t.side.value,
                    f"{t.size_base:.6f}",
                    f"{t.entry_price:.6f}",
                    "-" if t.exit_price is None else f"{t.exit_price:.6f}",
                    "-" if t.realized_pnl is None else f"{t.realized_pnl:.6f}",
                    f"{t.fees_quote:.6f}",
                    f"{t.slippage_quote:.6f}",
                    t.status,
                    f"{int(age)}s",
                ]
            )
        _print_table(rows)
    elif args.cmd == "export":
        tpath = pnl.export_csv(args.trades_csv)
        epath = pnl.export_equity_csv(args.equity_csv)
        print(f"Exported trades -> {tpath}")
        print(f"Exported equity -> {epath}")


if __name__ == "__main__":
    main()
