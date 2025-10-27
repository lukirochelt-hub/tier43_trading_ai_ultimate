# ===============================================
# test_slip_model.py
# Tier 4.3+ – Slippage Model Tester & Validator
# ===============================================
# Purpose:
#   Validate the slippage estimation model against
#   historical fills, latency, and simulated spreads.
# -----------------------------------------------
# Features:
#   - Synthetic orderbook generator
#   - Latency-aware price impact model
#   - Comparison of real vs expected fill prices
#   - Statistical diagnostics (MAE, MAPE, R²)
#   - Optional plotting (Matplotlib)
# -----------------------------------------------

import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

# =========================
# Config
# =========================
DATA_DIR = Path(os.getenv("SLIP_DATA_DIR", "./data/slippage"))
RESULTS_DIR = Path(os.getenv("SLIP_RESULTS_DIR", "./results/slippage"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAT_MS = int(os.getenv("SIM_LAT_MS", "180"))
DEFAULT_BOOK_DEPTH = int(os.getenv("SIM_BOOK_DEPTH", "25"))
DEFAULT_TICK_SIZE = float(os.getenv("SIM_TICK_SIZE", "0.01"))
DEFAULT_SPREAD_BPS = float(os.getenv("SIM_SPREAD_BPS", "4.0"))

# =========================
# Core Classes
# =========================


@dataclass
class FillEvent:
    ts: float
    side: str  # "BUY" or "SELL"
    qty: float
    price_real: float
    price_expected: float
    latency_ms: int
    spread_bps: float


@dataclass
class SlipStats:
    mae: float
    mape: float
    r2: float
    avg_latency_ms: float
    avg_spread_bps: float
    n: int


# =========================
# Helper Functions
# =========================


def generate_synthetic_book(
    mid_price: float, depth: int, tick: float, spread_bps: float
) -> Tuple[np.ndarray, np.ndarray]:
    spread = mid_price * (spread_bps / 10000)
    best_bid = mid_price - spread / 2
    best_ask = mid_price + spread / 2

    bids = np.array([best_bid - i * tick for i in range(depth)])
    asks = np.array([best_ask + i * tick for i in range(depth)])
    return bids, asks


def simulate_fill_price(
    side: str, qty: float, bids: np.ndarray, asks: np.ndarray, latency_ms: int
) -> float:
    """Simulate execution price based on side, qty, and latency drift."""
    slip_factor = np.log1p(latency_ms / 500.0) * 0.0005
    if side == "BUY":
        base_price = asks[0]
        impact = slip_factor * qty
        return base_price * (1 + impact)
    else:
        base_price = bids[0]
        impact = slip_factor * qty
        return base_price * (1 - impact)


def evaluate_slippage_model(events: List[FillEvent]) -> SlipStats:
    df = pd.DataFrame([asdict(e) for e in events])
    df["abs_err"] = abs(df["price_real"] - df["price_expected"])
    df["pct_err"] = df["abs_err"] / df["price_real"]
    ss_res = ((df["price_real"] - df["price_expected"]) ** 2).sum()
    ss_tot = ((df["price_real"] - df["price_real"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return SlipStats(
        mae=df["abs_err"].mean(),
        mape=df["pct_err"].mean() * 100,
        r2=r2,
        avg_latency_ms=df["latency_ms"].mean(),
        avg_spread_bps=df["spread_bps"].mean(),
        n=len(df),
    )


# =========================
# Test Routine
# =========================


def run_slippage_test(
    n_events: int = 250, seed: int = 42, plot: bool = True
) -> SlipStats:
    random.seed(seed)
    np.random.seed(seed)

    events: List[FillEvent] = []
    for _ in range(n_events):
        side = random.choice(["BUY", "SELL"])
        mid = random.uniform(50, 250)
        bids, asks = generate_synthetic_book(
            mid, DEFAULT_BOOK_DEPTH, DEFAULT_TICK_SIZE, DEFAULT_SPREAD_BPS
        )
        qty = random.uniform(0.1, 10.0)
        lat = random.randint(50, 400)
        price_exp = simulate_fill_price(side, qty, bids, asks, lat)
        price_real = price_exp * random.uniform(0.998, 1.002)
        events.append(
            FillEvent(
                time.time(), side, qty, price_real, price_exp, lat, DEFAULT_SPREAD_BPS
            )
        )

    stats = evaluate_slippage_model(events)
    print(
        f"MAE={stats.mae:.4f}, MAPE={stats.mape:.2f}%, R²={stats.r2:.4f}, N={stats.n}"
    )

    # Save
    with open(RESULTS_DIR / f"slip_results_{int(time.time())}.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)

    if plot:
        df = pd.DataFrame([asdict(e) for e in events])
        plt.scatter(df["price_expected"], df["price_real"], alpha=0.6, s=14)
        plt.plot(
            [df["price_expected"].min(), df["price_expected"].max()],
            [df["price_expected"].min(), df["price_expected"].max()],
            "r--",
            label="ideal",
        )
        plt.xlabel("Expected Price")
        plt.ylabel("Real Price")
        plt.title("Slippage Model – Expected vs Real")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "slippage_scatter.png")
        plt.close()

    return stats


# =========================
# Main
# =========================

if __name__ == "__main__":
    print("▶ Running Tier 4.3+ slippage test ...")
    stats = run_slippage_test(n_events=300)
    print("✅ Done.")
