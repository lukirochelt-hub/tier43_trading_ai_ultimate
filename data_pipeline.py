import asyncio
from adapters.adapters_feeds import get_candles_paged


async def main():
    df = await get_candles_paged("bybit", "BTCUSDT", "15", limit_total=5000)
    df = df.sort_values("open_time").reset_index(drop=True)
    df.set_index("open_time", inplace=True)
    df.to_parquet("data/bybit_BTCUSDT_15.parquet")
    print("Saved:", len(df), "rows -> data/bybit_BTCUSDT_15.parquet")


if __name__ == "__main__":
    asyncio.run(main())
