import asyncio
from adapters.adapters_alt_feeds import stream_liquidations_binance

async def main():
    try:
        await stream_liquidations_binance("BTCUSDT")
    except KeyboardInterrupt:
        print("bye 👋")

asyncio.run(main())
