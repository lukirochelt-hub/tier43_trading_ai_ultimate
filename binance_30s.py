import asyncio
from adapters.adapters_alt_feeds import stream_liquidations_binance

async def main():
    # 30 Sekunden streamen, dann sauber beenden
    task = asyncio.create_task(stream_liquidations_binance("BTCUSDT"))
    try:
        await asyncio.sleep(30)
    finally:
        task.cancel()
        try:
            await task
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
