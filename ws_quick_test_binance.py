import asyncio, json, websockets

async def main(sym='btcusdt'):
    uri = f"wss://fstream.binance.com/ws/{sym.lower()}@aggTrade"
    print("Connecting:", uri)
    async with websockets.connect(uri) as ws:
        for i in range(5):  # zeig 5 Trades und beende
            msg = await ws.recv()
            data = json.loads(msg)
            print({k: data.get(k) for k in ('e','s','p','q','T')})
asyncio.run(main('BTCUSDT'))
