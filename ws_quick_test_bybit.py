import asyncio, json, websockets

async def main(sym='BTCUSDT'):
    uri = "wss://stream.bybit.com/v5/public/linear"
    print("Connecting:", uri)
    async with websockets.connect(uri) as ws:
        sub = {"op":"subscribe","args":[f"publicTrade.{sym}"]}
        await ws.send(json.dumps(sub))
        got = 0
        while got < 5:
            msg = json.loads(await ws.recv())
            if 'data' in msg:
                got += 1
                print(msg['data'][0])
asyncio.run(main('BTCUSDT'))
