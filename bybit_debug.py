import asyncio, json
import websockets

async def bybit_debug(symbol='BTCUSDT'):
    uri = 'wss://stream.bybit.com/v5/public/linear'
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({'op':'subscribe','args':['liquidation.all']}))
        print('[bybit] subscribed liquidation.all; filtering for', symbol)
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            # Status / Ping-Pong
            if isinstance(data, dict) and (data.get('success') or data.get('op') in ('subscribe','ping','pong')):
                print('[bybit/status]', data); continue
            # Daten
            if 'topic' in data and data.get('data'):
                for d in data['data']:
                    if d.get('symbol') == symbol:
                        print(f"[BYBIT] {symbol} {d.get('side')} {d.get('price')} × {d.get('size')}")

asyncio.run(bybit_debug('BTCUSDT'))
