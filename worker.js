// Cloudflare Worker – Free Data Proxy + SSE
// Endpunkte:
//  GET /price?symbol=BTCUSDT
//  GET /oi?symbol=BTCUSDT
//  GET /funding?symbol=BTCUSDT
//  GET /liq24h?symbol=BTCUSDT
//  GET /dominance
//  GET /fng
//  POST /solana_tps   (optional GET /tps alias)
//  GET /sse?symbols=BTCUSDT,ETHUSDT,SOLUSDT   (Server-Sent Events Stream)

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const sendJSON = (obj, status = 200) =>
      new Response(JSON.stringify(obj), {
        status,
        headers: {
          "content-type": "application/json; charset=utf-8",
          "access-control-allow-origin": "*",
          "cache-control": "no-store",
        },
      });

    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET,POST,OPTIONS",
          "access-control-allow-headers": "*",
        },
      });
    }

    const timeoutMs = parseInt(env.UPSTREAM_TIMEOUT_MS || "12000", 10);

    const _getJSON = async (u, init = {}) => {
      const c = new AbortController();
      const t = setTimeout(() => c.abort("timeout"), timeoutMs);
      try {
        const r = await fetch(u, { ...init, signal: c.signal, cf: { cacheTtl: 0 } });
        if (!r.ok) return null;
        const j = await r.json();
        return j;
      } catch (_e) {
        return null;
      } finally {
        clearTimeout(t);
      }
    };

    const symbol = (url.searchParams.get("symbol") || "").toUpperCase();

    // ---- Simple GET endpoints ----
    if (url.pathname === "/price") {
      if (!symbol) return sendJSON({ error: "symbol required" }, 400);
      const j = await _getJSON(`https://fapi.binance.com/fapi/v1/ticker/price?symbol=${symbol}`);
      const price = j && j.price ? parseFloat(j.price) : null;
      return sendJSON({ symbol, price_usd: price });
    }

    if (url.pathname === "/oi") {
      if (!symbol) return sendJSON({ error: "symbol required" }, 400);
      const j = await _getJSON(`https://fapi.binance.com/fapi/v1/openInterest?symbol=${symbol}`);
      const oiBase = j && j.openInterest ? parseFloat(j.openInterest) : null;
      // optional gleich in USD w/ last price
      let oiUsd = null;
      if (oiBase !== null) {
        const p = await _getJSON(`https://fapi.binance.com/fapi/v1/ticker/price?symbol=${symbol}`);
        const px = p && p.price ? parseFloat(p.price) : null;
        if (px !== null) oiUsd = Math.abs(oiBase * px);
      }
      return sendJSON({ symbol, oi_base: oiBase, oi_usd: oiUsd === null ? null : Math.round(oiUsd * 100) / 100 });
    }

    if (url.pathname === "/funding") {
      if (!symbol) return sendJSON({ error: "symbol required" }, 400);
      const j = await _getJSON(`https://fapi.binance.com/fapi/v1/fundingRate?symbol=${symbol}&limit=1`);
      let fr = null;
      if (Array.isArray(j) && j.length) fr = parseFloat(j[0]?.fundingRate ?? "0");
      return sendJSON({ symbol, funding_8h: fr });
    }

    if (url.pathname === "/dominance") {
      const j = await _getJSON(`https://api.coingecko.com/api/v3/global`);
      let pct = null;
      try {
        pct = parseFloat(j.data.market_cap_percentage.btc);
      } catch (_e) {}
      return sendJSON({ btc_dominance_pct: pct });
    }

    if (url.pathname === "/fng") {
      const j = await _getJSON(`https://api.alternative.me/fng/`);
      let val = null;
      try {
        if (j && j.data && j.data.length) val = parseFloat(j.data[0].value);
      } catch (_e) {}
      return sendJSON({ fgi: val });
    }

    // Solana TPS – Public RPC, kurz halten
    if (url.pathname === "/tps" || url.pathname === "/solana_tps") {
      const body = { jsonrpc: "2.0", id: 1, method: "getRecentPerformanceSamples", params: [1] };
      const j = await _getJSON("https://api.mainnet-beta.solana.com", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      let tps = null;
      try {
        const s = j.result?.[0];
        const tx = parseFloat(String(s.numTransactions));
        const secs = parseFloat(String(s.samplePeriodSecs));
        if (secs > 0) tps = Math.round((tx / secs) * 100) / 100;
      } catch (_e) {}
      return sendJSON({ sol_tps: tps });
    }

    // 24h Liquidations USD (robust, aber gratis) – paginiert forceOrders/allForceOrders
    if (url.pathname === "/liq24h") {
      if (!symbol) return sendJSON({ error: "symbol required" }, 400);

      async function sumForceOrders(baseUrl) {
        const end = Date.now();
        let start = end - 24 * 60 * 60 * 1000;
        let total = 0.0;
        let pages = 0;

        while (start < end && pages < 30) {
          const params = new URLSearchParams({
            symbol,
            autoCloseType: "LIQUIDATION",
            startTime: String(start),
            endTime: String(end),
            limit: "1000",
          });
          let j = await _getJSON(`${baseUrl}/fapi/v1/forceOrders?${params.toString()}`);
          if (!Array.isArray(j) || j.length === 0) {
            j = await _getJSON(`${baseUrl}/fapi/v1/allForceOrders?${params.toString()}`);
          }
          if (!Array.isArray(j) || j.length === 0) break;

          let lastTs = start;
          for (const it of j) {
            const px = parseFloat(it.price ?? it.averagePrice ?? "0");
            const qty = parseFloat(it.executedQty ?? it.origQty ?? "0");
            if (px && qty) total += Math.abs(px * qty);
            const ts = Number(it.time ?? start);
            if (ts > lastTs) lastTs = ts;
          }
          if (lastTs <= start) break;
          start = lastTs + 1;
          pages += 1;
          await new Promise((r) => setTimeout(r, 120)); // freundlich drosseln
        }
        return total;
      }

      let total = await sumForceOrders("https://fapi.binance.com");
      if (total === 0) {
        // einmal nachlegen (manchmal verzögert)
        await new Promise((r) => setTimeout(r, 250));
        total = await sumForceOrders("https://fapi.binance.com");
      }
      return sendJSON({ symbol, liq_24h_usd: total > 0 ? Math.round(total * 100) / 100 : null });
    }

    // ---- SSE Realtime Stream ----
    if (url.pathname === "/sse") {
      const list = (url.searchParams.get("symbols") || "BTCUSDT,ETHUSDT,SOLUSDT")
        .split(",")
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean);

      const stream = new ReadableStream({
        async start(controller) {
          const enc = (obj) =>
            controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(obj)}\n\n`));
          const ping = () => controller.enqueue(new TextEncoder().encode(`: ping\n\n`));

          // sofort initiale Ladung
          await pushOnce();
          // dann alle 5s
          const iv = setInterval(pushOnce, 5000);
          const keepAlive = setInterval(ping, 15000);

          async function pushOnce() {
            try {
              const now = new Date().toISOString();
              // Parallel pro Symbol
              await Promise.all(
                list.map(async (sym) => {
                  const [p, oi, fr, liq] = await Promise.all([
                    _getJSON(`https://fapi.binance.com/fapi/v1/ticker/price?symbol=${sym}`),
                    _getJSON(`https://fapi.binance.com/fapi/v1/openInterest?symbol=${sym}`),
                    _getJSON(`https://fapi.binance.com/fapi/v1/fundingRate?symbol=${sym}&limit=1`),
                    _getJSON(
                      `https://fapi.binance.com/fapi/v1/forceOrders?symbol=${sym}&autoCloseType=LIQUIDATION&startTime=${
                        Date.now() - 24 * 60 * 60 * 1000
                      }&endTime=${Date.now()}&limit=1000`,
                    ),
                  ]);

                  // Quick liq Sum (nur 1 Page im Stream, genügt für „Trend“)
                  let liqUsd = 0;
                  if (Array.isArray(liq)) {
                    for (const it of liq) {
                      const px = parseFloat(it.price ?? it.averagePrice ?? "0");
                      const qty = parseFloat(it.executedQty ?? it.origQty ?? "0");
                      if (px && qty) liqUsd += Math.abs(px * qty);
                    }
                  }
                  const price = p?.price ? parseFloat(p.price) : null;
                  const oiBase = oi?.openInterest ? parseFloat(oi.openInterest) : null;
                  const funding = Array.isArray(fr) && fr.length ? parseFloat(fr[0].fundingRate) : null;

                  enc({
                    ts_utc: now,
                    symbol: sym,
                    price_usd: price,
                    oi_base: oiBase,
                    // optionale USD-Umrechnung live:
                    oi_usd: price !== null && oiBase !== null ? Math.round(price * oiBase * 100) / 100 : null,
                    funding_8h: funding,
                    liq_24h_usd_hint: liqUsd || null,
                  });
                }),
              );

              // globale Metriken
              const [dom, fng] = await Promise.all([
                _getJSON(`https://api.coingecko.com/api/v3/global`),
                _getJSON(`https://api.alternative.me/fng/`),
              ]);
              let domPct = null;
              try {
                domPct = parseFloat(dom.data.market_cap_percentage.btc);
              } catch (_e) {}
              let fngVal = null;
              try {
                if (fng?.data?.length) fngVal = parseFloat(fng.data[0].value);
              } catch (_e) {}

              enc({ type: "meta", ts_utc: new Date().toISOString(), btc_dominance_pct: domPct, fgi: fngVal });
            } catch (_e) {
              enc({ type: "error", msg: "push failed" });
            }
          }

          controller.enqueue(new TextEncoder().encode("retry: 3000\n\n")); // Reconnect delay
          // Cleanup
          controller.closed.finally(() => {
            clearInterval(iv);
            clearInterval(keepAlive);
          });
        },
      });

      return new Response(stream, {
        headers: {
          "content-type": "text/event-stream",
          "cache-control": "no-store",
          "access-control-allow-origin": "*",
          "connection": "keep-alive",
        },
      });
    }

    return sendJSON({ ok: true, info: "use /price, /oi, /funding, /liq24h, /dominance, /fng, /tps, /sse" });
  },
};
