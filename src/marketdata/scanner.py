"""Dynamic symbol scanner.

Fetches top USDT perpetual symbols from Bybit ranked by:
  1. 24h trading volume (liquidity)
  2. 24h price change % (volatility = scalping opportunity)

Also auto-fetches instrument specs (tick_size, min_qty, qty_step).
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Optional

import aiohttp

log = logging.getLogger(__name__)

BYBIT_BASE = "https://api.bybit.com"
BYBIT_TESTNET_BASE = "https://api-testnet.bybit.com"


async def fetch_tickers(testnet: bool = False) -> List[dict]:
    """Fetch 24h tickers for all USDT linear perpetuals."""
    base = BYBIT_TESTNET_BASE if testnet else BYBIT_BASE
    url = f"{base}/v5/market/tickers"
    params = {"category": "linear"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json(content_type=None)
            tickers = data.get("result", {}).get("list", [])
            return tickers


async def fetch_instruments(testnet: bool = False) -> Dict[str, dict]:
    """Fetch instrument specs for all USDT linear perpetuals."""
    base = BYBIT_TESTNET_BASE if testnet else BYBIT_BASE
    url = f"{base}/v5/market/instruments-info"
    params = {"category": "linear", "limit": 1000}
    specs = {}

    async with aiohttp.ClientSession() as session:
        cursor = ""
        while True:
            if cursor:
                params["cursor"] = cursor
            async with session.get(url, params=params) as resp:
                data = await resp.json(content_type=None)
                items = data.get("result", {}).get("list", [])
                for item in items:
                    sym = item.get("symbol", "")
                    if not sym.endswith("USDT"):
                        continue
                    lot = item.get("lotSizeFilter", {})
                    price = item.get("priceFilter", {})
                    specs[sym] = {
                        "symbol": sym,
                        "tick_size": float(price.get("tickSize", "0.01")),
                        "min_qty": float(lot.get("minOrderQty", "0.001")),
                        "qty_step": float(lot.get("qtyStep", "0.001")),
                        "min_notional": float(lot.get("minNotionalValue", "5")),
                    }
                cursor = data.get("result", {}).get("nextPageCursor", "")
                if not cursor or not items:
                    break

    return specs


async def scan_top_symbols(
    n: int = 10,
    min_volume_usd: float = 50_000_000,
    min_price: float = 0.01,
    testnet: bool = False,
    sort_by: str = "score",
) -> List[dict]:
    """Scan and rank top symbols for scalping.

    Ranking score = volume_rank * 0.6 + change_rank * 0.4
    (High volume = good liquidity, high change = good opportunity)

    Args:
        n: Number of top symbols to return
        min_volume_usd: Minimum 24h turnover in USD
        min_price: Minimum last price
        testnet: Use testnet API
        sort_by: "score" (combined), "volume", or "change"

    Returns:
        List of dicts with symbol info, sorted best-first
    """
    tickers, specs = await asyncio.gather(
        fetch_tickers(testnet),
        fetch_instruments(testnet),
    )

    candidates = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue

        # Skip stablecoins and leveraged tokens
        base = sym.replace("USDT", "")
        if base in ("USDC", "DAI", "TUSD", "BUSD", "FDUSD", "USDD", "PYUSD"):
            continue
        if any(x in base for x in ("1000", "10000", "UP", "DOWN", "BEAR", "BULL")):
            # Allow 1000-prefixed meme coins (1000PEPE, 1000SHIB etc)
            if base.startswith("1000"):
                pass  # keep these
            else:
                continue

        volume_24h = float(t.get("turnover24h", 0))
        last_price = float(t.get("lastPrice", 0))
        change_pct = abs(float(t.get("price24hPcnt", 0))) * 100

        if volume_24h < min_volume_usd:
            continue
        if last_price < min_price:
            continue

        spec = specs.get(sym, {})
        if not spec:
            continue

        candidates.append({
            "symbol": sym,
            "last_price": last_price,
            "volume_24h_usd": volume_24h,
            "change_24h_pct": change_pct,
            "tick_size": spec["tick_size"],
            "min_qty": spec["min_qty"],
            "qty_step": spec["qty_step"],
            "min_notional": spec.get("min_notional", 5),
        })

    if not candidates:
        log.warning("No candidates found matching criteria")
        return []

    # Rank by volume
    candidates.sort(key=lambda x: -x["volume_24h_usd"])
    for i, c in enumerate(candidates):
        c["volume_rank"] = i + 1

    # Rank by absolute change
    candidates.sort(key=lambda x: -x["change_24h_pct"])
    for i, c in enumerate(candidates):
        c["change_rank"] = i + 1

    # Combined score (lower is better)
    total = len(candidates)
    for c in candidates:
        vol_score = c["volume_rank"] / total
        chg_score = c["change_rank"] / total
        c["score"] = vol_score * 0.6 + chg_score * 0.4

    if sort_by == "volume":
        candidates.sort(key=lambda x: x["volume_rank"])
    elif sort_by == "change":
        candidates.sort(key=lambda x: x["change_rank"])
    else:
        candidates.sort(key=lambda x: x["score"])

    return candidates[:n]


def apply_specs_to_config(config: dict, symbols_info: List[dict]) -> None:
    """Update config with fetched instrument specs."""
    config.setdefault("tick_sizes", {})
    config.setdefault("min_qty", {})
    config.setdefault("qty_step", {})
    config["symbols"] = []

    for info in symbols_info:
        sym = info["symbol"]
        config["symbols"].append(sym)
        config["tick_sizes"][sym] = info["tick_size"]
        config["min_qty"][sym] = info["min_qty"]
        config["qty_step"][sym] = info["qty_step"]


async def print_scanner_report(n: int = 20, testnet: bool = False):
    """Print a formatted scanner report."""
    symbols = await scan_top_symbols(n=n, testnet=testnet)

    print(f"\n{'='*90}")
    print(f"  TOP {len(symbols)} USDT PERPETUALS FOR SCALPING")
    print(f"{'='*90}")
    print(f"  {'#':>3} {'Symbol':<14} {'Price':>12} {'24h Vol ($)':>16} "
          f"{'24h Chg':>8} {'Tick':>10} {'Score':>7}")
    print(f"  {'-'*87}")

    for i, s in enumerate(symbols, 1):
        vol_str = f"${s['volume_24h_usd']/1e6:,.0f}M" if s['volume_24h_usd'] >= 1e6 else f"${s['volume_24h_usd']:,.0f}"
        print(f"  {i:>3} {s['symbol']:<14} ${s['last_price']:>10,.4f} {vol_str:>16} "
              f"{s['change_24h_pct']:>+7.2f}% {s['tick_size']:>10} {s['score']:>7.3f}")

    print(f"{'='*90}")
    return symbols


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    asyncio.run(print_scanner_report(n))
