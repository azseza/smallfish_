"""Dynamic symbol scanner.

Fetches top USDT perpetual symbols ranked by:
  1. 24h trading volume (liquidity)
  2. 24h price change % (volatility = scalping opportunity)

Also auto-fetches instrument specs (tick_size, min_qty, qty_step).
Works with any exchange adapter implementing ExchangeREST.
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Optional

from gateway.base import ExchangeREST, InstrumentSpec, TickerInfo

log = logging.getLogger(__name__)


async def scan_top_symbols(
    rest: ExchangeREST,
    n: int = 10,
    min_volume_usd: float = 50_000_000,
    min_price: float = 0.01,
    sort_by: str = "score",
) -> List[dict]:
    """Scan and rank top symbols for scalping.

    Ranking score = volume_rank * 0.6 + change_rank * 0.4

    Args:
        rest: Exchange REST client
        n: Number of top symbols to return
        min_volume_usd: Minimum 24h turnover in USD
        min_price: Minimum last price
        sort_by: "score" (combined), "volume", or "change"

    Returns:
        List of dicts with symbol info, sorted best-first
    """
    tickers, specs_list = await asyncio.gather(
        rest.get_tickers(),
        rest.get_instruments(),
    )

    # Index specs by symbol
    specs: Dict[str, InstrumentSpec] = {s.symbol: s for s in specs_list}

    candidates = []
    for t in tickers:
        sym = t.symbol
        if not sym.endswith("USDT"):
            continue

        # Skip stablecoins and leveraged tokens
        base = sym.replace("USDT", "")
        if base in ("USDC", "DAI", "TUSD", "BUSD", "FDUSD", "USDD", "PYUSD"):
            continue
        if any(x in base for x in ("1000", "10000", "UP", "DOWN", "BEAR", "BULL")):
            if base.startswith("1000"):
                pass  # keep meme coins
            else:
                continue

        volume_24h = t.turnover_24h
        last_price = t.last_price
        change_pct = abs(t.price_change_pct) * 100

        if volume_24h < min_volume_usd:
            continue
        if last_price < min_price:
            continue

        spec = specs.get(sym)
        if not spec:
            continue

        candidates.append({
            "symbol": sym,
            "last_price": last_price,
            "volume_24h_usd": volume_24h,
            "change_24h_pct": change_pct,
            "tick_size": spec.tick_size,
            "min_qty": spec.min_qty,
            "qty_step": spec.qty_step,
            "min_notional": spec.min_notional,
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


async def print_scanner_report(rest: ExchangeREST, n: int = 20):
    """Print a formatted scanner report."""
    symbols = await scan_top_symbols(rest, n=n)

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
