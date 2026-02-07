"""Symbol format conversion between internal format and exchange-specific formats.

Internal format: BTCUSDT (no separator, USDT-margined)
MEXC futures:    BTC_USDT (underscore separator)
dYdX v4:         BTC-USD  (hyphen separator, USD not USDT)

All conversions happen at the adapter boundary — the core engine always
uses the internal BTCUSDT format.
"""
from __future__ import annotations

import re

# ── MEXC ──────────────────────────────────────────────────────────────

_MEXC_QUOTE = re.compile(r"(USDT|USDC|BUSD)$")


def to_mexc(symbol: str) -> str:
    """Convert internal symbol to MEXC futures format.

    BTCUSDT -> BTC_USDT
    """
    m = _MEXC_QUOTE.search(symbol)
    if m:
        base = symbol[: m.start()]
        quote = m.group(1)
        return f"{base}_{quote}"
    return symbol


def from_mexc(symbol: str) -> str:
    """Convert MEXC futures symbol to internal format.

    BTC_USDT -> BTCUSDT
    """
    return symbol.replace("_", "")


# ── dYdX v4 ───────────────────────────────────────────────────────────

_DYDX_USDT = re.compile(r"USDT$")


def to_dydx(symbol: str) -> str:
    """Convert internal symbol to dYdX v4 format.

    BTCUSDT -> BTC-USD
    ETHUSDT -> ETH-USD
    """
    # Strip USDT suffix, insert hyphen + USD
    base = _DYDX_USDT.sub("", symbol)
    return f"{base}-USD"


def from_dydx(symbol: str) -> str:
    """Convert dYdX v4 symbol to internal format.

    BTC-USD -> BTCUSDT
    ETH-USD -> ETHUSDT
    """
    # BTC-USD -> BTCUSDT
    base = symbol.replace("-USD", "")
    return f"{base}USDT"
