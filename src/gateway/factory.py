"""Factory functions for creating exchange-specific REST and WebSocket clients."""
from __future__ import annotations
from typing import List, Optional, Callable, Awaitable

from gateway.base import ExchangeREST, ExchangeWS
from core.types import WsEvent


def create_rest(
    exchange: str,
    api_key: str,
    api_secret: str,
    testnet: bool = False,
) -> ExchangeREST:
    """Create an exchange REST client.

    Args:
        exchange: "bybit" or "binance"
        api_key: API key
        api_secret: API secret
        testnet: Use testnet endpoints
    """
    exchange = exchange.lower()
    if exchange == "bybit":
        from gateway.rest import BybitREST
        return BybitREST(api_key, api_secret, testnet=testnet)
    elif exchange == "binance":
        from gateway.binance_rest import BinanceREST
        return BinanceREST(api_key, api_secret, testnet=testnet)
    else:
        raise ValueError(f"Unsupported exchange: {exchange!r}. Use 'bybit' or 'binance'.")


def create_ws(
    exchange: str,
    symbols: List[str],
    config: dict,
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
    on_event: Optional[Callable[[WsEvent], Awaitable[None]]] = None,
) -> ExchangeWS:
    """Create an exchange WebSocket client.

    Args:
        exchange: "bybit" or "binance"
        symbols: List of symbols to subscribe
        config: Application config dict
        api_key: API key (for private streams)
        api_secret: API secret
        testnet: Use testnet endpoints
        on_event: Optional callback for events
    """
    exchange = exchange.lower()
    if exchange == "bybit":
        from gateway.bybit_ws import BybitWS
        return BybitWS(symbols, config, api_key, api_secret, testnet, on_event)
    elif exchange == "binance":
        from gateway.binance_ws import BinanceWS
        return BinanceWS(symbols, config, api_key, api_secret, testnet, on_event)
    else:
        raise ValueError(f"Unsupported exchange: {exchange!r}. Use 'bybit' or 'binance'.")
