"""Gateway package — exchange-agnostic adapters.

Re-exports the abstract interfaces and factory so consumers can write::

    from gateway import ExchangeREST, ExchangeWS, create_rest, create_ws
"""
from gateway.base import (
    ExchangeREST,
    ExchangeWS,
    OrderResponse,
    WalletInfo,
    InstrumentSpec,
    TickerInfo,
)
from gateway.factory import create_rest, create_ws

__all__ = [
    "ExchangeREST",
    "ExchangeWS",
    "OrderResponse",
    "WalletInfo",
    "InstrumentSpec",
    "TickerInfo",
    "create_rest",
    "create_ws",
]
