"""Abstract base classes for exchange adapters.

Defines the contract that all exchange implementations (Bybit, Binance, etc.)
must follow. Consumers import these ABCs and normalized types so they remain
exchange-agnostic.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any

from core.types import Side, OrderType, TimeInForce, WsEvent


# ── Normalized response types ──────────────────────────────────────────

@dataclass(slots=True)
class OrderResponse:
    success: bool
    order_id: str = ""
    error_code: int = 0
    error_msg: str = ""
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class WalletInfo:
    equity: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class InstrumentSpec:
    symbol: str = ""
    tick_size: float = 0.01
    min_qty: float = 0.001
    qty_step: float = 0.001
    min_notional: float = 5.0


@dataclass(slots=True)
class TickerInfo:
    symbol: str = ""
    last_price: float = 0.0
    turnover_24h: float = 0.0
    price_change_pct: float = 0.0


# ── REST ABC ───────────────────────────────────────────────────────────

class ExchangeREST(ABC):
    """Abstract REST client for exchange operations."""

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        order_link_id: str = "",
    ) -> OrderResponse: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse: ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> OrderResponse: ...

    @abstractmethod
    async def amend_order(
        self, symbol: str, order_id: str,
        qty: Optional[float] = None,
        price: Optional[float] = None,
    ) -> OrderResponse: ...

    @abstractmethod
    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: int = 0,
        position_side: Optional[Side] = None,  # for Binance: explicit close side
    ) -> OrderResponse: ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> OrderResponse: ...

    @abstractmethod
    async def get_wallet_balance(self) -> WalletInfo: ...

    @abstractmethod
    async def get_positions(self, symbol: str = "") -> list: ...

    @abstractmethod
    async def get_open_orders(self, symbol: str = "") -> list: ...

    @abstractmethod
    async def get_instruments(self, symbol: str = "") -> List[InstrumentSpec]: ...

    @abstractmethod
    async def get_tickers(self) -> List[TickerInfo]: ...

    @abstractmethod
    async def get_server_time(self) -> int: ...

    @abstractmethod
    async def get_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> List[dict]: ...

    @abstractmethod
    async def cleanup_bracket(self, symbol: str, order_id: str) -> None: ...

    @abstractmethod
    async def withdraw(self, coin: str, chain: str, address: str,
                       amount: float) -> dict:
        """Submit withdrawal. Returns {success, tx_id, error_msg}."""
        ...

    @abstractmethod
    async def get_deposit_address(self, coin: str, chain: str) -> dict:
        """Get deposit address. Returns {address, chain, tag}."""
        ...

    async def get_orderbook(self, symbol: str, depth: int = 50) -> dict:
        """Fetch L2 orderbook snapshot via REST. Returns {b: [...], a: [...], seq: int}.

        Used as a periodic fallback to heal WS delta drift.
        Default returns empty dict (subclasses override).
        """
        return {}

    @abstractmethod
    async def close(self) -> None: ...


# ── WebSocket ABC ──────────────────────────────────────────────────────

class ExchangeWS(ABC):
    """Abstract WebSocket client for market data and private streams."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def next_event(self, timeout_s: float = 0.05) -> Optional[WsEvent]: ...

    @abstractmethod
    def drain_private(self) -> List[WsEvent]: ...

    @abstractmethod
    def latency_estimate_ms(self) -> int: ...

    async def resubscribe_book(self, symbol: str) -> None:
        """Resubscribe orderbook topic to force a fresh snapshot. Optional."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...
