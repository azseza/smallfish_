from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple, Optional


class Side(IntEnum):
    BUY = 1
    SELL = -1


class OrderStatus(str, Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


class OrderType(str, Enum):
    LIMIT = "Limit"
    MARKET = "Market"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "PostOnly"


class EventType(str, Enum):
    BOOK_SNAPSHOT = "book_snapshot"
    BOOK_DELTA = "book_delta"
    TRADE = "trade"
    ORDER = "order"
    EXECUTION = "execution"
    POSITION = "position"
    PING = "ping"


@dataclass(slots=True)
class Level:
    price: float
    size: float


@dataclass(slots=True)
class Trade:
    trade_id: str
    symbol: str
    price: float
    quantity: float
    side: Side
    timestamp: int  # ms epoch

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass(slots=True)
class Order:
    order_id: str
    symbol: str
    price: float
    quantity: float
    side: Side
    status: OrderStatus
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: int = 0
    updated_at: int = 0
    client_order_id: str = ""

    @property
    def remaining_qty(self) -> float:
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float
    entry_price: float
    side: Side
    stop_price: float = 0.0
    tp_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: int = 0
    stop_order_id: str = ""
    tp_order_id: str = ""
    trail_active: bool = False
    peak_favorable: float = 0.0  # best price seen (for trailing)
    worst_adverse: float = 0.0   # worst price seen (for MAE)
    signals_at_entry: dict = field(default_factory=dict)  # scores at entry time
    confidence_at_entry: float = 0.0

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity

    def mark_to_market(self, current_price: float) -> float:
        pnl = (current_price - self.entry_price) * self.quantity * int(self.side)
        self.unrealized_pnl = pnl

        # Track peak favorable (for MFE) and worst adverse (for MAE)
        if self.side == Side.BUY:
            if current_price > self.peak_favorable:
                self.peak_favorable = current_price
            if self.worst_adverse == 0 or current_price < self.worst_adverse:
                self.worst_adverse = current_price
        else:  # SELL
            if self.peak_favorable == 0 or current_price < self.peak_favorable:
                self.peak_favorable = current_price
            if current_price > self.worst_adverse:
                self.worst_adverse = current_price

        return pnl


@dataclass(slots=True)
class Execution:
    exec_id: str
    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    fee: float
    timestamp: int
    is_maker: bool = False


@dataclass(slots=True)
class BookSnapshot:
    symbol: str
    bids: List[Level]
    asks: List[Level]
    timestamp: int
    sequence: int


@dataclass(slots=True)
class WsEvent:
    event_type: EventType
    symbol: str
    data: object
    timestamp: int
    raw: Optional[dict] = None


@dataclass(slots=True)
class NormalizedExecution:
    """Exchange-agnostic execution (fill) event."""
    exec_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    price: float = 0.0
    quantity: float = 0.0
    is_maker: bool = False
    order_type: str = ""


@dataclass(slots=True)
class NormalizedOrderUpdate:
    """Exchange-agnostic order status update."""
    order_id: str = ""
    symbol: str = ""
    status: str = ""
    avg_price: float = 0.0
    filled_qty: float = 0.0
    reduce_only: bool = False
    stop_order_type: str = ""
    side: str = ""


@dataclass(slots=True)
class NormalizedPositionUpdate:
    """Exchange-agnostic position update."""
    symbol: str = ""
    size: float = 0.0
    entry_price: float = 0.0
    side: str = ""


@dataclass(slots=True)
class TradeResult:
    """Result of a completed round-trip trade."""
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_R: float  # in R-multiples
    entry_time: int
    exit_time: int
    duration_ms: int
    slippage_entry: float
    slippage_exit: float
    signals_at_entry: dict = field(default_factory=dict)
    exit_reason: str = ""
    mae: float = 0.0  # max adverse excursion
    mfe: float = 0.0  # max favorable excursion
