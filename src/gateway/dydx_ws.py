"""dYdX v4 Indexer WebSocket client.

Single WS connection for both public and private data (unlike Bybit/Binance
which use separate connections).

Channels:
- v4_orderbook: L2 orderbook updates
- v4_trades: trade events
- v4_subaccounts: private fills, positions, account updates

URL: wss://indexer.dydx.trade/v4/ws (mainnet)
     wss://indexer.v4testnet.dydx.exchange/v4/ws (testnet)

Symbol format: BTC-USD (hyphen separator, USD not USDT).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, List, Callable, Awaitable

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from core.types import (
    WsEvent, EventType, Trade, Side,
    NormalizedExecution, NormalizedOrderUpdate, NormalizedPositionUpdate,
)
from core.utils import time_now_ms
from gateway.base import ExchangeWS
from gateway.symbol_map import to_dydx, from_dydx

log = logging.getLogger(__name__)

WS_URL = "wss://indexer.dydx.trade/v4/ws"
WS_URL_TESTNET = "wss://indexer.v4testnet.dydx.exchange/v4/ws"


class DydxWS(ExchangeWS):
    """Async WebSocket client for dYdX v4 Indexer."""

    def __init__(
        self,
        symbols: List[str],
        config: dict,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        on_event: Optional[Callable[[WsEvent], Awaitable[None]]] = None,
    ):
        self.symbols = symbols
        self.config = config
        self.address = api_key  # dYdX address for private subs
        self.testnet = testnet
        self.on_event = on_event

        ws_cfg = config.get("ws", {})
        self.reconnect_delay = ws_cfg.get("reconnect_delay_s", 1)
        self.max_reconnect_delay = ws_cfg.get("max_reconnect_delay_s", 30)

        self._ws = None
        self._running = False
        self._event_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=10000)
        self._private_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=1000)
        self._last_msg_ts: int = 0
        self._first_book: dict[str, bool] = {}

    @property
    def _ws_url(self) -> str:
        return WS_URL_TESTNET if self.testnet else WS_URL

    # --- Connection Lifecycle ---

    async def start(self) -> None:
        self._running = True
        await self._run_ws()

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _run_ws(self) -> None:
        delay = self.reconnect_delay
        while self._running:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    delay = self.reconnect_delay
                    log.info("dYdX WS connected")
                    self._first_book = {}

                    # Subscribe to public channels
                    await self._subscribe_public(ws)

                    # Subscribe to private channels if address provided
                    if self.address:
                        await self._subscribe_private(ws)

                    await self._listen(ws)

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("dYdX WS disconnected: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _subscribe_public(self, ws) -> None:
        """Subscribe to orderbook and trades for all symbols."""
        for sym in self.symbols:
            dydx_sym = to_dydx(sym)
            # Orderbook channel
            await ws.send(orjson.dumps({
                "type": "subscribe",
                "channel": "v4_orderbook",
                "id": dydx_sym,
            }))
            # Trades channel
            await ws.send(orjson.dumps({
                "type": "subscribe",
                "channel": "v4_trades",
                "id": dydx_sym,
            }))

    async def _subscribe_private(self, ws) -> None:
        """Subscribe to subaccount channel for fills and positions."""
        await ws.send(orjson.dumps({
            "type": "subscribe",
            "channel": "v4_subaccounts",
            "id": f"{self.address}/0",  # subaccount 0
        }))

    async def _listen(self, ws) -> None:
        async for raw in ws:
            now = time_now_ms()
            self._last_msg_ts = now
            try:
                msg = orjson.loads(raw)
            except Exception:
                continue

            channel = msg.get("channel", "")
            msg_type = msg.get("type", "")

            # Skip connection/subscription confirmations
            if msg_type in ("connected", "subscribed", "unsubscribed"):
                continue

            if channel == "v4_orderbook":
                events = self._parse_orderbook(msg, now)
                self._enqueue_public(events)
            elif channel == "v4_trades":
                events = self._parse_trades(msg, now)
                self._enqueue_public(events)
            elif channel == "v4_subaccounts":
                events = self._parse_subaccount(msg, now)
                self._enqueue_private(events)

    def _enqueue_public(self, events: List[WsEvent]) -> None:
        for evt in events:
            try:
                self._event_queue.put_nowait(evt)
            except asyncio.QueueFull:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(evt)

    def _enqueue_private(self, events: List[WsEvent]) -> None:
        for evt in events:
            try:
                self._private_queue.put_nowait(evt)
            except asyncio.QueueFull:
                self._private_queue.get_nowait()
                self._private_queue.put_nowait(evt)

    # --- Parsers ---

    def _parse_orderbook(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse orderbook snapshot or update from dYdX.

        Initial: type=subscribed, contents={bids: [...], asks: [...]}
        Updates: type=channel_data, contents={bids: [...], asks: [...]}
        """
        events: List[WsEvent] = []
        ticker = msg.get("id", "")
        if not ticker:
            return events
        symbol = from_dydx(ticker)

        contents = msg.get("contents", {})
        msg_type = msg.get("type", "")

        bids = contents.get("bids", [])
        asks = contents.get("asks", [])

        # Convert to [[price, qty], ...] format
        book_data = {
            "b": [[str(b.get("price", b[0]) if isinstance(b, dict) else b[0]),
                    str(b.get("size", b[1]) if isinstance(b, dict) else b[1])]
                   for b in bids],
            "a": [[str(a.get("price", a[0]) if isinstance(a, dict) else a[0]),
                    str(a.get("size", a[1]) if isinstance(a, dict) else a[1])]
                   for a in asks],
        }

        if symbol not in self._first_book or msg_type == "subscribed":
            evt_type = EventType.BOOK_SNAPSHOT
            self._first_book[symbol] = True
        else:
            evt_type = EventType.BOOK_DELTA

        events.append(WsEvent(
            event_type=evt_type,
            symbol=symbol,
            data=book_data,
            timestamp=now,
            raw=contents,
        ))
        return events

    def _parse_trades(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse trade events from dYdX.

        Format: {channel: "v4_trades", id: "BTC-USD", contents: {trades: [...]}}
        """
        events: List[WsEvent] = []
        ticker = msg.get("id", "")
        if not ticker:
            return events
        symbol = from_dydx(ticker)

        contents = msg.get("contents", {})
        trades = contents.get("trades", [])

        for t in trades:
            side = Side.BUY if t.get("side") == "BUY" else Side.SELL
            created = t.get("createdAt", "")
            if created:
                try:
                    from datetime import datetime as dt
                    ts = int(dt.fromisoformat(created.replace("Z", "+00:00")).timestamp() * 1000)
                except Exception:
                    ts = now
            else:
                ts = now

            trade = Trade(
                trade_id=str(t.get("id", now)),
                symbol=symbol,
                price=float(t.get("price", 0)),
                quantity=float(t.get("size", 0)),
                side=side,
                timestamp=ts,
            )
            events.append(WsEvent(
                event_type=EventType.TRADE,
                symbol=symbol,
                data=trade,
                timestamp=ts,
            ))
        return events

    def _parse_subaccount(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse private subaccount updates (fills, positions, orders)."""
        events: List[WsEvent] = []
        contents = msg.get("contents", {})

        # Order fills
        fills = contents.get("fills", [])
        for fill in fills:
            ticker = fill.get("market", "")
            symbol = from_dydx(ticker) if ticker else ""
            side_str = fill.get("side", "")

            events.append(WsEvent(
                event_type=EventType.EXECUTION,
                symbol=symbol,
                data=NormalizedExecution(
                    exec_id=str(fill.get("id", "")),
                    order_id=str(fill.get("orderId", "")),
                    symbol=symbol,
                    side=Side.BUY if side_str == "BUY" else Side.SELL,
                    price=float(fill.get("price", 0)),
                    quantity=float(fill.get("size", 0)),
                    is_maker=fill.get("liquidity") == "MAKER",
                    order_type=fill.get("type", ""),
                ),
                timestamp=now,
            ))

        # Order updates
        orders = contents.get("orders", [])
        for order in orders:
            ticker = order.get("ticker", "")
            symbol = from_dydx(ticker) if ticker else ""
            status_map = {
                "OPEN": "New", "PENDING": "New", "BEST_EFFORT_OPENED": "New",
                "FILLED": "Filled", "BEST_EFFORT_CANCELED": "Cancelled",
                "CANCELED": "Cancelled", "UNTRIGGERED": "New",
            }
            status = status_map.get(order.get("status", ""), "New")
            side_str = order.get("side", "")

            events.append(WsEvent(
                event_type=EventType.ORDER,
                symbol=symbol,
                data=NormalizedOrderUpdate(
                    order_id=str(order.get("id", "")),
                    symbol=symbol,
                    status=status,
                    avg_price=float(order.get("price", 0)),
                    filled_qty=float(order.get("totalFilled", 0)),
                    reduce_only=order.get("reduceOnly", False),
                    stop_order_type="",
                    side=side_str.capitalize() if side_str else "",
                ),
                timestamp=now,
            ))

        # Position updates
        positions = contents.get("perpetualPositions", [])
        for pos in positions:
            ticker = pos.get("market", "")
            symbol = from_dydx(ticker) if ticker else ""
            size = float(pos.get("size", 0))
            entry = float(pos.get("entryPrice", 0))
            side = ""
            if size > 0:
                side = "Buy"
            elif size < 0:
                side = "Sell"

            events.append(WsEvent(
                event_type=EventType.POSITION,
                symbol=symbol,
                data=NormalizedPositionUpdate(
                    symbol=symbol,
                    size=abs(size),
                    entry_price=entry,
                    side=side,
                ),
                timestamp=now,
            ))

        return events

    # --- Public API ---

    async def next_event(self, timeout_s: float = 0.05) -> Optional[WsEvent]:
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    def drain_private(self) -> List[WsEvent]:
        events = []
        while not self._private_queue.empty():
            try:
                events.append(self._private_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    def latency_estimate_ms(self) -> int:
        now = time_now_ms()
        age = now - self._last_msg_ts if self._last_msg_ts > 0 else 999
        return min(age, 999)

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open
