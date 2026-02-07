"""MEXC Futures WebSocket client.

Handles both public (depth, deals) and private (orders, positions) streams.
Key differences from Bybit/Binance:
- URL: wss://contract.mexc.com/ws (single endpoint, JSON)
- Ping: send {"method": "ping"} every 15s
- Depth: subscribe "sub.contract.depth.{symbol}"
- Trades: subscribe "sub.contract.deal.{symbol}"
- Private: authenticate then subscribe sub.personal.*
- Symbol format: BTC_USDT (underscore separator)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
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
from gateway.symbol_map import from_mexc

log = logging.getLogger(__name__)

WS_URL = "wss://contract.mexc.com/edge"
PING_INTERVAL_S = 15


class MexcWS(ExchangeWS):
    """Async WebSocket client for MEXC Futures."""

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
        self.api_key = api_key
        self.api_secret = api_secret
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
                    WS_URL,
                    ping_interval=None,  # we handle pings manually
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    delay = self.reconnect_delay
                    log.info("MEXC WS connected")
                    self._first_book = {}

                    # Subscribe to public channels
                    await self._subscribe_public(ws)

                    # Authenticate and subscribe to private channels
                    if self.api_key and self.api_secret:
                        await self._authenticate(ws)
                        await self._subscribe_private(ws)

                    # Run listener and ping concurrently
                    listener = asyncio.create_task(self._listen(ws))
                    pinger = asyncio.create_task(self._ping_loop(ws))
                    done, pending = await asyncio.wait(
                        [listener, pinger],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("MEXC WS disconnected: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _subscribe_public(self, ws) -> None:
        """Subscribe to depth and trade channels for all symbols."""
        from gateway.symbol_map import to_mexc
        for sym in self.symbols:
            mexc_sym = to_mexc(sym)
            # Depth subscription
            await ws.send(orjson.dumps({
                "method": "sub.contract.depth",
                "param": {"symbol": mexc_sym},
            }))
            # Trade (deal) subscription
            await ws.send(orjson.dumps({
                "method": "sub.contract.deal",
                "param": {"symbol": mexc_sym},
            }))

    async def _authenticate(self, ws) -> None:
        """Authenticate for private channels."""
        timestamp = str(int(time.time() * 1000))
        sign_str = self.api_key + timestamp
        signature = hmac.new(
            self.api_secret.encode(),
            sign_str.encode(),
            hashlib.sha256,
        ).hexdigest()
        await ws.send(orjson.dumps({
            "method": "login",
            "param": {
                "apiKey": self.api_key,
                "reqTime": timestamp,
                "signature": signature,
            },
        }))

    async def _subscribe_private(self, ws) -> None:
        """Subscribe to private channels after auth."""
        for channel in ["sub.personal.order", "sub.personal.position",
                        "sub.personal.asset"]:
            await ws.send(orjson.dumps({"method": channel, "param": {}}))

    async def _ping_loop(self, ws) -> None:
        """Send ping every 15s to keep connection alive."""
        while self._running:
            try:
                await ws.send(orjson.dumps({"method": "ping"}))
            except Exception:
                break
            await asyncio.sleep(PING_INTERVAL_S)

    async def _listen(self, ws) -> None:
        """Listen for messages and dispatch to appropriate parser."""
        async for raw in ws:
            now = time_now_ms()
            self._last_msg_ts = now
            try:
                msg = orjson.loads(raw)
            except Exception:
                continue

            channel = msg.get("channel", "")

            if channel.startswith("push.contract.depth"):
                events = self._parse_depth(msg, now)
            elif channel.startswith("push.contract.deal"):
                events = self._parse_deals(msg, now)
            elif channel.startswith("push.personal.order"):
                events = self._parse_order(msg, now)
                self._enqueue_private(events)
                continue
            elif channel.startswith("push.personal.position"):
                events = self._parse_position(msg, now)
                self._enqueue_private(events)
                continue
            else:
                continue

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

    def _parse_depth(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse depth update from MEXC.

        Format: {channel: "push.contract.depth.BTC_USDT", data: {bids: [...], asks: [...]}}
        """
        events: List[WsEvent] = []
        data = msg.get("data", {})
        channel = msg.get("channel", "")

        # Extract symbol from channel name
        parts = channel.split(".")
        if len(parts) < 4:
            return events
        mexc_sym = parts[3]  # e.g. "BTC_USDT"
        symbol = from_mexc(mexc_sym)

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Convert to [[price, qty], ...] format
        book_data = {
            "b": [[str(b[0]), str(b[1])] for b in bids] if bids else [],
            "a": [[str(a[0]), str(a[1])] for a in asks] if asks else [],
        }

        if symbol not in self._first_book:
            evt_type = EventType.BOOK_SNAPSHOT
            self._first_book[symbol] = True
        else:
            evt_type = EventType.BOOK_DELTA

        events.append(WsEvent(
            event_type=evt_type,
            symbol=symbol,
            data=book_data,
            timestamp=now,
            raw=data,
        ))
        return events

    def _parse_deals(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse trade deals from MEXC.

        Format: {channel: "push.contract.deal.BTC_USDT", data: [{p: price, v: qty, T: 1/2, t: ts}, ...]}
        """
        events: List[WsEvent] = []
        data = msg.get("data", {})
        channel = msg.get("channel", "")

        parts = channel.split(".")
        if len(parts) < 4:
            return events
        mexc_sym = parts[3]
        symbol = from_mexc(mexc_sym)

        deals = data if isinstance(data, list) else [data]
        for deal in deals:
            # T: 1 = buy, 2 = sell
            side = Side.BUY if deal.get("T") == 1 else Side.SELL
            trade = Trade(
                trade_id=str(deal.get("t", now)),
                symbol=symbol,
                price=float(deal.get("p", 0)),
                quantity=float(deal.get("v", 0)),
                side=side,
                timestamp=int(deal.get("t", now)),
            )
            events.append(WsEvent(
                event_type=EventType.TRADE,
                symbol=symbol,
                data=trade,
                timestamp=trade.timestamp,
            ))
        return events

    def _parse_order(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse private order update."""
        events: List[WsEvent] = []
        data = msg.get("data", {})
        if not isinstance(data, dict):
            return events

        mexc_sym = data.get("symbol", "")
        symbol = from_mexc(mexc_sym)
        order_id = str(data.get("orderId", ""))

        # MEXC state: 1=uninformed, 2=uncompleted, 3=completed, 4=cancelled, 5=invalid
        state = data.get("state", 0)
        status_map = {1: "New", 2: "PartiallyFilled", 3: "Filled", 4: "Cancelled", 5: "Rejected"}
        status = status_map.get(state, "New")

        # MEXC side: 1=open_long, 2=close_long, 3=open_short, 4=close_short
        mexc_side = data.get("side", 0)
        side_str = "Buy" if mexc_side in (1, 4) else "Sell"

        filled_qty = float(data.get("dealVol", 0))
        avg_price = float(data.get("dealAvgPrice", 0))

        # Emit execution if filled
        if filled_qty > 0:
            events.append(WsEvent(
                event_type=EventType.EXECUTION,
                symbol=symbol,
                data=NormalizedExecution(
                    exec_id=order_id,
                    order_id=order_id,
                    symbol=symbol,
                    side=Side.BUY if side_str == "Buy" else Side.SELL,
                    price=avg_price,
                    quantity=filled_qty,
                    is_maker=data.get("makerFee", 0) > 0,
                    order_type=str(data.get("orderType", "")),
                ),
                timestamp=now,
            ))

        events.append(WsEvent(
            event_type=EventType.ORDER,
            symbol=symbol,
            data=NormalizedOrderUpdate(
                order_id=order_id,
                symbol=symbol,
                status=status,
                avg_price=avg_price,
                filled_qty=filled_qty,
                reduce_only=mexc_side in (2, 4),  # close sides
                stop_order_type="",
                side=side_str,
            ),
            timestamp=now,
        ))
        return events

    def _parse_position(self, msg: dict, now: int) -> List[WsEvent]:
        """Parse private position update."""
        events: List[WsEvent] = []
        data = msg.get("data", {})
        if not isinstance(data, dict):
            return events

        mexc_sym = data.get("symbol", "")
        symbol = from_mexc(mexc_sym)

        hold_vol = float(data.get("holdVol", 0))
        # MEXC position_type: 1=long, 2=short
        pos_type = data.get("positionType", 0)
        side = "Buy" if pos_type == 1 else "Sell" if pos_type == 2 else ""
        entry = float(data.get("openAvgPrice", 0))

        events.append(WsEvent(
            event_type=EventType.POSITION,
            symbol=symbol,
            data=NormalizedPositionUpdate(
                symbol=symbol,
                size=hold_vol,
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
