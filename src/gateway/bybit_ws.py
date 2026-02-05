"""Bybit WebSocket client with auto-reconnection, heartbeat, and message routing.

Handles both public (orderbook, trades) and private (orders, positions, executions)
WebSocket streams using Bybit V5 API.
"""
from __future__ import annotations
import asyncio
import hmac
import hashlib
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

log = logging.getLogger(__name__)

PUBLIC_URL = "wss://stream.bybit.com/v5/public/linear"
PUBLIC_URL_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"
PRIVATE_URL = "wss://stream.bybit.com/v5/private"
PRIVATE_URL_TESTNET = "wss://stream-testnet.bybit.com/v5/private"


class BybitWS(ExchangeWS):
    """Async WebSocket client for Bybit V5."""

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
        self.ping_interval = ws_cfg.get("ping_interval_s", 20)
        self.ping_timeout = ws_cfg.get("ping_timeout_s", 10)
        self.reconnect_delay = ws_cfg.get("reconnect_delay_s", 1)
        self.max_reconnect_delay = ws_cfg.get("max_reconnect_delay_s", 30)
        self.book_depth = ws_cfg.get("orderbook_depth", 50)

        self._pub_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._prv_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._event_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=10000)
        self._private_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=1000)
        self._last_pub_msg_ts: int = 0
        self._last_prv_msg_ts: int = 0

    @property
    def pub_url(self) -> str:
        return PUBLIC_URL_TESTNET if self.testnet else PUBLIC_URL

    @property
    def prv_url(self) -> str:
        return PRIVATE_URL_TESTNET if self.testnet else PRIVATE_URL

    # --- Connection Lifecycle ---

    async def start(self) -> None:
        self._running = True
        tasks = [asyncio.create_task(self._run_public())]
        if self.api_key and self.api_secret:
            tasks.append(asyncio.create_task(self._run_private()))
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False
        for ws in (self._pub_ws, self._prv_ws):
            if ws:
                await ws.close()

    async def _run_public(self) -> None:
        delay = self.reconnect_delay
        while self._running:
            try:
                async with websockets.connect(
                    self.pub_url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._pub_ws = ws
                    delay = self.reconnect_delay
                    log.info("Public WS connected: %s", self.pub_url)
                    await self._subscribe_public(ws)
                    await self._listen(ws, is_private=False)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("Public WS disconnected: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _run_private(self) -> None:
        delay = self.reconnect_delay
        while self._running:
            try:
                async with websockets.connect(
                    self.prv_url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                ) as ws:
                    self._prv_ws = ws
                    delay = self.reconnect_delay
                    log.info("Private WS connected: %s", self.prv_url)
                    await self._authenticate(ws)
                    await self._subscribe_private(ws)
                    await self._listen(ws, is_private=True)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("Private WS disconnected: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    # --- Authentication ---

    async def _authenticate(self, ws) -> None:
        expires = int(time.time() * 1000) + 10000
        signature = hmac.new(
            self.api_secret.encode(),
            f"GET/realtime{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()
        auth_msg = {
            "op": "auth",
            "args": [self.api_key, expires, signature],
        }
        await ws.send(orjson.dumps(auth_msg).decode())
        resp = await asyncio.wait_for(ws.recv(), timeout=5)
        data = orjson.loads(resp)
        if data.get("success"):
            log.info("Private WS authenticated successfully")
        else:
            log.error("Private WS auth failed: %s", data)

    # --- Subscriptions ---

    async def _subscribe_public(self, ws) -> None:
        args = []
        for sym in self.symbols:
            args.append(f"orderbook.{self.book_depth}.{sym}")
            args.append(f"publicTrade.{sym}")
        msg = {"op": "subscribe", "args": args}
        await ws.send(orjson.dumps(msg).decode())
        log.info("Public subscribed: %s", args)

    async def _subscribe_private(self, ws) -> None:
        args = ["order", "execution", "position"]
        msg = {"op": "subscribe", "args": args}
        await ws.send(orjson.dumps(msg).decode())
        log.info("Private subscribed: %s", args)

    # --- Message Loop ---

    async def _listen(self, ws, is_private: bool) -> None:
        async for raw in ws:
            now = time_now_ms()
            try:
                data = orjson.loads(raw)
            except Exception:
                continue

            if "op" in data:
                continue

            topic = data.get("topic", "")
            msg_data = data.get("data", {})
            msg_type = data.get("type", "")

            if is_private:
                self._last_prv_msg_ts = now
                events = self._parse_private(topic, msg_data, now)
            else:
                self._last_pub_msg_ts = now
                events = self._parse_public(topic, msg_type, msg_data, data, now)

            for evt in events:
                if is_private:
                    try:
                        self._private_queue.put_nowait(evt)
                    except asyncio.QueueFull:
                        self._private_queue.get_nowait()
                        self._private_queue.put_nowait(evt)
                else:
                    try:
                        self._event_queue.put_nowait(evt)
                    except asyncio.QueueFull:
                        self._event_queue.get_nowait()
                        self._event_queue.put_nowait(evt)

    # --- Parsers ---

    def _parse_public(self, topic: str, msg_type: str, data, raw_msg: dict,
                      now: int) -> List[WsEvent]:
        events: List[WsEvent] = []
        parts = topic.split(".")
        if len(parts) < 2:
            return events

        channel = parts[0]
        symbol = parts[-1]

        if channel == "orderbook":
            evt_type = EventType.BOOK_SNAPSHOT if msg_type == "snapshot" else EventType.BOOK_DELTA
            events.append(WsEvent(
                event_type=evt_type,
                symbol=symbol,
                data=data,
                timestamp=now,
                raw=raw_msg,
            ))

        elif channel == "publicTrade":
            trades_list = data if isinstance(data, list) else [data]
            for td in trades_list:
                trade = Trade(
                    trade_id=str(td.get("i", "")),
                    symbol=symbol,
                    price=float(td.get("p", 0)),
                    quantity=float(td.get("v", 0)),
                    side=Side.BUY if td.get("S") == "Buy" else Side.SELL,
                    timestamp=int(td.get("T", now)),
                )
                events.append(WsEvent(
                    event_type=EventType.TRADE,
                    symbol=symbol,
                    data=trade,
                    timestamp=trade.timestamp,
                ))

        return events

    def _parse_private(self, topic: str, data, now: int) -> List[WsEvent]:
        events: List[WsEvent] = []
        items = data if isinstance(data, list) else [data]

        for item in items:
            if topic == "execution":
                events.append(WsEvent(
                    event_type=EventType.EXECUTION,
                    symbol=item.get("symbol", ""),
                    data=NormalizedExecution(
                        exec_id=item.get("execId", ""),
                        order_id=item.get("orderId", ""),
                        symbol=item.get("symbol", ""),
                        side=Side.BUY if item.get("side") == "Buy" else Side.SELL,
                        price=float(item.get("execPrice", 0)),
                        quantity=float(item.get("execQty", 0)),
                        is_maker=item.get("isMaker", False),
                        order_type=item.get("orderType", ""),
                    ),
                    timestamp=now,
                ))
            elif topic == "order":
                events.append(WsEvent(
                    event_type=EventType.ORDER,
                    symbol=item.get("symbol", ""),
                    data=NormalizedOrderUpdate(
                        order_id=item.get("orderId", ""),
                        symbol=item.get("symbol", ""),
                        status=item.get("orderStatus", ""),
                        avg_price=float(item.get("avgPrice", 0)),
                        filled_qty=float(item.get("cumExecQty", 0)),
                        reduce_only=item.get("reduceOnly", False),
                        stop_order_type=item.get("stopOrderType", ""),
                        side=item.get("side", ""),
                    ),
                    timestamp=now,
                ))
            elif topic == "position":
                events.append(WsEvent(
                    event_type=EventType.POSITION,
                    symbol=item.get("symbol", ""),
                    data=NormalizedPositionUpdate(
                        symbol=item.get("symbol", ""),
                        size=float(item.get("size", 0)),
                        entry_price=float(item.get("entryPrice", 0)),
                        side=item.get("side", ""),
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
        pub_age = now - self._last_pub_msg_ts if self._last_pub_msg_ts > 0 else 999
        return min(pub_age, 999)

    @property
    def is_connected(self) -> bool:
        return self._pub_ws is not None and self._pub_ws.open
