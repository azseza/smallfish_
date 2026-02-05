"""Binance USDT-M Futures WebSocket client.

Handles both public (depth, aggTrade) and private (listenKey) streams.
Key differences from Bybit:
- Public: combined stream URL with lowercase symbols
- Book: depth20@100ms full snapshots (no incremental deltas)
- Trades: aggTrade (aggregated), side inversion (m=true → SELL)
- Private: listenKey-based auth, refreshed every 30 min via REST
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional, List, Callable, Awaitable

import aiohttp
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

PUBLIC_URL = "wss://fstream.binance.com/stream"
PUBLIC_URL_TESTNET = "wss://fstream.binancefuture.com/stream"
PRIVATE_URL = "wss://fstream.binance.com/ws/"
PRIVATE_URL_TESTNET = "wss://fstream.binancefuture.com/ws/"
REST_URL = "https://fapi.binance.com"
REST_URL_TESTNET = "https://demo-fapi.binance.com"

LISTEN_KEY_REFRESH_S = 30 * 60  # 30 minutes


# --- Status mapping ---

_STATUS_MAP = {
    "NEW": "New",
    "PARTIALLY_FILLED": "PartiallyFilled",
    "FILLED": "Filled",
    "CANCELED": "Cancelled",
    "CANCELLED": "Cancelled",
    "REJECTED": "Rejected",
    "EXPIRED": "Cancelled",
    "EXPIRED_IN_MATCH": "Cancelled",
}

_STOP_TYPE_MAP = {
    "TAKE_PROFIT_MARKET": "TakeProfit",
    "TAKE_PROFIT": "TakeProfit",
    "STOP_MARKET": "StopLoss",
    "STOP": "StopLoss",
}


class BinanceWS(ExchangeWS):
    """Async WebSocket client for Binance USDT-M Futures."""

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

        self._pub_ws = None
        self._prv_ws = None
        self._running = False
        self._event_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=10000)
        self._private_queue: asyncio.Queue[WsEvent] = asyncio.Queue(maxsize=1000)
        self._last_pub_msg_ts: int = 0
        self._last_prv_msg_ts: int = 0
        self._listen_key: str = ""
        self._first_book: dict[str, bool] = {}  # symbol → True after first msg

    @property
    def _rest_base(self) -> str:
        return REST_URL_TESTNET if self.testnet else REST_URL

    @property
    def _pub_base(self) -> str:
        return PUBLIC_URL_TESTNET if self.testnet else PUBLIC_URL

    @property
    def _prv_base(self) -> str:
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

    # --- Public Stream ---

    def _build_public_url(self) -> str:
        streams = []
        for sym in self.symbols:
            s = sym.lower()
            streams.append(f"{s}@depth20@100ms")
            streams.append(f"{s}@aggTrade")
        return f"{self._pub_base}?streams={'/'.join(streams)}"

    async def _run_public(self) -> None:
        delay = self.reconnect_delay
        while self._running:
            try:
                url = self._build_public_url()
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._pub_ws = ws
                    delay = self.reconnect_delay
                    log.info("Binance public WS connected")
                    self._first_book = {}
                    await self._listen_public(ws)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("Binance public WS disconnected: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _listen_public(self, ws) -> None:
        async for raw in ws:
            now = time_now_ms()
            self._last_pub_msg_ts = now
            try:
                msg = orjson.loads(raw)
            except Exception:
                continue

            stream = msg.get("stream", "")
            data = msg.get("data", {})
            if not stream:
                continue

            events = self._parse_public(stream, data, now)
            for evt in events:
                try:
                    self._event_queue.put_nowait(evt)
                except asyncio.QueueFull:
                    self._event_queue.get_nowait()
                    self._event_queue.put_nowait(evt)

    def _parse_public(self, stream: str, data: dict, now: int) -> List[WsEvent]:
        events: List[WsEvent] = []

        # stream format: "btcusdt@depth20@100ms" or "btcusdt@aggTrade"
        parts = stream.split("@")
        if len(parts) < 2:
            return events
        symbol = parts[0].upper()

        if "depth" in parts[1]:
            # Depth snapshot — first one = BOOK_SNAPSHOT, subsequent = BOOK_DELTA
            # so that app.py handles them identically to Bybit
            if symbol not in self._first_book:
                evt_type = EventType.BOOK_SNAPSHOT
                self._first_book[symbol] = True
            else:
                evt_type = EventType.BOOK_DELTA

            # Convert Binance format to Bybit-like format
            bids = data.get("b", [])  # [[price, qty], ...]
            asks = data.get("a", [])

            book_data = {"b": bids, "a": asks}
            events.append(WsEvent(
                event_type=evt_type,
                symbol=symbol,
                data=book_data,
                timestamp=now,
                raw=data,
            ))

        elif parts[1] == "aggTrade":
            # aggTrade: m=true means buyer is maker → trade was a SELL
            trade = Trade(
                trade_id=str(data.get("a", "")),
                symbol=symbol,
                price=float(data.get("p", 0)),
                quantity=float(data.get("q", 0)),
                side=Side.SELL if data.get("m", False) else Side.BUY,
                timestamp=int(data.get("T", now)),
            )
            events.append(WsEvent(
                event_type=EventType.TRADE,
                symbol=symbol,
                data=trade,
                timestamp=trade.timestamp,
            ))

        return events

    # --- Private Stream ---

    async def _get_listen_key(self) -> str:
        """Create a listenKey via REST."""
        async with aiohttp.ClientSession() as session:
            url = f"{self._rest_base}/fapi/v1/listenKey"
            headers = {"X-MBX-APIKEY": self.api_key}
            async with session.post(url, headers=headers) as resp:
                data = await resp.json(content_type=None)
                return data.get("listenKey", "")

    async def _refresh_listen_key(self) -> None:
        """Keep listenKey alive via PUT."""
        async with aiohttp.ClientSession() as session:
            url = f"{self._rest_base}/fapi/v1/listenKey"
            headers = {"X-MBX-APIKEY": self.api_key}
            async with session.put(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Failed to refresh listenKey: %s", await resp.text())

    async def _run_private(self) -> None:
        delay = self.reconnect_delay
        while self._running:
            try:
                self._listen_key = await self._get_listen_key()
                if not self._listen_key:
                    log.error("Failed to get listenKey")
                    await asyncio.sleep(delay)
                    continue

                url = f"{self._prv_base}{self._listen_key}"
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._prv_ws = ws
                    delay = self.reconnect_delay
                    log.info("Binance private WS connected (listenKey)")

                    # Run listener and key refresh concurrently
                    listener = asyncio.create_task(self._listen_private(ws))
                    refresher = asyncio.create_task(self._key_refresh_loop())
                    done, pending = await asyncio.wait(
                        [listener, refresher],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                log.warning("Binance private WS disconnected: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _key_refresh_loop(self) -> None:
        """Refresh listenKey every 30 minutes."""
        while self._running:
            await asyncio.sleep(LISTEN_KEY_REFRESH_S)
            try:
                await self._refresh_listen_key()
                log.debug("listenKey refreshed")
            except Exception as e:
                log.warning("listenKey refresh failed: %s", e)

    async def _listen_private(self, ws) -> None:
        async for raw in ws:
            now = time_now_ms()
            self._last_prv_msg_ts = now
            try:
                data = orjson.loads(raw)
            except Exception:
                continue

            events = self._parse_private(data, now)
            for evt in events:
                try:
                    self._private_queue.put_nowait(evt)
                except asyncio.QueueFull:
                    self._private_queue.get_nowait()
                    self._private_queue.put_nowait(evt)

    def _parse_private(self, data: dict, now: int) -> List[WsEvent]:
        events: List[WsEvent] = []
        event_type = data.get("e", "")

        if event_type == "ORDER_TRADE_UPDATE":
            o = data.get("o", {})
            symbol = o.get("s", "")
            side_str = o.get("S", "")
            status = _STATUS_MAP.get(o.get("X", ""), o.get("X", ""))
            order_type = o.get("ot", o.get("o", ""))
            stop_type = _STOP_TYPE_MAP.get(order_type, "")

            # Emit execution if there was a fill
            last_qty = float(o.get("l", 0))
            if last_qty > 0:
                events.append(WsEvent(
                    event_type=EventType.EXECUTION,
                    symbol=symbol,
                    data=NormalizedExecution(
                        exec_id=str(o.get("t", "")),
                        order_id=str(o.get("i", "")),
                        symbol=symbol,
                        side=Side.BUY if side_str == "BUY" else Side.SELL,
                        price=float(o.get("L", 0)),   # last filled price
                        quantity=last_qty,
                        is_maker=o.get("m", False),
                        order_type=order_type,
                    ),
                    timestamp=now,
                ))

            # Always emit order update
            events.append(WsEvent(
                event_type=EventType.ORDER,
                symbol=symbol,
                data=NormalizedOrderUpdate(
                    order_id=str(o.get("i", "")),
                    symbol=symbol,
                    status=status,
                    avg_price=float(o.get("ap", 0)),
                    filled_qty=float(o.get("z", 0)),
                    reduce_only=o.get("R", False),
                    stop_order_type=stop_type,
                    side=side_str.capitalize() if side_str else "",
                ),
                timestamp=now,
            ))

        elif event_type == "ACCOUNT_UPDATE":
            positions = data.get("a", {}).get("P", [])
            for p in positions:
                symbol = p.get("s", "")
                size = float(p.get("pa", 0))
                entry = float(p.get("ep", 0))
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
        pub_age = now - self._last_pub_msg_ts if self._last_pub_msg_ts > 0 else 999
        return min(pub_age, 999)

    @property
    def is_connected(self) -> bool:
        return self._pub_ws is not None and self._pub_ws.open
