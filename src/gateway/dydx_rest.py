"""dYdX v4 (Cosmos-based) REST adapter.

Hybrid architecture:
- Public data (klines, instruments, orderbook): Indexer REST API (no auth)
- Trading (place/cancel orders): dydx-v4-client SDK (optional dependency)

The dYdX SDK is imported lazily and guarded with ImportError so that MEXC
and other exchanges work without it installed.

Symbol format: BTC-USD (hyphen separator, USD not USDT).
Denomination: USD (treated as equivalent to USDT at 1:1 peg).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Dict, List

import aiohttp

from core.types import Side, OrderType, TimeInForce
from gateway.base import (
    ExchangeREST, OrderResponse, WalletInfo, InstrumentSpec, TickerInfo,
)
from gateway.symbol_map import to_dydx, from_dydx

log = logging.getLogger(__name__)

INDEXER_URL = "https://indexer.dydx.trade"
INDEXER_URL_TESTNET = "https://indexer.v4testnet.dydx.exchange"

# dYdX candle resolution map
_RESOLUTION_MAP = {
    "1": "1MIN", "3": "5MINS", "5": "5MINS", "15": "15MINS",
    "30": "30MINS", "60": "1HOUR",
    "1m": "1MIN", "3m": "5MINS", "5m": "5MINS", "15m": "15MINS",
    "30m": "30MINS", "60m": "1HOUR", "1h": "1HOUR",
}


def _fmt(value: float) -> str:
    """Format a float: no scientific notation, no trailing zeros."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


class DydxREST(ExchangeREST):
    """Async REST client for dYdX v4.

    Args:
        api_key: dYdX address (cosmos address, e.g. dydx1...)
        api_secret: Mnemonic phrase for signing transactions
        testnet: Use testnet indexer/validator
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.address = api_key  # dYdX uses address instead of API key
        self.mnemonic = api_secret  # mnemonic for transaction signing
        self.testnet = testnet
        self.indexer_url = INDEXER_URL_TESTNET if testnet else INDEXER_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._composite_client = None
        self._tp_sl_orders: Dict[str, Dict[str, str]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_composite_client(self):
        """Lazily initialize dYdX CompositeClient for trading operations."""
        if self._composite_client is not None:
            return self._composite_client

        try:
            from dydx_v4_client import CompositeClient, Network
            network = Network.testnet() if self.testnet else Network.mainnet()
            self._composite_client = CompositeClient(network, self.mnemonic)
            return self._composite_client
        except ImportError:
            log.error("dydx-v4-client not installed. Install with: pip install dydx-v4-client")
            return None

    # --- HTTP helpers (Indexer — public, no auth) ---

    async def _indexer_get(self, path: str, params: dict = None) -> dict:
        session = await self._get_session()
        url = f"{self.indexer_url}{path}"
        async with session.get(url, params=params or {}) as resp:
            data = await resp.json(content_type=None)
            return data if isinstance(data, dict) else {"data": data}

    def _to_order_response(self, data: dict) -> OrderResponse:
        if not isinstance(data, dict):
            return OrderResponse(success=False, error_msg=str(data))

        if "error" in data or "errors" in data:
            msg = data.get("error", data.get("errors", ""))
            return OrderResponse(
                success=False, order_id="",
                error_code=-1, error_msg=str(msg), raw=data,
            )
        order_id = str(data.get("id", data.get("orderId", "")))
        return OrderResponse(
            success=True, order_id=order_id,
            error_code=0, error_msg="", raw=data,
        )

    # --- Order Management ---

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
    ) -> OrderResponse:
        client = self._get_composite_client()
        if client is None:
            return OrderResponse(
                success=False, error_msg="dydx-v4-client not installed",
            )

        try:
            dydx_sym = to_dydx(symbol)
            dydx_side = "BUY" if side == Side.BUY else "SELL"

            if order_type == OrderType.MARKET or price is None:
                # Short-term market order
                result = await asyncio.to_thread(
                    client.place_short_term_order,
                    dydx_sym, dydx_side, float(_fmt(qty)), 0,  # price=0 for market
                    reduce_only=reduce_only,
                )
            else:
                result = await asyncio.to_thread(
                    client.place_short_term_order,
                    dydx_sym, dydx_side, float(_fmt(qty)), float(_fmt(price)),
                    reduce_only=reduce_only,
                )

            resp = self._to_order_response(result if isinstance(result, dict) else {"id": str(result)})
            if resp.success:
                log.info("Order placed: %s %s %s qty=%s px=%s -> %s",
                         side.name, symbol, order_type.value, qty, price, resp.order_id)
            return resp
        except Exception as e:
            log.error("dYdX place_order failed: %s", e)
            return OrderResponse(success=False, error_msg=str(e))

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        client = self._get_composite_client()
        if client is None:
            return OrderResponse(success=False, error_msg="dydx-v4-client not installed")
        try:
            result = await asyncio.to_thread(client.cancel_order, order_id)
            return self._to_order_response(result if isinstance(result, dict) else {})
        except Exception as e:
            log.error("dYdX cancel_order failed: %s", e)
            return OrderResponse(success=False, error_msg=str(e))

    async def cancel_all_orders(self, symbol: str) -> OrderResponse:
        # dYdX doesn't have a cancel-all endpoint; cancel individually
        orders = await self.get_open_orders(symbol)
        for order in orders:
            oid = order.get("id", "")
            if oid:
                await self.cancel_order(symbol, oid)
        return OrderResponse(success=True)

    async def amend_order(self, symbol: str, order_id: str,
                          qty: Optional[float] = None,
                          price: Optional[float] = None) -> OrderResponse:
        # dYdX doesn't support amendment — cancel and re-place
        cancel_resp = await self.cancel_order(symbol, order_id)
        if not cancel_resp.success:
            return cancel_resp
        return OrderResponse(
            success=True, order_id="",
            error_msg="Cancelled for amend — re-place required",
        )

    # --- TP/SL Emulation ---

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: int = 0,
        position_side: Optional[Side] = None,
    ) -> OrderResponse:
        """Emulate TP/SL with conditional orders via SDK."""
        existing = self._tp_sl_orders.get(symbol, {})

        if take_profit is not None and existing.get("tp"):
            await self.cancel_order(symbol, existing["tp"])
        if stop_loss is not None and existing.get("sl"):
            await self.cancel_order(symbol, existing["sl"])

        bracket = self._tp_sl_orders.setdefault(symbol, {})
        last_resp = OrderResponse(success=True)

        if position_side is not None:
            close_side = Side.SELL if position_side == Side.BUY else Side.BUY
        elif take_profit is not None and stop_loss is not None:
            close_side = Side.SELL if take_profit > stop_loss else Side.BUY
        else:
            close_side = Side.SELL

        # For dYdX, we place conditional orders through the SDK
        client = self._get_composite_client()
        if client is None:
            return OrderResponse(success=False, error_msg="dydx-v4-client not installed")

        dydx_sym = to_dydx(symbol)
        dydx_close = "SELL" if close_side == Side.BUY else "BUY"

        try:
            if take_profit is not None:
                result = await asyncio.to_thread(
                    client.place_short_term_order,
                    dydx_sym, dydx_close, 0, float(_fmt(take_profit)),
                    reduce_only=True,
                )
                resp = self._to_order_response(result if isinstance(result, dict) else {"id": str(result)})
                if resp.success:
                    bracket["tp"] = resp.order_id
                last_resp = resp

            if stop_loss is not None:
                result = await asyncio.to_thread(
                    client.place_short_term_order,
                    dydx_sym, dydx_close, 0, float(_fmt(stop_loss)),
                    reduce_only=True,
                )
                resp = self._to_order_response(result if isinstance(result, dict) else {"id": str(result)})
                if resp.success:
                    bracket["sl"] = resp.order_id
                last_resp = resp
        except Exception as e:
            log.error("dYdX set_trading_stop failed: %s", e)
            return OrderResponse(success=False, error_msg=str(e))

        return last_resp

    async def cleanup_bracket(self, symbol: str, order_id: str) -> None:
        bracket = self._tp_sl_orders.get(symbol, {})
        if not bracket:
            return

        if bracket.get("tp") == order_id:
            sl_id = bracket.get("sl", "")
            if sl_id:
                try:
                    await self.cancel_order(symbol, sl_id)
                except Exception as e:
                    log.warning("Failed to cancel paired SL %s: %s", sl_id, e)
            self._tp_sl_orders.pop(symbol, None)
        elif bracket.get("sl") == order_id:
            tp_id = bracket.get("tp", "")
            if tp_id:
                try:
                    await self.cancel_order(symbol, tp_id)
                except Exception as e:
                    log.warning("Failed to cancel paired TP %s: %s", tp_id, e)
            self._tp_sl_orders.pop(symbol, None)

    async def set_leverage(self, symbol: str, leverage: int) -> OrderResponse:
        # dYdX handles leverage via margin — no explicit set_leverage call
        log.debug("dYdX: set_leverage is a no-op (margin-based leverage)")
        return OrderResponse(success=True)

    # --- Account Info (Indexer REST) ---

    async def get_wallet_balance(self) -> WalletInfo:
        if not self.address:
            return WalletInfo(equity=0.0)
        data = await self._indexer_get(
            f"/v4/addresses/{self.address}/subaccountNumber/0",
        )
        equity = float(data.get("equity", data.get("freeCollateral", 0)))
        return WalletInfo(equity=equity, raw=data)

    async def get_positions(self, symbol: str = "") -> list:
        if not self.address:
            return []
        data = await self._indexer_get(
            f"/v4/addresses/{self.address}/subaccountNumber/0/perpetualPositions",
        )
        positions = data.get("positions", [])
        if symbol:
            dydx_sym = to_dydx(symbol)
            positions = [p for p in positions if p.get("market") == dydx_sym]
        return positions

    async def get_open_orders(self, symbol: str = "") -> list:
        if not self.address:
            return []
        params: Dict[str, str] = {}
        if symbol:
            params["ticker"] = to_dydx(symbol)
        data = await self._indexer_get(
            f"/v4/addresses/{self.address}/subaccountNumber/0/orders",
            params,
        )
        return data if isinstance(data, list) else data.get("orders", [])

    async def get_instruments(self, symbol: str = "") -> List[InstrumentSpec]:
        data = await self._indexer_get("/v4/perpetualMarkets")
        specs: List[InstrumentSpec] = []
        markets = data.get("markets", {})
        for ticker, info in markets.items():
            sym = from_dydx(ticker)
            if symbol and sym != symbol:
                continue

            tick_size = float(info.get("tickSize", 0.01))
            step_size = float(info.get("stepSize", 0.001))
            min_order_size = float(info.get("minOrderSize", step_size))

            specs.append(InstrumentSpec(
                symbol=sym,
                tick_size=tick_size,
                min_qty=min_order_size,
                qty_step=step_size,
                min_notional=5.0,  # dYdX doesn't expose this directly
            ))
        return specs

    async def get_tickers(self) -> List[TickerInfo]:
        data = await self._indexer_get("/v4/perpetualMarkets")
        tickers: List[TickerInfo] = []
        markets = data.get("markets", {})
        for ticker, info in markets.items():
            sym = from_dydx(ticker)
            oracle_price = float(info.get("oraclePrice", 0))
            volume_24h = float(info.get("volume24H", 0))
            price_change = float(info.get("priceChange24H", 0))
            pct = price_change / oracle_price if oracle_price > 0 else 0
            tickers.append(TickerInfo(
                symbol=sym,
                last_price=oracle_price,
                turnover_24h=volume_24h,
                price_change_pct=pct,
            ))
        return tickers

    async def get_server_time(self) -> int:
        return int(time.time() * 1000)

    async def get_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> List[dict]:
        """Download kline data from dYdX Indexer.

        Endpoint: GET /v4/candles/perpetualMarkets/{ticker}
        """
        dydx_sym = to_dydx(symbol)
        resolution = _RESOLUTION_MAP.get(interval, "5MINS")

        all_klines: List[dict] = []
        cursor_end = end_ms

        while True:
            from datetime import datetime, timezone
            start_iso = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(cursor_end / 1000, tz=timezone.utc).isoformat()

            params = {
                "resolution": resolution,
                "fromISO": start_iso,
                "toISO": end_iso,
                "limit": "100",
            }
            data = await self._indexer_get(
                f"/v4/candles/perpetualMarkets/{dydx_sym}", params,
            )

            candles = data.get("candles", [])
            if not candles:
                break

            for c in candles:
                # dYdX returns ISO timestamps
                started = c.get("startedAt", "")
                if started:
                    from datetime import datetime as dt
                    ts_dt = dt.fromisoformat(started.replace("Z", "+00:00"))
                    ts = int(ts_dt.timestamp() * 1000)
                else:
                    continue

                all_klines.append({
                    "ts": ts,
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                    "volume": float(c.get("baseTokenVolume", 0)),
                    "turnover": float(c.get("usdVolume", 0)),
                })

            # Paginate backward: dYdX returns newest first
            oldest_ts = min(k["ts"] for k in all_klines) if all_klines else start_ms
            if oldest_ts <= start_ms:
                break
            cursor_end = oldest_ts
            await asyncio.sleep(0.1)  # rate limit

        # Deduplicate and sort ascending
        seen: set = set()
        unique: List[dict] = []
        for k in all_klines:
            if k["ts"] not in seen:
                seen.add(k["ts"])
                unique.append(k)
        unique.sort(key=lambda x: x["ts"])
        return unique

    async def get_orderbook(self, symbol: str, depth: int = 50) -> dict:
        dydx_sym = to_dydx(symbol)
        data = await self._indexer_get(
            f"/v4/orderbooks/perpetualMarkets/{dydx_sym}",
        )
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        return {
            "b": [[b.get("price", "0"), b.get("size", "0")] for b in bids[:depth]],
            "a": [[a.get("price", "0"), a.get("size", "0")] for a in asks[:depth]],
            "seq": 0,
        }

    async def withdraw(self, coin: str, chain: str, address: str,
                       amount: float) -> dict:
        # dYdX withdrawals are Cosmos chain transfers via SDK
        return {"success": False, "tx_id": "", "error_msg": "Use dYdX bridge for withdrawals"}

    async def get_deposit_address(self, coin: str, chain: str) -> dict:
        return {"address": self.address, "chain": "dydx", "tag": ""}
