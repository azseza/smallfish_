"""Bybit V5 REST API client with HMAC-SHA256 authentication.

Used for order placement, cancellation, position queries, and account info.
All methods are async for non-blocking operation in the event loop.
"""
from __future__ import annotations
import hashlib
import hmac
import logging
import time
from typing import Optional, Dict, Any

import aiohttp
import orjson

from core.types import Side, OrderType, TimeInForce
from core.utils import time_now_ms

log = logging.getLogger(__name__)

BASE_URL = "https://api.bybit.com"
BASE_URL_TESTNET = "https://api-testnet.bybit.com"
RECV_WINDOW = "5000"


class BybitREST:
    """Async REST client for Bybit V5 Unified Trading API."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BASE_URL_TESTNET if testnet else BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                json_serialize=lambda x: orjson.dumps(x).decode(),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Authentication ---

    def _sign(self, timestamp: str, payload: str) -> str:
        param_str = f"{timestamp}{self.api_key}{RECV_WINDOW}{payload}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, payload: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        sign = self._sign(timestamp, payload)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "Content-Type": "application/json",
        }

    # --- HTTP helpers ---

    async def _post(self, path: str, payload: dict) -> dict:
        session = await self._get_session()
        body = orjson.dumps(payload).decode()
        headers = self._auth_headers(body)
        url = f"{self.base_url}{path}"
        async with session.post(url, headers=headers, data=body) as resp:
            data = await resp.json(content_type=None)
            if data.get("retCode", -1) != 0:
                log.error("REST POST %s failed: %s", path, data)
            return data

    async def _get(self, path: str, params: dict = None) -> dict:
        session = await self._get_session()
        # Build query string for signing
        params = params or {}
        query_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        headers = self._auth_headers(query_str)
        url = f"{self.base_url}{path}"
        async with session.get(url, headers=headers, params=params) as resp:
            data = await resp.json(content_type=None)
            if data.get("retCode", -1) != 0:
                log.error("REST GET %s failed: %s", path, data)
            return data

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
    ) -> dict:
        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side == Side.BUY else "Sell",
            "orderType": order_type.value,
            "qty": str(qty),
            "timeInForce": time_in_force.value,
        }
        if price is not None:
            payload["price"] = str(price)
        if reduce_only:
            payload["reduceOnly"] = True
        if close_on_trigger:
            payload["closeOnTrigger"] = True
        if order_link_id:
            payload["orderLinkId"] = order_link_id

        result = await self._post("/v5/order/create", payload)
        order_id = result.get("result", {}).get("orderId", "")
        if order_id:
            log.info("Order placed: %s %s %s qty=%s px=%s → %s",
                     side.name, symbol, order_type.value, qty, price, order_id)
        return result

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        payload = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        result = await self._post("/v5/order/cancel", payload)
        log.info("Order cancelled: %s", order_id)
        return result

    async def cancel_all_orders(self, symbol: str) -> dict:
        payload = {
            "category": "linear",
            "symbol": symbol,
        }
        return await self._post("/v5/order/cancel-all", payload)

    async def amend_order(self, symbol: str, order_id: str,
                          qty: Optional[float] = None,
                          price: Optional[float] = None) -> dict:
        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        if qty is not None:
            payload["qty"] = str(qty)
        if price is not None:
            payload["price"] = str(price)
        return await self._post("/v5/order/amend", payload)

    # --- Position Management ---

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: int = 0,
    ) -> dict:
        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx,
        }
        if take_profit is not None:
            payload["takeProfit"] = str(take_profit)
        if stop_loss is not None:
            payload["stopLoss"] = str(stop_loss)
        if trailing_stop is not None:
            payload["trailingStop"] = str(trailing_stop)
        return await self._post("/v5/position/trading-stop", payload)

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        payload = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        return await self._post("/v5/position/set-leverage", payload)

    # --- Account Info ---

    async def get_wallet_balance(self) -> dict:
        return await self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})

    async def get_positions(self, symbol: str = "") -> dict:
        params: Dict[str, str] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/v5/position/list", params)

    async def get_open_orders(self, symbol: str = "") -> dict:
        params: Dict[str, str] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/v5/order/realtime", params)

    async def get_instruments(self, symbol: str = "") -> dict:
        params: Dict[str, str] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/v5/market/instruments-info", params)

    async def get_server_time(self) -> int:
        """Returns server time in ms for latency estimation."""
        session = await self._get_session()
        url = f"{self.base_url}/v5/market/time"
        async with session.get(url) as resp:
            data = await resp.json(content_type=None)
            return int(data.get("result", {}).get("timeNano", "0")) // 1_000_000
