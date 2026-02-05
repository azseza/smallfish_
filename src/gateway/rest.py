"""Bybit V5 REST API client with HMAC-SHA256 authentication.

Used for order placement, cancellation, position queries, and account info.
All methods are async for non-blocking operation in the event loop.
"""
from __future__ import annotations
import asyncio
import hashlib
import hmac
import logging
import time
from typing import Optional, Dict, Any, List

import aiohttp
import orjson

from core.types import Side, OrderType, TimeInForce
from core.utils import time_now_ms
from gateway.base import (
    ExchangeREST, OrderResponse, WalletInfo, InstrumentSpec, TickerInfo,
)

log = logging.getLogger(__name__)


def _fmt(value: float) -> str:
    """Format a float for Bybit API: no scientific notation, no trailing zeros."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


BASE_URL = "https://api.bybit.com"
BASE_URL_TESTNET = "https://api-testnet.bybit.com"
RECV_WINDOW = "5000"


class BybitREST(ExchangeREST):
    """Async REST client for Bybit V5 Unified Trading API."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
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

    async def _post(self, path: str, payload: dict, *,
                    ok_codes: set[int] | None = None) -> dict:
        session = await self._get_session()
        body = orjson.dumps(payload).decode()
        headers = self._auth_headers(body)
        url = f"{self.base_url}{path}"
        async with session.post(url, headers=headers, data=body) as resp:
            data = await resp.json(content_type=None)
            ret = data.get("retCode", -1)
            if ret != 0:
                if ok_codes and ret in ok_codes:
                    log.debug("REST POST %s: %s", path, data.get("retMsg", ""))
                else:
                    log.error("REST POST %s failed: %s", path, data)
            return data

    async def _get(self, path: str, params: dict = None) -> dict:
        session = await self._get_session()
        params = params or {}
        query_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        headers = self._auth_headers(query_str)
        url = f"{self.base_url}{path}"
        async with session.get(url, headers=headers, params=params) as resp:
            data = await resp.json(content_type=None)
            if data.get("retCode", -1) != 0:
                log.error("REST GET %s failed: %s", path, data)
            return data

    def _to_order_response(self, data: dict) -> OrderResponse:
        """Convert raw Bybit response to OrderResponse."""
        ret = data.get("retCode", -1)
        return OrderResponse(
            success=(ret == 0),
            order_id=data.get("result", {}).get("orderId", ""),
            error_code=ret,
            error_msg=data.get("retMsg", ""),
            raw=data,
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
        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side == Side.BUY else "Sell",
            "orderType": order_type.value,
            "qty": _fmt(qty),
            "timeInForce": time_in_force.value,
        }
        if price is not None:
            payload["price"] = _fmt(price)
        if reduce_only:
            payload["reduceOnly"] = True
        if close_on_trigger:
            payload["closeOnTrigger"] = True
        if order_link_id:
            payload["orderLinkId"] = order_link_id

        result = await self._post("/v5/order/create", payload)
        resp = self._to_order_response(result)
        if resp.order_id:
            log.info("Order placed: %s %s %s qty=%s px=%s -> %s",
                     side.name, symbol, order_type.value, qty, price, resp.order_id)
        return resp

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        payload = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        result = await self._post("/v5/order/cancel", payload)
        resp = self._to_order_response(result)
        if resp.success:
            log.info("Order cancelled: %s", order_id)
        return resp

    async def cancel_all_orders(self, symbol: str) -> OrderResponse:
        payload = {
            "category": "linear",
            "symbol": symbol,
        }
        result = await self._post("/v5/order/cancel-all", payload)
        return self._to_order_response(result)

    async def amend_order(self, symbol: str, order_id: str,
                          qty: Optional[float] = None,
                          price: Optional[float] = None) -> OrderResponse:
        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        if qty is not None:
            payload["qty"] = _fmt(qty)
        if price is not None:
            payload["price"] = _fmt(price)
        result = await self._post("/v5/order/amend", payload)
        return self._to_order_response(result)

    # --- Position Management ---

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: int = 0,
    ) -> OrderResponse:
        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx,
        }
        if take_profit is not None:
            payload["takeProfit"] = _fmt(take_profit)
        if stop_loss is not None:
            payload["stopLoss"] = _fmt(stop_loss)
        if trailing_stop is not None:
            payload["trailingStop"] = _fmt(trailing_stop)
        result = await self._post("/v5/position/trading-stop", payload)
        return self._to_order_response(result)

    async def set_leverage(self, symbol: str, leverage: int) -> OrderResponse:
        payload = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        result = await self._post("/v5/position/set-leverage", payload,
                                  ok_codes={110043})
        return self._to_order_response(result)

    # --- Account Info ---

    async def get_wallet_balance(self) -> WalletInfo:
        data = await self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        coins = data.get("result", {}).get("list", [{}])[0].get("coin", [])
        equity = 0.0
        for c in coins:
            if c.get("coin") == "USDT":
                equity = float(c.get("walletBalance", 0))
                break
        return WalletInfo(equity=equity, raw=data)

    async def get_positions(self, symbol: str = "") -> list:
        params: Dict[str, str] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/v5/position/list", params)
        return data.get("result", {}).get("list", [])

    async def get_open_orders(self, symbol: str = "") -> list:
        params: Dict[str, str] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/v5/order/realtime", params)
        return data.get("result", {}).get("list", [])

    async def get_instruments(self, symbol: str = "") -> List[InstrumentSpec]:
        params: Dict[str, str] = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        specs: List[InstrumentSpec] = []
        cursor = ""
        while True:
            if cursor:
                params["cursor"] = cursor
            data = await self._get("/v5/market/instruments-info", params)
            items = data.get("result", {}).get("list", [])
            for item in items:
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                lot = item.get("lotSizeFilter", {})
                price_f = item.get("priceFilter", {})
                specs.append(InstrumentSpec(
                    symbol=sym,
                    tick_size=float(price_f.get("tickSize", "0.01")),
                    min_qty=float(lot.get("minOrderQty", "0.001")),
                    qty_step=float(lot.get("qtyStep", "0.001")),
                    min_notional=float(lot.get("minNotionalValue", "5")),
                ))
            cursor = data.get("result", {}).get("nextPageCursor", "")
            if not cursor or not items:
                break
        return specs

    async def get_tickers(self) -> List[TickerInfo]:
        data = await self._get("/v5/market/tickers", {"category": "linear"})
        tickers: List[TickerInfo] = []
        for t in data.get("result", {}).get("list", []):
            tickers.append(TickerInfo(
                symbol=t.get("symbol", ""),
                last_price=float(t.get("lastPrice", 0)),
                turnover_24h=float(t.get("turnover24h", 0)),
                price_change_pct=float(t.get("price24hPcnt", 0)),
            ))
        return tickers

    async def get_server_time(self) -> int:
        session = await self._get_session()
        url = f"{self.base_url}/v5/market/time"
        async with session.get(url) as resp:
            data = await resp.json(content_type=None)
            return int(data.get("result", {}).get("timeNano", "0")) // 1_000_000

    async def get_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> List[dict]:
        """Download kline data. Returns list of dicts sorted ascending by ts."""
        all_klines: List[dict] = []
        cursor_end = end_ms
        while cursor_end > start_ms:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "start": str(start_ms),
                "end": str(cursor_end),
                "limit": "1000",
            }
            data = await self._get("/v5/market/kline", params)
            klines = data.get("result", {}).get("list", [])
            if not klines:
                break
            for k in klines:
                all_klines.append({
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "turnover": float(k[6]),
                })
            oldest_ts = int(klines[-1][0])
            if oldest_ts >= cursor_end:
                break
            cursor_end = oldest_ts - 1
            await asyncio.sleep(0.05)

        seen: set = set()
        unique: List[dict] = []
        for k in all_klines:
            if k["ts"] not in seen:
                seen.add(k["ts"])
                unique.append(k)
        unique.sort(key=lambda x: x["ts"])
        return unique

    async def cleanup_bracket(self, symbol: str, order_id: str) -> None:
        """No-op for Bybit — native TP/SL handles cleanup automatically."""
        pass
