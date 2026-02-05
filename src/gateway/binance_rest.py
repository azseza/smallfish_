"""Binance USDT-M Futures REST API client with HMAC-SHA256 authentication.

Implements ExchangeREST for Binance Futures. Handles the key differences from Bybit:
- Auth: X-MBX-APIKEY header + HMAC signature as query param
- TP/SL: No native set_trading_stop — emulated with separate stop orders
- PostOnly: Uses GTX time-in-force
- Cancel: DELETE method instead of POST
- Amend: PUT method instead of POST
"""
from __future__ import annotations
import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Optional, Dict, Any, List

import aiohttp
import orjson

from core.types import Side, OrderType, TimeInForce
from gateway.base import (
    ExchangeREST, OrderResponse, WalletInfo, InstrumentSpec, TickerInfo,
)

log = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"
BASE_URL_TESTNET = "https://demo-fapi.binance.com"
RECV_WINDOW = 5000


def _fmt(value: float) -> str:
    """Format a float: no scientific notation, no trailing zeros."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _tif_to_binance(tif: TimeInForce) -> str:
    """Map our TimeInForce to Binance's values."""
    if tif == TimeInForce.POST_ONLY:
        return "GTX"
    return tif.value


def _otype_to_binance(ot: OrderType) -> str:
    return ot.value.upper()


class BinanceREST(ExchangeREST):
    """Async REST client for Binance USDT-M Futures API."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = BASE_URL_TESTNET if testnet else BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        # Track TP/SL order IDs per symbol for cleanup_bracket
        self._tp_sl_orders: Dict[str, Dict[str, str]] = {}  # symbol → {"tp": id, "sl": id}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Authentication ---

    def _sign(self, params: dict) -> str:
        """HMAC-SHA256 signature over query string."""
        query = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _signed_params(self, params: dict) -> dict:
        """Add timestamp and signature to params."""
        params["timestamp"] = str(int(time.time() * 1000))
        params["recvWindow"] = str(RECV_WINDOW)
        params["signature"] = self._sign(params)
        return params

    # --- HTTP helpers ---

    async def _request(self, method: str, path: str, params: dict = None, *,
                       signed: bool = True) -> dict:
        session = await self._get_session()
        params = params or {}
        headers = self._auth_headers() if signed else {}
        if signed:
            params = self._signed_params(params)
        url = f"{self.base_url}{path}"

        async with session.request(method, url, params=params, headers=headers) as resp:
            data = await resp.json(content_type=None)
            if isinstance(data, dict) and "code" in data and data["code"] != 200:
                code = data.get("code", -1)
                if code != -2021:  # -2021 = order not found (already cancelled)
                    log.error("REST %s %s failed: %s", method, path, data)
            return data

    def _to_order_response(self, data) -> OrderResponse:
        """Convert Binance response to OrderResponse."""
        if isinstance(data, dict) and "code" in data:
            return OrderResponse(
                success=False,
                order_id="",
                error_code=data.get("code", -1),
                error_msg=data.get("msg", ""),
                raw=data,
            )
        order_id = str(data.get("orderId", "")) if isinstance(data, dict) else ""
        return OrderResponse(
            success=True,
            order_id=order_id,
            error_code=0,
            error_msg="",
            raw=data if isinstance(data, dict) else {},
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
        params: Dict[str, str] = {
            "symbol": symbol,
            "side": "BUY" if side == Side.BUY else "SELL",
            "type": _otype_to_binance(order_type),
            "quantity": _fmt(qty),
        }
        if order_type == OrderType.LIMIT:
            params["timeInForce"] = _tif_to_binance(time_in_force)
        if price is not None:
            params["price"] = _fmt(price)
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_link_id:
            params["newClientOrderId"] = order_link_id

        data = await self._request("POST", "/fapi/v1/order", params)
        resp = self._to_order_response(data)
        if resp.order_id:
            log.info("Order placed: %s %s %s qty=%s px=%s -> %s",
                     side.name, symbol, order_type.value, qty, price, resp.order_id)
        return resp

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        params = {"symbol": symbol, "orderId": order_id}
        data = await self._request("DELETE", "/fapi/v1/order", params)
        resp = self._to_order_response(data)
        if resp.success:
            log.info("Order cancelled: %s", order_id)
        return resp

    async def cancel_all_orders(self, symbol: str) -> OrderResponse:
        params = {"symbol": symbol}
        data = await self._request("DELETE", "/fapi/v1/allOpenOrders", params)
        return self._to_order_response(data)

    async def amend_order(self, symbol: str, order_id: str,
                          qty: Optional[float] = None,
                          price: Optional[float] = None) -> OrderResponse:
        params: Dict[str, str] = {"symbol": symbol, "orderId": order_id}
        if qty is not None:
            params["quantity"] = _fmt(qty)
        if price is not None:
            params["price"] = _fmt(price)
        data = await self._request("PUT", "/fapi/v1/order", params)
        return self._to_order_response(data)

    # --- TP/SL Emulation ---

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: int = 0,
    ) -> OrderResponse:
        """Emulate Bybit's set_trading_stop by placing separate stop orders.

        Binance has no single API for TP+SL. We place:
        - TAKE_PROFIT_MARKET for TP
        - STOP_MARKET for SL
        and track their IDs for cleanup_bracket.
        """
        existing = self._tp_sl_orders.get(symbol, {})

        # Cancel existing orders before placing new ones
        if take_profit is not None and existing.get("tp"):
            await self.cancel_order(symbol, existing["tp"])
        if stop_loss is not None and existing.get("sl"):
            await self.cancel_order(symbol, existing["sl"])

        bracket = self._tp_sl_orders.setdefault(symbol, {})
        last_resp = OrderResponse(success=True)

        if take_profit is not None:
            tp_params: Dict[str, str] = {
                "symbol": symbol,
                "side": "SELL",  # will be overridden by position
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": _fmt(take_profit),
                "closePosition": "true",
            }
            # Determine side from stop price relative to take profit
            # TP above current = long position → close with SELL
            # TP below current = short position → close with BUY
            if stop_loss is not None:
                if take_profit > stop_loss:
                    tp_params["side"] = "SELL"  # long position
                else:
                    tp_params["side"] = "BUY"   # short position

            data = await self._request("POST", "/fapi/v1/order", tp_params)
            resp = self._to_order_response(data)
            if resp.success:
                bracket["tp"] = resp.order_id
            last_resp = resp

        if stop_loss is not None:
            sl_params: Dict[str, str] = {
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP_MARKET",
                "stopPrice": _fmt(stop_loss),
                "closePosition": "true",
            }
            if take_profit is not None:
                if take_profit > stop_loss:
                    sl_params["side"] = "SELL"  # long position → SL sells
                else:
                    sl_params["side"] = "BUY"   # short position → SL buys
            elif existing.get("tp"):
                # Infer from existing TP
                pass

            data = await self._request("POST", "/fapi/v1/order", sl_params)
            resp = self._to_order_response(data)
            if resp.success:
                bracket["sl"] = resp.order_id
            last_resp = resp

        return last_resp

    async def cleanup_bracket(self, symbol: str, order_id: str) -> None:
        """Cancel the paired TP or SL order when one side fills.

        Called from app.py when a TP or SL fill is detected.
        """
        bracket = self._tp_sl_orders.get(symbol, {})
        if not bracket:
            return

        # If TP filled, cancel SL and vice versa
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
        params = {"symbol": symbol, "leverage": str(leverage)}
        data = await self._request("POST", "/fapi/v1/leverage", params)
        return self._to_order_response(data)

    # --- Account Info ---

    async def get_wallet_balance(self) -> WalletInfo:
        data = await self._request("GET", "/fapi/v2/balance")
        equity = 0.0
        if isinstance(data, list):
            for entry in data:
                if entry.get("asset") == "USDT":
                    equity = float(entry.get("balance", 0))
                    break
        return WalletInfo(equity=equity, raw=data if isinstance(data, dict) else {"list": data})

    async def get_positions(self, symbol: str = "") -> list:
        params: Dict[str, str] = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._request("GET", "/fapi/v2/positionRisk", params)
        return data if isinstance(data, list) else []

    async def get_open_orders(self, symbol: str = "") -> list:
        params: Dict[str, str] = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._request("GET", "/fapi/v1/openOrders", params)
        return data if isinstance(data, list) else []

    async def get_instruments(self, symbol: str = "") -> List[InstrumentSpec]:
        data = await self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        specs: List[InstrumentSpec] = []
        for sym_info in data.get("symbols", []):
            sym = sym_info.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if symbol and sym != symbol:
                continue
            if sym_info.get("contractType") != "PERPETUAL":
                continue

            tick_size = 0.01
            min_qty = 0.001
            qty_step = 0.001
            min_notional = 5.0

            for f in sym_info.get("filters", []):
                ft = f.get("filterType", "")
                if ft == "PRICE_FILTER":
                    tick_size = float(f.get("tickSize", "0.01"))
                elif ft == "LOT_SIZE":
                    min_qty = float(f.get("minQty", "0.001"))
                    qty_step = float(f.get("stepSize", "0.001"))
                elif ft == "MIN_NOTIONAL":
                    min_notional = float(f.get("notional", "5"))

            specs.append(InstrumentSpec(
                symbol=sym,
                tick_size=tick_size,
                min_qty=min_qty,
                qty_step=qty_step,
                min_notional=min_notional,
            ))
        return specs

    async def get_tickers(self) -> List[TickerInfo]:
        data = await self._request("GET", "/fapi/v1/ticker/24hr", signed=False)
        tickers: List[TickerInfo] = []
        if isinstance(data, list):
            for t in data:
                tickers.append(TickerInfo(
                    symbol=t.get("symbol", ""),
                    last_price=float(t.get("lastPrice", 0)),
                    turnover_24h=float(t.get("quoteVolume", 0)),
                    price_change_pct=float(t.get("priceChangePercent", 0)) / 100,
                ))
        return tickers

    async def get_server_time(self) -> int:
        data = await self._request("GET", "/fapi/v1/time", signed=False)
        return data.get("serverTime", 0)

    async def get_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> List[dict]:
        """Download kline data from Binance Futures.

        Binance returns arrays: [openTime, open, high, low, close, volume, closeTime, ...]
        Returns ascending by ts.
        """
        # Map interval: our "1" = Binance "1m"
        bi = interval + "m" if interval.isdigit() else interval

        all_klines: List[dict] = []
        cursor_start = start_ms
        while cursor_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": bi,
                "startTime": str(cursor_start),
                "endTime": str(end_ms),
                "limit": "1500",
            }
            data = await self._request("GET", "/fapi/v1/klines", params, signed=False)
            if not isinstance(data, list) or not data:
                break

            for k in data:
                all_klines.append({
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "turnover": float(k[7]),  # quote asset volume
                })

            newest_ts = int(data[-1][0])
            if newest_ts <= cursor_start:
                break
            cursor_start = newest_ts + 60_000  # advance by 1 candle
            await asyncio.sleep(0.05)

        # Deduplicate and sort
        seen: set = set()
        unique: List[dict] = []
        for k in all_klines:
            if k["ts"] not in seen:
                seen.add(k["ts"])
                unique.append(k)
        unique.sort(key=lambda x: x["ts"])
        return unique
