"""MEXC Futures REST API client with HMAC-SHA256 authentication.

Implements ExchangeREST for MEXC Futures. Key differences from Bybit/Binance:
- Auth: X-MEXC-APIKEY header + HMAC-SHA256 signature via request param
- Symbol format: BTC_USDT (underscore separator)
- Side mapping: 1=open long, 2=close long, 3=open short, 4=close short
- Kline intervals: Min1, Min5, Min15, Min30, Hour1 etc
- TP/SL: Emulated with separate orders (same as Binance pattern)
- No testnet available for futures
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Optional, Dict, List

import aiohttp
import orjson

from core.types import Side, OrderType, TimeInForce
from gateway.base import (
    ExchangeREST, OrderResponse, WalletInfo, InstrumentSpec, TickerInfo,
)
from gateway.symbol_map import to_mexc, from_mexc

log = logging.getLogger(__name__)

BASE_URL = "https://contract.mexc.com"

# Interval map: our "1" -> MEXC "Min1"
_INTERVAL_MAP = {
    "1": "Min1", "3": "Min5", "5": "Min5", "15": "Min15",
    "30": "Min30", "60": "Hour1",
    "1m": "Min1", "3m": "Min5", "5m": "Min5", "15m": "Min15",
    "30m": "Min30", "60m": "Hour1", "1h": "Hour1",
}

# Seconds per candle for pagination advancement
_INTERVAL_SECS = {
    "Min1": 60, "Min5": 300, "Min15": 900,
    "Min30": 1800, "Hour1": 3600,
}


def _fmt(value: float) -> str:
    """Format a float: no scientific notation, no trailing zeros."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _mexc_side(side: Side, reduce_only: bool) -> int:
    """Map Side + reduce_only to MEXC's 4-directional side codes.

    1 = open long, 2 = close long (reduce), 3 = open short, 4 = close short (reduce)
    """
    if side == Side.BUY:
        return 2 if reduce_only else 1  # close short vs open long
    else:
        return 4 if reduce_only else 3  # close long vs open short


def _mexc_order_type(ot: OrderType) -> int:
    """Map OrderType to MEXC order type codes.

    MEXC uses numeric codes: 1=limit, 2=post-only, 3=IOC, 5=market
    """
    if ot == OrderType.MARKET:
        return 5
    return 1  # limit


class MexcREST(ExchangeREST):
    """Async REST client for MEXC Futures API."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet  # MEXC has no futures testnet, kept for interface compat
        self.base_url = BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._tp_sl_orders: Dict[str, Dict[str, str]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Authentication ---

    def _sign(self, timestamp: str, params_str: str = "") -> str:
        """HMAC-SHA256 signature: sign(accessKey + timestamp + paramString)."""
        message = self.api_key + timestamp + params_str
        return hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, params_str: str = "") -> Dict[str, str]:
        """Build authenticated headers for MEXC."""
        timestamp = str(int(time.time() * 1000))
        signature = self._sign(timestamp, params_str)
        return {
            "Content-Type": "application/json",
            "ApiKey": self.api_key,
            "Request-Time": timestamp,
            "Signature": signature,
        }

    # --- HTTP helpers ---

    async def _request(self, method: str, path: str, params: dict = None, *,
                       signed: bool = True, body: dict = None) -> dict:
        session = await self._get_session()
        params = params or {}
        url = f"{self.base_url}{path}"

        if signed:
            # Build param string for signature
            if body:
                import json
                params_str = json.dumps(body, separators=(",", ":"))
            elif params:
                params_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            else:
                params_str = ""
            headers = self._auth_headers(params_str)
        else:
            headers = {"Content-Type": "application/json"}

        if method in ("POST", "PUT") and body:
            async with session.request(method, url, json=body, headers=headers) as resp:
                http_status = resp.status
                data = await resp.json(content_type=None)
        else:
            async with session.request(method, url, params=params, headers=headers) as resp:
                http_status = resp.status
                data = await resp.json(content_type=None)

        # Check HTTP-level errors (auth failures, server errors, rate limits)
        if http_status >= 400:
            log.error("REST %s %s HTTP %d: %s", method, path, http_status, data)
            return {"code": http_status, "msg": f"HTTP {http_status}", "data": data}

        if isinstance(data, dict) and data.get("code") not in (0, 200, None):
            log.error("REST %s %s failed: %s", method, path, data)
        return data

    def _to_order_response(self, data: dict) -> OrderResponse:
        """Convert MEXC response to OrderResponse."""
        if not isinstance(data, dict):
            return OrderResponse(success=False, error_msg=str(data))

        code = data.get("code", -1)
        if code != 0:
            return OrderResponse(
                success=False,
                order_id="",
                error_code=code,
                error_msg=data.get("msg", ""),
                raw=data,
            )

        result = data.get("data", "")
        order_id = str(result) if result else ""
        if not order_id:
            return OrderResponse(
                success=False,
                order_id="",
                error_code=0,
                error_msg="API returned code=0 but no order ID in data field",
                raw=data,
            )
        return OrderResponse(
            success=True,
            order_id=order_id,
            error_code=0,
            error_msg="",
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
        body: Dict = {
            "symbol": to_mexc(symbol),
            "side": _mexc_side(side, reduce_only),
            "type": _mexc_order_type(order_type),
            "vol": int(qty) if qty == int(qty) else float(qty),
            "openType": 2,  # cross margin
        }
        if price is not None and order_type != OrderType.MARKET:
            body["price"] = float(_fmt(price))
        if order_link_id:
            body["externalOid"] = order_link_id

        # Post-only
        if time_in_force == TimeInForce.POST_ONLY:
            body["type"] = 2  # post-only maker

        data = await self._request("POST", "/api/v1/private/order/submit", body=body)
        resp = self._to_order_response(data)
        if resp.success:
            log.info("Order placed: %s %s %s qty=%s px=%s -> %s",
                     side.name, symbol, order_type.value, qty, price, resp.order_id)
        return resp

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        body = [{"symbol": to_mexc(symbol), "orderId": order_id}]
        data = await self._request("POST", "/api/v1/private/order/cancel", body=body)
        return self._to_order_response(data)

    async def cancel_all_orders(self, symbol: str) -> OrderResponse:
        body = {"symbol": to_mexc(symbol)}
        data = await self._request("POST", "/api/v1/private/order/cancel_all", body=body)
        return self._to_order_response(data)

    async def amend_order(self, symbol: str, order_id: str,
                          qty: Optional[float] = None,
                          price: Optional[float] = None) -> OrderResponse:
        # MEXC doesn't support order amendment — cancel and re-place
        cancel_resp = await self.cancel_order(symbol, order_id)
        if not cancel_resp.success:
            return cancel_resp
        # Caller must re-place order with new params
        return OrderResponse(
            success=True, order_id="", error_msg="Cancelled for amend — re-place required",
        )

    # --- TP/SL Emulation (same pattern as Binance) ---

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: int = 0,
        position_side: Optional[Side] = None,
    ) -> OrderResponse:
        """Emulate TP/SL with separate stop orders."""
        existing = self._tp_sl_orders.get(symbol, {})

        if take_profit is not None and existing.get("tp"):
            await self.cancel_order(symbol, existing["tp"])
        if stop_loss is not None and existing.get("sl"):
            await self.cancel_order(symbol, existing["sl"])

        bracket = self._tp_sl_orders.setdefault(symbol, {})
        last_resp = OrderResponse(success=True)

        # Determine close side
        if position_side is not None:
            close_side = Side.SELL if position_side == Side.BUY else Side.BUY
        elif take_profit is not None and stop_loss is not None:
            close_side = Side.SELL if take_profit > stop_loss else Side.BUY
        else:
            close_side = Side.SELL

        if take_profit is not None:
            body = {
                "symbol": to_mexc(symbol),
                "side": _mexc_side(close_side, reduce_only=True),
                "type": 5,  # market
                "vol": 0,   # close full position
                "triggerPrice": float(_fmt(take_profit)),
                "triggerType": 1,  # mark price
                "openType": 2,
                "executeCycle": 2,  # 24h
            }
            data = await self._request("POST", "/api/v1/private/order/submit", body=body)
            resp = self._to_order_response(data)
            if resp.success:
                bracket["tp"] = resp.order_id
            last_resp = resp

        if stop_loss is not None:
            body = {
                "symbol": to_mexc(symbol),
                "side": _mexc_side(close_side, reduce_only=True),
                "type": 5,
                "vol": 0,
                "triggerPrice": float(_fmt(stop_loss)),
                "triggerType": 1,
                "openType": 2,
                "executeCycle": 2,
            }
            data = await self._request("POST", "/api/v1/private/order/submit", body=body)
            resp = self._to_order_response(data)
            if resp.success:
                bracket["sl"] = resp.order_id
            last_resp = resp

        return last_resp

    async def cleanup_bracket(self, symbol: str, order_id: str) -> None:
        """Cancel the paired TP or SL order when one side fills."""
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
        body = {"symbol": to_mexc(symbol), "leverage": leverage, "openType": 2}
        data = await self._request("POST", "/api/v1/private/position/change_leverage", body=body)
        return self._to_order_response(data)

    # --- Account Info ---

    async def get_wallet_balance(self) -> WalletInfo:
        data = await self._request("GET", "/api/v1/private/account/assets")
        equity = 0.0
        if isinstance(data, dict):
            for asset in data.get("data", []):
                if asset.get("currency") == "USDT":
                    equity = float(asset.get("equity", 0))
                    break
        return WalletInfo(equity=equity, raw=data if isinstance(data, dict) else {"raw": data})

    async def get_positions(self, symbol: str = "") -> list:
        params: Dict[str, str] = {}
        if symbol:
            params["symbol"] = to_mexc(symbol)
        data = await self._request("GET", "/api/v1/private/position/open_positions", params)
        result = data.get("data", []) if isinstance(data, dict) else []
        return result if isinstance(result, list) else []

    async def get_open_orders(self, symbol: str = "") -> list:
        params: Dict[str, str] = {}
        if symbol:
            params["symbol"] = to_mexc(symbol)
        data = await self._request("GET", "/api/v1/private/order/list/open_orders", params)
        result = data.get("data", []) if isinstance(data, dict) else []
        return result if isinstance(result, list) else []

    async def get_instruments(self, symbol: str = "") -> List[InstrumentSpec]:
        data = await self._request("GET", "/api/v1/contract/detail", signed=False)
        specs: List[InstrumentSpec] = []
        items = data.get("data", []) if isinstance(data, dict) else []
        for info in items:
            sym_raw = info.get("symbol", "")
            sym = from_mexc(sym_raw)
            if not sym.endswith("USDT"):
                continue
            if symbol and sym != symbol:
                continue

            tick_size = float(info.get("priceUnit", 0.01))
            min_qty = float(info.get("volUnit", 1))
            qty_step = float(info.get("volUnit", 1))
            min_notional = float(info.get("minVol", 1)) * tick_size

            specs.append(InstrumentSpec(
                symbol=sym,
                tick_size=tick_size,
                min_qty=min_qty,
                qty_step=qty_step,
                min_notional=min_notional,
            ))
        return specs

    async def get_tickers(self) -> List[TickerInfo]:
        data = await self._request("GET", "/api/v1/contract/ticker", signed=False)
        tickers: List[TickerInfo] = []
        items = data.get("data", []) if isinstance(data, dict) else []
        for t in items:
            sym_raw = t.get("symbol", "")
            sym = from_mexc(sym_raw)
            tickers.append(TickerInfo(
                symbol=sym,
                last_price=float(t.get("lastPrice", 0)),
                turnover_24h=float(t.get("amount24", 0)),
                price_change_pct=float(t.get("riseFallRate", 0)),
            ))
        return tickers

    async def get_server_time(self) -> int:
        return int(time.time() * 1000)

    async def get_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> List[dict]:
        """Download kline data from MEXC Futures.

        MEXC endpoint: GET /api/v1/contract/kline/{symbol}
        Returns: {data: {time: [...], open: [...], ...}}
        """
        mexc_sym = to_mexc(symbol)
        mexc_interval = _INTERVAL_MAP.get(interval, "Min5")
        step_secs = _INTERVAL_SECS.get(mexc_interval, 300)  # seconds per candle

        all_klines: List[dict] = []
        cursor_start = start_ms // 1000  # MEXC uses seconds
        cursor_end = end_ms // 1000

        while cursor_start < cursor_end:
            params = {
                "interval": mexc_interval,
                "start": str(cursor_start),
                "end": str(cursor_end),
            }
            data = await self._request(
                "GET", f"/api/v1/contract/kline/{mexc_sym}", params, signed=False,
            )

            result = data.get("data", {}) if isinstance(data, dict) else {}
            times = result.get("time", [])
            opens = result.get("open", [])
            highs = result.get("high", [])
            lows = result.get("low", [])
            closes = result.get("close", [])
            volumes = result.get("vol", [])

            if not times:
                break

            for i in range(len(times)):
                ts = int(times[i]) * 1000  # convert to ms
                all_klines.append({
                    "ts": ts,
                    "open": float(opens[i]) if i < len(opens) else 0,
                    "high": float(highs[i]) if i < len(highs) else 0,
                    "low": float(lows[i]) if i < len(lows) else 0,
                    "close": float(closes[i]) if i < len(closes) else 0,
                    "volume": float(volumes[i]) if i < len(volumes) else 0,
                    "turnover": 0.0,  # MEXC doesn't return quote volume
                })

            newest_ts = int(times[-1])
            if newest_ts <= cursor_start:
                break
            cursor_start = newest_ts + step_secs  # advance by 1 candle
            await asyncio.sleep(0.1)  # rate limit

        # Deduplicate and sort
        seen: set = set()
        unique: List[dict] = []
        for k in all_klines:
            if k["ts"] not in seen:
                seen.add(k["ts"])
                unique.append(k)
        unique.sort(key=lambda x: x["ts"])
        return unique

    async def withdraw(self, coin: str, chain: str, address: str,
                       amount: float) -> dict:
        # MEXC withdrawal is via spot API, not futures
        return {"success": False, "tx_id": "", "error_msg": "Withdraw via MEXC spot API"}

    async def get_deposit_address(self, coin: str, chain: str) -> dict:
        return {"address": "", "chain": chain, "tag": ""}

    async def get_leverage(self, symbol: str) -> int:
        """Get current leverage for symbol."""
        positions = await self.get_positions(symbol)
        for p in positions:
            if from_mexc(p.get("symbol", "")) == symbol:
                return int(p.get("leverage", 1))
        return 1
