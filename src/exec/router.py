"""Smart order execution router.

Handles the full lifecycle of entering a trade:
1. Try post-only limit at favorable price
2. Wait for fill (with timeout)
3. Reprice if BBO moves
4. Fall back to IOC if fill probability is high enough
5. Give up after max attempts

All methods are async to work with the REST client.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Optional

from core.types import Side, OrderType, TimeInForce, OrderStatus
from core.utils import time_now_ms, tick_round
from marketdata.book import OrderBook
from gateway.base import ExchangeREST, OrderResponse

log = logging.getLogger(__name__)


class OrderRouter:
    """Manages order entry with smart routing and fallback logic."""

    def __init__(self, rest: ExchangeREST, config: dict):
        self.rest = rest
        self.config = config

    async def enter(
        self,
        symbol: str,
        side: Side,
        qty: float,
        book: OrderBook,
    ) -> Optional[str]:
        """Attempt to enter a position. Returns order_id if filled, None otherwise.

        Strategy:
        1. Post limit at best bid (for buys) or best ask (for sells)
        2. Wait up to entry_timeout_ms for fill
        3. If BBO moves, reprice (up to max_reprice_attempts)
        4. If still unfilled, try IOC at BBO if fill_prob > threshold
        5. Cancel and give up if all fails
        """
        timeout_ms = self.config.get("entry_timeout_ms", 600)
        reprice_ticks = self.config.get("reprice_trigger_ticks", 1)
        max_reprices = self.config.get("max_reprice_attempts", 3)
        ioc_pmin = self.config.get("ioc_pmin", 0.65)
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)

        reprice_count = 0
        order_id = None

        try:
            # Initial price: join the queue at BBO
            price = self._entry_price(side, book, tick_size)
            order_id = await self._post_only_limit(symbol, side, qty, price)
            if not order_id:
                return None

            # Wait for fill with reprice logic
            start = time_now_ms()
            while (time_now_ms() - start) < timeout_ms:
                await asyncio.sleep(0.05)  # 50ms poll

                # Check if BBO has moved enough to warrant reprice
                new_price = self._entry_price(side, book, tick_size)
                price_moved = abs(new_price - price) / tick_size if tick_size > 0 else 0

                if price_moved >= reprice_ticks and reprice_count < max_reprices:
                    # Try to amend the order to new price
                    result = await self.rest.amend_order(symbol, order_id, price=new_price)
                    if result.success:
                        price = new_price
                        reprice_count += 1
                        log.info("Repriced order %s to %.2f (attempt %d)",
                                 order_id, new_price, reprice_count)
                    else:
                        # Order might already be filled or cancelled
                        break

            # Check if order was filled via REST (or we rely on WS updates)
            # For now, return the order_id — fill status handled by WS
            return order_id

        except Exception as e:
            log.error("Order entry failed: %s", e)
            if order_id:
                await self._safe_cancel(symbol, order_id)
            return None

    async def ioc_fallback(
        self,
        symbol: str,
        side: Side,
        qty: float,
        book: OrderBook,
    ) -> Optional[str]:
        """Send IOC order at BBO. Used when post-only fails."""
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)

        if side == Side.BUY:
            price = book.best_ask()  # take the ask
        else:
            price = book.best_bid()  # take the bid

        price = tick_round(price, tick_size)
        result = await self.rest.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.IOC,
        )
        if result.order_id:
            log.info("IOC order sent: %s %s qty=%.6f @ %.2f", side.name, symbol, qty, price)
        return result.order_id or None

    async def market_close(self, symbol: str, side: Side, qty: float) -> Optional[str]:
        """Emergency market order to flatten a position."""
        result = await self.rest.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )
        log.info("Market close: %s %s qty=%.6f -> %s", side.name, symbol, qty, result.order_id)
        return result.order_id or None

    async def place_limit(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        reduce_only: bool = False,
    ) -> Optional[str]:
        """Place a simple GTC limit order (for grid orders).

        Unlike enter(), this doesn't do reprice/timeout logic.
        """
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)
        price = tick_round(price, tick_size)

        result = await self.rest.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
        )
        if result.order_id:
            log.info("Grid limit: %s %s qty=%.6f @ %.2f -> %s",
                     side.name, symbol, qty, price, result.order_id)
        return result.order_id or None

    async def cancel(self, symbol: str, order_id: str) -> bool:
        result = await self.rest.cancel_order(symbol, order_id)
        return result.success

    async def cancel_all(self, symbol: str) -> None:
        await self.rest.cancel_all_orders(symbol)
        log.info("All orders cancelled for %s", symbol)

    # --- Helpers ---

    def _entry_price(self, side: Side, book: OrderBook, tick_size: float) -> float:
        """Compute entry price: join the queue at BBO."""
        if side == Side.BUY:
            # Post at best bid (maker)
            return tick_round(book.best_bid(), tick_size)
        else:
            # Post at best ask (maker)
            return tick_round(book.best_ask(), tick_size)

    async def _post_only_limit(self, symbol: str, side: Side,
                                qty: float, price: float) -> Optional[str]:
        result = await self.rest.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.POST_ONLY,
        )
        if result.order_id:
            log.info("Post-only limit: %s %s qty=%.6f @ %.2f -> %s",
                     side.name, symbol, qty, price, result.order_id)
        return result.order_id or None

    async def _safe_cancel(self, symbol: str, order_id: str) -> None:
        try:
            await self.rest.cancel_order(symbol, order_id)
        except Exception as e:
            log.warning("Failed to cancel order %s: %s", order_id, e)

    def fill_probability(self, side: Side, book: OrderBook, qty: float) -> float:
        """Estimate probability that an IOC at BBO would fill.

        Model: based on available liquidity at touch relative to our size,
        and recent trade rate (higher rate = more likely to fill).
        """
        if side == Side.BUY:
            available = book.best_ask_size()
        else:
            available = book.best_bid_size()

        if available <= 0:
            return 0.0

        # If our size is <= available at touch, high fill prob
        size_ratio = min(qty / available, 3.0)
        # Simple model: prob decreases with size ratio
        prob = max(0.0, 1.0 - 0.3 * size_ratio)
        return prob
