"""OCO (One-Cancels-Other) bracket order management.

Manages TP/SL brackets and trailing stops via the Bybit trading-stop API.
"""
from __future__ import annotations
import logging

from core.types import Position, Side, OrderType
from core.utils import tick_round
from gateway.base import ExchangeREST, OrderResponse

log = logging.getLogger(__name__)


class OcoManager:
    """Manages TP/SL and trailing stop for open positions."""

    def __init__(self, rest: ExchangeREST, config: dict):
        self.rest = rest
        self.config = config

    async def attach(self, position: Position) -> None:
        """Attach TP and SL to a new position via exchange trading-stop API."""
        if position.tp_price <= 0 or position.stop_price <= 0:
            log.warning("Cannot attach OCO: missing TP or SL price")
            return

        result = await self.rest.set_trading_stop(
            symbol=position.symbol,
            take_profit=position.tp_price,
            stop_loss=position.stop_price,
            position_side=position.side,  # pass position side for Binance
        )
        if result.success:
            log.info("OCO attached: %s TP=%.2f SL=%.2f",
                     position.symbol, position.tp_price, position.stop_price)
        else:
            log.error("OCO attach failed: code=%d %s", result.error_code, result.error_msg)

    async def update_trailing(self, position: Position, current_price: float,
                              confidence: float,
                              avg_range: float = 0.0) -> None:
        """Update trailing stop based on current price.

        When a profile is active and avg_range is provided, uses
        volatility-adapted trailing (avg_range * trail_pct) matching the
        backtest engine. Otherwise falls back to fixed-tick trailing.
        """
        config = self.config
        tick_size = config.get("tick_sizes", {}).get(position.symbol, 0.01)

        # Update peak favorable
        if position.side == Side.BUY:
            if current_price > position.peak_favorable:
                position.peak_favorable = current_price
        else:
            if position.peak_favorable == 0 or current_price < position.peak_favorable:
                position.peak_favorable = current_price

        # Compute trail distance
        profile = config.get("profile")
        if profile and avg_range > 0:
            # Volatility-adapted trailing — matches BacktestEngine._manage_position
            trail_distance = avg_range * profile.get("trail_pct", 0.25)
        else:
            # Fallback: fixed-tick trailing (only when confidence is high)
            c_trail = config.get("C_trail", 0.85)
            if confidence < c_trail:
                return
            trail_step = config.get("trail_step_ticks", 1)
            trail_distance = trail_step * tick_size

        # Compute new stop
        if position.side == Side.BUY:
            new_stop = position.peak_favorable - trail_distance
            if new_stop <= position.stop_price:
                return
        else:
            new_stop = position.peak_favorable + trail_distance
            if new_stop >= position.stop_price:
                return

        new_stop = tick_round(new_stop, tick_size)

        # Update via API
        result = await self.rest.set_trading_stop(
            symbol=position.symbol,
            stop_loss=new_stop,
            position_side=position.side,  # pass position side for Binance
        )
        if result.success:
            old_stop = position.stop_price
            position.stop_price = new_stop
            position.trail_active = True
            log.info("Trailing stop updated: %s %.2f → %.2f (peak=%.2f, conf=%.3f)",
                     position.symbol, old_stop, new_stop,
                     position.peak_favorable, confidence)

    async def tighten_on_confidence_drop(self, position: Position,
                                          confidence: float) -> bool:
        """If confidence drops below C_exit, flatten the position.

        Returns True if position should be exited.
        """
        c_exit = self.config.get("C_exit", 0.40)
        deadband = self.config.get("deadband", 0.05)

        if confidence < (c_exit - deadband):
            log.info("Confidence below exit threshold: %.3f < %.3f — signaling exit",
                     confidence, c_exit)
            return True
        return False

    async def market_flatten(self, position: Position) -> None:
        """Flatten a position with a market order."""
        close_side = Side.SELL if position.side == Side.BUY else Side.BUY
        result = await self.rest.place_order(
            symbol=position.symbol,
            side=close_side,
            qty=position.quantity,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )
        log.info("Market flatten: %s %s qty=%.6f -> %s",
                 close_side.name, position.symbol, position.quantity, result.order_id)

    async def flatten_all(self, positions: dict) -> None:
        """Emergency: flatten all positions."""
        for symbol, pos in list(positions.items()):
            await self.market_flatten(pos)
