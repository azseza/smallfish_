"""Multigrid Strategy — layered grid trading with signal-fusion bias.

Places buy and sell limit orders at regular price intervals around a dynamic
center (VWAP or mid-price). The grid bias shifts based on the fused signal
confidence: bullish signals shift the center upward (more buy levels below),
bearish signals shift it downward (more sell levels above).

Grid profits are harvested as price oscillates through levels. Works best
in ranging/low-volatility regimes. In trending regimes, the signal bias
prevents the grid from fighting the trend.

Usage:
    grid = MultigridStrategy(config, state)
    grid.recalculate(book, direction, confidence)
    orders_to_place = grid.pending_orders(symbol)
    grid.on_fill(symbol, price, side)
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.types import Side
from core.utils import tick_round, qty_round, clamp

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GridLevel:
    """A single level in the grid."""
    price: float
    side: Side
    quantity: float
    is_filled: bool = False
    order_id: str = ""
    fill_price: float = 0.0
    pair_price: float = 0.0  # the opposite side price for this grid pair


@dataclass(slots=True)
class GridState:
    """Per-symbol grid state."""
    center: float = 0.0
    levels: List[GridLevel] = field(default_factory=list)
    total_pnl: float = 0.0
    round_trips: int = 0
    active: bool = False
    last_recalc_ts: int = 0


class MultigridStrategy:
    """Multi-level grid trading with directional bias from signal fusion."""

    def __init__(self, config: dict):
        gc = config.get("multigrid", {})
        self.enabled = gc.get("enabled", False)
        self.num_levels = gc.get("levels", 5)
        self.spacing_pct = gc.get("spacing_pct", 0.001)  # 0.1% between levels
        self.qty_per_level_pct = gc.get("qty_per_level_pct", 0.01)  # 1% equity per level
        self.max_open_levels = gc.get("max_open_levels", 10)
        self.bias_strength = gc.get("bias_strength", 0.5)  # how much signal shifts center
        self.recalc_interval_ms = gc.get("recalc_interval_ms", 60_000)
        self.harvest_pct = gc.get("harvest_pct", 0.001)  # min profit to harvest (0.1%)
        self.regime_filter = gc.get("regime_filter", True)  # disable in extreme vol

        self.grids: Dict[str, GridState] = {}
        self.config = config

    def init_symbol(self, symbol: str) -> None:
        """Initialize grid state for a symbol."""
        if symbol not in self.grids:
            self.grids[symbol] = GridState()

    def recalculate(
        self,
        symbol: str,
        mid_price: float,
        direction: int,
        confidence: float,
        equity: float,
        vol_regime: str,
        ts: int,
    ) -> None:
        """Recalculate grid levels around the current price with directional bias.

        Args:
            symbol: Trading pair
            mid_price: Current mid-price from orderbook
            direction: Signal direction (+1 long, -1 short, 0 neutral)
            confidence: Signal confidence (0-1)
            equity: Current account equity
            vol_regime: Current volatility regime
            ts: Current timestamp in ms
        """
        gs = self.grids.get(symbol)
        if not gs:
            return

        # Don't recalculate too often
        if ts - gs.last_recalc_ts < self.recalc_interval_ms and gs.active:
            return

        # Disable grid in extreme volatility
        if self.regime_filter and vol_regime == "extreme":
            gs.active = False
            gs.levels.clear()
            return

        # Widen spacing in high vol, tighten in low vol
        vol_mult = {
            "low": 0.7,
            "normal": 1.0,
            "high": 1.5,
            "extreme": 0.0,  # disabled above
        }.get(vol_regime, 1.0)
        spacing = self.spacing_pct * vol_mult

        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)
        qty_step = self.config.get("qty_step", {}).get(symbol, 0.001)
        min_qty = self.config.get("min_qty", {}).get(symbol, 0.001)

        # Bias: shift center in the direction of the signal
        # Positive direction = shift center down (more buys below, expecting upward move)
        # Negative direction = shift center up (more sells above, expecting downward move)
        bias_shift = -direction * confidence * self.bias_strength * spacing * mid_price
        center = mid_price + bias_shift
        gs.center = center

        # Calculate quantity per level
        qty_per_level = (equity * self.qty_per_level_pct) / mid_price
        qty_per_level = qty_round(qty_per_level, qty_step)
        if qty_per_level < min_qty:
            gs.active = False
            return

        # Build symmetric grid levels around center
        levels: List[GridLevel] = []
        price_step = center * spacing

        for i in range(1, self.num_levels + 1):
            # Buy levels below center
            buy_price = tick_round(center - i * price_step, tick_size)
            # Corresponding sell (take-profit) price
            sell_tp = tick_round(buy_price + price_step, tick_size)

            levels.append(GridLevel(
                price=buy_price,
                side=Side.BUY,
                quantity=qty_per_level,
                pair_price=sell_tp,
            ))

            # Sell levels above center
            sell_price = tick_round(center + i * price_step, tick_size)
            # Corresponding buy (take-profit) price
            buy_tp = tick_round(sell_price - price_step, tick_size)

            levels.append(GridLevel(
                price=sell_price,
                side=Side.SELL,
                quantity=qty_per_level,
                pair_price=buy_tp,
            ))

        # Carry over fill state from old levels if prices match
        old_filled = {(lv.price, lv.side): lv for lv in gs.levels if lv.is_filled}
        for lv in levels:
            key = (lv.price, lv.side)
            if key in old_filled:
                old = old_filled[key]
                lv.is_filled = True
                lv.order_id = old.order_id
                lv.fill_price = old.fill_price

        gs.levels = levels
        gs.active = True
        gs.last_recalc_ts = ts

        log.info("GRID %s: center=%.2f levels=%d spacing=%.4f%% bias=%+.4f",
                 symbol, center, len(levels), spacing * 100, bias_shift)

    def pending_orders(self, symbol: str) -> List[GridLevel]:
        """Return grid levels that need orders placed (not yet filled, no order_id)."""
        gs = self.grids.get(symbol)
        if not gs or not gs.active:
            return []

        pending = []
        open_count = sum(1 for lv in gs.levels if lv.order_id and not lv.is_filled)
        for lv in gs.levels:
            if not lv.is_filled and not lv.order_id and open_count < self.max_open_levels:
                pending.append(lv)
                open_count += 1
        return pending

    def harvest_orders(self, symbol: str) -> List[GridLevel]:
        """Return filled levels that are ready to harvest (place opposite order)."""
        gs = self.grids.get(symbol)
        if not gs or not gs.active:
            return []

        harvests = []
        for lv in gs.levels:
            if lv.is_filled and lv.pair_price > 0:
                harvests.append(lv)
        return harvests

    def on_fill(self, symbol: str, price: float, side: Side, order_id: str) -> Optional[GridLevel]:
        """Record a grid order fill. Returns the filled level."""
        gs = self.grids.get(symbol)
        if not gs:
            return None

        for lv in gs.levels:
            if lv.order_id == order_id and not lv.is_filled:
                lv.is_filled = True
                lv.fill_price = price
                log.info("GRID FILL %s: %s @ %.2f (pair target: %.2f)",
                         symbol, side.name, price, lv.pair_price)
                return lv
        return None

    def on_harvest(self, symbol: str, level: GridLevel, fill_price: float) -> float:
        """Record a harvest (opposite side fill). Returns PnL."""
        gs = self.grids.get(symbol)
        if not gs:
            return 0.0

        if level.side == Side.BUY:
            pnl = (fill_price - level.fill_price) * level.quantity
        else:
            pnl = (level.fill_price - fill_price) * level.quantity

        gs.total_pnl += pnl
        gs.round_trips += 1

        # Reset level for reuse
        level.is_filled = False
        level.order_id = ""
        level.fill_price = 0.0

        log.info("GRID HARVEST %s: pnl=%.4f total=%.4f trips=%d",
                 symbol, pnl, gs.total_pnl, gs.round_trips)
        return pnl

    def mark_order(self, symbol: str, price: float, side: Side, order_id: str) -> None:
        """Mark a grid level as having a pending order."""
        gs = self.grids.get(symbol)
        if not gs:
            return
        for lv in gs.levels:
            if abs(lv.price - price) < 1e-10 and lv.side == side and not lv.order_id:
                lv.order_id = order_id
                return

    def cancel_all(self, symbol: str) -> List[str]:
        """Return all active order IDs for cancellation."""
        gs = self.grids.get(symbol)
        if not gs:
            return []
        order_ids = [lv.order_id for lv in gs.levels if lv.order_id and not lv.is_filled]
        return order_ids

    def status(self, symbol: str) -> dict:
        """Return grid status for monitoring/dashboard."""
        gs = self.grids.get(symbol)
        if not gs:
            return {"active": False}

        buy_levels = [lv for lv in gs.levels if lv.side == Side.BUY]
        sell_levels = [lv for lv in gs.levels if lv.side == Side.SELL]

        return {
            "active": gs.active,
            "center": round(gs.center, 2),
            "buy_levels": len(buy_levels),
            "sell_levels": len(sell_levels),
            "filled_buys": sum(1 for lv in buy_levels if lv.is_filled),
            "filled_sells": sum(1 for lv in sell_levels if lv.is_filled),
            "pending_orders": sum(1 for lv in gs.levels if lv.order_id and not lv.is_filled),
            "total_pnl": round(gs.total_pnl, 4),
            "round_trips": gs.round_trips,
        }

    def all_status(self) -> Dict[str, dict]:
        """Return status for all symbols."""
        return {sym: self.status(sym) for sym in self.grids}
