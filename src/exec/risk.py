"""Risk management: position sizing, pre-trade checks, and kill switch logic.

Key principles:
- Never risk more than X% of equity per trade
- Scale down during drawdown
- Scale down in high volatility
- Hard daily loss limit in R-multiples
- Kill switch on slippage breaches
"""
from __future__ import annotations
import logging
from core.types import Side
from core.utils import clamp, qty_round
from marketdata.book import OrderBook
from core.state import RuntimeState

log = logging.getLogger(__name__)


def position_size(state: RuntimeState, book: OrderBook, stop_ticks: int,
                  symbol: str) -> float:
    """Calculate position size based on risk parameters.

    Size = risk_dollars / (stop_distance_in_price * 1)
    With adjustments for drawdown tier and volatility.
    """
    config = state.config
    equity = state.equity
    tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)
    qty_step = config.get("qty_step", {}).get(symbol, 0.001)
    min_qty = config.get("min_qty", {}).get(symbol, 0.001)

    if stop_ticks <= 0 or equity <= 0:
        return 0.0

    # Base risk
    risk_frac = config.get("risk_per_trade", 0.005)
    max_risk = config.get("max_risk_dollars", 5.0)
    risk_dollars = min(risk_frac * equity, max_risk)

    # Drawdown tier adjustment
    dd_mult = state.size_multiplier()
    risk_dollars *= dd_mult

    # Rolling edge check: reduce or skip if recent win rate is poor
    wr = state.rolling_win_rate(20)
    if wr < 0.20:
        return 0.0  # too many losses — skip entry
    if wr < 0.30:
        risk_dollars *= 0.5  # halve size on thin edge

    if risk_dollars <= 0:
        return 0.0

    # Stop distance in price terms
    stop_distance = stop_ticks * tick_size
    if stop_distance <= 0:
        return 0.0

    # Raw quantity: risk / stop_distance (for linear perps, 1 qty = 1 unit exposure)
    mid = book.mid_price()
    if mid <= 0:
        return 0.0

    # For USDT linear perps: PnL = qty * price_change
    # So qty = risk_dollars / stop_distance
    qty = risk_dollars / stop_distance

    # Apply leverage cap
    leverage = config.get("leverage", 10)
    max_notional = equity * leverage
    max_qty = max_notional / mid
    qty = min(qty, max_qty)

    # Round to step size
    qty = qty_round(qty, qty_step)

    if qty < min_qty:
        return 0.0

    return qty


def can_trade(state: RuntimeState, book: OrderBook, symbol: str) -> tuple[bool, str]:
    """Pre-trade gate check. Returns (allowed, reason_if_blocked)."""
    config = state.config

    if state.kill_switch:
        return False, f"kill_switch: {state.kill_reason}"

    if state.has_position(symbol):
        return False, "already_in_position"

    if len(state.active_orders) >= config.get("max_open_orders", 2):
        return False, "max_open_orders"

    if not book.is_fresh():
        return False, "stale_book"

    if book.spread_ticks() > config.get("max_spread", 2):
        return False, f"spread_too_wide: {book.spread_ticks():.1f}"

    if state.latency_ms > config.get("max_latency_ms", 80):
        return False, f"high_latency: {state.latency_ms}ms"

    if state.in_funding_window(symbol):
        return False, "funding_window"

    if state.in_cooldown():
        return False, "loss_cooldown"

    if state.daily_loss_R >= config.get("max_daily_R", 10):
        return False, f"daily_loss_limit: {state.daily_loss_R:.1f}R"

    # Size multiplier check (0 = drawdown too deep)
    if state.size_multiplier() <= 0:
        return False, "drawdown_limit"

    return True, ""


def compute_stops(book: OrderBook, direction: int, config: dict,
                  symbol: str) -> tuple[float, float, float]:
    """Compute stop-loss and take-profit prices.

    Returns (stop_price, tp_price, stop_ticks).
    """
    tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)
    base_stop = config.get("base_stop_ticks", 3)
    tp_mult = config.get("tp_ticks_multiplier", 1.5)
    tp_ticks = base_stop * tp_mult

    mid = book.mid_price()

    if direction == 1:  # long
        stop_price = mid - base_stop * tick_size
        tp_price = mid + tp_ticks * tick_size
    else:  # short
        stop_price = mid + base_stop * tick_size
        tp_price = mid - tp_ticks * tick_size

    return round(stop_price, 8), round(tp_price, 8), base_stop


def check_slippage(expected_price: float, fill_price: float,
                   direction: int, tick_size: float,
                   max_slip_ticks: int) -> tuple[float, bool]:
    """Check if slippage exceeds threshold.

    Returns (slippage_ticks, is_breach).
    """
    if tick_size <= 0:
        return 0.0, False
    # Adverse slippage: for longs, fill > expected is bad
    slip = (fill_price - expected_price) * direction / tick_size
    return slip, abs(slip) > max_slip_ticks
