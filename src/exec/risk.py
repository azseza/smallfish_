"""Risk management: position sizing, pre-trade checks, and kill switch logic.

Key principles:
- Never risk more than X% of equity per trade
- Scale down during drawdown
- Scale down in high volatility
- Hard daily loss limit in R-multiples
- Kill switch on slippage breaches
- Volatility-adapted stops matching backtest engine
"""
from __future__ import annotations
import logging
from core.types import Side
from core.utils import clamp, qty_round, time_now_ms
from marketdata.book import OrderBook
from marketdata.tape import TradeTape
from core.state import RuntimeState

log = logging.getLogger(__name__)


def estimate_avg_range(tape: TradeTape, tick_size: float,
                       window_minutes: int = 20) -> float:
    """Estimate average 1-minute price range from recent trade data.

    Splits the last *window_minutes* of trades into 1-minute buckets,
    computes high-low per bucket, and returns the mean range.
    This matches the backtest's per-candle range tracking.
    """
    now = time_now_ms()
    cutoff = now - window_minutes * 60 * 1000

    minute_ranges: dict[int, list[float]] = {}  # minute_key -> [low, high]
    for t in tape.trades:
        if t.timestamp < cutoff:
            continue
        mk = t.timestamp // 60000
        if mk not in minute_ranges:
            minute_ranges[mk] = [t.price, t.price]
        else:
            if t.price < minute_ranges[mk][0]:
                minute_ranges[mk][0] = t.price
            if t.price > minute_ranges[mk][1]:
                minute_ranges[mk][1] = t.price

    ranges = [hi - lo for lo, hi in minute_ranges.values() if hi > lo]
    if ranges:
        return max(sum(ranges) / len(ranges), tick_size * 5)
    return tick_size * 30  # safe fallback


def position_size(state: RuntimeState, book: OrderBook,
                  stop_distance: float, symbol: str,
                  confidence: float = 0.0) -> float:
    """Calculate position size based on risk parameters.

    Args:
        stop_distance: stop-loss distance in price terms (NOT ticks).
        confidence: entry confidence for optional confidence-scaled sizing.
    """
    config = state.config
    equity = state.equity
    qty_step = config.get("qty_step", {}).get(symbol, 0.001)
    min_qty = config.get("min_qty", {}).get(symbol, 0.001)

    if stop_distance <= 0 or equity <= 0:
        return 0.0

    # Base risk
    risk_frac = config.get("risk_per_trade", 0.005)
    max_risk = config.get("max_risk_dollars", 5.0)
    risk_dollars = min(risk_frac * equity, max_risk)

    # Drawdown tier adjustment
    dd_mult = state.size_multiplier()
    risk_dollars *= dd_mult

    # Confidence-scaled sizing (matches backtest when profile active)
    profile = config.get("profile")
    if profile and profile.get("conf_scale") and confidence > 0.6:
        conf_mult = 1.0 + (confidence - 0.6) * 0.5  # 0.6→1.0x, 0.8→1.1x
        risk_dollars *= min(conf_mult, 1.2)

    # Rolling edge check: reduce or skip if recent win rate is poor
    wr = state.rolling_win_rate(20)
    if wr < 0.20:
        return 0.0  # too many consecutive losses — sit out
    if wr < 0.30:
        risk_dollars *= 0.5  # halve size on thin edge

    if risk_dollars <= 0:
        return 0.0

    # Raw quantity: risk / stop_distance
    mid = book.mid_price()
    if mid <= 0:
        return 0.0

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

    # Max concurrent positions (from profile or config)
    profile = config.get("profile")
    max_pos = config.get("max_open_orders", 2)
    if profile:
        max_pos = profile.get("max_positions", max_pos)
    if state.open_position_count() >= max_pos:
        return False, "max_positions"

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
                  symbol: str, avg_range: float = 0.0) -> tuple[float, float, float]:
    """Compute stop-loss and take-profit prices.

    When a profile is active (config["profile"] exists) and avg_range > 0,
    uses volatility-adapted stops matching the backtest engine:
        stop_distance = avg_range * sl_range_mult
        tp_distance   = avg_range * tp_range_mult

    Otherwise falls back to fixed-tick stops.

    Returns (stop_price, tp_price, stop_distance_in_price).
    """
    tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)
    mid = book.mid_price()

    profile = config.get("profile")
    if profile and avg_range > 0:
        # Volatility-adapted stops — matches BacktestEngine._try_enter()
        sl_mult = profile.get("sl_range_mult", 0.50)
        tp_mult = profile.get("tp_range_mult", 1.60)
        avg_range = max(avg_range, tick_size * 5)

        stop_distance = avg_range * sl_mult
        tp_distance = avg_range * tp_mult
    else:
        # Fallback: fixed-tick stops (no profile or no range data yet)
        base_stop = config.get("base_stop_ticks", 3)
        tp_ticks_mult = config.get("tp_ticks_multiplier", 1.5)
        stop_distance = base_stop * tick_size
        tp_distance = base_stop * tp_ticks_mult * tick_size

    if direction == 1:  # long
        stop_price = mid - stop_distance
        tp_price = mid + tp_distance
    else:  # short
        stop_price = mid + stop_distance
        tp_price = mid - tp_distance

    return round(stop_price, 8), round(tp_price, 8), stop_distance


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
