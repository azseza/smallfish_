"""Real-time trading metrics and performance analytics.

Computes slippage, fill ratios, edge per signal, Sharpe ratio,
and other live metrics from completed trades.
"""
from __future__ import annotations
import math
import logging
from typing import Dict, List, Optional
from core.types import TradeResult
from core.ringbuffer import RingBuffer
from core.utils import safe_div

log = logging.getLogger(__name__)


def slippage_ticks(expected: float, actual: float, direction: int,
                   tick_size: float) -> float:
    """Signed slippage in ticks. Positive = adverse."""
    if tick_size <= 0:
        return 0.0
    return (actual - expected) * direction / tick_size


def avg_slippage(trades: RingBuffer[TradeResult], n: int = 50) -> float:
    """Average entry slippage over last N trades."""
    recent = trades.last(min(n, len(trades)))
    if not recent:
        return 0.0
    return sum(t.slippage_entry for t in recent) / len(recent)


def fill_ratio(filled: int, total: int) -> float:
    return safe_div(filled, total, 1.0)


def win_rate(trades: RingBuffer[TradeResult], n: int = 100) -> float:
    recent = trades.last(min(n, len(trades)))
    if not recent:
        return 0.0
    wins = sum(1 for t in recent if t.pnl > 0)
    return wins / len(recent)


def profit_factor(trades: RingBuffer[TradeResult], n: int = 100) -> float:
    """Gross profits / gross losses."""
    recent = trades.last(min(n, len(trades)))
    if not recent:
        return 0.0
    gross_profit = sum(t.pnl for t in recent if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in recent if t.pnl < 0))
    return safe_div(gross_profit, gross_loss)


def avg_pnl_R(trades: RingBuffer[TradeResult], n: int = 100) -> float:
    recent = trades.last(min(n, len(trades)))
    if not recent:
        return 0.0
    return sum(t.pnl_R for t in recent) / len(recent)


def sharpe_ratio(trades: RingBuffer[TradeResult], n: int = 100) -> float:
    """Sharpe ratio from R-multiples (risk-free rate = 0)."""
    recent = trades.last(min(n, len(trades)))
    if len(recent) < 2:
        return 0.0
    returns = [t.pnl_R for t in recent]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return safe_div(mean, std)


def edge_per_signal(trades: RingBuffer[TradeResult], n: int = 100) -> Dict[str, float]:
    """Average PnL contribution attributed to each signal.

    For each trade, we examine the signal scores at entry and compute
    the weighted contribution.
    """
    recent = trades.last(min(n, len(trades)))
    if not recent:
        return {}

    signal_names = [
        "obi_long", "obi_short", "prt_long", "prt_short",
        "umom_long", "umom_short", "ltb_long", "ltb_short",
        "sweep_up", "sweep_down", "ice_long", "ice_short",
        "vwap_long", "vwap_short",
    ]

    # Map to base signal names
    base_signals = {
        "obi": ["obi_long", "obi_short"],
        "prt": ["prt_long", "prt_short"],
        "umom": ["umom_long", "umom_short"],
        "ltb": ["ltb_long", "ltb_short"],
        "sweep": ["sweep_up", "sweep_down"],
        "ice": ["ice_long", "ice_short"],
        "vwap": ["vwap_long", "vwap_short"],
    }

    edge: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for name in base_signals:
        edge[name] = 0.0
        counts[name] = 0

    for trade in recent:
        scores = trade.signals_at_entry
        if not scores:
            continue

        for base_name, signal_keys in base_signals.items():
            # Get the active signal score (the one matching trade direction)
            active = 0.0
            for key in signal_keys:
                s = scores.get(key, 0.0)
                if s > 0:
                    active = max(active, s)

            if active > 0.1:  # signal was meaningfully active
                # Attribute PnL proportionally to signal strength
                edge[base_name] += trade.pnl_R * active
                counts[base_name] += 1

    # Average
    for name in edge:
        if counts[name] > 0:
            edge[name] /= counts[name]

    return edge


def avg_trade_duration_ms(trades: RingBuffer[TradeResult], n: int = 50) -> float:
    recent = trades.last(min(n, len(trades)))
    if not recent:
        return 0.0
    return sum(t.duration_ms for t in recent) / len(recent)


def compute_report(trades: RingBuffer[TradeResult], n: int = 100) -> Dict[str, float]:
    """Generate a full metrics report."""
    return {
        "win_rate": round(win_rate(trades, n), 4),
        "profit_factor": round(profit_factor(trades, n), 3),
        "avg_pnl_R": round(avg_pnl_R(trades, n), 4),
        "sharpe": round(sharpe_ratio(trades, n), 3),
        "avg_slippage": round(avg_slippage(trades, n), 3),
        "avg_duration_ms": round(avg_trade_duration_ms(trades, n), 0),
        "n_trades": min(n, len(trades)),
    }
