"""Volatility Regime signal.

This is a meta-signal that doesn't directly produce long/short scores.
Instead it produces a quality multiplier that scales other signals:

- Low vol → boost mean-reversion signals, dampen momentum
- Normal vol → standard weights
- High vol → boost momentum signals, dampen mean-reversion
- Extreme vol → kill all signals (stay flat)

It also influences position sizing via the state's vol_regime field.
"""
from core.utils import clamp


def calculate(features: dict) -> tuple[float, float]:
    """Returns (regime_long_bias, regime_short_bias).
    These are modifiers rather than directional signals.
    Positive = conditions favor trading, negative = stay out.
    """
    vol_regime = features.get("vol_regime", 0.5)
    rv = features.get("rv_regime", 0.0)

    # Extreme vol → zero (don't trade)
    if vol_regime < 0:
        return 0.0, 0.0

    # Trade rate as a proxy for "interesting market"
    trade_rate = features.get("trade_rate", 0.0)
    # If trade rate is very low, market is dead — reduce signals
    activity_gate = clamp(trade_rate / 5.0, 0.0, 1.0)  # 5 trades/s = full activity

    # In low vol, slight long bias (markets tend to grind up in low vol)
    if vol_regime <= 0.25:
        return 0.3 * activity_gate, 0.1 * activity_gate
    elif vol_regime <= 0.75:
        return 0.5 * activity_gate, 0.5 * activity_gate
    else:
        # High vol: symmetrical, both directions viable
        return 0.7 * activity_gate, 0.7 * activity_gate
