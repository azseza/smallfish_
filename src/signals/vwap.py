"""VWAP Deviation signal.

Core insight: Price tends to revert to VWAP (volume-weighted average price).
When price deviates significantly above VWAP, short. Below VWAP, long.
This is a mean-reversion signal that works best in range-bound markets.

In trending markets (high vol regime), we reduce or invert the signal
to follow momentum instead.
"""
from core.utils import clamp, sigmoid


def calculate(features: dict) -> tuple[float, float]:
    deviation = features.get("vwap_deviation_ticks", 0.0)
    vol_regime = features.get("vol_regime", 0.5)  # 0=low, 0.5=normal, 1=high, -1=extreme

    # Don't trade VWAP in extreme volatility
    if vol_regime < 0:
        return 0.0, 0.0

    # In low vol (mean-revert mode): fade the deviation
    # In high vol (momentum mode): follow the deviation
    if vol_regime <= 0.25:
        # Strong mean-reversion: go long when below VWAP, short when above
        mean_revert_strength = 1.0
    elif vol_regime <= 0.75:
        # Normal: mild mean-reversion
        mean_revert_strength = 0.5
    else:
        # High vol: flip to momentum — go with the deviation
        mean_revert_strength = -0.3

    # Normalize: 5-tick deviation is a strong signal
    norm_dev = deviation / 5.0

    # Mean-reversion: negative deviation (below VWAP) → long
    raw_long = -norm_dev * mean_revert_strength
    raw_short = norm_dev * mean_revert_strength

    s_long = clamp(sigmoid(2.0 * raw_long) - 0.3, 0.0, 1.0)
    s_short = clamp(sigmoid(2.0 * raw_short) - 0.3, 0.0, 1.0)

    return s_long, s_short
