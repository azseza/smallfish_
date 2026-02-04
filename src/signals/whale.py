"""Whale activity signals: Large-Trade-Burst, Sweep, Iceberg.

LTB: Detects bursts of large trades from institutional players.
SWEEP: Detects when aggressive orders clear multiple book levels.
ICE: Detects hidden iceberg orders absorbing pressure at a level.
"""
from core.utils import clamp, sigmoid, safe_div


def calculate(features: dict) -> tuple[float, float, float, float, float, float]:
    """Returns (ltb_long, ltb_short, sweep_up, sweep_down, ice_long, ice_short)."""

    # --- Large Trade Burst ---
    buy_burst = features.get("buy_burst", 0.0)
    sell_burst = features.get("sell_burst", 0.0)
    total_burst = buy_burst + sell_burst

    if total_burst > 0:
        s_ltb_long = clamp(buy_burst / (total_burst + 1e-9), 0.0, 1.0)
        s_ltb_short = clamp(sell_burst / (total_burst + 1e-9), 0.0, 1.0)
        # Scale by magnitude: more burst = stronger signal
        magnitude = sigmoid(total_burst * 0.1) * 2.0  # soft scaling
        s_ltb_long = clamp(s_ltb_long * magnitude, 0.0, 1.0)
        s_ltb_short = clamp(s_ltb_short * magnitude, 0.0, 1.0)
    else:
        s_ltb_long = 0.0
        s_ltb_short = 0.0

    # --- Sweep ---
    sweep_up = clamp(features.get("sweep_up", 0.0), 0.0, 1.0)
    sweep_down = clamp(features.get("sweep_down", 0.0), 0.0, 1.0)

    # --- Iceberg ---
    ice_absorb = features.get("ice_absorb", 0.0)  # +1 = bids absorbing, -1 = asks absorbing
    # Bids absorbing sell pressure = bullish (long)
    s_ice_long = clamp(ice_absorb, 0.0, 1.0)
    # Asks absorbing buy pressure = bearish (short)
    s_ice_short = clamp(-ice_absorb, 0.0, 1.0)

    return s_ltb_long, s_ltb_short, sweep_up, sweep_down, s_ice_long, s_ice_short
