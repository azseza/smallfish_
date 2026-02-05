"""TPS (Trade Intensity / Trades Per Second) signal.

Core insight: High trade intensity validates momentum; low intensity
means dead market — suppress everything.  NOT directional alone: gates
on urgency, then uses buy/sell ratio for direction.
"""
from core.utils import clamp


def calculate(features: dict) -> tuple[float, float]:
    tps_ratio = features.get("tps_ratio", 1.0)
    buy_sell_ratio = features.get("buy_sell_ratio", 0.5)

    # Dead market: suppress everything
    if tps_ratio < 0.5:
        return 0.0, 0.0

    # Middle range: weak/no signal
    if tps_ratio < 1.5:
        return 0.0, 0.0

    # High urgency: direction from buy/sell ratio
    # Urgency intensity scales 1.5→3.0 → 0→1
    urgency = clamp((tps_ratio - 1.5) / 1.5, 0.0, 1.0)

    s_long = 0.0
    s_short = 0.0

    if buy_sell_ratio > 0.55:
        # Buy-heavy + high urgency → long
        directional = clamp((buy_sell_ratio - 0.55) / 0.15, 0.0, 1.0)
        s_long = clamp(urgency * directional, 0.0, 1.0)
    elif buy_sell_ratio < 0.45:
        # Sell-heavy + high urgency → short
        directional = clamp((0.45 - buy_sell_ratio) / 0.15, 0.0, 1.0)
        s_short = clamp(urgency * directional, 0.0, 1.0)

    return s_long, s_short
