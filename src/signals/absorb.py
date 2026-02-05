"""ABS (Absorption Ratio) signal.

Core insight: When aggressive volume is being absorbed without price
movement, a limit-order participant is defending a level.  This is
a strong confirmation signal for hidden institutional support/resistance.

- High absorption + flat price + more sell-side absorbed → bullish (bid defense)
- High absorption + flat price + more buy-side absorbed → bearish (ask defense)
- Gate: only fires when |mid_change_ticks| < 1 (price relatively flat)
"""
from core.utils import clamp, sigmoid, safe_div


def calculate(features: dict) -> tuple[float, float]:
    absorption = features.get("absorption", 0.0)
    mid_change = features.get("mid_change_ticks", 0.0)
    ice_absorb_bid = features.get("ice_absorb_bid", 0.0)
    ice_absorb_ask = features.get("ice_absorb_ask", 0.0)
    buy_sell_ratio = features.get("buy_sell_ratio", 0.5)

    # Gate: only fire when price is relatively flat
    if abs(mid_change) >= 1.0:
        return 0.0, 0.0

    # Need meaningful absorption to fire
    # Absorption values vary widely; use sigmoid to soft-cap
    abs_intensity = sigmoid(absorption * 0.01) - 0.5  # ~0 to 0.5 range
    if abs_intensity < 0.05:
        return 0.0, 0.0

    # Direction from ice_absorb: which side is absorbing more?
    total_absorb = ice_absorb_bid + ice_absorb_ask
    if total_absorb < 1e-9:
        # Fallback: use buy_sell_ratio — if more selling is absorbed, bullish
        if buy_sell_ratio < 0.45:
            # More sellers → bids absorbing → bullish
            return clamp(abs_intensity * 2.0, 0.0, 1.0), 0.0
        elif buy_sell_ratio > 0.55:
            # More buyers → asks absorbing → bearish
            return 0.0, clamp(abs_intensity * 2.0, 0.0, 1.0)
        return 0.0, 0.0

    # Bid absorbing more = bids defending = bullish
    absorb_ratio = safe_div(ice_absorb_bid - ice_absorb_ask, total_absorb)

    s_long = clamp(absorb_ratio * abs_intensity * 2.0, 0.0, 1.0)
    s_short = clamp(-absorb_ratio * abs_intensity * 2.0, 0.0, 1.0)

    return s_long, s_short
