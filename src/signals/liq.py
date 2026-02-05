"""LIQ (Liquidity Thinness) signal.

Core insight: Thin order books move more easily.  When one side is
significantly thicker than the other, price is more likely to move
*away* from the thick side (it acts as a wall).

Thick book overall → (0, 0) — hard to move, skip momentum trades.
Thin book + positive liq_asymmetry (more bids) → long (easier up).
Thin book + negative liq_asymmetry (more asks) → short (easier down).
"""
from core.utils import clamp, sigmoid


def calculate(features: dict) -> tuple[float, float]:
    liq_thinness = features.get("liq_thinness", 0.0)
    liq_asymmetry = features.get("liq_asymmetry", 0.0)
    obi = features.get("obi", 0.0)

    # Thick book gate: if depth is very high (thinness very low), no signal
    # Typical thinness range: 0.001 (thick) to 1.0+ (thin)
    # Only fire when book is meaningfully thin
    if liq_thinness < 0.01:
        return 0.0, 0.0

    # Thinness intensity: soft cap via sigmoid
    thin_intensity = sigmoid(10.0 * liq_thinness) - 0.5  # 0→0.5 range

    # Direction from asymmetry
    # Positive asymmetry = more bids → support below → long
    # Negative asymmetry = more asks → resistance above → short
    raw_long = thin_intensity * clamp(liq_asymmetry, 0.0, 1.0)
    raw_short = thin_intensity * clamp(-liq_asymmetry, 0.0, 1.0)

    # Cross-reference with OBI for confirmation boost
    if obi > 0.1 and raw_long > 0:
        raw_long *= 1.2
    elif obi < -0.1 and raw_short > 0:
        raw_short *= 1.2

    s_long = clamp(raw_long * 2.0, 0.0, 1.0)
    s_short = clamp(raw_short * 2.0, 0.0, 1.0)

    return s_long, s_short
