"""Pull-&-Replace Trap (PRT) signal.

Core insight: When one side of the book shows heavy cancellations
(spoofing) without corresponding price movement, the "real" pressure
is on the opposite side. Fade the fake side.

Example: Heavy ask cancellations + no down-move = spoof selling → go long.
"""
from core.utils import sigmoid, clamp


def calculate(features: dict, config: dict = None) -> tuple[float, float]:
    cfg = (config or {}).get("prt", {})
    threshold = cfg.get("cancel_rate_threshold", 0.6)
    mid_max = cfg.get("mid_change_max_ticks", 2)

    cancel_rate_ask = features.get("cancel_rate_ask", 0.0)
    cancel_rate_bid = features.get("cancel_rate_bid", 0.0)
    mid_change = features.get("mid_change_ticks", 0.0)

    s_long = 0.0
    s_short = 0.0

    # Ask-side spoofing → long signal (fake selling pressure)
    if cancel_rate_ask > threshold and abs(mid_change) < mid_max:
        # Scale by how extreme the cancel rate is
        intensity = (cancel_rate_ask - threshold) / (1.0 - threshold + 1e-9)
        s_long = clamp(sigmoid(3.0 * intensity) - 0.3, 0.0, 1.0)

    # Bid-side spoofing → short signal (fake buying pressure)
    if cancel_rate_bid > threshold and abs(mid_change) < mid_max:
        intensity = (cancel_rate_bid - threshold) / (1.0 - threshold + 1e-9)
        s_short = clamp(sigmoid(3.0 * intensity) - 0.3, 0.0, 1.0)

    return s_long, s_short
