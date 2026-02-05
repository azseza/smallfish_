"""MVR (Micro Volatility Ratio) signal.

Core insight: RV_short / RV_long tells us whether price is breaking out
(ratio >> 1) or chopping (ratio << 1).

- MVR >> 1 (breakout): amplify momentum direction
- MVR << 1 (chop): amplify mean-reversion direction (fade momentum)
- MVR ≈ 1: neutral → (0, 0)

This is the key fix for UMOM firing in chop — MVR acts as a regime
gate that either confirms or counters the momentum signal.
"""
from core.utils import clamp, sigmoid


def calculate(features: dict) -> tuple[float, float]:
    mvr = features.get("mvr", 1.0)
    momentum = features.get("momentum", 0.0)
    ema_fast = features.get("ema_fast", 0.0)
    ema_slow = features.get("ema_slow", 0.0)

    if ema_slow <= 0:
        return 0.0, 0.0

    # Momentum in basis points
    mom_bps = (ema_fast - ema_slow) / ema_slow * 10000

    # Breakout regime: MVR > 1.5 → follow momentum
    if mvr > 1.5:
        intensity = clamp((mvr - 1.5) / 2.0, 0.0, 1.0)
        if mom_bps > 0:
            return clamp(sigmoid(2.0 * mom_bps) * intensity, 0.0, 1.0), 0.0
        else:
            return 0.0, clamp(sigmoid(-2.0 * mom_bps) * intensity, 0.0, 1.0)

    # Chop regime: MVR < 0.5 → fade momentum (mean-revert)
    if mvr < 0.5:
        intensity = clamp((0.5 - mvr) / 0.5, 0.0, 1.0)
        # Fade: if momentum is up, signal short (expect reversion)
        if mom_bps > 0.5:
            return 0.0, clamp(sigmoid(2.0 * mom_bps) * intensity * 0.5, 0.0, 1.0)
        elif mom_bps < -0.5:
            return clamp(sigmoid(-2.0 * mom_bps) * intensity * 0.5, 0.0, 1.0), 0.0
        return 0.0, 0.0

    # Neutral regime (0.5 <= mvr <= 1.5): no signal
    return 0.0, 0.0
