"""CVD (Cumulative Volume Delta) signal.

Core insight: Persistent buy/sell aggression imbalance reveals hidden
institutional flow.  CVD acceleration (recent delta > older delta) is
the primary trigger; raw CVD direction confirms.  Price-vs-CVD divergence
adds a contrarian overlay (e.g. price falling while CVD rising → bullish).
"""
from core.utils import sigmoid, clamp


def calculate(features: dict) -> tuple[float, float]:
    cvd_accel = features.get("cvd_accel", 0.0)
    cvd_norm = features.get("cvd_norm", 0.0)
    cvd_divergence = features.get("cvd_divergence", 0.0)  # +1 bullish, -1 bearish
    total_vol = features.get("cvd", 0.0)  # raw cvd used for extreme-vol gate

    # Extreme volume gate: if total vol is near zero, no signal
    vol_window = abs(features.get("cvd", 0.0))
    buy_vol = features.get("buy_sell_ratio", 0.5)
    # Use trade_rate as activity proxy
    trade_rate = features.get("trade_rate", 0.0)
    if trade_rate < 0.5:
        return 0.0, 0.0

    # Primary: CVD acceleration — sigmoid-scaled
    raw_long = sigmoid(5.0 * cvd_accel) - 0.5
    raw_short = sigmoid(-5.0 * cvd_accel) - 0.5

    # Confirming CVD direction boosts by 1.3x
    if cvd_norm > 0.05 and raw_long > 0:
        raw_long *= 1.3
    elif cvd_norm < -0.05 and raw_short > 0:
        raw_short *= 1.3

    # Divergence overlay: bullish divergence adds to long, bearish to short
    if cvd_divergence > 0:
        raw_long += 0.2
    elif cvd_divergence < 0:
        raw_short += 0.2

    s_long = clamp(raw_long * 2.0, 0.0, 1.0)
    s_short = clamp(raw_short * 2.0, 0.0, 1.0)

    return s_long, s_short
