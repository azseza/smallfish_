"""Micro-Momentum (UMOM) signal.

Core insight: On 1-5 second timescales, fast EMA crossing above slow EMA
predicts continued price movement — but only when volatility is moderate
and spreads are tight. In choppy/wide-spread markets, momentum signals
are noise.
"""
from core.utils import sigmoid, clamp


def calculate(features: dict) -> tuple[float, float]:
    ema_fast = features.get("ema_fast", 0.0)
    ema_slow = features.get("ema_slow", 0.0)
    rv_10s = features.get("rv_10s", 0.0)
    spread_ticks = features.get("spread_ticks", 0.0)
    buy_sell_ratio = features.get("buy_sell_ratio", 0.5)

    # Volatility gate: only trade in moderate vol + tight spreads
    rv_min, rv_max = 0.05, 2.0
    spread_max = 2.5
    vol_ok = rv_min <= rv_10s <= rv_max
    spread_ok = spread_ticks <= spread_max
    gate = 1.0 if (vol_ok and spread_ok) else 0.0

    if ema_slow <= 0:
        return 0.0, 0.0

    # Momentum in tick-normalized space
    tick = features.get("spread_ticks", 1.0)
    tick = max(tick, 0.1)  # avoid div-by-zero
    momentum = (ema_fast - ema_slow) / ema_slow * 10000  # basis points

    # Trade flow confirmation: if buy/sell ratio agrees with momentum, boost
    flow_confirm = 1.0
    if momentum > 0 and buy_sell_ratio > 0.55:
        flow_confirm = 1.3
    elif momentum < 0 and buy_sell_ratio < 0.45:
        flow_confirm = 1.3
    elif (momentum > 0 and buy_sell_ratio < 0.4) or (momentum < 0 and buy_sell_ratio > 0.6):
        flow_confirm = 0.5  # momentum contradicts flow — reduce confidence

    raw_long = sigmoid(2.0 * momentum) - 0.5  # center around 0
    raw_short = sigmoid(-2.0 * momentum) - 0.5

    s_long = clamp(raw_long * 2.0 * flow_confirm * gate, 0.0, 1.0)
    s_short = clamp(raw_short * 2.0 * flow_confirm * gate, 0.0, 1.0)

    return s_long, s_short
