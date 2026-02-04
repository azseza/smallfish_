"""Order-Book Imbalance (OBI) signal.

Core insight: When bid volume significantly outweighs ask volume,
price tends to move up (and vice versa). The rate of change of
imbalance is an even stronger predictor.
"""
from core.utils import clamp


def calculate(features: dict) -> tuple[float, float]:
    obi = features.get("obi", 0.0)
    d_obi_dt = features.get("d_obi_dt", 0.0)
    spread_penalty = features.get("spread_penalty", 0.0)
    micro_pressure = features.get("micro_pressure", 0.0)

    # Core signal: imbalance + momentum of imbalance + micro-pressure
    # k1=0.6 weights momentum, k2=0.15 penalizes wide spreads, k3=0.2 adds micro-pressure
    raw_long = 0.5 * (obi + 0.6 * d_obi_dt + 0.2 * micro_pressure) - 0.15 * spread_penalty
    raw_short = 0.5 * (-obi - 0.6 * d_obi_dt - 0.2 * micro_pressure) - 0.15 * spread_penalty

    s_long = clamp(raw_long, 0.0, 1.0)
    s_short = clamp(raw_short, 0.0, 1.0)

    return s_long, s_short
