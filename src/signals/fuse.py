"""Signal Fusion Engine.

Combines all signal scores into a single directional decision
with a confidence level. Supports adaptive weight adjustment
based on recent per-signal performance (edge tracking).
"""
from __future__ import annotations
import logging
from typing import Dict, Tuple
from core.utils import sigmoid, clamp, safe_div

log = logging.getLogger(__name__)


def get_weights(config: dict, signal_edges: Dict[str, float] = None,
                adaptive_cfg: dict = None) -> dict:
    """Return current signal weights, optionally adjusted by recent edge.

    Adaptive logic: weight_i = base_weight_i * clamp(edge_i / baseline, floor, ceiling)
    """
    weights = {
        "w": list(config["weights"]["w"]),
        "v": list(config["weights"]["v"]),
        "x": list(config["weights"].get("x", [0.07, 0.03])),
        "t": list(config["weights"].get("t", [0.07, 0.05, 0.06, 0.06, 0.06])),
    }

    if not signal_edges or not adaptive_cfg or not adaptive_cfg.get("enabled"):
        return weights

    floor = adaptive_cfg.get("weight_floor", 0.5)
    ceiling = adaptive_cfg.get("weight_ceiling", 2.0)
    min_edge = adaptive_cfg.get("min_edge", 0.0001)

    # Map signal names to weight indices
    signal_map = {
        "obi": ("w", 0),
        "prt": ("w", 1),
        "umom": ("w", 2),
        "ltb": ("v", 0),
        "sweep": ("v", 1),
        "ice": ("v", 2),
        "vwap": ("x", 0),
        "regime": ("x", 1),
        "cvd": ("t", 0),
        "tps": ("t", 1),
        "liq": ("t", 2),
        "mvr": ("t", 3),
        "absorb": ("t", 4),
    }

    # Compute baseline edge (average across all signals)
    edges = [e for e in signal_edges.values() if abs(e) > 1e-12]
    if not edges:
        return weights
    baseline = sum(abs(e) for e in edges) / len(edges)
    if baseline < min_edge:
        return weights

    for name, (group, idx) in signal_map.items():
        edge = signal_edges.get(name, 0.0)
        if abs(edge) < 1e-12:
            multiplier = floor  # no data → floor weight
        else:
            multiplier = clamp(abs(edge) / baseline, floor, ceiling)
        weights[group][idx] *= multiplier

    # Re-normalize so total weights sum to 1
    total = sum(weights["w"]) + sum(weights["v"]) + sum(weights["x"]) + sum(weights["t"])
    if total > 0:
        for group in ("w", "v", "x", "t"):
            weights[group] = [w / total for w in weights[group]]

    return weights


def score_all(features: dict, config: dict) -> dict:
    """Compute all signal scores from features."""
    import signals.obi as obi_mod
    import signals.prt as prt_mod
    import signals.umom as umom_mod
    import signals.whale as whale_mod
    import signals.vwap as vwap_mod
    import signals.regime as regime_mod
    import signals.cvd as cvd_mod
    import signals.tps as tps_mod
    import signals.liq as liq_mod
    import signals.mvr as mvr_mod
    import signals.absorb as absorb_mod

    s: dict = {}

    obi_l, obi_s = obi_mod.calculate(features)
    s["obi_long"], s["obi_short"] = obi_l, obi_s

    prt_l, prt_s = prt_mod.calculate(features, config)
    s["prt_long"], s["prt_short"] = prt_l, prt_s

    umom_l, umom_s = umom_mod.calculate(features)
    s["umom_long"], s["umom_short"] = umom_l, umom_s

    ltb_l, ltb_s, sw_up, sw_dn, ice_l, ice_s = whale_mod.calculate(features)
    s["ltb_long"], s["ltb_short"] = ltb_l, ltb_s
    s["sweep_up"], s["sweep_down"] = sw_up, sw_dn
    s["ice_long"], s["ice_short"] = ice_l, ice_s

    vwap_l, vwap_s = vwap_mod.calculate(features)
    s["vwap_long"], s["vwap_short"] = vwap_l, vwap_s

    regime_l, regime_s = regime_mod.calculate(features)
    s["regime_long"], s["regime_short"] = regime_l, regime_s

    cvd_l, cvd_s = cvd_mod.calculate(features)
    s["cvd_long"], s["cvd_short"] = cvd_l, cvd_s

    tps_l, tps_s = tps_mod.calculate(features)
    s["tps_long"], s["tps_short"] = tps_l, tps_s

    liq_l, liq_s = liq_mod.calculate(features)
    s["liq_long"], s["liq_short"] = liq_l, liq_s

    mvr_l, mvr_s = mvr_mod.calculate(features)
    s["mvr_long"], s["mvr_short"] = mvr_l, mvr_s

    absorb_l, absorb_s = absorb_mod.calculate(features)
    s["absorb_long"], s["absorb_short"] = absorb_l, absorb_s

    return s


def meta(scores: dict, weights: dict) -> Tuple[float, float]:
    """Weighted combination of all signal scores."""
    w = weights["w"]
    v = weights["v"]
    x = weights.get("x", [0.0, 0.0])
    t = weights.get("t", [0.0, 0.0, 0.0, 0.0, 0.0])

    meta_long = (
        w[0] * scores.get("obi_long", 0.0) +
        w[1] * scores.get("prt_long", 0.0) +
        w[2] * scores.get("umom_long", 0.0) +
        v[0] * scores.get("ltb_long", 0.0) +
        v[1] * scores.get("sweep_up", 0.0) +
        v[2] * scores.get("ice_long", 0.0) +
        x[0] * scores.get("vwap_long", 0.0) +
        x[1] * scores.get("regime_long", 0.0) +
        t[0] * scores.get("cvd_long", 0.0) +
        t[1] * scores.get("tps_long", 0.0) +
        t[2] * scores.get("liq_long", 0.0) +
        t[3] * scores.get("mvr_long", 0.0) +
        t[4] * scores.get("absorb_long", 0.0)
    )

    meta_short = (
        w[0] * scores.get("obi_short", 0.0) +
        w[1] * scores.get("prt_short", 0.0) +
        w[2] * scores.get("umom_short", 0.0) +
        v[0] * scores.get("ltb_short", 0.0) +
        v[1] * scores.get("sweep_down", 0.0) +
        v[2] * scores.get("ice_short", 0.0) +
        x[0] * scores.get("vwap_short", 0.0) +
        x[1] * scores.get("regime_short", 0.0) +
        t[0] * scores.get("cvd_short", 0.0) +
        t[1] * scores.get("tps_short", 0.0) +
        t[2] * scores.get("liq_short", 0.0) +
        t[3] * scores.get("mvr_short", 0.0) +
        t[4] * scores.get("absorb_short", 0.0)
    )

    return meta_long, meta_short


def confidence(raw: float, alpha: float, quality_gate: float) -> float:
    """Map raw directional score to confidence in [0, 1]."""
    return sigmoid(alpha * abs(raw)) * quality_gate


def count_agreement(scores: dict, direction: int) -> int:
    """Count how many of the 13 signal categories agree with the direction.

    Each signal category (obi, prt, umom, ltb, sweep, ice, vwap, regime,
    cvd, tps, liq, mvr, absorb) counts as agreeing if its score in the
    given direction exceeds a minimum threshold (0.05).
    """
    if direction == 0:
        return 0

    threshold = 0.05

    if direction == 1:
        pairs = [
            scores.get("obi_long", 0.0),
            scores.get("prt_long", 0.0),
            scores.get("umom_long", 0.0),
            scores.get("ltb_long", 0.0),
            scores.get("sweep_up", 0.0),
            scores.get("ice_long", 0.0),
            scores.get("vwap_long", 0.0),
            scores.get("regime_long", 0.0),
            scores.get("cvd_long", 0.0),
            scores.get("tps_long", 0.0),
            scores.get("liq_long", 0.0),
            scores.get("mvr_long", 0.0),
            scores.get("absorb_long", 0.0),
        ]
    else:
        pairs = [
            scores.get("obi_short", 0.0),
            scores.get("prt_short", 0.0),
            scores.get("umom_short", 0.0),
            scores.get("ltb_short", 0.0),
            scores.get("sweep_down", 0.0),
            scores.get("ice_short", 0.0),
            scores.get("vwap_short", 0.0),
            scores.get("regime_short", 0.0),
            scores.get("cvd_short", 0.0),
            scores.get("tps_short", 0.0),
            scores.get("liq_short", 0.0),
            scores.get("mvr_short", 0.0),
            scores.get("absorb_short", 0.0),
        ]

    return sum(1 for s in pairs if s > threshold)


def decide(scores: dict, weights: dict, alpha: float,
           quality_gate: float,
           min_signals: int = 0) -> Tuple[int, float, float]:
    """Full decision pipeline.

    Returns:
        direction: +1 (long), -1 (short), 0 (no trade)
        conf: confidence score [0, 1]
        raw: raw directional score
    """
    meta_long, meta_short = meta(scores, weights)
    raw = meta_long - meta_short

    if abs(raw) < 1e-6:
        return 0, 0.0, raw

    direction = 1 if raw > 0 else -1
    conf = confidence(raw, alpha, quality_gate)

    # Signal agreement gate: require min_signals categories to agree
    if min_signals > 0 and count_agreement(scores, direction) < min_signals:
        return 0, 0.0, raw

    return direction, conf, raw
