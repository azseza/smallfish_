"""Tests for signal fusion engine."""
import pytest
from signals.fuse import get_weights, meta, confidence, decide, score_all


class TestMeta:
    def test_meta_long_positive_with_long_signals(self):
        scores = {
            "obi_long": 0.8, "obi_short": 0.0,
            "prt_long": 0.5, "prt_short": 0.0,
            "umom_long": 0.6, "umom_short": 0.0,
            "ltb_long": 0.0, "ltb_short": 0.0,
            "sweep_up": 0.0, "sweep_down": 0.0,
            "ice_long": 0.0, "ice_short": 0.0,
            "vwap_long": 0.0, "vwap_short": 0.0,
            "regime_long": 0.5, "regime_short": 0.5,
        }
        weights = {"w": [0.30, 0.15, 0.20], "v": [0.12, 0.08, 0.05], "x": [0.07, 0.03]}
        ml, ms = meta(scores, weights)
        assert ml > ms
        assert ml > 0

    def test_meta_short_positive_with_short_signals(self):
        scores = {
            "obi_long": 0.0, "obi_short": 0.9,
            "prt_long": 0.0, "prt_short": 0.3,
            "umom_long": 0.0, "umom_short": 0.7,
            "ltb_long": 0.0, "ltb_short": 0.0,
            "sweep_up": 0.0, "sweep_down": 0.0,
            "ice_long": 0.0, "ice_short": 0.0,
            "vwap_long": 0.0, "vwap_short": 0.0,
            "regime_long": 0.5, "regime_short": 0.5,
        }
        weights = {"w": [0.30, 0.15, 0.20], "v": [0.12, 0.08, 0.05], "x": [0.07, 0.03]}
        ml, ms = meta(scores, weights)
        assert ms > ml

    def test_zero_scores_give_zero_meta(self):
        scores = {k: 0.0 for k in [
            "obi_long", "obi_short", "prt_long", "prt_short",
            "umom_long", "umom_short", "ltb_long", "ltb_short",
            "sweep_up", "sweep_down", "ice_long", "ice_short",
            "vwap_long", "vwap_short", "regime_long", "regime_short",
        ]}
        weights = {"w": [0.30, 0.15, 0.20], "v": [0.12, 0.08, 0.05], "x": [0.07, 0.03]}
        ml, ms = meta(scores, weights)
        assert ml == 0.0
        assert ms == 0.0


class TestConfidence:
    def test_confidence_increases_with_raw(self):
        c1 = confidence(0.1, 5.0, 1.0)
        c2 = confidence(0.5, 5.0, 1.0)
        assert c2 > c1

    def test_quality_gate_zero_kills_confidence(self):
        c = confidence(0.5, 5.0, 0.0)
        assert c == 0.0

    def test_confidence_bounded_0_1(self):
        c = confidence(100.0, 5.0, 1.0)
        assert 0.0 <= c <= 1.0


class TestDecide:
    def test_long_decision(self):
        scores = {
            "obi_long": 0.9, "obi_short": 0.0,
            "prt_long": 0.0, "prt_short": 0.0,
            "umom_long": 0.8, "umom_short": 0.0,
            "ltb_long": 0.0, "ltb_short": 0.0,
            "sweep_up": 0.0, "sweep_down": 0.0,
            "ice_long": 0.0, "ice_short": 0.0,
            "vwap_long": 0.0, "vwap_short": 0.0,
            "regime_long": 0.5, "regime_short": 0.5,
        }
        weights = {"w": [0.30, 0.15, 0.20], "v": [0.12, 0.08, 0.05], "x": [0.07, 0.03]}
        direction, conf, raw = decide(scores, weights, 5.0, 1.0)
        assert direction == 1
        assert conf > 0.5
        assert raw > 0

    def test_no_trade_on_zero_signals(self):
        scores = {k: 0.0 for k in [
            "obi_long", "obi_short", "prt_long", "prt_short",
            "umom_long", "umom_short", "ltb_long", "ltb_short",
            "sweep_up", "sweep_down", "ice_long", "ice_short",
            "vwap_long", "vwap_short", "regime_long", "regime_short",
        ]}
        weights = {"w": [0.30, 0.15, 0.20], "v": [0.12, 0.08, 0.05], "x": [0.07, 0.03]}
        direction, conf, raw = decide(scores, weights, 5.0, 1.0)
        assert direction == 0
        assert conf == 0.0


class TestAdaptiveWeights:
    def test_no_edges_returns_base_weights(self, config):
        weights = get_weights(config, None, None)
        assert weights["w"] == config["weights"]["w"]

    def test_adaptive_disabled_returns_base(self, config):
        edges = {"obi": 0.01, "prt": 0.005}
        adaptive = {"enabled": False}
        weights = get_weights(config, edges, adaptive)
        assert weights["w"] == config["weights"]["w"]

    def test_adaptive_adjusts_weights(self, config):
        edges = {"obi": 0.02, "prt": 0.001, "umom": 0.01,
                 "ltb": 0.005, "sweep": 0.003, "ice": 0.002, "vwap": 0.004}
        adaptive = {
            "enabled": True, "lookback_trades": 100,
            "min_edge": 0.0001, "weight_floor": 0.5, "weight_ceiling": 2.0,
        }
        weights = get_weights(config, edges, adaptive)
        # OBI has highest edge → should get boosted relative to PRT
        # (after normalization, obi should still be > prt weight)
        assert weights["w"][0] > weights["w"][1]  # obi > prt
