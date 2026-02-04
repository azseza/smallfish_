"""Tests for signal generation modules."""
import pytest
from signals.obi import calculate as obi_calc
from signals.prt import calculate as prt_calc
from signals.umom import calculate as umom_calc
from signals.whale import calculate as whale_calc
from signals.vwap import calculate as vwap_calc
from signals.regime import calculate as regime_calc


class TestObiSignal:
    def test_strong_bid_imbalance_gives_long(self):
        features = {"obi": 0.8, "d_obi_dt": 0.1, "spread_penalty": 0.0, "micro_pressure": 0.1}
        long_s, short_s = obi_calc(features)
        assert long_s > 0.3
        assert short_s == 0.0

    def test_strong_ask_imbalance_gives_short(self):
        features = {"obi": -0.8, "d_obi_dt": -0.1, "spread_penalty": 0.0, "micro_pressure": -0.1}
        long_s, short_s = obi_calc(features)
        assert short_s > 0.3
        assert long_s == 0.0

    def test_balanced_book_gives_zero(self):
        features = {"obi": 0.0, "d_obi_dt": 0.0, "spread_penalty": 0.0, "micro_pressure": 0.0}
        long_s, short_s = obi_calc(features)
        assert long_s == 0.0
        assert short_s == 0.0

    def test_spread_penalty_reduces_signal(self):
        features = {"obi": 0.5, "d_obi_dt": 0.0, "spread_penalty": 3.0, "micro_pressure": 0.0}
        long_s, _ = obi_calc(features)
        features_no_penalty = {"obi": 0.5, "d_obi_dt": 0.0, "spread_penalty": 0.0, "micro_pressure": 0.0}
        long_np, _ = obi_calc(features_no_penalty)
        assert long_s < long_np

    def test_scores_bounded_0_1(self):
        features = {"obi": 10.0, "d_obi_dt": 5.0, "spread_penalty": 0.0, "micro_pressure": 5.0}
        long_s, short_s = obi_calc(features)
        assert 0.0 <= long_s <= 1.0
        assert 0.0 <= short_s <= 1.0


class TestPrtSignal:
    def test_ask_spoofing_gives_long(self):
        features = {"cancel_rate_ask": 0.9, "cancel_rate_bid": 0.0, "mid_change_ticks": 0.5}
        config = {"prt": {"cancel_rate_threshold": 0.6, "mid_change_max_ticks": 2}}
        long_s, short_s = prt_calc(features, config)
        assert long_s > 0.0
        assert short_s == 0.0

    def test_bid_spoofing_gives_short(self):
        features = {"cancel_rate_ask": 0.0, "cancel_rate_bid": 0.9, "mid_change_ticks": 0.0}
        config = {"prt": {"cancel_rate_threshold": 0.6, "mid_change_max_ticks": 2}}
        long_s, short_s = prt_calc(features, config)
        assert short_s > 0.0
        assert long_s == 0.0

    def test_no_signal_when_mid_moves(self):
        features = {"cancel_rate_ask": 0.9, "cancel_rate_bid": 0.0, "mid_change_ticks": 5.0}
        config = {"prt": {"cancel_rate_threshold": 0.6, "mid_change_max_ticks": 2}}
        long_s, short_s = prt_calc(features, config)
        assert long_s == 0.0

    def test_below_threshold_no_signal(self):
        features = {"cancel_rate_ask": 0.3, "cancel_rate_bid": 0.0, "mid_change_ticks": 0.0}
        config = {"prt": {"cancel_rate_threshold": 0.6, "mid_change_max_ticks": 2}}
        long_s, _ = prt_calc(features, config)
        assert long_s == 0.0


class TestUmomSignal:
    def test_upward_momentum_gives_long(self):
        features = {
            "ema_fast": 50001.0, "ema_slow": 50000.0,
            "rv_10s": 0.5, "spread_ticks": 1.0, "buy_sell_ratio": 0.6,
        }
        long_s, short_s = umom_calc(features)
        assert long_s > short_s

    def test_downward_momentum_gives_short(self):
        features = {
            "ema_fast": 49999.0, "ema_slow": 50000.0,
            "rv_10s": 0.5, "spread_ticks": 1.0, "buy_sell_ratio": 0.4,
        }
        long_s, short_s = umom_calc(features)
        assert short_s > long_s

    def test_gate_kills_signal_on_wide_spread(self):
        features = {
            "ema_fast": 50010.0, "ema_slow": 50000.0,
            "rv_10s": 0.5, "spread_ticks": 5.0, "buy_sell_ratio": 0.5,
        }
        long_s, short_s = umom_calc(features)
        assert long_s == 0.0
        assert short_s == 0.0

    def test_gate_kills_signal_on_extreme_vol(self):
        features = {
            "ema_fast": 50010.0, "ema_slow": 50000.0,
            "rv_10s": 5.0, "spread_ticks": 1.0, "buy_sell_ratio": 0.5,
        }
        long_s, short_s = umom_calc(features)
        assert long_s == 0.0


class TestWhaleSignal:
    def test_buy_burst_gives_long(self):
        features = {"buy_burst": 10.0, "sell_burst": 1.0, "sweep_up": 0.0, "ice_absorb": 0.0}
        ltb_l, ltb_s, _, _, _, _ = whale_calc(features)
        assert ltb_l > ltb_s

    def test_no_burst_gives_zero(self):
        features = {"buy_burst": 0.0, "sell_burst": 0.0, "sweep_up": 0.0, "ice_absorb": 0.0}
        ltb_l, ltb_s, _, _, _, _ = whale_calc(features)
        assert ltb_l == 0.0
        assert ltb_s == 0.0

    def test_sweep_passthrough(self):
        features = {"buy_burst": 0.0, "sell_burst": 0.0, "sweep_up": 0.8, "sweep_down": 0.2, "ice_absorb": 0.0}
        _, _, sw_up, sw_dn, _, _ = whale_calc(features)
        assert sw_up == 0.8
        assert sw_dn == 0.2

    def test_ice_absorb_bullish(self):
        features = {"buy_burst": 0.0, "sell_burst": 0.0, "sweep_up": 0.0, "ice_absorb": 0.7}
        _, _, _, _, ice_l, ice_s = whale_calc(features)
        assert ice_l > 0.0
        assert ice_s == 0.0


class TestVwapSignal:
    def test_below_vwap_in_low_vol_goes_long(self):
        features = {"vwap_deviation_ticks": -5.0, "vol_regime": 0.1}
        long_s, short_s = vwap_calc(features)
        assert long_s > short_s

    def test_above_vwap_in_low_vol_goes_short(self):
        features = {"vwap_deviation_ticks": 5.0, "vol_regime": 0.1}
        long_s, short_s = vwap_calc(features)
        assert short_s > long_s

    def test_extreme_vol_gives_zero(self):
        features = {"vwap_deviation_ticks": -5.0, "vol_regime": -1.0}
        long_s, short_s = vwap_calc(features)
        assert long_s == 0.0
        assert short_s == 0.0


class TestRegimeSignal:
    def test_low_vol_slight_long_bias(self):
        features = {"vol_regime": 0.1, "rv_regime": 0.2, "trade_rate": 10.0}
        long_s, short_s = regime_calc(features)
        assert long_s > short_s

    def test_extreme_vol_zero(self):
        features = {"vol_regime": -1.0, "rv_regime": 5.0, "trade_rate": 10.0}
        long_s, short_s = regime_calc(features)
        assert long_s == 0.0
        assert short_s == 0.0

    def test_low_activity_reduces_signal(self):
        features = {"vol_regime": 0.5, "rv_regime": 0.5, "trade_rate": 0.5}
        long_s, short_s = regime_calc(features)
        features_active = {"vol_regime": 0.5, "rv_regime": 0.5, "trade_rate": 10.0}
        long_a, _ = regime_calc(features_active)
        assert long_s < long_a
