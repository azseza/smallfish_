"""Tests for the 5 new microstructure signals: CVD, TPS, LIQ, MVR, ABS."""
import pytest
from signals.cvd import calculate as cvd_calc
from signals.tps import calculate as tps_calc
from signals.liq import calculate as liq_calc
from signals.mvr import calculate as mvr_calc
from signals.absorb import calculate as absorb_calc


# ── CVD ──────────────────────────────────────────────────────────

class TestCVD:
    def test_positive_accel_gives_long(self):
        f = {"cvd_accel": 0.3, "cvd_norm": 0.2, "cvd_divergence": 0.0,
             "cvd": 100.0, "buy_sell_ratio": 0.6, "trade_rate": 5.0}
        l, s = cvd_calc(f)
        assert l > 0
        assert l > s

    def test_negative_accel_gives_short(self):
        f = {"cvd_accel": -0.3, "cvd_norm": -0.2, "cvd_divergence": 0.0,
             "cvd": -100.0, "buy_sell_ratio": 0.4, "trade_rate": 5.0}
        l, s = cvd_calc(f)
        assert s > 0
        assert s > l

    def test_bullish_divergence_boosts_long(self):
        f_base = {"cvd_accel": 0.1, "cvd_norm": 0.1, "cvd_divergence": 0.0,
                  "cvd": 50.0, "buy_sell_ratio": 0.55, "trade_rate": 5.0}
        f_div = dict(f_base, cvd_divergence=1.0)
        l_base, _ = cvd_calc(f_base)
        l_div, _ = cvd_calc(f_div)
        assert l_div > l_base

    def test_bearish_divergence_boosts_short(self):
        f_base = {"cvd_accel": -0.1, "cvd_norm": -0.1, "cvd_divergence": 0.0,
                  "cvd": -50.0, "buy_sell_ratio": 0.45, "trade_rate": 5.0}
        f_div = dict(f_base, cvd_divergence=-1.0)
        _, s_base = cvd_calc(f_base)
        _, s_div = cvd_calc(f_div)
        assert s_div > s_base

    def test_dead_market_returns_zero(self):
        """Low trade rate → no signal."""
        f = {"cvd_accel": 0.5, "cvd_norm": 0.3, "cvd_divergence": 0.0,
             "cvd": 200.0, "buy_sell_ratio": 0.7, "trade_rate": 0.3}
        l, s = cvd_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_confirming_direction_boosts(self):
        """CVD_norm matching accel direction should amplify the signal."""
        f_confirm = {"cvd_accel": 0.2, "cvd_norm": 0.1, "cvd_divergence": 0.0,
                     "cvd": 80.0, "buy_sell_ratio": 0.5, "trade_rate": 5.0}
        f_contra = {"cvd_accel": 0.2, "cvd_norm": -0.1, "cvd_divergence": 0.0,
                    "cvd": 80.0, "buy_sell_ratio": 0.5, "trade_rate": 5.0}
        l_confirm, _ = cvd_calc(f_confirm)
        l_contra, _ = cvd_calc(f_contra)
        assert l_confirm > l_contra


# ── TPS ──────────────────────────────────────────────────────────

class TestTPS:
    def test_high_urgency_buy_heavy_gives_long(self):
        f = {"tps_ratio": 2.5, "buy_sell_ratio": 0.65}
        l, s = tps_calc(f)
        assert l > 0
        assert s == 0.0

    def test_high_urgency_sell_heavy_gives_short(self):
        f = {"tps_ratio": 2.5, "buy_sell_ratio": 0.35}
        l, s = tps_calc(f)
        assert s > 0
        assert l == 0.0

    def test_dead_market_suppresses(self):
        f = {"tps_ratio": 0.3, "buy_sell_ratio": 0.7}
        l, s = tps_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_moderate_urgency_no_signal(self):
        f = {"tps_ratio": 1.0, "buy_sell_ratio": 0.6}
        l, s = tps_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_high_urgency_balanced_no_signal(self):
        """High urgency but balanced buy/sell → no directional signal."""
        f = {"tps_ratio": 3.0, "buy_sell_ratio": 0.50}
        l, s = tps_calc(f)
        assert l == 0.0
        assert s == 0.0


# ── LIQ ──────────────────────────────────────────────────────────

class TestLIQ:
    def test_thin_book_positive_asymmetry_long(self):
        f = {"liq_thinness": 0.5, "liq_asymmetry": 0.4, "obi": 0.2}
        l, s = liq_calc(f)
        assert l > 0
        assert l > s

    def test_thin_book_negative_asymmetry_short(self):
        f = {"liq_thinness": 0.5, "liq_asymmetry": -0.4, "obi": -0.2}
        l, s = liq_calc(f)
        assert s > 0
        assert s > l

    def test_thick_book_no_signal(self):
        f = {"liq_thinness": 0.005, "liq_asymmetry": 0.8, "obi": 0.5}
        l, s = liq_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_obi_confirmation_boost(self):
        """OBI confirming liq direction should boost signal."""
        f_confirm = {"liq_thinness": 0.3, "liq_asymmetry": 0.3, "obi": 0.2}
        f_noconfirm = {"liq_thinness": 0.3, "liq_asymmetry": 0.3, "obi": 0.0}
        l_confirm, _ = liq_calc(f_confirm)
        l_noconfirm, _ = liq_calc(f_noconfirm)
        assert l_confirm > l_noconfirm

    def test_output_bounded_0_1(self):
        f = {"liq_thinness": 100.0, "liq_asymmetry": 1.0, "obi": 0.9}
        l, s = liq_calc(f)
        assert 0.0 <= l <= 1.0
        assert 0.0 <= s <= 1.0


# ── MVR ──────────────────────────────────────────────────────────

class TestMVR:
    def test_breakout_positive_momentum_long(self):
        f = {"mvr": 3.0, "momentum": 5.0, "ema_fast": 50010, "ema_slow": 50000}
        l, s = mvr_calc(f)
        assert l > 0
        assert s == 0.0

    def test_breakout_negative_momentum_short(self):
        f = {"mvr": 3.0, "momentum": -5.0, "ema_fast": 49990, "ema_slow": 50000}
        l, s = mvr_calc(f)
        assert s > 0
        assert l == 0.0

    def test_chop_positive_momentum_fades_to_short(self):
        """In chop (mvr < 0.5), positive momentum should mean-revert → short."""
        f = {"mvr": 0.2, "momentum": 5.0, "ema_fast": 50010, "ema_slow": 50000}
        l, s = mvr_calc(f)
        assert s > 0
        assert l == 0.0

    def test_chop_negative_momentum_fades_to_long(self):
        """In chop, negative momentum should mean-revert → long."""
        f = {"mvr": 0.2, "momentum": -5.0, "ema_fast": 49990, "ema_slow": 50000}
        l, s = mvr_calc(f)
        assert l > 0
        assert s == 0.0

    def test_neutral_no_signal(self):
        f = {"mvr": 1.0, "momentum": 5.0, "ema_fast": 50010, "ema_slow": 50000}
        l, s = mvr_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_zero_ema_slow_returns_zero(self):
        f = {"mvr": 3.0, "momentum": 5.0, "ema_fast": 100, "ema_slow": 0}
        l, s = mvr_calc(f)
        assert l == 0.0
        assert s == 0.0


# ── ABS (Absorption) ─────────────────────────────────────────────

class TestAbsorption:
    def test_bid_defense_bullish(self):
        """High absorption + flat price + bids absorbing → long."""
        f = {"absorption": 500.0, "mid_change_ticks": 0.2,
             "ice_absorb_bid": 100.0, "ice_absorb_ask": 20.0,
             "buy_sell_ratio": 0.5}
        l, s = absorb_calc(f)
        assert l > 0
        assert l > s

    def test_ask_defense_bearish(self):
        """High absorption + flat price + asks absorbing → short."""
        f = {"absorption": 500.0, "mid_change_ticks": 0.2,
             "ice_absorb_bid": 20.0, "ice_absorb_ask": 100.0,
             "buy_sell_ratio": 0.5}
        l, s = absorb_calc(f)
        assert s > 0
        assert s > l

    def test_volatile_price_gate(self):
        """Price moving >= 1 tick → gate kills the signal."""
        f = {"absorption": 500.0, "mid_change_ticks": 1.5,
             "ice_absorb_bid": 100.0, "ice_absorb_ask": 20.0,
             "buy_sell_ratio": 0.5}
        l, s = absorb_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_low_absorption_no_signal(self):
        """Very low absorption → nothing fires."""
        f = {"absorption": 0.1, "mid_change_ticks": 0.1,
             "ice_absorb_bid": 1.0, "ice_absorb_ask": 0.5,
             "buy_sell_ratio": 0.5}
        l, s = absorb_calc(f)
        assert l == 0.0
        assert s == 0.0

    def test_fallback_to_buy_sell_ratio(self):
        """When ice_absorb data is zero, falls back to buy_sell_ratio."""
        f = {"absorption": 500.0, "mid_change_ticks": 0.1,
             "ice_absorb_bid": 0.0, "ice_absorb_ask": 0.0,
             "buy_sell_ratio": 0.35}
        l, s = absorb_calc(f)
        # More sellers → bids absorbing → bullish
        assert l > 0

    def test_output_bounded_0_1(self):
        f = {"absorption": 99999.0, "mid_change_ticks": 0.0,
             "ice_absorb_bid": 1000.0, "ice_absorb_ask": 0.0,
             "buy_sell_ratio": 0.5}
        l, s = absorb_calc(f)
        assert 0.0 <= l <= 1.0
        assert 0.0 <= s <= 1.0
