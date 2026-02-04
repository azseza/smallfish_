"""Tests for feature extraction from market data."""
import pytest
from marketdata.features import obi, umom, prt, whales, vwap_features, vol_regime, compute_all
from core.types import Trade, Side


class TestObiFeatures:
    def test_returns_required_keys(self, sample_book):
        result = obi(sample_book)
        assert "obi" in result
        assert "d_obi_dt" in result
        assert "spread_penalty" in result
        assert "spread_ticks" in result

    def test_obi_positive_when_bids_heavy(self, sample_book):
        # Default fixture has more bid volume
        result = obi(sample_book)
        assert result["obi"] > 0  # bids heavier

    def test_spread_ticks(self, sample_book):
        result = obi(sample_book)
        # spread = 50000.1 - 50000.0 = 0.1, tick_size = 0.10
        assert abs(result["spread_ticks"] - 1.0) < 0.01

    def test_micro_pressure(self, sample_book):
        result = obi(sample_book)
        assert "micro_pressure" in result


class TestUmomFeatures:
    def test_returns_required_keys(self, sample_tape, sample_book, config):
        result = umom(sample_tape, sample_book)
        assert "ema_fast" in result
        assert "ema_slow" in result
        assert "momentum" in result
        assert "rv_10s" in result
        assert "trade_rate" in result
        assert "buy_sell_ratio" in result

    def test_ema_values_nonzero_with_data(self, sample_tape, sample_book):
        result = umom(sample_tape, sample_book)
        assert result["ema_fast"] > 0
        assert result["ema_slow"] > 0


class TestPrtFeatures:
    def test_returns_required_keys(self, sample_book, config):
        result = prt(sample_book, config)
        assert "cancel_rate_ask" in result
        assert "cancel_rate_bid" in result
        assert "mid_change_ticks" in result


class TestWhaleFeatures:
    def test_returns_required_keys(self, sample_tape, sample_book, config):
        result = whales(sample_tape, sample_book, config)
        assert "buy_burst" in result
        assert "sell_burst" in result
        assert "sweep_up" in result
        assert "sweep_down" in result
        assert "ice_absorb" in result


class TestVwapFeatures:
    def test_returns_required_keys(self, sample_tape, sample_book, config):
        result = vwap_features(sample_tape, sample_book, config)
        assert "vwap" in result
        assert "vwap_deviation_ticks" in result

    def test_vwap_positive(self, sample_tape, sample_book, config):
        result = vwap_features(sample_tape, sample_book, config)
        assert result["vwap"] > 0


class TestVolRegime:
    def test_returns_required_keys(self, sample_tape, config):
        result = vol_regime(sample_tape, config)
        assert "rv_regime" in result
        assert "vol_regime" in result
        assert "vol_regime_name" in result

    def test_regime_name_valid(self, sample_tape, config):
        result = vol_regime(sample_tape, config)
        assert result["vol_regime_name"] in ("low", "normal", "high", "extreme")


class TestComputeAll:
    def test_computes_all_features(self, sample_book, sample_tape, config):
        result = compute_all(sample_book, sample_tape, config)
        # Should have features from all modules
        assert "obi" in result
        assert "ema_fast" in result
        assert "cancel_rate_ask" in result
        assert "buy_burst" in result
        assert "vwap" in result
        assert "vol_regime" in result
