"""Tests for order execution router and risk management."""
import pytest
from core.types import Side
from core.state import RuntimeState
from marketdata.book import OrderBook
import exec.risk as risk


class TestPositionSize:
    def test_basic_sizing(self, config):
        state = RuntimeState(config)
        state.equity = 1000
        book = OrderBook(symbol="BTCUSDT", depth=10, tick_size=0.10)
        book.on_snapshot(
            [[50000.0, 1.0], [49999.9, 1.0]],
            [[50000.1, 1.0], [50000.2, 1.0]],
        )
        # stop_distance in price terms: 3 ticks * $0.10 = $0.30
        qty = risk.position_size(state, book, 0.30, symbol="BTCUSDT")
        # risk = min(0.005 * 1000, 5.0) = $5.0
        # qty = 5.0 / 0.30 = 16.666 → rounded down to step
        assert qty > 0
        assert qty <= 200  # leverage cap

    def test_zero_stop_gives_zero(self, config):
        state = RuntimeState(config)
        book = OrderBook(symbol="BTCUSDT", depth=10, tick_size=0.10)
        book.on_snapshot([[50000.0, 1.0]], [[50000.1, 1.0]])
        qty = risk.position_size(state, book, 0.0, symbol="BTCUSDT")
        assert qty == 0.0

    def test_drawdown_reduces_size(self, config):
        state = RuntimeState(config)
        state.equity = 1000
        book = OrderBook(symbol="BTCUSDT", depth=10, tick_size=0.10)
        book.on_snapshot([[50000.0, 1.0]], [[50000.1, 1.0]])

        # stop_distance in price terms: 3 ticks * $0.10 = $0.30
        qty_full = risk.position_size(state, book, 0.30, "BTCUSDT")

        # Simulate 6% drawdown (0.25 multiplier tier)
        state.peak_equity = 1000
        state.equity = 940
        state.drawdown = 0.06

        qty_reduced = risk.position_size(state, book, 0.30, "BTCUSDT")
        assert qty_reduced < qty_full


class TestCanTrade:
    def test_allows_normal_conditions(self, config, sample_book):
        state = RuntimeState(config)
        state.latency_ms = 10
        allowed, _ = risk.can_trade(state, sample_book, "BTCUSDT")
        assert allowed

    def test_blocks_on_kill_switch(self, config, sample_book):
        state = RuntimeState(config)
        state.kill_switch = True
        state.kill_reason = "test"
        allowed, reason = risk.can_trade(state, sample_book, "BTCUSDT")
        assert not allowed
        assert "kill_switch" in reason

    def test_blocks_on_wide_spread(self, config):
        state = RuntimeState(config)
        state.latency_ms = 10
        book = OrderBook(symbol="BTCUSDT", depth=10, tick_size=0.10)
        book.on_snapshot([[49990.0, 1.0]], [[50010.0, 1.0]])  # 200 tick spread
        allowed, reason = risk.can_trade(state, book, "BTCUSDT")
        assert not allowed
        assert "spread" in reason

    def test_blocks_on_high_latency(self, config, sample_book):
        state = RuntimeState(config)
        state.latency_ms = 200  # > 80ms limit
        allowed, reason = risk.can_trade(state, sample_book, "BTCUSDT")
        assert not allowed
        assert "latency" in reason

    def test_blocks_existing_position(self, config, sample_book):
        state = RuntimeState(config)
        state.latency_ms = 10
        state.on_enter("BTCUSDT", Side.BUY, 50000.0, 0.01, 0.8, {})
        allowed, reason = risk.can_trade(state, sample_book, "BTCUSDT")
        assert not allowed
        assert "position" in reason

    def test_blocks_on_daily_loss(self, config, sample_book):
        state = RuntimeState(config)
        state.latency_ms = 10
        state.daily_loss_R = 11  # > max_daily_R=10
        allowed, reason = risk.can_trade(state, sample_book, "BTCUSDT")
        assert not allowed
        assert "daily_loss" in reason


class TestComputeStops:
    def test_long_stops(self, config, sample_book):
        stop, tp, stop_distance = risk.compute_stops(sample_book, 1, config, "BTCUSDT")
        mid = sample_book.mid_price()
        assert stop < mid
        assert tp > mid
        # Without profile, falls back to fixed-tick stops:
        # stop_distance = base_stop_ticks(3) * tick_size(0.10) = 0.30
        assert abs(stop_distance - 0.30) < 0.01

    def test_short_stops(self, config, sample_book):
        stop, tp, stop_distance = risk.compute_stops(sample_book, -1, config, "BTCUSDT")
        mid = sample_book.mid_price()
        assert stop > mid
        assert tp < mid

    def test_profile_volatility_adapted_stops(self, config, sample_book):
        """When profile is active and avg_range provided, uses volatility-adapted stops."""
        config["profile"] = {
            "name": "aggressive",
            "sl_range_mult": 0.50,
            "tp_range_mult": 1.60,
        }
        # avg_range of $100 → stop_distance = 100 * 0.50 = $50
        stop, tp, stop_distance = risk.compute_stops(
            sample_book, 1, config, "BTCUSDT", avg_range=100.0)
        mid = sample_book.mid_price()
        assert abs(stop_distance - 50.0) < 0.01
        assert stop < mid
        assert tp > mid
        assert abs(mid - stop - 50.0) < 0.01
        assert abs(tp - mid - 160.0) < 0.01


class TestSlippage:
    def test_no_slippage(self):
        slip, breach = risk.check_slippage(50000.0, 50000.0, 1, 0.10, 2)
        assert slip == 0.0
        assert not breach

    def test_adverse_slippage_detected(self):
        # Long: fill higher than expected = adverse
        slip, breach = risk.check_slippage(50000.0, 50000.5, 1, 0.10, 2)
        assert slip == 5.0
        assert breach

    def test_within_threshold(self):
        slip, breach = risk.check_slippage(50000.0, 50000.1, 1, 0.10, 2)
        assert abs(slip - 1.0) < 0.01
        assert not breach
