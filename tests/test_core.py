"""Tests for core utilities, ringbuffer, and state management."""
import pytest
from core.utils import (
    sigmoid, clamp, ema_update, ema_alpha, ewma_series,
    realized_vol, z_score, tick_round, qty_round, safe_div,
)
from core.ringbuffer import RingBuffer
from core.state import RuntimeState
from core.types import Side, TradeResult


class TestUtils:
    def test_sigmoid_at_zero(self):
        assert sigmoid(0) == 0.5

    def test_sigmoid_positive(self):
        assert sigmoid(5) > 0.99

    def test_sigmoid_negative(self):
        assert sigmoid(-5) < 0.01

    def test_sigmoid_bounded(self):
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            assert 0.0 <= sigmoid(x) <= 1.0

    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_ema_update(self):
        result = ema_update(10.0, 12.0, 0.5)
        assert result == 11.0

    def test_ewma_series_single(self):
        assert ewma_series([5.0], 3) == 5.0

    def test_ewma_series_multiple(self):
        result = ewma_series([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert 3.0 < result < 5.0  # should be biased toward recent

    def test_realized_vol_constant_price(self):
        prices = [100.0] * 10
        assert realized_vol(prices) == 0.0

    def test_realized_vol_positive_for_varying(self):
        prices = [100.0, 101.0, 99.0, 100.5, 98.0, 102.0]
        assert realized_vol(prices) > 0

    def test_z_score(self):
        assert z_score(10, 10, 1) == 0.0
        assert z_score(11, 10, 1) == 1.0
        assert z_score(10, 10, 0) == 0.0  # zero std

    def test_tick_round(self):
        assert tick_round(50000.13, 0.10) == 50000.1
        assert tick_round(50000.06, 0.10) == 50000.1  # rounds to nearest
        assert tick_round(100.0, 0) == 100.0

    def test_qty_round(self):
        assert qty_round(1.567, 0.01) == 1.56  # floor
        assert qty_round(1.999, 0.01) == 1.99

    def test_safe_div(self):
        assert safe_div(10, 2) == 5.0
        assert safe_div(10, 0) == 0.0
        assert safe_div(10, 0, default=-1.0) == -1.0


class TestRingBuffer:
    def test_append_and_len(self):
        rb = RingBuffer(5)
        rb.append(1)
        rb.append(2)
        rb.append(3)
        assert len(rb) == 3

    def test_overflow(self):
        rb = RingBuffer(3)
        for i in range(5):
            rb.append(i)
        assert len(rb) == 3
        assert rb.get() == [2, 3, 4]

    def test_last(self):
        rb = RingBuffer(10)
        for i in range(5):
            rb.append(i)
        last2 = rb.last(2)
        assert last2 == [4, 3]

    def test_peek(self):
        rb = RingBuffer(5)
        rb.append(42)
        assert rb.peek() == 42

    def test_peek_empty(self):
        rb = RingBuffer(5)
        assert rb.peek() is None

    def test_filter(self):
        rb = RingBuffer(10)
        for i in range(10):
            rb.append(i)
        evens = rb.filter(lambda x: x % 2 == 0)
        assert evens == [0, 2, 4, 6, 8]

    def test_iteration(self):
        rb = RingBuffer(5)
        rb.extend([1, 2, 3])
        assert list(rb) == [1, 2, 3]


class TestRuntimeState:
    def test_initial_state(self, config):
        state = RuntimeState(config)
        assert state.equity == 1000
        assert state.kill_switch is False
        assert state.no_position("BTCUSDT")

    def test_enter_and_exit(self, config):
        state = RuntimeState(config)
        state.on_enter("BTCUSDT", Side.BUY, 50000.0, 0.01, 0.8, {}, 49997.0, 50004.5)
        assert state.has_position("BTCUSDT")
        assert not state.no_position("BTCUSDT")

        result = state.on_exit("BTCUSDT", 50004.0, "tp_hit")
        assert result is not None
        assert result.pnl > 0
        assert state.no_position("BTCUSDT")
        assert state.win_count == 1

    def test_losing_trade_updates_daily_loss(self, config):
        state = RuntimeState(config)
        state.on_enter("BTCUSDT", Side.BUY, 50000.0, 0.01, 0.8, {})
        result = state.on_exit("BTCUSDT", 49990.0, "sl_hit")
        assert result.pnl < 0
        assert state.daily_loss_R > 0
        assert state.loss_count == 1

    def test_kill_switch(self, config):
        state = RuntimeState(config)
        state.trigger_kill_switch("test")
        assert state.kill_switch is True
        assert state.kill_reason == "test"

    def test_drawdown_tiers(self, config):
        state = RuntimeState(config)
        state.drawdown = 0.01  # 1% DD — below first tier
        assert state.size_multiplier() == 1.0

        state.drawdown = 0.03  # 3% DD — in [0.02, 0.05) tier
        assert state.size_multiplier() == 1.0

        state.drawdown = 0.06  # 6% DD — in [0.05, 0.10) tier
        assert state.size_multiplier() == 0.5

        state.drawdown = 0.11  # 11% DD — in [0.10, 1.0) tier
        assert state.size_multiplier() == 0.25

        state.drawdown = 1.0  # 100% DD — final tier
        assert state.size_multiplier() == 0.0

    def test_daily_reset(self, config):
        state = RuntimeState(config)
        state.daily_pnl = 100
        state.trade_count = 50
        state.kill_switch = True
        state.reset_daily()
        assert state.daily_pnl == 0
        assert state.trade_count == 0
        assert state.kill_switch is False

    def test_funding_window(self, config):
        state = RuntimeState(config)
        # This test is time-dependent; just ensure it doesn't crash
        result = state.in_funding_window()
        assert isinstance(result, bool)

    def test_summary(self, config):
        state = RuntimeState(config)
        summary = state.summary()
        assert "equity" in summary
        assert "win_rate" in summary
        assert "kill_switch" in summary

    def test_rolling_win_rate_not_enough_trades(self, config):
        """With fewer than 10 trades, rolling_win_rate returns 1.0 (no restriction)."""
        state = RuntimeState(config)
        # Add 9 losing trades (below minimum 10 needed for WR estimate)
        for i in range(9):
            state.completed_trades.append(TradeResult(
                symbol="BTCUSDT", side=Side.BUY,
                entry_price=50000.0, exit_price=49990.0,
                quantity=0.01, pnl=-0.10, pnl_R=-1.0,
                entry_time=0, exit_time=1000,
                duration_ms=1000, slippage_entry=0, slippage_exit=0,
                signals_at_entry={}, exit_reason="sl_hit",
            ))
        # Even 9 consecutive losses should NOT trigger WR gate
        assert state.rolling_win_rate(20) == 1.0

    def test_rolling_win_rate_all_wins(self, config):
        state = RuntimeState(config)
        for i in range(20):
            state.completed_trades.append(TradeResult(
                symbol="BTCUSDT", side=Side.BUY,
                entry_price=50000.0, exit_price=50010.0,
                quantity=0.01, pnl=0.10, pnl_R=1.0,
                entry_time=0, exit_time=1000,
                duration_ms=1000, slippage_entry=0, slippage_exit=0,
                signals_at_entry={}, exit_reason="tp_hit",
            ))
        assert state.rolling_win_rate(20) == 1.0

    def test_rolling_win_rate_all_losses(self, config):
        state = RuntimeState(config)
        for i in range(20):
            state.completed_trades.append(TradeResult(
                symbol="BTCUSDT", side=Side.BUY,
                entry_price=50000.0, exit_price=49990.0,
                quantity=0.01, pnl=-0.10, pnl_R=-1.0,
                entry_time=0, exit_time=1000,
                duration_ms=1000, slippage_entry=0, slippage_exit=0,
                signals_at_entry={}, exit_reason="sl_hit",
            ))
        assert state.rolling_win_rate(20) == 0.0

    def test_rolling_win_rate_mixed(self, config):
        state = RuntimeState(config)
        # 6 wins, 14 losses = 30% WR
        for i in range(20):
            pnl = 0.10 if i < 6 else -0.10
            state.completed_trades.append(TradeResult(
                symbol="BTCUSDT", side=Side.BUY,
                entry_price=50000.0,
                exit_price=50010.0 if pnl > 0 else 49990.0,
                quantity=0.01, pnl=pnl, pnl_R=1.0 if pnl > 0 else -1.0,
                entry_time=0, exit_time=1000,
                duration_ms=1000, slippage_entry=0, slippage_exit=0,
                signals_at_entry={}, exit_reason="tp_hit",
            ))
        assert state.rolling_win_rate(20) == pytest.approx(0.30)
