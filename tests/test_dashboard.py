"""Tests for plotext terminal dashboard."""
import pytest
import time
from unittest.mock import patch

from monitor.dashboard import TerminalDashboard, _ansi_len, _ansi_pad, _pnl_col
from core.state import RuntimeState
from core.types import Side, TradeResult


@pytest.fixture
def state(config):
    s = RuntimeState(config)
    s.equity = 1050.0
    s.peak_equity = 1100.0
    s.daily_start_equity = 1000.0
    s.realized_pnl = 50.0
    s.daily_pnl = 12.5
    s.trade_count = 10
    s.win_count = 6
    s.loss_count = 4
    s.vol_regime = "normal"
    s.latency_ms = 15
    s.last_scores = {
        "obi": 0.3, "prt": -0.1, "umom": 0.5,
        "ltb": 0.0, "sweep": 0.0, "ice": 0.0,
        "vwap_dev": 0.2, "vol_regime": 0.1,
    }
    s.last_direction = 1
    s.last_confidence = 0.72
    return s


@pytest.fixture
def dashboard(state, config):
    config["dashboard"] = {"enabled": True}
    return TerminalDashboard(state, config, refresh_s=1.0)


class TestDashboardInit:
    def test_creates_with_defaults(self, dashboard):
        assert dashboard.enabled is True
        assert dashboard.refresh_s == 1.0
        assert len(dashboard._prices) == 0
        assert len(dashboard._equity_history) == 0

    def test_disabled_by_default(self, state, config):
        config["dashboard"] = {"enabled": False}
        d = TerminalDashboard(state, config)
        assert d.enabled is False


class TestPriceData:
    def test_add_price_point(self, dashboard):
        dashboard.add_price_point(50000.0)
        dashboard.add_price_point(50001.0)
        assert len(dashboard._prices) == 2
        assert dashboard._prices[-1] == 50001.0

    def test_price_buffer_bounded(self, dashboard):
        for i in range(400):
            dashboard.add_price_point(50000.0 + i)
        assert len(dashboard._prices) <= 300

    def test_add_order_marker_buy(self, dashboard):
        dashboard.add_price_point(50000.0)
        dashboard.add_order_marker("BUY", 50000.0)
        assert len(dashboard._buy_markers_x) == 1
        assert dashboard._buy_markers_y[0] == 50000.0

    def test_add_order_marker_sell(self, dashboard):
        dashboard.add_price_point(50000.0)
        dashboard.add_order_marker("SELL", 49999.0)
        assert len(dashboard._sell_markers_x) == 1
        assert dashboard._sell_markers_y[0] == 49999.0

    def test_order_markers_bounded(self, dashboard):
        for i in range(250):
            dashboard.add_order_marker("BUY", 50000.0 + i)
        assert len(dashboard._buy_markers_x) <= 200


class TestEquitySnapshot:
    def test_record_snapshot(self, dashboard):
        dashboard._record_snapshot()
        assert len(dashboard._equity_history) == 1
        assert dashboard._equity_history[0] == 1050.0

    def test_equity_history_bounded(self, dashboard):
        for _ in range(250):
            dashboard._record_snapshot()
        assert len(dashboard._equity_history) <= 200


class TestAnsiHelpers:
    def test_ansi_len_plain(self):
        assert _ansi_len("hello") == 5

    def test_ansi_len_with_codes(self):
        colored = "\033[32mhello\033[0m"
        assert _ansi_len(colored) == 5

    def test_ansi_pad_plain(self):
        assert _ansi_pad("hi", 5) == "hi   "

    def test_ansi_pad_colored(self):
        colored = "\033[31mhi\033[0m"
        padded = _ansi_pad(colored, 5)
        assert _ansi_len(padded) == 5

    def test_ansi_pad_already_wide(self):
        result = _ansi_pad("hello", 3)
        assert result == "hello"

    def test_pnl_col_positive(self):
        result = _pnl_col(1.0, "+$1.00")
        assert "\033[32m" in result  # green

    def test_pnl_col_negative(self):
        result = _pnl_col(-1.0, "-$1.00")
        assert "\033[31m" in result  # red


class TestLeftPanel:
    def test_build_left_lines_returns_correct_height(self, dashboard):
        lines = dashboard._build_left_lines(40, 30)
        assert len(lines) == 30

    def test_build_left_lines_correct_width(self, dashboard):
        lines = dashboard._build_left_lines(40, 20)
        for line in lines:
            assert _ansi_len(line) == 40

    def test_left_panel_shows_equity(self, dashboard):
        lines = dashboard._build_left_lines(40, 30)
        joined = "\n".join(lines)
        assert "1050.00" in joined

    def test_left_panel_shows_profile(self, dashboard, config):
        config["profile"] = {"name": "aggressive"}
        lines = dashboard._build_left_lines(40, 30)
        joined = "\n".join(lines)
        assert "AGGRESSIVE" in joined

    def test_left_panel_shows_trades(self, dashboard, state):
        now_ms = int(time.time() * 1000)
        trade = TradeResult(
            symbol="BTCUSDT", side=Side.BUY,
            entry_price=50000.0, exit_price=50010.0,
            quantity=0.001, pnl=0.01, pnl_R=0.5,
            entry_time=now_ms - 1000, exit_time=now_ms,
            duration_ms=1000, slippage_entry=0.0,
            slippage_exit=0.0, exit_reason="tp",
        )
        state.completed_trades.append(trade)
        lines = dashboard._build_left_lines(40, 40)
        joined = "\n".join(lines)
        assert "BTCUSD" in joined


class TestSevenDayPerf:
    def test_calc_7d_empty(self, dashboard):
        result = dashboard._calc_7d()
        assert result == []

    def test_calc_7d_with_trades(self, dashboard, state):
        now_ms = int(time.time() * 1000)
        for i in range(5):
            trade = TradeResult(
                symbol="BTCUSDT", side=Side.BUY,
                entry_price=50000.0, exit_price=50010.0,
                quantity=0.001, pnl=0.5 if i % 2 == 0 else -0.3,
                pnl_R=0.5 if i % 2 == 0 else -0.3,
                entry_time=now_ms - 2000, exit_time=now_ms - 1000,
                duration_ms=1000, slippage_entry=0.0,
                slippage_exit=0.0, exit_reason="tp",
            )
            state.completed_trades.append(trade)
        result = dashboard._calc_7d()
        assert len(result) >= 1
        # Check today has trades
        label, pnl, wins, losses = result[-1]
        assert wins + losses == 5


class TestStaticRenderers:
    def test_render_backtest_trades(self):
        trades = [
            {"pnl": 0.5, "side": "BUY", "symbol": "BTCUSDT"},
            {"pnl": -0.3, "side": "SELL", "symbol": "ETHUSDT"},
        ]
        result = TerminalDashboard.render_backtest_trades(trades)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_backtest_trades_empty(self):
        result = TerminalDashboard.render_backtest_trades([])
        assert "no trades" in result

    def test_render_equity_curve(self):
        curve = [(1, 100.0), (2, 105.0), (3, 102.0), (4, 110.0)]
        result = TerminalDashboard.render_equity_curve(curve)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_equity_curve_empty(self):
        result = TerminalDashboard.render_equity_curve([])
        assert result == ""

    def test_render_signal_weights(self, config):
        result = TerminalDashboard.render_signal_weights(config)
        assert isinstance(result, str)
        assert len(result) > 0


class TestLifecycle:
    def test_start_when_disabled(self, state, config):
        config["dashboard"] = {"enabled": False}
        d = TerminalDashboard(state, config)
        d.start()
        assert d._task is None

    def test_stop(self, dashboard):
        dashboard._running = True
        dashboard.stop()
        assert dashboard._running is False
