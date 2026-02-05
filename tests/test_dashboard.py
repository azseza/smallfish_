"""Tests for rich terminal dashboard."""
import pytest
import time
from unittest.mock import patch

from rich.panel import Panel

from monitor.dashboard import TerminalDashboard
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
        assert len(dashboard._candles) == 0
        assert len(dashboard._equity_history) == 0

    def test_disabled_by_default(self, state, config):
        config["dashboard"] = {"enabled": False}
        d = TerminalDashboard(state, config)
        assert d.enabled is False


class TestPriceData:
    def test_add_price_point_creates_candle(self, dashboard):
        dashboard.add_price_point(50000.0)
        dashboard.add_price_point(50001.0)
        assert dashboard._current_candle is not None
        assert dashboard._current_candle.close == 50001.0
        assert dashboard._current_candle.volume == 2

    def test_candle_ohlc_correct(self, dashboard):
        dashboard.add_price_point(100.0)
        dashboard.add_price_point(105.0)
        dashboard.add_price_point(98.0)
        dashboard.add_price_point(102.0)
        c = dashboard._current_candle
        assert c.open == 100.0
        assert c.high == 105.0
        assert c.low == 98.0
        assert c.close == 102.0

    def test_add_order_marker_buy(self, dashboard):
        dashboard.add_price_point(50000.0)
        dashboard.add_order_marker("BUY", 50000.0)
        assert len(dashboard._buy_markers) == 1
        assert dashboard._buy_markers[0][1] == 50000.0

    def test_add_order_marker_sell(self, dashboard):
        dashboard.add_price_point(50000.0)
        dashboard.add_order_marker("SELL", 49999.0)
        assert len(dashboard._sell_markers) == 1
        assert dashboard._sell_markers[0][1] == 49999.0

    def test_order_markers_bounded(self, dashboard):
        for i in range(150):
            dashboard.add_order_marker("BUY", 50000.0 + i)
        assert len(dashboard._buy_markers) <= 100


class TestEquitySnapshot:
    def test_record_snapshot(self, dashboard):
        dashboard._record_snapshot()
        assert len(dashboard._equity_history) == 1
        assert dashboard._equity_history[0] == 1050.0

    def test_equity_history_bounded(self, dashboard):
        for _ in range(250):
            dashboard._record_snapshot()
        assert len(dashboard._equity_history) <= 200


class TestStatusPanel:
    def test_returns_panel(self, dashboard):
        panel = dashboard._build_status_panel()
        assert isinstance(panel, Panel)
        assert "SMALLFISH" in panel.title


class TestStatsPanel:
    def test_returns_panel(self, dashboard):
        panel = dashboard._build_stats_panel()
        assert isinstance(panel, Panel)
        assert "METRICS" in panel.title

    def test_panel_shows_metrics(self, dashboard, config):
        config["profile"] = {"name": "aggressive"}
        panel = dashboard._build_stats_panel()
        assert isinstance(panel, Panel)


class TestTradesPanel:
    def test_returns_panel_empty(self, dashboard):
        panel = dashboard._build_trades_panel()
        assert isinstance(panel, Panel)

    def test_returns_panel_with_trades(self, dashboard, state):
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
        panel = dashboard._build_trades_panel()
        assert isinstance(panel, Panel)


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
        label, pnl, wins, losses = result[-1]
        assert wins + losses == 5


class TestCandlePanel:
    def test_returns_panel_waiting(self, dashboard):
        panel = dashboard._build_candle_panel()
        assert isinstance(panel, Panel)

    def test_returns_panel_with_data(self, dashboard):
        # Simulate enough ticks for at least 2 candles
        from monitor.dashboard import OHLCCandle
        dashboard._candles.append(
            OHLCCandle(ts=1000, open=100, high=105, low=98, close=102, volume=10))
        dashboard._candles.append(
            OHLCCandle(ts=1060, open=102, high=108, low=101, close=106, volume=15))
        panel = dashboard._build_candle_panel()
        assert isinstance(panel, Panel)


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
