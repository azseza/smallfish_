"""Tests for multigrid strategy."""
import pytest
from strategies.multigrid import MultigridStrategy, GridLevel, GridState
from core.types import Side


@pytest.fixture
def grid_config(config):
    config["multigrid"] = {
        "enabled": True,
        "levels": 3,
        "spacing_pct": 0.001,
        "qty_per_level_pct": 0.01,
        "max_open_levels": 6,
        "bias_strength": 0.5,
        "recalc_interval_ms": 1000,
        "harvest_pct": 0.001,
        "regime_filter": True,
    }
    return config


@pytest.fixture
def grid(grid_config):
    g = MultigridStrategy(grid_config)
    g.init_symbol("BTCUSDT")
    return g


class TestMultigridInit:
    def test_init_loads_config(self, grid_config):
        g = MultigridStrategy(grid_config)
        assert g.enabled is True
        assert g.num_levels == 3
        assert g.spacing_pct == 0.001

    def test_init_disabled_by_default(self, config):
        g = MultigridStrategy(config)
        assert g.enabled is False

    def test_init_symbol_creates_state(self, grid):
        assert "BTCUSDT" in grid.grids
        assert isinstance(grid.grids["BTCUSDT"], GridState)

    def test_init_symbol_idempotent(self, grid):
        grid.grids["BTCUSDT"].total_pnl = 42.0
        grid.init_symbol("BTCUSDT")
        # Should not overwrite existing state
        assert grid.grids["BTCUSDT"].total_pnl == 42.0


class TestMultigridRecalculate:
    def test_recalculate_creates_levels(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        gs = grid.grids["BTCUSDT"]
        assert gs.active is True
        # 3 levels per side = 6 total
        assert len(gs.levels) == 6

    def test_recalculate_symmetric_sides(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        gs = grid.grids["BTCUSDT"]
        buys = [lv for lv in gs.levels if lv.side == Side.BUY]
        sells = [lv for lv in gs.levels if lv.side == Side.SELL]
        assert len(buys) == 3
        assert len(sells) == 3

    def test_buy_levels_below_center(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.0, 100000.0, "normal", 10000)
        gs = grid.grids["BTCUSDT"]
        for lv in gs.levels:
            if lv.side == Side.BUY:
                assert lv.price < 50000.0

    def test_sell_levels_above_center(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.0, 100000.0, "normal", 10000)
        gs = grid.grids["BTCUSDT"]
        for lv in gs.levels:
            if lv.side == Side.SELL:
                assert lv.price > 50000.0

    def test_bias_shifts_center(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 1, 0.8, 100000.0, "normal", 10000)
        gs_long = grid.grids["BTCUSDT"]
        center_long = gs_long.center

        grid.recalculate("BTCUSDT", 50000.0, -1, 0.8, 100000.0, "normal", 20000)
        gs_short = grid.grids["BTCUSDT"]
        center_short = gs_short.center

        # Long bias shifts center down, short shifts up
        assert center_long < center_short

    def test_extreme_vol_disables_grid(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "extreme", 10000)
        gs = grid.grids["BTCUSDT"]
        assert gs.active is False
        assert len(gs.levels) == 0

    def test_high_vol_widens_spacing(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        normal_levels = list(grid.grids["BTCUSDT"].levels)

        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "high", 20000)
        high_levels = list(grid.grids["BTCUSDT"].levels)

        # High vol should have wider spacing between buy levels
        normal_buys = sorted([lv.price for lv in normal_levels if lv.side == Side.BUY])
        high_buys = sorted([lv.price for lv in high_levels if lv.side == Side.BUY])
        if len(normal_buys) >= 2 and len(high_buys) >= 2:
            normal_gap = normal_buys[1] - normal_buys[0]
            high_gap = high_buys[1] - high_buys[0]
            assert abs(high_gap) > abs(normal_gap)

    def test_cooldown_prevents_recalc(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        first_center = grid.grids["BTCUSDT"].center

        # Within cooldown — should not recalculate
        grid.recalculate("BTCUSDT", 51000.0, 0, 0.5, 100000.0, "normal", 10500)
        assert grid.grids["BTCUSDT"].center == first_center

    def test_small_equity_disables_grid(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 0.001, "normal", 10000)
        assert grid.grids["BTCUSDT"].active is False


class TestMultigridOrders:
    def test_pending_orders_returns_unfilled(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        pending = grid.pending_orders("BTCUSDT")
        assert len(pending) > 0
        for lv in pending:
            assert not lv.is_filled
            assert lv.order_id == ""

    def test_pending_orders_empty_when_inactive(self, grid):
        assert grid.pending_orders("BTCUSDT") == []

    def test_pending_orders_empty_for_unknown_symbol(self, grid):
        assert grid.pending_orders("UNKNOWN") == []

    def test_mark_order_sets_order_id(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        pending = grid.pending_orders("BTCUSDT")
        lv = pending[0]
        grid.mark_order("BTCUSDT", lv.price, lv.side, "order123")

        found = False
        for l in grid.grids["BTCUSDT"].levels:
            if l.order_id == "order123":
                found = True
        assert found


class TestMultigridFills:
    def test_on_fill_records_fill(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        pending = grid.pending_orders("BTCUSDT")
        lv = pending[0]
        grid.mark_order("BTCUSDT", lv.price, lv.side, "order456")

        result = grid.on_fill("BTCUSDT", lv.price, lv.side, "order456")
        assert result is not None
        assert result.is_filled is True
        assert result.fill_price == lv.price

    def test_on_fill_unknown_order(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        result = grid.on_fill("BTCUSDT", 50000.0, Side.BUY, "nonexistent")
        assert result is None

    def test_on_harvest_calculates_pnl(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        gs = grid.grids["BTCUSDT"]

        # Manually set up a filled buy level
        buy_level = None
        for lv in gs.levels:
            if lv.side == Side.BUY:
                buy_level = lv
                break
        assert buy_level is not None

        buy_level.is_filled = True
        buy_level.fill_price = 49950.0
        buy_level.quantity = 0.001

        # Harvest at higher price
        pnl = grid.on_harvest("BTCUSDT", buy_level, 49960.0)
        assert pnl > 0
        assert gs.round_trips == 1
        assert gs.total_pnl > 0
        # Level should be reset
        assert buy_level.is_filled is False

    def test_on_harvest_short_pnl(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        gs = grid.grids["BTCUSDT"]

        sell_level = None
        for lv in gs.levels:
            if lv.side == Side.SELL:
                sell_level = lv
                break
        assert sell_level is not None

        sell_level.is_filled = True
        sell_level.fill_price = 50050.0
        sell_level.quantity = 0.001

        pnl = grid.on_harvest("BTCUSDT", sell_level, 50040.0)
        assert pnl > 0


class TestMultigridStatus:
    def test_status_inactive(self, grid):
        status = grid.status("BTCUSDT")
        assert status["active"] is False

    def test_status_active(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        status = grid.status("BTCUSDT")
        assert status["active"] is True
        assert status["buy_levels"] == 3
        assert status["sell_levels"] == 3
        assert status["total_pnl"] == 0.0
        assert status["round_trips"] == 0

    def test_status_unknown_symbol(self, grid):
        status = grid.status("UNKNOWN")
        assert status["active"] is False

    def test_all_status(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        all_s = grid.all_status()
        assert "BTCUSDT" in all_s

    def test_cancel_all_returns_order_ids(self, grid):
        grid.recalculate("BTCUSDT", 50000.0, 0, 0.5, 100000.0, "normal", 10000)
        pending = grid.pending_orders("BTCUSDT")
        for i, lv in enumerate(pending[:3]):
            grid.mark_order("BTCUSDT", lv.price, lv.side, f"order_{i}")

        cancel_ids = grid.cancel_all("BTCUSDT")
        assert len(cancel_ids) >= 1
        assert all(oid.startswith("order_") for oid in cancel_ids)
