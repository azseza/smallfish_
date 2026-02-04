"""Shared test fixtures."""
import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def config():
    return {
        "symbols": ["BTCUSDT"],
        "tick_sizes": {"BTCUSDT": 0.10, "ETHUSDT": 0.01},
        "min_qty": {"BTCUSDT": 0.001, "ETHUSDT": 0.01},
        "qty_step": {"BTCUSDT": 0.001, "ETHUSDT": 0.01},
        "leverage": 10,
        "initial_equity": 1000,
        "max_spread": 2,
        "risk_per_trade": 0.005,
        "max_risk_dollars": 5.0,
        "max_daily_R": 10,
        "base_stop_ticks": 3,
        "tp_ticks_multiplier": 1.5,
        "trail_step_ticks": 1,
        "kill_switch_slip_breaches": 3,
        "max_slippage_ticks": 2,
        "max_open_orders": 2,
        "cooldown_after_loss_ms": 5000,
        "drawdown_tiers": [
            {"threshold": 0.02, "multiplier": 1.0},
            {"threshold": 0.05, "multiplier": 0.5},
            {"threshold": 0.10, "multiplier": 0.25},
            {"threshold": 1.00, "multiplier": 0.0},
        ],
        "C_enter": 0.65,
        "C_trail": 0.85,
        "C_exit": 0.40,
        "deadband": 0.05,
        "alpha": 5,
        "entry_timeout_ms": 600,
        "reprice_trigger_ticks": 1,
        "ioc_pmin": 0.65,
        "max_reprice_attempts": 3,
        "weights": {
            "w": [0.30, 0.15, 0.20],
            "v": [0.12, 0.08, 0.05],
            "x": [0.07, 0.03],
        },
        "vol_regime": {
            "lookback_s": 60,
            "low_vol_threshold": 0.3,
            "high_vol_threshold": 1.2,
            "extreme_vol_threshold": 3.0,
        },
        "vwap": {"lookback_trades": 500, "deviation_ticks_threshold": 5},
        "whale": {
            "large_trade_multiplier": 5.0,
            "large_trade_min_usd": 50000,
            "burst_window_ms": 2000,
            "sweep_levels": 3,
            "ice_refill_count": 4,
            "ice_window_ms": 5000,
        },
        "prt": {
            "cancel_window_ms": 500,
            "cancel_rate_threshold": 0.6,
            "mid_change_max_ticks": 2,
        },
        "max_latency_ms": 80,
        "funding_window_minutes": 3,
        "adaptive": {
            "enabled": False,
            "lookback_trades": 100,
            "min_edge": 0.0001,
            "weight_floor": 0.5,
            "weight_ceiling": 2.0,
        },
    }


@pytest.fixture
def sample_book():
    from marketdata.book import OrderBook
    book = OrderBook(symbol="BTCUSDT", depth=10, tick_size=0.10)
    bids = [
        [50000.0, 1.5],
        [49999.9, 2.0],
        [49999.8, 3.0],
        [49999.7, 1.0],
        [49999.6, 0.5],
    ]
    asks = [
        [50000.1, 1.2],
        [50000.2, 2.5],
        [50000.3, 1.8],
        [50000.4, 0.8],
        [50000.5, 0.3],
    ]
    book.on_snapshot(bids, asks, seq=1)
    return book


@pytest.fixture
def sample_tape():
    from marketdata.tape import TradeTape
    from core.types import Trade, Side
    tape = TradeTape(capacity=1000)
    base_ts = 1700000000000
    for i in range(100):
        trade = Trade(
            trade_id=str(i),
            symbol="BTCUSDT",
            price=50000.0 + (i % 10) * 0.1 - 0.5,
            quantity=0.01 + (i % 5) * 0.005,
            side=Side.BUY if i % 3 != 0 else Side.SELL,
            timestamp=base_ts + i * 100,
        )
        tape.add_trade(trade)
    return tape
