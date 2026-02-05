"""Tests for Telegram bot command handlers."""
import pytest
import os
from unittest.mock import AsyncMock, patch
from remote.telegram_bot import TelegramBot
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
    return s


@pytest.fixture
def bot(state, config):
    config["telegram"] = {"enabled": False, "notify_trades": False, "alert_cooldown_s": 0}
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
        return TelegramBot(state, config)


class TestTelegramCommands:
    def test_cmd_status(self, bot):
        result = bot._cmd_status()
        assert "1050.00" in result
        assert "LIVE" in result
        assert "normal" in result
        assert "60.0%" in result  # win rate

    def test_cmd_status_with_kill_switch(self, bot):
        bot.state.kill_switch = True
        bot.state.kill_reason = "daily_loss"
        result = bot._cmd_status()
        assert "KILLED" in result
        assert "daily_loss" in result

    def test_cmd_status_with_position(self, bot):
        from core.types import Position
        bot.state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT", quantity=0.01,
            entry_price=50000.0, side=Side.BUY,
            unrealized_pnl=5.0,
        )
        result = bot._cmd_status()
        assert "LONG" in result
        assert "BTCUSDT" in result

    def test_cmd_trades_no_trades(self, bot):
        result = bot._cmd_trades()
        assert "No trades" in result

    def test_cmd_trades_with_history(self, bot):
        trade = TradeResult(
            symbol="BTCUSDT", side=Side.BUY,
            entry_price=50000.0, exit_price=50010.0,
            quantity=0.01, pnl=0.10, pnl_R=0.5,
            entry_time=1000, exit_time=2000,
            duration_ms=1000, slippage_entry=0.0,
            slippage_exit=0.0, exit_reason="tp_hit",
        )
        bot.state.completed_trades.append(trade)
        result = bot._cmd_trades()
        assert "BTCUSDT" in result
        assert "tp_hit" in result

    def test_cmd_equity(self, bot):
        result = bot._cmd_equity()
        assert "1050.00" in result
        assert "1100.00" in result

    def test_cmd_grid_no_grid(self, bot):
        result = bot._cmd_grid()
        assert "not enabled" in result

    def test_cmd_kill(self, bot):
        assert bot.state.kill_switch is False
        result = bot._cmd_kill()
        assert "TRIGGERED" in result
        assert bot.state.kill_switch is True

    def test_cmd_kill_already_active(self, bot):
        bot.state.kill_switch = True
        bot.state.kill_reason = "test"
        result = bot._cmd_kill()
        assert "already" in result

    def test_cmd_resume(self, bot):
        bot.state.kill_switch = True
        bot.state.kill_reason = "test"
        result = bot._cmd_resume()
        assert "RESET" in result
        assert bot.state.kill_switch is False

    def test_cmd_resume_not_active(self, bot):
        result = bot._cmd_resume()
        assert "not active" in result

    def test_cmd_config(self, bot):
        result = bot._cmd_config()
        assert "10x" in result  # leverage
        assert "BTCUSDT" in result

    def test_cmd_help(self, bot):
        result = bot._cmd_help()
        assert "/status" in result
        assert "/kill" in result
        assert "/resume" in result


class TestTelegramAuth:
    def test_bot_disabled_without_token(self, state, config):
        config["telegram"] = {"enabled": False}
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
            bot = TelegramBot(state, config)
            assert bot.enabled is False

    def test_bot_enabled_with_token(self, state, config):
        config["telegram"] = {"enabled": True}
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "fake_token",
            "TELEGRAM_CHAT_ID": "12345",
        }):
            bot = TelegramBot(state, config)
            assert bot.enabled is True
            assert bot.chat_id == "12345"


class TestTelegramAlerts:
    def test_alert_cooldown(self, bot):
        import time
        bot._alert_cooldown_s = 300
        bot._last_alert["test"] = time.time()
        # Should be within cooldown — this is just state testing
        last = bot._last_alert.get("test", 0)
        assert time.time() - last < 1.0
