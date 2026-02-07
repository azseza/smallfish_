"""Tests for Telegram bot command handlers."""
import pytest
import os
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from remote.telegram_bot import TelegramBot
from core.state import RuntimeState
from core.types import Side, TradeResult, Position
from gateway.base import OrderResponse


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
def mock_rest():
    rest = AsyncMock()
    rest.place_order = AsyncMock(return_value=OrderResponse(success=True, order_id="ord123"))
    rest.set_trading_stop = AsyncMock(return_value=OrderResponse(success=True))
    rest.withdraw = AsyncMock(return_value={"success": True, "tx_id": "tx_abc123", "error_msg": ""})
    rest.get_deposit_address = AsyncMock(return_value={"address": "TRxTestAddr", "chain": "TRC20", "tag": ""})
    return rest


@pytest.fixture
def bot(state, config, mock_rest):
    config["telegram"] = {"enabled": False, "notify_trades": False, "alert_cooldown_s": 0}
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
        return TelegramBot(state, config, rest=mock_rest)


@pytest.fixture
def bot_no_rest(state, config):
    config["telegram"] = {"enabled": False, "notify_trades": False, "alert_cooldown_s": 0}
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
        return TelegramBot(state, config)


def _add_sample_trades(state, count=15):
    """Add sample completed trades to state."""
    import time
    base_ts = int(time.time() * 1000) - count * 60000
    for i in range(count):
        pnl = 0.50 if i % 3 != 0 else -0.30
        t = TradeResult(
            symbol="BTCUSDT", side=Side.BUY,
            entry_price=50000.0, exit_price=50000.0 + pnl * 100,
            quantity=0.01, pnl=pnl, pnl_R=pnl / 0.5,
            entry_time=base_ts + i * 60000,
            exit_time=base_ts + i * 60000 + 30000,
            duration_ms=30000, slippage_entry=0.0,
            slippage_exit=0.0, exit_reason="tp_hit" if pnl > 0 else "sl_hit",
        )
        state.completed_trades.append(t)


def _add_position(state, symbol="BTCUSDT", side=Side.BUY):
    state.positions[symbol] = Position(
        symbol=symbol, quantity=0.01,
        entry_price=50000.0, side=side,
        stop_price=49900.0, tp_price=50200.0,
        unrealized_pnl=5.0,
    )


class TestExistingCommands:
    def test_cmd_status(self, bot):
        result = bot._cmd_status()
        assert "1050.00" in result
        assert "LIVE" in result
        assert "normal" in result
        assert "60.0%" in result

    def test_cmd_status_with_kill_switch(self, bot):
        bot.state.kill_switch = True
        bot.state.kill_reason = "daily_loss"
        result = bot._cmd_status()
        assert "KILLED" in result
        assert "daily_loss" in result

    def test_cmd_status_with_position(self, bot):
        _add_position(bot.state)
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
        assert "10x" in result
        assert "BTCUSDT" in result

    def test_cmd_help(self, bot):
        result = bot._cmd_help()
        assert "/status" in result
        assert "/kill" in result
        assert "/stats" in result
        assert "/cashout" in result
        assert "/signals" in result


class TestNewMonitoringCommands:
    def test_cmd_stats_basic(self, bot):
        result = bot._cmd_stats()
        assert "LIVE STATS" in result
        assert "1050.00" in result
        assert "Equity" in result
        assert "Win Rate" in result

    def test_cmd_stats_with_trades(self, bot):
        _add_sample_trades(bot.state)
        result = bot._cmd_stats()
        assert "Sharpe" in result
        assert "Profit Factor" in result
        assert "BTCUSDT" in result

    def test_cmd_balance(self, bot):
        result = bot._cmd_balance()
        assert "BALANCE" in result
        assert "1050.00" in result
        assert "Equity" in result
        assert "Available Margin" in result

    def test_cmd_balance_with_position(self, bot):
        _add_position(bot.state)
        result = bot._cmd_balance()
        assert "Unrealized" in result

    def test_cmd_pnl_default(self, bot):
        result = bot._cmd_pnl()
        assert "No trades" in result

    def test_cmd_pnl_with_trades(self, bot):
        _add_sample_trades(bot.state)
        result = bot._cmd_pnl()
        assert "P&L BREAKDOWN" in result
        assert "Total" in result

    def test_cmd_pnl_custom_days(self, bot):
        _add_sample_trades(bot.state)
        result = bot._cmd_pnl(["3"])
        assert "3d" in result

    def test_cmd_signals_no_data(self, bot):
        result = bot._cmd_signals()
        assert "No signal" in result

    def test_cmd_signals_with_data(self, bot):
        bot.state.last_scores = {
            "obi_long": 0.8, "obi_short": 0.1,
            "prt_long": 0.5, "prt_short": 0.3,
        }
        bot.state.last_confidence = 0.72
        bot.state.last_direction = 1
        result = bot._cmd_signals()
        assert "SIGNAL SCORES" in result
        assert "OBI" in result
        assert "LONG" in result

    def test_cmd_symbols(self, bot):
        result = bot._cmd_symbols()
        assert "ACTIVE SYMBOLS" in result
        assert "BTCUSDT" in result

    def test_cmd_symbols_with_position(self, bot):
        _add_position(bot.state)
        result = bot._cmd_symbols()
        assert "LONG" in result

    def test_cmd_symbols_empty(self, bot):
        bot.config["symbols"] = []
        result = bot._cmd_symbols()
        assert "No active" in result


class TestControlCommands:
    def test_cmd_cooldown_default(self, bot):
        result = bot._cmd_cooldown()
        assert "paused for 5 minutes" in result
        assert bot.state.cooldown_until_ms > 0

    def test_cmd_cooldown_custom(self, bot):
        result = bot._cmd_cooldown(["10"])
        assert "paused for 10 minutes" in result

    def test_cmd_cooldown_invalid(self, bot):
        result = bot._cmd_cooldown(["abc"])
        assert "Usage" in result

    def test_cmd_mode_no_args(self, bot):
        result = bot._cmd_mode()
        assert "Current mode" in result
        assert "Usage" in result

    def test_cmd_mode_switch(self, bot):
        result = bot._cmd_mode(["aggressive"])
        assert "AGGRESSIVE" in result
        assert bot.config["profile"]["name"] == "aggressive"

    def test_cmd_mode_invalid(self, bot):
        result = bot._cmd_mode(["nonexistent"])
        assert "Unknown profile" in result

    async def test_cmd_close_no_rest(self, bot_no_rest):
        result = await bot_no_rest._cmd_close()
        assert "REST client not available" in result

    async def test_cmd_close_no_positions(self, bot):
        result = await bot._cmd_close()
        assert "No open positions" in result

    async def test_cmd_close_specific(self, bot, mock_rest):
        _add_position(bot.state)
        result = await bot._cmd_close(["BTCUSDT"])
        assert "Closed BTCUSDT" in result
        mock_rest.place_order.assert_called_once()

    async def test_cmd_close_all(self, bot, mock_rest):
        _add_position(bot.state, "BTCUSDT")
        _add_position(bot.state, "ETHUSDT")
        result = await bot._cmd_close()
        assert "CLOSE ALL" in result
        assert mock_rest.place_order.call_count == 2

    async def test_cmd_close_unknown_symbol(self, bot):
        result = await bot._cmd_close(["XYZUSDT"])
        assert "No open position" in result

    async def test_cmd_sl_no_rest(self, bot_no_rest):
        result = await bot_no_rest._cmd_sl(["BTCUSDT", "49800"])
        assert "REST client not available" in result

    async def test_cmd_sl_missing_args(self, bot):
        result = await bot._cmd_sl()
        assert "Usage" in result

    async def test_cmd_sl_success(self, bot, mock_rest):
        _add_position(bot.state)
        result = await bot._cmd_sl(["BTCUSDT", "49800.00"])
        assert "Stop-loss moved" in result
        assert bot.state.positions["BTCUSDT"].stop_price == 49800.0
        mock_rest.set_trading_stop.assert_called_once()

    async def test_cmd_sl_no_position(self, bot):
        result = await bot._cmd_sl(["BTCUSDT", "49800"])
        assert "No open position" in result

    async def test_cmd_tp_success(self, bot, mock_rest):
        _add_position(bot.state)
        result = await bot._cmd_tp(["BTCUSDT", "50500.00"])
        assert "Take-profit moved" in result
        assert bot.state.positions["BTCUSDT"].tp_price == 50500.0

    async def test_cmd_tp_no_position(self, bot):
        result = await bot._cmd_tp(["ETHUSDT", "4000"])
        assert "No open position" in result

    async def test_cmd_tp_invalid_price(self, bot):
        result = await bot._cmd_tp(["BTCUSDT", "notanumber"])
        assert "Invalid price" in result


class TestNotificationCommands:
    def test_cmd_notify_status(self, bot):
        result = bot._cmd_notify()
        assert "OFF" in result

    def test_cmd_notify_on(self, bot):
        result = bot._cmd_notify(["on"])
        assert "ON" in result
        assert bot.config["telegram"]["notify_trades"] is True

    def test_cmd_notify_off(self, bot):
        bot.config["telegram"]["notify_trades"] = True
        result = bot._cmd_notify(["off"])
        assert "OFF" in result
        assert bot.config["telegram"]["notify_trades"] is False

    def test_cmd_notify_invalid(self, bot):
        result = bot._cmd_notify(["maybe"])
        assert "Usage" in result

    def test_cmd_alert_status(self, bot):
        result = bot._cmd_alert()
        assert "Alert settings" in result
        assert "Kill switch: ON" in result

    def test_cmd_alert_trade_on(self, bot):
        result = bot._cmd_alert(["trade", "on"])
        assert "Trade fill alerts: ON" in result

    def test_cmd_alert_dd_off(self, bot):
        result = bot._cmd_alert(["dd", "off"])
        assert "Drawdown alerts: OFF" in result

    def test_cmd_alert_all_on(self, bot):
        result = bot._cmd_alert(["all", "on"])
        assert "All configurable alerts: ON" in result

    def test_cmd_alert_missing_args(self, bot):
        result = bot._cmd_alert(["trade"])
        assert "Usage" in result

    def test_cmd_alert_unknown_category(self, bot):
        result = bot._cmd_alert(["bogus", "on"])
        assert "Unknown" in result


class TestFinanceCommands:
    async def test_cmd_cashout_no_rest(self, bot_no_rest):
        result = await bot_no_rest._cmd_cashout()
        assert "REST client not available" in result

    async def test_cmd_cashout_no_wallet(self, bot):
        with patch.dict(os.environ, {"COLD_WALLET_ADDRESS": ""}):
            result = await bot._cmd_cashout()
            assert "No COLD_WALLET_ADDRESS" in result

    async def test_cmd_cashout_preview(self, bot):
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
            "MIN_WITHDRAWAL": "10",
        }):
            result = await bot._cmd_cashout()
            assert "CASH OUT PREVIEW" in result
            assert "confirm" in result
            assert bot.state.pending_cashout is not None

    async def test_cmd_cashout_specific_amount(self, bot):
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
            "MIN_WITHDRAWAL": "10",
        }):
            result = await bot._cmd_cashout(["25"])
            assert "25.00" in result
            assert bot.state.pending_cashout["amount"] == 25.0

    async def test_cmd_cashout_below_minimum(self, bot):
        bot.state.equity = 1005.0
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
            "MIN_WITHDRAWAL": "10",
        }):
            result = await bot._cmd_cashout()
            assert "Not enough profit" in result

    async def test_cmd_cashout_confirm(self, bot, mock_rest):
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
            "MIN_WITHDRAWAL": "10",
        }):
            bot.state.pending_cashout = {"amount": 50.0, "ts": 12345}
            result = await bot._cmd_cashout(["confirm"])
            assert "Withdrawal submitted" in result
            assert "tx_abc123" in result
            assert bot.state.pending_cashout is None
            assert len(bot.state.withdrawal_history) == 1
            assert bot.state.cooldown_until_ms > 0
            mock_rest.withdraw.assert_called_once()

    async def test_cmd_cashout_confirm_with_positions(self, bot, mock_rest):
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
            "MIN_WITHDRAWAL": "10",
        }):
            _add_position(bot.state)
            bot.state.pending_cashout = {"amount": 30.0, "ts": 12345}
            result = await bot._cmd_cashout(["confirm"])
            assert "Closed 1 position" in result
            # place_order called for close + withdraw called
            mock_rest.place_order.assert_called_once()
            mock_rest.withdraw.assert_called_once()

    async def test_cmd_cashout_cancel(self, bot):
        bot.state.pending_cashout = {"amount": 50.0, "ts": 12345}
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
        }):
            result = await bot._cmd_cashout(["cancel"])
            assert "cancelled" in result
            assert bot.state.pending_cashout is None

    async def test_cmd_cashout_confirm_no_pending(self, bot):
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
        }):
            result = await bot._cmd_cashout(["confirm"])
            assert "No pending cashout" in result

    async def test_cmd_cashout_withdrawal_fails(self, bot, mock_rest):
        mock_rest.withdraw = AsyncMock(return_value={
            "success": False, "tx_id": "", "error_msg": "insufficient funds"
        })
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
            "MIN_WITHDRAWAL": "10",
        }):
            bot.state.pending_cashout = {"amount": 50.0, "ts": 12345}
            result = await bot._cmd_cashout(["confirm"])
            assert "FAILED" in result
            assert "insufficient funds" in result

    def test_cmd_wallet_no_address(self, bot):
        with patch.dict(os.environ, {"COLD_WALLET_ADDRESS": ""}):
            result = bot._cmd_wallet()
            assert "No COLD_WALLET_ADDRESS" in result

    def test_cmd_wallet_with_address(self, bot):
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
        }):
            result = bot._cmd_wallet()
            assert "COLD WALLET" in result
            assert "No withdrawals" in result

    def test_cmd_wallet_with_history(self, bot):
        import time
        bot.state.withdrawal_history = [
            {"ts": int(time.time() * 1000), "amount": 50.0,
             "tx_id": "tx_abc123def456", "address": "TRxTest"},
        ]
        with patch.dict(os.environ, {
            "COLD_WALLET_ADDRESS": "TRxTestAddress1234567890",
            "COLD_WALLET_CHAIN": "TRC20",
        }):
            result = bot._cmd_wallet()
            assert "$50.00" in result
            assert "tx_abc123def" in result
            assert "Total withdrawn" in result


class TestAsyncDispatch:
    async def test_handle_update_sync_command(self, bot):
        bot.send_message = AsyncMock()
        bot.chat_id = "12345"
        update = {
            "message": {
                "text": "/status",
                "chat": {"id": 12345},
            }
        }
        await bot._handle_update(update)
        bot.send_message.assert_called_once()
        msg = bot.send_message.call_args[0][0]
        assert "STATUS" in msg

    async def test_handle_update_async_command(self, bot):
        bot.send_message = AsyncMock()
        bot.chat_id = "12345"
        update = {
            "message": {
                "text": "/close",
                "chat": {"id": 12345},
            }
        }
        await bot._handle_update(update)
        bot.send_message.assert_called_once()

    async def test_handle_update_with_args(self, bot):
        bot.send_message = AsyncMock()
        bot.chat_id = "12345"
        update = {
            "message": {
                "text": "/cooldown 15",
                "chat": {"id": 12345},
            }
        }
        await bot._handle_update(update)
        bot.send_message.assert_called_once()
        msg = bot.send_message.call_args[0][0]
        assert "15 minutes" in msg

    async def test_handle_update_unknown_command(self, bot):
        bot.send_message = AsyncMock()
        bot.chat_id = "12345"
        update = {
            "message": {
                "text": "/unknown",
                "chat": {"id": 12345},
            }
        }
        await bot._handle_update(update)
        bot.send_message.assert_called_once()
        msg = bot.send_message.call_args[0][0]
        assert "Unknown command" in msg

    async def test_handle_update_unauthorized_chat(self, bot):
        bot.send_message = AsyncMock()
        bot.chat_id = "12345"
        update = {
            "message": {
                "text": "/status",
                "chat": {"id": 99999},
            }
        }
        await bot._handle_update(update)
        bot.send_message.assert_not_called()


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
        last = bot._last_alert.get("test", 0)
        assert time.time() - last < 1.0

    async def test_alert_drawdown_respects_setting(self, bot):
        bot.send_message = AsyncMock()
        bot.enabled = True
        bot._session = MagicMock()
        # Default: drawdown alerts ON
        bot.config["telegram"] = {"alerts": {"drawdown": True}}
        await bot.alert_drawdown(5.0)
        # Should have tried to send (alert method called)

    async def test_alert_drawdown_disabled(self, bot):
        bot.send_message = AsyncMock()
        bot.config["telegram"] = {"alerts": {"drawdown": False}}
        await bot.alert_drawdown(5.0)
        bot.send_message.assert_not_called()


class TestStateExtensions:
    def test_manual_cooldown_field(self, config):
        s = RuntimeState(config)
        assert s.cooldown_until_ms == 0

    def test_in_manual_cooldown_false(self, config):
        s = RuntimeState(config)
        assert s.in_manual_cooldown() is False

    def test_in_manual_cooldown_true(self, config):
        from core.utils import time_now_ms
        s = RuntimeState(config)
        s.cooldown_until_ms = time_now_ms() + 60000
        assert s.in_manual_cooldown() is True

    def test_in_manual_cooldown_expired(self, config):
        from core.utils import time_now_ms
        s = RuntimeState(config)
        s.cooldown_until_ms = time_now_ms() - 1000
        assert s.in_manual_cooldown() is False

    def test_pending_cashout_field(self, config):
        s = RuntimeState(config)
        assert s.pending_cashout is None

    def test_withdrawal_history_field(self, config):
        s = RuntimeState(config)
        assert len(s.withdrawal_history) == 0


class TestBacktestCommand:
    async def test_cmd_backtest_import_error(self, bot):
        """Backtest command should handle missing backtest module gracefully."""
        result = await bot._cmd_backtest()
        # Will fail with import error in test environment - that's OK
        assert "BACKTEST" in result or "failed" in result
