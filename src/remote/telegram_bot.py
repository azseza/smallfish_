"""Telegram Bot — remote control and monitoring for Smallfish.

Provides commands to check status, view trades, trigger kill switch,
and receive alerts from anywhere via Telegram.

Setup:
    1. Create a bot via @BotFather on Telegram
    2. Set TELEGRAM_BOT_TOKEN in .env
    3. Send /start to your bot, note the chat ID
    4. Set TELEGRAM_CHAT_ID in .env
    5. Run with --telegram flag

Commands:
    /status  - Account summary (equity, PnL, positions, regime)
    /trades  - Recent trade history with PnL
    /equity  - Equity and drawdown info
    /grid    - Multigrid status
    /kill    - Trigger kill switch
    /resume  - Reset kill switch
    /config  - View current profile parameters
    /help    - List commands
"""
from __future__ import annotations
import asyncio
import logging
import os
from typing import Optional

from core.state import RuntimeState

log = logging.getLogger(__name__)


class TelegramBot:
    """Async Telegram bot for remote Smallfish control."""

    def __init__(self, state: RuntimeState, config: dict, grid_strategy=None):
        self.state = state
        self.config = config
        self.grid = grid_strategy
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._session = None
        self._offset = 0

        # Alert cooldowns (prevent spam)
        self._last_alert: dict[str, float] = {}
        self._alert_cooldown_s = config.get("telegram", {}).get("alert_cooldown_s", 300)

    async def start(self) -> None:
        """Start the Telegram bot polling loop."""
        if not self.enabled:
            log.warning("Telegram bot disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
            return

        import aiohttp
        self._session = aiohttp.ClientSession()
        self._running = True
        self._task = asyncio.ensure_future(self._poll_loop())

        # Send startup message
        await self.send_message(
            "><(((o> *SMALLFISH ONLINE* <o)))><\n"
            f"Equity: ${self.state.equity:.2f}\n"
            f"Symbols: {', '.join(self.config.get('symbols', []))}\n"
            "Send /help for commands"
        )
        log.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the bot."""
        self._running = False
        if self._task:
            self._task.cancel()
        if self.enabled:
            await self.send_message("Smallfish shutting down.")
        if self._session:
            await self._session.close()

    async def _poll_loop(self) -> None:
        """Long-poll for Telegram updates."""
        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug("Telegram poll error: %s", e)
                await asyncio.sleep(5)

    async def _get_updates(self) -> list:
        """Fetch updates from Telegram API."""
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"offset": self._offset, "timeout": 10, "limit": 10}
        try:
            async with self._session.get(url, params=params, timeout=15) as resp:
                data = await resp.json()
                updates = data.get("result", [])
                if updates:
                    self._offset = updates[-1]["update_id"] + 1
                return updates
        except Exception:
            await asyncio.sleep(2)
            return []

    async def _handle_update(self, update: dict) -> None:
        """Route incoming message to command handler."""
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        # Only respond to authorized chat
        if chat_id != self.chat_id:
            return

        if not text.startswith("/"):
            return

        cmd = text.split()[0].lower()
        handlers = {
            "/status": self._cmd_status,
            "/trades": self._cmd_trades,
            "/equity": self._cmd_equity,
            "/grid": self._cmd_grid,
            "/kill": self._cmd_kill,
            "/resume": self._cmd_resume,
            "/config": self._cmd_config,
            "/help": self._cmd_help,
            "/start": self._cmd_help,
        }

        handler = handlers.get(cmd)
        if handler:
            response = handler()
            await self.send_message(response)
        else:
            await self.send_message(f"Unknown command: {cmd}\nSend /help for available commands")

    # --- Command handlers ---

    def _cmd_status(self) -> str:
        s = self.state
        wr = s.win_count / max(s.trade_count, 1) * 100
        kill_str = f"KILLED: {s.kill_reason}" if s.kill_switch else "LIVE"

        positions = ""
        if s.positions:
            for sym, p in s.positions.items():
                side = "LONG" if p.side == 1 else "SHORT"
                positions += (
                    f"\n  {side} {sym} qty={p.quantity:.4f} "
                    f"entry={p.entry_price:.2f} uPnL=${p.unrealized_pnl:+.4f}"
                )
        else:
            positions = "\n  (no open positions)"

        return (
            f"><(((o> STATUS\n"
            f"State: {kill_str}\n"
            f"Equity: ${s.equity:.2f}\n"
            f"Peak: ${s.peak_equity:.2f}\n"
            f"Drawdown: {s.drawdown*100:.2f}%\n"
            f"Daily PnL: ${s.daily_pnl:+.4f}\n"
            f"Total PnL: ${s.realized_pnl:+.4f}\n"
            f"Trades: {s.trade_count} (W:{s.win_count} L:{s.loss_count})\n"
            f"Win Rate: {wr:.1f}%\n"
            f"Daily R: {s.daily_loss_R:.1f}/{self.config.get('max_daily_R', 10)}\n"
            f"Regime: {s.vol_regime}\n"
            f"Latency: {s.latency_ms}ms\n"
            f"Positions:{positions}"
        )

    def _cmd_trades(self) -> str:
        trades = list(self.state.completed_trades.get())
        if not trades:
            return "No trades yet."

        recent = trades[-10:]
        lines = ["RECENT TRADES:"]
        for t in recent:
            side = "L" if t.side == 1 else "S"
            emoji = "+" if t.pnl >= 0 else "-"
            lines.append(
                f"  {emoji} {side} {t.symbol} "
                f"${t.pnl:+.4f} ({t.pnl_R:+.2f}R) "
                f"{t.exit_reason}"
            )

        total = sum(t.pnl for t in recent)
        lines.append(f"\nLast {len(recent)} total: ${total:+.4f}")
        return "\n".join(lines)

    def _cmd_equity(self) -> str:
        s = self.state
        initial = self.config.get("initial_equity", 1000)
        total_return = (s.equity - initial) / initial * 100 if initial > 0 else 0

        return (
            f"EQUITY\n"
            f"Initial: ${initial:.2f}\n"
            f"Current: ${s.equity:.2f}\n"
            f"Peak: ${s.peak_equity:.2f}\n"
            f"Return: {total_return:+.2f}%\n"
            f"Drawdown: {s.drawdown*100:.2f}%\n"
            f"Max DD: {s.max_drawdown*100:.2f}%"
        )

    def _cmd_grid(self) -> str:
        if not self.grid:
            return "Multigrid is not enabled."

        all_status = self.grid.all_status()
        if not any(gs.get("active") for gs in all_status.values()):
            return "Multigrid: all grids inactive."

        lines = ["MULTIGRID STATUS:"]
        for sym, gs in all_status.items():
            if not gs.get("active"):
                continue
            lines.append(
                f"  {sym}:\n"
                f"    Center: {gs['center']:.2f}\n"
                f"    Buys: {gs['filled_buys']}/{gs['buy_levels']}\n"
                f"    Sells: {gs['filled_sells']}/{gs['sell_levels']}\n"
                f"    Pending: {gs['pending_orders']}\n"
                f"    PnL: ${gs['total_pnl']:+.4f}\n"
                f"    Round trips: {gs['round_trips']}"
            )
        return "\n".join(lines)

    def _cmd_kill(self) -> str:
        if self.state.kill_switch:
            return f"Kill switch already active: {self.state.kill_reason}"
        self.state.trigger_kill_switch("telegram_remote")
        return "Kill switch TRIGGERED via Telegram. All trading stopped."

    def _cmd_resume(self) -> str:
        if not self.state.kill_switch:
            return "Kill switch is not active. Trading is running."
        self.state.kill_switch = False
        self.state.kill_reason = ""
        return "Kill switch RESET. Trading resumed."

    def _cmd_config(self) -> str:
        c = self.config
        return (
            f"CONFIG\n"
            f"Symbols: {', '.join(c.get('symbols', []))}\n"
            f"Leverage: {c.get('leverage', 10)}x\n"
            f"Risk/Trade: {c.get('risk_per_trade', 0.005)*100:.1f}%\n"
            f"Max Risk: ${c.get('max_risk_dollars', 5)}\n"
            f"Max Spread: {c.get('max_spread', 2)} ticks\n"
            f"Max Latency: {c.get('max_latency_ms', 80)}ms\n"
            f"Daily R Limit: {c.get('max_daily_R', 10)}\n"
            f"C_enter: {c.get('C_enter', 0.65)}\n"
            f"C_exit: {c.get('C_exit', 0.40)}\n"
            f"Multigrid: {'ON' if c.get('multigrid', {}).get('enabled') else 'OFF'}\n"
            f"Dashboard: {'ON' if c.get('dashboard', {}).get('enabled') else 'OFF'}"
        )

    def _cmd_help(self) -> str:
        return (
            "><(((o> SMALLFISH COMMANDS\n"
            "/status  - Account summary\n"
            "/trades  - Recent trades\n"
            "/equity  - Equity details\n"
            "/grid    - Multigrid status\n"
            "/kill    - Trigger kill switch\n"
            "/resume  - Reset kill switch\n"
            "/config  - View configuration\n"
            "/help    - This message"
        )

    # --- Alert sending ---

    async def send_message(self, text: str) -> None:
        """Send a message to the configured chat."""
        if not self.enabled or not self._session:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        try:
            async with self._session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    log.debug("Telegram send failed: %d", resp.status)
        except Exception as e:
            log.debug("Telegram send error: %s", e)

    async def alert(self, alert_type: str, message: str) -> None:
        """Send an alert with cooldown to prevent spam."""
        import time
        now = time.time()
        last = self._last_alert.get(alert_type, 0)
        if now - last < self._alert_cooldown_s:
            return
        self._last_alert[alert_type] = now
        await self.send_message(f"ALERT [{alert_type}]\n{message}")

    async def alert_kill_switch(self, reason: str) -> None:
        """Alert when kill switch is triggered."""
        await self.alert("kill_switch", f"Kill switch triggered: {reason}\nEquity: ${self.state.equity:.2f}")

    async def alert_drawdown(self, dd_pct: float) -> None:
        """Alert on significant drawdown."""
        await self.alert("drawdown", f"Drawdown: {dd_pct:.2f}%\nEquity: ${self.state.equity:.2f}")

    async def alert_trade(self, trade_result) -> None:
        """Alert on trade completion (optional, can be noisy)."""
        notify_trades = self.config.get("telegram", {}).get("notify_trades", False)
        if not notify_trades:
            return
        side = "LONG" if trade_result.side == 1 else "SHORT"
        emoji = "+" if trade_result.pnl >= 0 else "-"
        await self.send_message(
            f"{emoji} {side} {trade_result.symbol}\n"
            f"PnL: ${trade_result.pnl:+.4f} ({trade_result.pnl_R:+.2f}R)\n"
            f"Reason: {trade_result.exit_reason}"
        )
