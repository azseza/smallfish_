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
    /status    - Account summary (equity, PnL, positions, regime)
    /stats     - Live stats: equity, PnL, WR, PF, DD, Sharpe
    /balance   - Detailed balance & margin info
    /equity    - Equity and drawdown info
    /trades    - Recent trade history with PnL
    /pnl [7]   - Daily P&L breakdown
    /signals   - Current signal scores
    /symbols   - Active symbols + spreads
    /kill      - Trigger kill switch
    /resume    - Reset kill switch
    /close     - Close position(s)
    /cooldown  - Pause trading N minutes
    /mode      - Switch risk profile
    /sl        - Move stop-loss
    /tp        - Move take-profit
    /notify    - Toggle trade notifications
    /alert     - Fine-grained alert control
    /config    - View current profile parameters
    /grid      - Multigrid status
    /cashout   - Withdraw profits to cold wallet
    /wallet    - Cold wallet info + history
    /backtest  - Quick backtest report
    /help      - List commands
"""
from __future__ import annotations
import asyncio
import inspect
import logging
import math
import os
import time as _time
from datetime import datetime, timezone, timedelta
from typing import Optional

from core.state import RuntimeState
from core.utils import time_now_ms

log = logging.getLogger(__name__)


class TelegramBot:
    """Async Telegram bot for remote Smallfish control."""

    def __init__(self, state: RuntimeState, config: dict,
                 grid_strategy=None, rest=None):
        self.state = state
        self.config = config
        self.grid = grid_strategy
        self.rest = rest  # ExchangeREST instance for /close, /sl, /tp, /cashout
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

        # Session health tracking for auto-reconnect
        self._consecutive_failures = 0
        self._max_failures_before_reconnect = 5

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
        """Long-poll for Telegram updates with auto-reconnection."""
        while self._running:
            try:
                updates = await self._get_updates()
                if updates is not None:
                    self._consecutive_failures = 0
                    for update in updates:
                        await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                log.info("Telegram poll error (#%d): %s",
                         self._consecutive_failures, e)
                if self._consecutive_failures >= self._max_failures_before_reconnect:
                    log.warning("Telegram: %d consecutive failures — recreating session",
                                self._consecutive_failures)
                    await self._recreate_session()
                    self._consecutive_failures = 0
                await asyncio.sleep(5)

    async def _recreate_session(self) -> None:
        """Close and recreate the aiohttp session (network recovery)."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass
        import aiohttp
        self._session = aiohttp.ClientSession()
        log.info("Telegram session recreated")

    async def _get_updates(self) -> list | None:
        """Fetch updates from Telegram API. Returns None on failure."""
        if not self._session or self._session.closed:
            return None
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
            return None

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

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            "/status": self._cmd_status,
            "/stats": self._cmd_stats,
            "/balance": self._cmd_balance,
            "/trades": self._cmd_trades,
            "/equity": self._cmd_equity,
            "/pnl": self._cmd_pnl,
            "/signals": self._cmd_signals,
            "/symbols": self._cmd_symbols,
            "/grid": self._cmd_grid,
            "/kill": self._cmd_kill,
            "/resume": self._cmd_resume,
            "/close": self._cmd_close,
            "/cooldown": self._cmd_cooldown,
            "/mode": self._cmd_mode,
            "/sl": self._cmd_sl,
            "/tp": self._cmd_tp,
            "/notify": self._cmd_notify,
            "/alert": self._cmd_alert,
            "/config": self._cmd_config,
            "/cashout": self._cmd_cashout,
            "/wallet": self._cmd_wallet,
            "/backtest": self._cmd_backtest,
            "/help": self._cmd_help,
            "/start": self._cmd_help,
        }

        handler = handlers.get(cmd)
        if handler:
            result = handler(args)
            if inspect.isawaitable(result):
                result = await result
            await self.send_message(result)
        else:
            await self.send_message(f"Unknown command: {cmd}\nSend /help for available commands")

    # ── Monitoring commands ────────────────────────────────────────

    def _cmd_status(self, args: list = None) -> str:
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

    def _cmd_stats(self, args: list = None) -> str:
        s = self.state
        c = self.config
        initial = c.get("initial_equity", 1000)
        total_pnl = s.equity - initial
        total_pct = total_pnl / max(initial, 0.01) * 100
        daily_pct = s.daily_pnl / max(s.daily_start_equity, 0.01) * 100

        completed = list(s.completed_trades.get())
        wins = [t.pnl for t in completed if t.pnl > 0]
        losses = [t.pnl for t in completed if t.pnl < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_win / max(gross_loss, 0.01) if gross_loss > 0 else 0
        best = max((t.pnl for t in completed), default=0)
        worst = min((t.pnl for t in completed), default=0)
        best_sym = next((t.symbol for t in completed if t.pnl == best), "") if completed else ""
        worst_sym = next((t.symbol for t in completed if t.pnl == worst), "") if completed else ""

        # Sharpe approximation (daily returns if enough data)
        sharpe_str = "n/a"
        if len(completed) >= 10:
            returns = [t.pnl for t in completed]
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = math.sqrt(var_r) if var_r > 0 else 0
            if std_r > 0:
                sharpe_str = f"{mean_r / std_r:.2f}"

        dd = s.drawdown * 100
        mdd = s.max_drawdown * 100
        wr = s.win_count / max(s.trade_count, 1) * 100

        # Uptime
        uptime_ms = time_now_ms() - s.started_at
        uptime_s = uptime_ms // 1000
        h, rem = divmod(uptime_s, 3600)
        m, _ = divmod(rem, 60)

        return (
            f"><(((o> LIVE STATS\n\n"
            f"Equity:       ${s.equity:.2f} (peak ${s.peak_equity:.2f})\n"
            f"Daily PnL:    ${s.daily_pnl:+.2f} ({daily_pct:+.1f}%)\n"
            f"Total PnL:    ${total_pnl:+.2f} ({total_pct:+.1f}%)\n"
            f"Trades:       {s.trade_count} ({s.win_count}W / {s.loss_count}L)\n"
            f"Win Rate:     {wr:.1f}%\n"
            f"Profit Factor: {pf:.2f}\n"
            f"Avg Win:      ${avg_win:+.2f}\n"
            f"Avg Loss:     ${avg_loss:+.2f}\n"
            f"Best Trade:   ${best:+.2f} ({best_sym})\n"
            f"Worst Trade:  ${worst:+.2f} ({worst_sym})\n"
            f"Sharpe:       {sharpe_str}\n"
            f"Max Drawdown: {mdd:.1f}%\n"
            f"Current DD:   {dd:.1f}%\n"
            f"Daily R:      {s.daily_loss_R:.1f}R / {c.get('max_daily_R', 10)}R\n"
            f"Vol Regime:   {s.vol_regime}\n"
            f"Uptime:       {h}h{m:02d}m"
        )

    def _cmd_balance(self, args: list = None) -> str:
        s = self.state
        initial = self.config.get("initial_equity", 1000)
        unrealized = sum(p.unrealized_pnl for p in s.positions.values())
        available = s.equity - abs(unrealized)

        return (
            f"BALANCE\n"
            f"Equity: ${s.equity:.2f}\n"
            f"Available Margin: ${available:.2f}\n"
            f"Unrealized PnL: ${unrealized:+.4f}\n"
            f"Realized PnL: ${s.realized_pnl:+.4f}\n"
            f"Initial: ${initial:.2f}\n"
            f"Profit: ${s.equity - initial:+.2f}"
        )

    def _cmd_trades(self, args: list = None) -> str:
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

    def _cmd_equity(self, args: list = None) -> str:
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

    def _cmd_pnl(self, args: list = None) -> str:
        days = 7
        if args:
            try:
                days = int(args[0])
            except ValueError:
                pass
        days = min(days, 30)

        completed = list(self.state.completed_trades.get())
        if not completed:
            return "No trades yet."

        now = datetime.now(timezone.utc)
        buckets: dict[str, list[float]] = {}
        for i in range(days):
            key = (now - timedelta(days=i)).strftime("%m/%d")
            buckets[key] = []

        for t in completed:
            key = datetime.fromtimestamp(
                t.exit_time / 1000, tz=timezone.utc
            ).strftime("%m/%d")
            if key in buckets:
                buckets[key].append(t.pnl)

        lines = [f"P&L BREAKDOWN ({days}d)"]
        total = 0.0
        for i in range(days - 1, -1, -1):
            key = (now - timedelta(days=i)).strftime("%m/%d")
            pnls = buckets[key]
            day_pnl = sum(pnls)
            total += day_pnl
            w = sum(1 for p in pnls if p >= 0)
            l = sum(1 for p in pnls if p < 0)
            mark = "+" if day_pnl >= 0 else "-"
            lines.append(f"  {key}  {mark}${abs(day_pnl):.2f}  {w}W/{l}L")

        lines.append(f"\nTotal: ${total:+.2f}")
        return "\n".join(lines)

    def _cmd_signals(self, args: list = None) -> str:
        scores = self.state.last_scores
        if not scores:
            return "No signal data yet."

        signal_pairs = [
            ("OBI",    "obi_long",    "obi_short"),
            ("PRT",    "prt_long",    "prt_short"),
            ("UMOM",   "umom_long",   "umom_short"),
            ("LTB",    "ltb_long",    "ltb_short"),
            ("SWEEP",  "sweep_up",    "sweep_down"),
            ("ICE",    "ice_long",    "ice_short"),
            ("VWAP",   "vwap_long",   "vwap_short"),
            ("REGIME", "regime_long", "regime_short"),
            ("CVD",    "cvd_long",    "cvd_short"),
            ("TPS",    "tps_long",    "tps_short"),
            ("LIQ",    "liq_long",    "liq_short"),
            ("MVR",    "mvr_long",    "mvr_short"),
            ("ABSORB", "absorb_long", "absorb_short"),
        ]

        lines = ["SIGNAL SCORES"]
        for label, lk, sk in signal_pairs:
            lv = scores.get(lk, 0.0)
            sv = scores.get(sk, 0.0)
            net = lv - sv
            arrow = ">" if net > 0.01 else "<" if net < -0.01 else "="
            lines.append(f"  {label:<7} L={lv:.2f}  S={sv:.2f}  net={net:+.2f} {arrow}")

        conf = self.state.last_confidence
        d = self.state.last_direction
        dir_s = "LONG" if d == 1 else "SHORT" if d == -1 else "FLAT"
        lines.append(f"\nDirection: {dir_s}  Confidence: {conf:.3f}")
        return "\n".join(lines)

    def _cmd_symbols(self, args: list = None) -> str:
        symbols = self.config.get("symbols", [])
        if not symbols:
            return "No active symbols."

        lines = ["ACTIVE SYMBOLS"]
        for sym in symbols:
            pos = self.state.positions.get(sym)
            pos_str = ""
            if pos:
                side = "LONG" if pos.side == 1 else "SHORT"
                pos_str = f"  [{side} qty={pos.quantity:.4f}]"
            lines.append(f"  {sym}{pos_str}")
        return "\n".join(lines)

    # ── Control commands ──────────────────────────────────────────

    def _cmd_grid(self, args: list = None) -> str:
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

    def _cmd_kill(self, args: list = None) -> str:
        if self.state.kill_switch:
            return f"Kill switch already active: {self.state.kill_reason}"
        self.state.trigger_kill_switch("telegram_remote")
        return "Kill switch TRIGGERED via Telegram. All trading stopped."

    def _cmd_resume(self, args: list = None) -> str:
        if not self.state.kill_switch:
            return "Kill switch is not active. Trading is running."
        self.state.kill_switch = False
        self.state.kill_reason = ""
        return "Kill switch RESET. Trading resumed."

    async def _cmd_close(self, args: list = None) -> str:
        if not self.rest:
            return "REST client not available."

        from core.types import Side

        if args:
            symbol = args[0].upper()
            pos = self.state.positions.get(symbol)
            if not pos:
                return f"No open position for {symbol}."
            close_side = Side.SELL if pos.side == Side.BUY else Side.BUY
            from core.types import OrderType
            resp = await self.rest.place_order(
                symbol=symbol, side=close_side, qty=pos.quantity,
                order_type=OrderType.MARKET, reduce_only=True,
            )
            if resp.success:
                return f"Closed {symbol} position. Order ID: {resp.order_id}"
            return f"Failed to close {symbol}: {resp.error_msg}"
        else:
            if not self.state.positions:
                return "No open positions to close."
            results = []
            for sym, pos in list(self.state.positions.items()):
                close_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                from core.types import OrderType
                resp = await self.rest.place_order(
                    symbol=sym, side=close_side, qty=pos.quantity,
                    order_type=OrderType.MARKET, reduce_only=True,
                )
                if resp.success:
                    results.append(f"  Closed {sym}")
                else:
                    results.append(f"  Failed {sym}: {resp.error_msg}")
            return "CLOSE ALL POSITIONS\n" + "\n".join(results)

    def _cmd_cooldown(self, args: list = None) -> str:
        minutes = 5
        if args:
            try:
                minutes = int(args[0])
            except ValueError:
                return "Usage: /cooldown [minutes]"
        minutes = max(1, min(minutes, 1440))

        self.state.cooldown_until_ms = time_now_ms() + minutes * 60 * 1000
        return f"Trading paused for {minutes} minutes.\nResumes at cooldown end or /resume."

    def _cmd_mode(self, args: list = None) -> str:
        if not args:
            current = self.config.get("profile", {}).get("name", "default")
            return (
                f"Current mode: {current}\n"
                f"Usage: /mode <conservative|balanced|aggressive|ultra>"
            )

        profile_name = args[0].lower()
        try:
            from core.profiles import apply_profile
            apply_profile(self.config, profile_name)
            return f"Profile switched to: {profile_name.upper()}"
        except ValueError as e:
            return str(e)

    async def _cmd_sl(self, args: list = None) -> str:
        if not self.rest:
            return "REST client not available."
        if not args or len(args) < 2:
            return "Usage: /sl <symbol> <price>"

        symbol = args[0].upper()
        try:
            price = float(args[1])
        except ValueError:
            return "Invalid price."

        pos = self.state.positions.get(symbol)
        if not pos:
            return f"No open position for {symbol}."

        resp = await self.rest.set_trading_stop(symbol=symbol, stop_loss=price)
        if resp.success:
            pos.stop_price = price
            return f"Stop-loss moved to {price:.2f} for {symbol}"
        return f"Failed to move SL: {resp.error_msg}"

    async def _cmd_tp(self, args: list = None) -> str:
        if not self.rest:
            return "REST client not available."
        if not args or len(args) < 2:
            return "Usage: /tp <symbol> <price>"

        symbol = args[0].upper()
        try:
            price = float(args[1])
        except ValueError:
            return "Invalid price."

        pos = self.state.positions.get(symbol)
        if not pos:
            return f"No open position for {symbol}."

        resp = await self.rest.set_trading_stop(symbol=symbol, take_profit=price)
        if resp.success:
            pos.tp_price = price
            return f"Take-profit moved to {price:.2f} for {symbol}"
        return f"Failed to move TP: {resp.error_msg}"

    # ── Notification commands ─────────────────────────────────────

    def _cmd_notify(self, args: list = None) -> str:
        tg = self.config.setdefault("telegram", {})
        current = tg.get("notify_trades", False)

        if not args:
            status = "ON" if current else "OFF"
            toggle = "off" if current else "on"
            return f"Trade notifications: {status}\nUse /notify {toggle} to toggle"

        val = args[0].lower()
        if val == "on":
            tg["notify_trades"] = True
            return "Trade notifications: ON\nYou will receive a message for every completed trade."
        elif val == "off":
            tg["notify_trades"] = False
            return "Trade notifications: OFF"
        return "Usage: /notify [on|off]"

    def _cmd_alert(self, args: list = None) -> str:
        tg = self.config.setdefault("telegram", {})
        alerts = tg.setdefault("alerts", {
            "trade": False,
            "drawdown": True,
            "kill_switch": True,
        })

        if not args:
            trade_s = "ON" if alerts.get("trade", False) else "OFF"
            dd_s = "ON" if alerts.get("drawdown", True) else "OFF"
            return (
                f"Alert settings:\n"
                f"  Trade fills: {trade_s}\n"
                f"  Drawdown >5%: {dd_s}\n"
                f"  Kill switch: ON (always on)"
            )

        category = args[0].lower()
        if len(args) < 2:
            return "Usage: /alert [dd|trade|all] [on|off]"

        val = args[1].lower() == "on"

        if category == "trade":
            alerts["trade"] = val
            return f"Trade fill alerts: {'ON' if val else 'OFF'}"
        elif category == "dd":
            alerts["drawdown"] = val
            return f"Drawdown alerts: {'ON' if val else 'OFF'}"
        elif category == "all":
            alerts["trade"] = val
            alerts["drawdown"] = val
            return f"All configurable alerts: {'ON' if val else 'OFF'}\n(Kill switch alerts always stay ON)"
        return "Unknown alert category. Use: trade, dd, all"

    # ── Config commands ───────────────────────────────────────────

    def _cmd_config(self, args: list = None) -> str:
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

    # ── Finance commands ──────────────────────────────────────────

    async def _cmd_cashout(self, args: list = None) -> str:
        if not self.rest:
            return "REST client not available."

        s = self.state
        initial = self.config.get("initial_equity", 1000)
        profit = s.equity - initial
        cold_addr = os.environ.get("COLD_WALLET_ADDRESS", "")
        chain = os.environ.get("COLD_WALLET_CHAIN", "TRC20")
        min_wd = float(os.environ.get("MIN_WITHDRAWAL", "10"))

        if not cold_addr:
            return "No COLD_WALLET_ADDRESS configured in .env"

        addr_short = cold_addr[:6] + "..." + cold_addr[-4:] if len(cold_addr) > 12 else cold_addr

        # Handle confirm/cancel
        if args and args[0].lower() == "confirm":
            pending = s.pending_cashout
            if not pending:
                return "No pending cashout. Use /cashout to preview first."

            amount = pending["amount"]

            # Close all positions first
            closed = 0
            if s.positions:
                from core.types import Side, OrderType
                for sym, pos in list(s.positions.items()):
                    close_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                    await self.rest.place_order(
                        symbol=sym, side=close_side, qty=pos.quantity,
                        order_type=OrderType.MARKET, reduce_only=True,
                    )
                    closed += 1

            # Submit withdrawal
            result = await self.rest.withdraw(
                coin="USDT", chain=chain,
                address=cold_addr, amount=amount,
            )

            s.pending_cashout = None

            if not result["success"]:
                return f"Withdrawal FAILED: {result['error_msg']}"

            # Record withdrawal
            s.withdrawal_history.append({
                "ts": time_now_ms(),
                "amount": amount,
                "tx_id": result["tx_id"],
                "address": cold_addr,
            })

            # Pause trading for 5 minutes
            s.cooldown_until_ms = time_now_ms() + 5 * 60 * 1000

            lines = ["WITHDRAWING..."]
            if closed > 0:
                lines.append(f"  Closed {closed} position(s)")
            lines.append(f"  Withdrawal submitted: ${amount:.2f} USDT -> {addr_short}")
            lines.append(f"  TX ID: {result['tx_id']}")
            lines.append(f"  Trading paused for 5 min (equity sync)")
            return "\n".join(lines)

        if args and args[0].lower() == "cancel":
            s.pending_cashout = None
            return "Cashout cancelled."

        # Preview
        if profit < min_wd:
            return (
                f"CASH OUT\n"
                f"Available profit: ${profit:.2f}\n"
                f"Minimum withdrawal: ${min_wd:.2f}\n"
                f"Not enough profit to withdraw."
            )

        # Specific amount
        amount = profit
        if args:
            try:
                amount = float(args[0])
                amount = min(amount, profit)
            except ValueError:
                pass

        if amount < min_wd:
            return f"Amount ${amount:.2f} is below minimum ${min_wd:.2f}"

        pos_count = len(s.positions)
        s.pending_cashout = {"amount": amount, "ts": time_now_ms()}

        lines = [
            f"CASH OUT PREVIEW",
            f"Current Equity: ${s.equity:.2f}",
            f"Initial Equity: ${initial:.2f}",
            f"Available Profit: ${profit:.2f}",
            f"Amount to withdraw: ${amount:.2f}",
            f"Destination: {addr_short} ({chain})",
        ]
        if pos_count > 0:
            lines.append(f"\nOpen positions: {pos_count} (will be closed)")
        lines.append(f"\nSend /cashout confirm to proceed")
        lines.append(f"Send /cashout cancel to abort")
        return "\n".join(lines)

    def _cmd_wallet(self, args: list = None) -> str:
        cold_addr = os.environ.get("COLD_WALLET_ADDRESS", "")
        chain = os.environ.get("COLD_WALLET_CHAIN", "TRC20")

        if not cold_addr:
            return "No COLD_WALLET_ADDRESS configured in .env"

        addr_short = cold_addr[:6] + "..." + cold_addr[-4:] if len(cold_addr) > 12 else cold_addr

        lines = [
            f"COLD WALLET",
            f"  Address: {addr_short} ({chain})",
            f"\nWITHDRAWAL HISTORY",
        ]

        history = self.state.withdrawal_history
        if not history:
            lines.append("  No withdrawals yet.")
        else:
            total = 0.0
            for w in history:
                dt = datetime.fromtimestamp(w["ts"] / 1000, tz=timezone.utc)
                lines.append(f"  {dt.strftime('%m/%d')} ${w['amount']:.2f}  TX: {w['tx_id'][:12]}...")
                total += w["amount"]
            lines.append(f"\n  Total withdrawn: ${total:.2f}")
        return "\n".join(lines)

    async def _cmd_backtest(self, args: list = None) -> str:
        symbol = "BTCUSDT"
        days = 7
        if args:
            if len(args) >= 1:
                symbol = args[0].upper()
            if len(args) >= 2:
                try:
                    days = int(args[1])
                except ValueError:
                    pass
        days = min(days, 30)

        try:
            from backtest import BacktestEngine
            mode = self.config.get("profile", {}).get("name", "aggressive")
            equity = self.config.get("initial_equity", 50)
            exchange = self.config.get("exchange", "bybit")
            engine = BacktestEngine(
                symbol=symbol, days=days, mode=mode,
                initial_equity=equity, exchange=exchange,
            )
            result = await engine.run()
            roi = result.get("roi_pct", 0)
            trades = result.get("total_trades", 0)
            wr = result.get("win_rate", 0) * 100
            pf = result.get("profit_factor", 0)
            return (
                f"BACKTEST: {symbol} ({days}d)\n"
                f"ROI: {roi:+.1f}%\n"
                f"Trades: {trades}\n"
                f"Win Rate: {wr:.1f}%\n"
                f"Profit Factor: {pf:.2f}"
            )
        except Exception as e:
            return f"Backtest failed: {e}"

    # ── Help ──────────────────────────────────────────────────────

    def _cmd_help(self, args: list = None) -> str:
        return (
            "><(((o> SMALLFISH COMMANDS\n\n"
            "MONITORING\n"
            "  /status    - Account summary + positions\n"
            "  /stats     - Live stats: equity, PnL, WR, PF, DD, Sharpe\n"
            "  /balance   - Detailed balance & margin info\n"
            "  /equity    - Equity curve & drawdown\n"
            "  /trades    - Recent trade history\n"
            "  /pnl [7]   - Daily P&L breakdown\n"
            "  /signals   - Current signal scores\n"
            "  /symbols   - Active symbols + spreads\n\n"
            "CONTROL\n"
            "  /kill      - Emergency stop all trading\n"
            "  /resume    - Resume after kill switch\n"
            "  /close [sym] - Close position(s)\n"
            "  /cooldown [5] - Pause trading N minutes\n"
            "  /mode <profile> - Switch risk profile\n\n"
            "POSITION\n"
            "  /sl <sym> <price> - Move stop-loss\n"
            "  /tp <sym> <price> - Move take-profit\n\n"
            "NOTIFICATIONS\n"
            "  /notify [on|off] - Toggle trade notifications\n"
            "  /alert [dd|trade|all] [on|off] - Fine-grained alerts\n\n"
            "CONFIG\n"
            "  /config    - View current settings\n"
            "  /grid      - Multigrid status\n\n"
            "FINANCE\n"
            "  /cashout [amount] - Withdraw profits to cold wallet\n"
            "  /wallet    - Cold wallet info + history\n"
            "  /backtest [sym] [days] - Quick backtest report"
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
        now = _time.time()
        last = self._last_alert.get(alert_type, 0)
        if now - last < self._alert_cooldown_s:
            return
        self._last_alert[alert_type] = now
        await self.send_message(f"ALERT [{alert_type}]\n{message}")

    async def alert_kill_switch(self, reason: str) -> None:
        """Alert when kill switch is triggered — NO COOLDOWN, always delivers."""
        # Kill switch alerts bypass the generic cooldown.  You must always
        # know when trading stops, especially on a headless RPi.
        last_reason = self._last_alert.get("_kill_reason", "")
        if reason == last_reason:
            return  # same kill reason already sent, don't spam on every loop iter
        self._last_alert["_kill_reason"] = reason
        await self.send_message(
            f"ALERT [KILL SWITCH]\n"
            f"Kill switch triggered: {reason}\n"
            f"Equity: ${self.state.equity:.2f}\n"
            f"Send /resume to re-enable trading"
        )

    async def alert_drawdown(self, dd_pct: float) -> None:
        """Alert on significant drawdown."""
        tg = self.config.get("telegram", {})
        alerts = tg.get("alerts", {})
        if not alerts.get("drawdown", True):
            return
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
