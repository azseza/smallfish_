"""Terminal Dashboard — split-panel visualization for Smallfish.

Layout (rich.Layout):
  root.split_row(
    left(ratio=14).split_column(
      status(size=12),       <- system info + bot status
      stats(ratio=1),        <- today metrics + positions + 7D perf
      trades(size=14),       <- recent trades table
    ),
    right(ratio=20).split_column(
      candles(ratio=3),      <- candlestick chart (py-candlestick-chart)
      bottom.split_row(
        equity(ratio=1),     <- equity curve (plotext)
        signals(ratio=1),    <- signal bars (plotext)
      ),
    ),
  )

Usage:
    dashboard = TerminalDashboard(state, config)
    dashboard.start()   # launches async refresh loop
    dashboard.stop()
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import plotext as plt
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.state import RuntimeState
from core.types import TradeResult

log = logging.getLogger(__name__)

# Rolling buffer sizes
_MAX_CANDLES = 120       # ~2h of 1m candles
_MAX_EQUITY_POINTS = 200
_CANDLE_INTERVAL_S = 60  # 1-minute candles


@dataclass(slots=True)
class OHLCCandle:
    ts: int         # candle open time (unix seconds)
    open: float
    high: float
    low: float
    close: float
    volume: int     # tick count within candle


def _get_system_info() -> dict:
    """Gather system info (CPU, temp, memory, uptime). Safe on any platform."""
    info: dict = {}
    try:
        import psutil
        info["cpu"] = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        info["mem_pct"] = mem.percent
        info["mem_used_mb"] = round(mem.used / 1024**2)
        info["mem_total_mb"] = round(mem.total / 1024**2)
        boot = psutil.boot_time()
        up_s = int(time.time() - boot)
        h, rem = divmod(up_s, 3600)
        m, s = divmod(rem, 60)
        info["uptime"] = f"{h}h{m:02d}m"
        # Temperature — works on RPi and most Linux
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Prefer cpu_thermal (RPi), k10temp (AMD), coretemp (Intel)
                for key in ("cpu_thermal", "k10temp", "coretemp", "acpitz"):
                    if key in temps and temps[key]:
                        info["temp"] = temps[key][0].current
                        break
                if "temp" not in info:
                    first = next(iter(temps.values()))
                    if first:
                        info["temp"] = first[0].current
        except Exception:
            pass
        # Fallback: read thermal zone (RPi)
        if "temp" not in info:
            tz_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(tz_path):
                with open(tz_path) as f:
                    info["temp"] = int(f.read().strip()) / 1000
    except ImportError:
        info["cpu"] = -1
    return info


class TerminalDashboard:
    """Async terminal dashboard with split-panel layout using rich."""

    def __init__(self, state: RuntimeState, config: dict,
                 grid_strategy=None, refresh_s: float = 0.5):
        self.state = state
        self.config = config
        self.grid = grid_strategy
        self.refresh_s = refresh_s
        self.enabled = config.get("dashboard", {}).get("enabled", False)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._live: Optional[Live] = None

        # OHLCV candle buffer for candlestick chart
        self._candles: deque[OHLCCandle] = deque(maxlen=_MAX_CANDLES)
        self._current_candle: Optional[OHLCCandle] = None

        # Order markers on candle chart
        self._buy_markers: list[Tuple[int, float]] = []   # (candle_idx, price)
        self._sell_markers: list[Tuple[int, float]] = []

        # Equity history for equity curve
        self._equity_history: list[float] = []

        # System info cache (refreshed every 5s to avoid overhead)
        self._sys_info: dict = {}
        self._sys_info_ts: float = 0

        # Backfill ROI from 7D backtest (set by app.py at startup)
        self._backfill_roi: Optional[float] = None
        self._backfill_trades: int = 0
        self._backfill_wr: float = 0.0

    # ── Public data-feed methods ──────────────────────────────────

    def load_historical_candles(self, klines: list[dict]) -> None:
        """Pre-fill candle chart with historical kline data from REST.

        Args:
            klines: list of dicts with keys: ts, open, high, low, close, volume
        """
        for k in klines:
            candle = OHLCCandle(
                ts=int(k["ts"]) // 1000,  # ms → seconds
                open=k["open"],
                high=k["high"],
                low=k["low"],
                close=k["close"],
                volume=int(k.get("volume", 0)),
            )
            self._candles.append(candle)
        # Set current candle to last historical candle so live ticks extend it
        if self._candles:
            self._current_candle = None  # will start fresh on next tick

    def set_backfill_roi(self, roi_pct: float, trade_count: int,
                         win_rate: float) -> None:
        """Store 7D backtest ROI for display in stats panel."""
        self._backfill_roi = roi_pct
        self._backfill_trades = trade_count
        self._backfill_wr = win_rate

    def add_price_point(self, price: float, quantity: float = 0.0) -> None:
        """Feed a trade price. Aggregates into OHLCV candles.

        Should be called with actual trade prices (not mid-prices) so candles
        have realistic OHLC ranges matching exchange kline data.
        """
        now = int(time.time())
        candle_ts = now - (now % _CANDLE_INTERVAL_S)

        if self._current_candle is None or self._current_candle.ts != candle_ts:
            # Close previous candle
            if self._current_candle is not None:
                self._candles.append(self._current_candle)
            # Start new candle
            self._current_candle = OHLCCandle(
                ts=candle_ts, open=price, high=price,
                low=price, close=price, volume=1,
            )
        else:
            c = self._current_candle
            c.high = max(c.high, price)
            c.low = min(c.low, price)
            c.close = price
            c.volume += 1

    def add_order_marker(self, side: str, price: float) -> None:
        """Mark an order on the candle chart."""
        idx = len(self._candles)
        if side.upper() == "BUY":
            self._buy_markers.append((idx, price))
            if len(self._buy_markers) > 100:
                self._buy_markers = self._buy_markers[-50:]
        else:
            self._sell_markers.append((idx, price))
            if len(self._sell_markers) > 100:
                self._sell_markers = self._sell_markers[-50:]

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._refresh_loop())
        log.info("Terminal dashboard started (refresh every %.1fs)", self.refresh_s)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _refresh_loop(self) -> None:
        root = logging.getLogger()
        suppressed: list[logging.Handler] = []
        for h in root.handlers[:]:
            if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
                root.removeHandler(h)
                suppressed.append(h)
        try:
            try:
                initial = self._render()
            except Exception as exc:
                initial = Text(f"Dashboard starting... (init error: {exc})")

            with Live(initial, refresh_per_second=10, screen=True) as live:
                self._live = live
                while self._running:
                    try:
                        self._record_snapshot()
                        live.update(self._render())
                    except Exception:
                        pass
                    await asyncio.sleep(self.refresh_s)
        except Exception as exc:
            for h in suppressed:
                root.addHandler(h)
            suppressed.clear()
            log.error("Dashboard failed to start: %s", exc, exc_info=True)
        finally:
            for h in suppressed:
                root.addHandler(h)

    def _record_snapshot(self) -> None:
        self._equity_history.append(self.state.equity)
        if len(self._equity_history) > _MAX_EQUITY_POINTS:
            self._equity_history = self._equity_history[-_MAX_EQUITY_POINTS:]

    # ── Layout construction ───────────────────────────────────────

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=14),
            Layout(name="right", ratio=20),
        )
        layout["left"].split_column(
            Layout(name="status", size=12),
            Layout(name="stats", ratio=1),
            Layout(name="trades", size=15),
        )
        layout["right"].split_column(
            Layout(name="candles", ratio=3),
            Layout(name="bottom", ratio=2),
        )
        layout["bottom"].split_row(
            Layout(name="equity", ratio=1),
            Layout(name="signals", ratio=1),
        )
        return layout

    def _render(self) -> Layout:
        layout = self._make_layout()
        layout["status"].update(self._build_status_panel())
        layout["stats"].update(self._build_stats_panel())
        layout["trades"].update(self._build_trades_panel())
        layout["candles"].update(self._build_candle_panel())
        layout["equity"].update(self._build_equity_panel())
        layout["signals"].update(self._build_signals_panel())
        return layout

    # ── Status panel (system + bot info) ──────────────────────────

    def _build_status_panel(self) -> Panel:
        st = self.state
        cfg = self.config

        # Refresh system info every 5 seconds
        now = time.time()
        if now - self._sys_info_ts > 5:
            self._sys_info = _get_system_info()
            self._sys_info_ts = now
        si = self._sys_info

        profile = cfg.get("profile", {}).get("name", "default")
        exchange = cfg.get("exchange", "bybit").upper()
        symbols = cfg.get("symbols", [])

        status_text = Text("KILLED", style="bold red") if st.kill_switch else Text("LIVE", style="bold green")
        regime = Text(st.vol_regime or "normal", style="yellow")

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", min_width=9)
        grid.add_column(min_width=20)
        grid.add_row("Status", status_text)
        grid.add_row("Exchange", Text(f"{exchange} | {profile.upper()}", style="bold"))
        grid.add_row("Symbols", Text(", ".join(s[:6] for s in symbols[:5]), style="cyan"))
        grid.add_row("Regime", regime)
        grid.add_row("Latency", Text(f"{st.latency_ms}ms (ema {st.latency_ema:.0f}ms)"))

        # System info line
        cpu_s = f"{si.get('cpu', 0):.0f}%" if si.get("cpu", -1) >= 0 else "n/a"
        temp_s = f"{si['temp']:.0f}C" if "temp" in si else "n/a"
        mem_s = f"{si.get('mem_pct', 0):.0f}%"
        up_s = si.get("uptime", "n/a")
        grid.add_row("System",
                      Text(f"CPU {cpu_s}  Temp {temp_s}  Mem {mem_s}  Up {up_s}", style="dim"))
        grid.add_row("Time", Text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), style="dim"))

        return Panel(grid, title="SMALLFISH", title_align="left", border_style="cyan")

    # ── Stats panel (metrics + positions + 7D) ────────────────────

    def _build_stats_panel(self) -> Panel:
        st = self.state
        cfg = self.config

        daily = st.daily_pnl
        daily_pct = daily / max(st.daily_start_equity, 0.01) * 100
        pnl_style = "green" if daily >= 0 else "red"

        tc = st.trade_count
        wc, lc = st.win_count, st.loss_count
        wr = wc / max(tc, 1) * 100
        dd = st.drawdown * 100
        mdd = st.max_drawdown * 100

        # Compute avg win, avg loss, profit factor from completed trades
        completed = list(st.completed_trades.get())
        wins = [t.pnl for t in completed if t.pnl > 0]
        losses = [t.pnl for t in completed if t.pnl < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_win / max(gross_loss, 0.01) if gross_loss > 0 else 0
        best = max((t.pnl for t in completed), default=0)
        worst = min((t.pnl for t in completed), default=0)

        dir_map = {1: ("LONG", "green"), -1: ("SHORT", "red")}
        dir_text, dir_style = dir_map.get(st.last_direction, ("FLAT", "dim"))

        # Count signal agreement
        scores = st.last_scores
        agree_l = sum(1 for v in scores.values() if v > 0.05) if scores else 0
        agree_s = sum(1 for v in scores.values() if v < -0.05) if scores else 0
        min_sig = cfg.get("min_signals", 3)

        body = Table.grid()
        body.add_column()

        # ── Today section ──
        today = Table.grid(padding=(0, 1))
        today.add_column(style="dim", min_width=10)
        today.add_column()
        today.add_row("Equity", Text(f"${st.equity:.2f}", style="bold"))
        today.add_row("PnL", Text(f"${daily:+.2f} ({daily_pct:+.1f}%)", style=pnl_style))
        today.add_row("Trades", f"{tc}  (W:{wc}  L:{lc})")
        today.add_row("Win Rate", f"{wr:.1f}%")
        today.add_row("Avg W / L",
                       Text(f"${avg_win:+.2f}", style="green") +
                       Text(" / ") +
                       Text(f"${avg_loss:+.2f}", style="red"))
        today.add_row("PF", f"{pf:.2f}" if pf > 0 else "n/a")
        today.add_row("Best/Worst",
                       Text(f"${best:+.2f}", style="green") +
                       Text(" / ") +
                       Text(f"${worst:+.2f}", style="red"))
        today.add_row("DD", f"{dd:.2f}%  (max {mdd:.2f}%)")
        today.add_row("Daily R", f"{st.daily_loss_R:.1f}R / {cfg.get('max_daily_R', 10)}R")
        today.add_row("Signal",
                       Text(f"{dir_text} ", style=dir_style) +
                       Text(f"conf={st.last_confidence:.2f}  ") +
                       Text(f"agree={agree_l}L/{agree_s}S (min {min_sig})", style="dim"))

        body.add_row(Text("── TODAY ──", style="bold cyan"))
        body.add_row(today)

        # ── Positions ──
        body.add_row(Text("── POSITIONS ──", style="bold cyan"))
        if st.positions:
            for sym, pos in st.positions.items():
                side_str = "LONG" if pos.side.value == 1 else "SHORT"
                side_style = "green" if pos.side.value == 1 else "red"
                up = pos.unrealized_pnl
                up_style = "green" if up >= 0 else "red"
                body.add_row(Text.assemble(
                    ("  " + side_str + " ", side_style), f"{sym}\n",
                    f"    qty {pos.quantity:.6f}  @{pos.entry_price:.2f}\n",
                    "    uPnL ", (f"${up:+.4f}", up_style),
                    f"  SL {pos.stop_price:.2f}  TP {pos.tp_price:.2f}",
                ))
        else:
            body.add_row(Text("  No open positions", style="dim"))

        # ── 7D perf ──
        body.add_row(Text("── 7D PERFORMANCE ──", style="bold cyan"))

        # Show backtest ROI if available
        if self._backfill_roi is not None:
            roi_style = "green" if self._backfill_roi >= 0 else "red"
            body.add_row(Text.assemble(
                "  Backtest ROI  ",
                (f"{self._backfill_roi:+.1f}%", roi_style),
                f"  {self._backfill_trades} trades  WR {self._backfill_wr:.0f}%",
            ))

        perf = self._calc_7d()
        if perf:
            for label, pnl, w, l in perf:
                ps = f"${pnl:+.2f}"
                style = "green" if pnl >= 0 else "red"
                body.add_row(Text.assemble(
                    f"  {label}  ", (f"{ps:>9}", style), f"  {w}W/{l}L",
                ))
            tot_p = sum(p for _, p, _, _ in perf)
            tot_w = sum(w for _, _, w, _ in perf)
            tot_l = sum(l for _, _, _, l in perf)
            ts = f"${tot_p:+.2f}"
            style = "green" if tot_p >= 0 else "red"
            body.add_row(Text.assemble(
                "  TOTAL  ", (f"{ts:>9}", style), f"  {tot_w}W/{tot_l}L",
            ))
        elif self._backfill_roi is None:
            body.add_row(Text("  No trade history", style="dim"))

        return Panel(body, title="METRICS", title_align="left", border_style="cyan")

    # ── Trades panel ──────────────────────────────────────────────

    def _build_trades_panel(self) -> Panel:
        completed = list(self.state.completed_trades.get())

        if not completed:
            return Panel(
                Text("  No completed trades", style="dim"),
                title="RECENT TRADES",
                title_align="left",
                border_style="yellow",
            )

        table = Table(
            show_header=True, header_style="bold dim",
            box=None, padding=(0, 1),
        )
        table.add_column("S", width=1)
        table.add_column("Symbol", min_width=7)
        table.add_column("R", justify="right", min_width=7)
        table.add_column("PnL", justify="right", min_width=10)
        table.add_column("Reason", min_width=5)

        for t in reversed(completed[-10:]):
            sd = "B" if t.side.value == 1 else "S"
            sym = t.symbol[:7]
            r_s = f"{t.pnl_R:+.2f}R"
            p_s = f"${t.pnl:+.4f}"
            rsn = (t.exit_reason or "")[:8]
            style = "green" if t.pnl >= 0 else "red"
            table.add_row(sd, sym, Text(r_s, style=style),
                          Text(p_s, style=style), Text(rsn, style="dim"))

        return Panel(table, title="RECENT TRADES", title_align="left",
                     border_style="yellow")

    # ── Candlestick chart panel ───────────────────────────────────

    def _build_candle_panel(self) -> Panel:
        candles = list(self._candles)
        # Include current forming candle
        if self._current_candle is not None:
            candles.append(self._current_candle)

        if len(candles) < 2:
            return Panel(
                Text("  Waiting for price data...", style="dim"),
                title="PRICE (1m candles)",
                title_align="left",
                border_style="blue",
            )

        try:
            from candlestick_chart import Candle, Chart

            try:
                cols, rows = os.get_terminal_size()
            except OSError:
                cols, rows = 160, 50

            # Size the chart to fill the panel
            chart_w = max(40, cols * 20 // 34 - 4)
            chart_h = max(10, int(rows * 0.55) - 4)

            # Show last N candles that fit
            max_candles = max(10, chart_w // 3)
            display = candles[-max_candles:]

            cc_candles = [
                Candle(open=c.open, close=c.close, high=c.high, low=c.low)
                for c in display
            ]

            # Determine symbol from active positions or config
            syms = list(self.state.positions.keys())
            sym_label = syms[0] if syms else (
                self.config.get("symbols", [""])[0] if self.config.get("symbols") else ""
            )
            last_price = display[-1].close

            chart = Chart(cc_candles, title=f"{sym_label} ${last_price:,.2f}")
            chart.update_size(chart_w, chart_h)
            chart.set_bull_color(0, 200, 80)
            chart.set_bear_color(200, 60, 60)
            chart.set_volume_pane_enabled(False)

            # Draw position lines info as subtitle
            pos_info = ""
            for pos in self.state.positions.values():
                side_s = "L" if pos.side.value == 1 else "S"
                pos_info += (f"  [{side_s}] entry={pos.entry_price:.2f}"
                             f" SL={pos.stop_price:.2f} TP={pos.tp_price:.2f}")

            title = "PRICE (1m candles)"
            if pos_info:
                title += pos_info

            return Panel(chart, title=title, title_align="left",
                         border_style="blue")
        except ImportError:
            # Fallback to plotext line if candlestick-chart not installed
            return self._build_candle_fallback(candles)

    def _build_candle_fallback(self, candles: list) -> Panel:
        """Fallback price chart using plotext when candlestick-chart is missing."""
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 160, 50

        chart_w = max(40, cols * 20 // 34 - 4)
        chart_h = max(8, int(rows * 0.55) - 4)

        plt.clf()
        plt.clt()
        plt.plotsize(chart_w, chart_h)
        plt.title("Price (1m)")
        plt.theme("dark")

        closes = [c.close for c in candles[-60:]]
        xs = list(range(len(closes)))
        plt.plot(xs, closes, color="white", label="close")

        for pos in self.state.positions.values():
            plt.hline(pos.entry_price, color="cyan")
            plt.hline(pos.stop_price, color="red")
            plt.hline(pos.tp_price, color="green")

        return Panel(Text.from_ansi(plt.build()), title="PRICE (1m)",
                     title_align="left", border_style="blue")

    # ── Equity curve panel ────────────────────────────────────────

    def _build_equity_panel(self) -> Panel:
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 160, 50

        chart_w = max(30, cols * 10 // 34 - 4)
        chart_h = max(8, int(rows * 0.35) - 4)

        plt.clf()
        plt.clt()
        plt.plotsize(chart_w, chart_h)
        plt.theme("dark")

        eq = self._equity_history
        if len(eq) < 2:
            plt.scatter([0], [self.state.equity], marker="dot", label="current")
        else:
            xs = list(range(len(eq)))
            color = "green" if eq[-1] >= eq[0] else "red"
            pct = (eq[-1] - eq[0]) / max(eq[0], 0.01) * 100
            plt.plot(xs, eq, color=color,
                     label=f"${eq[-1]:.2f} ({pct:+.1f}%)")

        return Panel(Text.from_ansi(plt.build()), title="EQUITY",
                     title_align="left", border_style="green")

    # ── Signals panel ─────────────────────────────────────────────

    def _build_signals_panel(self) -> Panel:
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 160, 50

        chart_w = max(30, cols * 10 // 34 - 4)
        chart_h = max(8, int(rows * 0.35) - 4)

        plt.clf()
        plt.clt()
        plt.plotsize(chart_w, chart_h)
        plt.theme("dark")

        scores = self.state.last_scores
        if not scores:
            plt.scatter([0], [0], marker="dot", label="waiting...")
        else:
            # Signal keys: each has _long/_short — show net (long - short)
            signal_pairs = [
                ("OBI",    "obi_long",    "obi_short"),
                ("PRT",    "prt_long",    "prt_short"),
                ("UMOM",   "umom_long",   "umom_short"),
                ("LTB",    "ltb_long",    "ltb_short"),
                ("SWEEP",  "sweep_up",    "sweep_down"),
                ("ICE",    "ice_long",    "ice_short"),
                ("VWAP",   "vwap_long",   "vwap_short"),
                ("RGIME",  "regime_long", "regime_short"),
                ("CVD",    "cvd_long",    "cvd_short"),
                ("TPS",    "tps_long",    "tps_short"),
                ("LIQ",    "liq_long",    "liq_short"),
                ("MVR",    "mvr_long",    "mvr_short"),
                ("ABSRB",  "absorb_long", "absorb_short"),
            ]
            names = []
            vals = []
            colors = []
            for label, long_key, short_key in signal_pairs:
                net = scores.get(long_key, 0.0) - scores.get(short_key, 0.0)
                names.append(label)
                vals.append(net)
                colors.append("green" if net > 0.01 else "red" if net < -0.01 else "gray")
            plt.bar(names, vals, color=colors, orientation="horizontal")

        return Panel(Text.from_ansi(plt.build()), title="SIGNALS",
                     title_align="left", border_style="magenta")

    # ── 7-day performance ─────────────────────────────────────────

    def _calc_7d(self) -> list[tuple[str, float, int, int]]:
        completed = list(self.state.completed_trades.get())
        if not completed:
            return []

        now = datetime.now(timezone.utc)
        buckets: dict[str, list[float]] = {}
        for i in range(7):
            buckets[(now - timedelta(days=i)).strftime("%m/%d")] = []

        for t in completed:
            key = datetime.fromtimestamp(
                t.exit_time / 1000, tz=timezone.utc
            ).strftime("%m/%d")
            if key in buckets:
                buckets[key].append(t.pnl)

        result: list[tuple[str, float, int, int]] = []
        for i in range(6, -1, -1):
            key = (now - timedelta(days=i)).strftime("%m/%d")
            pnls = buckets[key]
            if pnls:
                result.append((
                    key,
                    sum(pnls),
                    sum(1 for p in pnls if p >= 0),
                    sum(1 for p in pnls if p < 0),
                ))
        return result

    # ── Static one-shot renders for backtest / reporting ──────────

    @staticmethod
    def render_backtest_trades(trades: List[dict], width: int = 60) -> str:
        if not trades:
            return "  (no trades)"

        plt.clf()
        plt.clt()
        plt.plotsize(width, 20)
        plt.title("Trade P&L")
        plt.theme("dark")

        recent = trades[-30:]
        pnls = [t["pnl"] for t in recent]
        colors = ["green" if p >= 0 else "red" for p in pnls]
        labels = [f"{t.get('side', '?')[0]}{t.get('symbol', '?')[:4]}" for t in recent]

        plt.bar(labels, pnls, color=colors)
        return plt.build()

    @staticmethod
    def render_equity_curve(equity_curve: List[Tuple[int, float]],
                            width: int = 60) -> str:
        if not equity_curve:
            return ""

        plt.clf()
        plt.clt()
        plt.plotsize(width, 15)
        plt.title("Equity Curve")
        plt.theme("dark")

        values = [e for _, e in equity_curve]
        xs = list(range(len(values)))
        color = "green" if values[-1] >= values[0] else "red"
        plt.plot(xs, values, color=color,
                 label=f"${values[0]:.2f} -> ${values[-1]:.2f}")

        return plt.build()

    @staticmethod
    def render_signal_weights(config: dict) -> str:
        weights = config.get("weights", {})
        w_vals = weights.get("w", [0.22, 0.10, 0.15])
        v_vals = weights.get("v", [0.08, 0.05, 0.03])
        x_vals = weights.get("x", [0.05, 0.02])
        t_vals = weights.get("t", [0.07, 0.05, 0.06, 0.06, 0.06])

        names = ["OBI", "PRT", "UMOM", "LTB", "SWEEP", "ICE", "VWAP", "REGIME",
                 "CVD", "TPS", "LIQ", "MVR", "ABSORB"]
        vals = w_vals + v_vals + x_vals + t_vals

        plt.clf()
        plt.clt()
        plt.plotsize(50, 15)
        plt.title("Signal Weights")
        plt.theme("dark")

        plt.bar(names, vals, color="blue", orientation="horizontal")
        return plt.build()
