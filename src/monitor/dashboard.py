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

# Y-axis stability: max allowed change per frame (prevents jumps)
_Y_AXIS_MAX_CHANGE_PCT = 0.02  # 2% max Y-axis change per render

# Screen size breakpoints
_TINY_COLS = 50      # < 50 cols = tiny (3.5" display, ~40-50 cols)
_SMALL_COLS = 80     # < 80 cols = small
_TINY_ROWS = 20      # < 20 rows = tiny


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

        # OHLCV candle buffer for candlestick chart (single symbol only)
        self._candles: deque[OHLCCandle] = deque(maxlen=_MAX_CANDLES)
        self._current_candle: Optional[OHLCCandle] = None
        self._last_valid_price: float = 0.0  # for outlier rejection
        symbols = config.get("symbols", [])
        self.chart_symbol: str = symbols[0] if symbols else ""

        # Order markers on candle chart
        self._buy_markers: list[Tuple[int, float]] = []   # (candle_idx, price)
        self._sell_markers: list[Tuple[int, float]] = []

        # Equity history for equity curve
        self._equity_history: list[float] = []

        # System info cache (refreshed every 5s to avoid overhead)
        self._sys_info: dict = {}
        self._sys_info_ts: float = 0

        # Cached terminal size (refreshed each render cycle)
        self._term_cols: int = 160
        self._term_rows: int = 50

        # Backfill ROI from 7D backtest (set by app.py at startup)
        self._backfill_roi: Optional[float] = None
        self._backfill_trades: int = 0
        self._backfill_wr: float = 0.0

        # Y-axis smoothing state (prevents sudden jumps)
        self._y_lo_smooth: float = 0.0
        self._y_hi_smooth: float = 0.0

    # ── Public data-feed methods ──────────────────────────────────

    def load_historical_candles(self, klines: list[dict]) -> None:
        """Pre-fill candle chart with historical kline data from REST.

        Args:
            klines: list of dicts with keys: ts, open, high, low, close, volume
        """
        last_close = 0.0
        seen_ts: set[int] = set()  # dedupe by timestamp

        for k in klines:
            o, h, l, c = k["open"], k["high"], k["low"], k["close"]
            ts = int(k["ts"]) // 1000

            # Skip duplicate timestamps
            if ts in seen_ts:
                continue

            # Sanity check: reject candles where any OHLC deviates >10% from
            # previous close (catches bad data while allowing normal volatility)
            if last_close > 0:
                reject = False
                for val in (o, h, l, c):
                    if abs(val - last_close) / last_close > 0.10:
                        log.debug("Rejected historical candle ts=%s (val=%.2f vs last=%.2f)",
                                  k.get("ts"), val, last_close)
                        reject = True
                        break
                if reject:
                    continue

                # Also reject if OHLC values are internally inconsistent
                if not (l <= o <= h and l <= c <= h):
                    log.debug("Rejected malformed candle ts=%s (O=%.2f H=%.2f L=%.2f C=%.2f)",
                              k.get("ts"), o, h, l, c)
                    continue

            candle = OHLCCandle(
                ts=ts, open=o, high=h, low=l, close=c,
                volume=int(k.get("volume", 0)),
            )
            self._candles.append(candle)
            seen_ts.add(ts)
            last_close = c

        # Seed last_valid_price from historical data for live outlier rejection
        if self._candles:
            self._last_valid_price = self._candles[-1].close
            self._current_candle = None  # will start fresh on next tick
            # Initialize Y-axis smoothing from historical data
            self._y_lo_smooth = min(c.low for c in self._candles)
            self._y_hi_smooth = max(c.high for c in self._candles)

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
        Rejects outlier prices (>10% deviation from last known price to prevent
        Y-axis jumps from bad ticks while allowing normal intra-candle volatility).
        """
        if price <= 0:
            return

        # Outlier rejection: skip any price that deviates >10% from last known
        # (allows normal volatility but catches bad data / wrong symbol prices)
        if self._last_valid_price > 0:
            deviation = abs(price - self._last_valid_price) / self._last_valid_price
            if deviation > 0.10:
                log.debug("Rejected outlier price %.2f (last valid %.2f, dev %.1f%%)",
                          price, self._last_valid_price, deviation * 100)
                return
        self._last_valid_price = price

        now = int(time.time())
        candle_ts = now - (now % _CANDLE_INTERVAL_S)

        if self._current_candle is None or self._current_candle.ts != candle_ts:
            # Close previous candle — but only append if not already in buffer
            if self._current_candle is not None:
                # Avoid duplicates: check if last candle in buffer has same ts
                if not self._candles or self._candles[-1].ts != self._current_candle.ts:
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
                    except Exception as exc:
                        log.debug("Dashboard render error: %s", exc)
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

    def _is_tiny(self) -> bool:
        """Check if terminal is tiny (3.5" display)."""
        return self._term_cols < _TINY_COLS or self._term_rows < _TINY_ROWS

    def _is_small(self) -> bool:
        """Check if terminal is small."""
        return self._term_cols < _SMALL_COLS

    def _make_layout(self) -> Layout:
        layout = Layout()

        # Tiny screens: single column, NO charts - just info
        if self._is_tiny():
            layout.split_column(
                Layout(name="status", size=4),
                Layout(name="stats", size=10),
                Layout(name="trades", ratio=1),
            )
            return layout

        # Small screens: two columns, NO charts - detailed info
        if self._is_small():
            layout.split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )
            layout["left"].split_column(
                Layout(name="status", size=6),
                Layout(name="stats", ratio=1),
            )
            layout["right"].split_column(
                Layout(name="signals", size=12),
                Layout(name="trades", ratio=1),
            )
            return layout

        # Normal/large screens: full layout with charts
        layout.split_row(
            Layout(name="left", ratio=14),
            Layout(name="right", ratio=20),
        )
        layout["left"].split_column(
            Layout(name="status", size=10),
            Layout(name="stats", ratio=1),
            Layout(name="trades", size=14),
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

    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal size, cached per render cycle."""
        try:
            return os.get_terminal_size()
        except OSError:
            return (160, 50)

    def _render(self) -> Layout:
        # Cache terminal size once per render cycle BEFORE making layout
        self._term_cols, self._term_rows = self._get_terminal_size()
        layout = self._make_layout()

        # Build panels based on screen size
        if self._is_tiny():
            panels = {
                "status": self._build_status_panel_compact,
                "stats": self._build_stats_panel_tiny,
                "trades": self._build_trades_panel_compact,
            }
        elif self._is_small():
            panels = {
                "status": self._build_status_panel_compact,
                "stats": self._build_stats_panel_small,
                "signals": self._build_signals_panel_text,
                "trades": self._build_trades_panel_compact,
            }
        else:
            panels = {
                "status": self._build_status_panel,
                "stats": self._build_stats_panel,
                "trades": self._build_trades_panel,
                "candles": self._build_candle_panel,
                "equity": self._build_equity_panel,
                "signals": self._build_signals_panel,
            }

        for name, builder in panels.items():
            try:
                layout[name].update(builder())
            except Exception as exc:
                log.debug("Panel %s error: %s", name, exc)
                layout[name].update(Panel(Text(f"err", style="red")))
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

        profile_cfg = cfg.get("profile")
        profile = profile_cfg.get("name", "default") if isinstance(profile_cfg, dict) else str(profile_cfg or "default")
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

    def _build_status_panel_compact(self) -> Panel:
        """Compact status for tiny/small screens."""
        st = self.state
        cfg = self.config

        profile_cfg = cfg.get("profile")
        profile = (profile_cfg.get("name", "def") if isinstance(profile_cfg, dict) else str(profile_cfg or "def"))[:3]
        exchange = cfg.get("exchange", "bybit")[:3].upper()
        sym = st.last_symbol or (cfg.get("symbols", [""])[0][:6] if cfg.get("symbols") else "")

        status = "X" if st.kill_switch else "OK"
        status_style = "red" if st.kill_switch else "green"

        # Single line compact info
        line1 = Text.assemble(
            (f"[{status}]", status_style), " ",
            (f"{exchange}/{profile}", "bold"), " ",
            (f"{sym}", "cyan"),
        )

        dir_map = {1: ("L", "green"), -1: ("S", "red"), 0: ("-", "dim")}
        dir_s, dir_style = dir_map.get(st.last_direction, ("-", "dim"))

        line2 = Text.assemble(
            (f"${st.equity:.0f}", "bold"), " ",
            (f"{dir_s}", dir_style),
            (f"{st.last_confidence:.0%}", "yellow"), " ",
            (f"{st.latency_ms}ms", "dim"),
        )

        content = Text.assemble(line1, "\n", line2)
        return Panel(content, title="SF", border_style="cyan", padding=(0, 0))

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

        # Get focused symbol (the one with the most recent scores)
        focused_sym = st.last_symbol or (cfg.get("symbols", [""])[0] if cfg.get("symbols") else "")

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
        today.add_row("Focus", Text(focused_sym or "none", style="cyan bold"))
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

        # ── Grid status (if active) ──
        if self.grid and self.grid.enabled:
            body.add_row(Text("── GRID ──", style="bold cyan"))
            for sym, status in self.grid.all_status().items():
                if status.get("active"):
                    grid_pnl = status.get("total_pnl", 0)
                    grid_style = "green" if grid_pnl >= 0 else "red"
                    body.add_row(Text.assemble(
                        f"  {sym[:8]} ",
                        f"center={status.get('center', 0):.2f} ",
                        f"trips={status.get('round_trips', 0)} ",
                        ("pnl=", "dim"), (f"${grid_pnl:+.4f}", grid_style),
                    ))
                    body.add_row(Text.assemble(
                        f"    buys={status.get('buy_levels', 0)} ",
                        f"(filled {status.get('filled_buys', 0)}) ",
                        f"sells={status.get('sell_levels', 0)} ",
                        f"(filled {status.get('filled_sells', 0)}) ",
                        f"pending={status.get('pending_orders', 0)}",
                    ))

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

    def _build_stats_panel_compact(self) -> Panel:
        """Compact stats for tiny screens."""
        st = self.state

        daily = st.daily_pnl
        daily_pct = daily / max(st.daily_start_equity, 0.01) * 100
        pnl_style = "green" if daily >= 0 else "red"

        tc = st.trade_count
        wc = st.win_count
        wr = wc / max(tc, 1) * 100
        dd = st.drawdown * 100

        # Very compact: 3-4 lines max
        lines = []

        # Line 1: PnL
        lines.append(Text.assemble(
            "PnL ", (f"${daily:+.2f}", pnl_style),
            (f" ({daily_pct:+.0f}%)", pnl_style),
        ))

        # Line 2: Trades + WR
        lines.append(Text.assemble(
            f"T:{tc} W:{wc} ",
            (f"WR:{wr:.0f}%", "green" if wr >= 50 else "red"),
        ))

        # Line 3: DD + Daily R
        max_R = self.config.get('max_daily_R', 10)
        lines.append(Text.assemble(
            f"DD:{dd:.1f}% ",
            f"R:{st.daily_loss_R:.1f}/{max_R}",
        ))

        # Line 4: Position (if any)
        if st.positions:
            sym, pos = next(iter(st.positions.items()))
            side_s = "L" if pos.side.value == 1 else "S"
            side_style = "green" if pos.side.value == 1 else "red"
            up = pos.unrealized_pnl
            up_style = "green" if up >= 0 else "red"
            lines.append(Text.assemble(
                (side_s, side_style), f" {sym[:5]} ",
                (f"${up:+.2f}", up_style),
            ))

        content = Text("\n").join(lines)
        return Panel(content, title="STATS", border_style="cyan", padding=(0, 0))

    def _build_stats_panel_tiny(self) -> Panel:
        """Detailed stats panel for tiny screens (no charts)."""
        st = self.state
        cfg = self.config

        daily = st.daily_pnl
        daily_pct = daily / max(st.daily_start_equity, 0.01) * 100
        pnl_style = "green" if daily >= 0 else "red"

        # Calculate ROI from start
        roi = (st.equity - cfg.get("initial_equity", st.equity)) / max(cfg.get("initial_equity", 1), 1) * 100
        roi_style = "green" if roi >= 0 else "red"

        tc = st.trade_count
        wc, lc = st.win_count, st.loss_count
        wr = wc / max(tc, 1) * 100
        dd = st.drawdown * 100
        max_R = cfg.get('max_daily_R', 10)

        # Direction and confidence
        dir_map = {1: ("LONG", "green"), -1: ("SHORT", "red"), 0: ("FLAT", "dim")}
        dir_text, dir_style = dir_map.get(st.last_direction, ("FLAT", "dim"))

        lines = []
        # Row 1: Equity + ROI
        lines.append(Text.assemble(
            (f"${st.equity:.0f}", "bold"), " ",
            ("ROI:", "dim"), (f"{roi:+.1f}%", roi_style),
        ))
        # Row 2: Today PnL
        lines.append(Text.assemble(
            ("PnL:", "dim"), (f"${daily:+.2f}", pnl_style),
            (f"({daily_pct:+.0f}%)", pnl_style),
        ))
        # Row 3: Trades + WR
        lines.append(Text.assemble(
            f"T:{tc} ", ("W:", "green"), f"{wc} ", ("L:", "red"), f"{lc} ",
            (f"WR:{wr:.0f}%", "green" if wr >= 50 else "red"),
        ))
        # Row 4: DD + Daily R
        lines.append(Text.assemble(
            f"DD:{dd:.1f}% R:{st.daily_loss_R:.1f}/{max_R}",
        ))
        # Row 5: Signal direction + confidence
        lines.append(Text.assemble(
            (dir_text, dir_style), " ",
            ("conf:", "dim"), (f"{st.last_confidence:.0%}", "yellow"),
        ))
        # Row 6: Focused symbol
        focused = st.last_symbol or cfg.get("symbols", [""])[0] if cfg.get("symbols") else ""
        lines.append(Text.assemble(
            ("Focus:", "dim"), (f" {focused}", "cyan"),
        ))
        # Row 7: Position info (if any)
        if st.positions:
            sym, pos = next(iter(st.positions.items()))
            side_s = "L" if pos.side.value == 1 else "S"
            side_style = "green" if pos.side.value == 1 else "red"
            up = pos.unrealized_pnl
            up_style = "green" if up >= 0 else "red"
            lines.append(Text.assemble(
                ("Pos:", "dim"), (f" {side_s}", side_style),
                f" {sym[:6]} ", (f"${up:+.2f}", up_style),
            ))

        content = Text("\n").join(lines)
        return Panel(content, title="INFO", border_style="cyan", padding=(0, 0))

    def _build_stats_panel_small(self) -> Panel:
        """Detailed stats panel for small screens (no charts)."""
        st = self.state
        cfg = self.config

        daily = st.daily_pnl
        daily_pct = daily / max(st.daily_start_equity, 0.01) * 100
        pnl_style = "green" if daily >= 0 else "red"

        # Calculate ROI
        initial = cfg.get("initial_equity", st.equity)
        roi = (st.equity - initial) / max(initial, 1) * 100
        roi_style = "green" if roi >= 0 else "red"

        tc = st.trade_count
        wc, lc = st.win_count, st.loss_count
        wr = wc / max(tc, 1) * 100
        dd = st.drawdown * 100
        mdd = st.max_drawdown * 100
        max_R = cfg.get('max_daily_R', 10)

        # Avg win/loss from completed trades
        completed = list(st.completed_trades.get())
        wins = [t.pnl for t in completed if t.pnl > 0]
        losses = [t.pnl for t in completed if t.pnl < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_win / max(gross_loss, 0.01) if gross_loss > 0 else 0

        # Direction and confidence
        dir_map = {1: ("LONG", "green"), -1: ("SHORT", "red"), 0: ("FLAT", "dim")}
        dir_text, dir_style = dir_map.get(st.last_direction, ("FLAT", "dim"))

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", min_width=8)
        grid.add_column()

        grid.add_row("Equity", Text(f"${st.equity:.2f}", style="bold"))
        grid.add_row("ROI", Text(f"{roi:+.1f}%", style=roi_style))
        grid.add_row("PnL", Text(f"${daily:+.2f} ({daily_pct:+.1f}%)", style=pnl_style))
        grid.add_row("Trades", f"{tc} (W:{wc} L:{lc})")
        grid.add_row("WinRate", Text(f"{wr:.1f}%", style="green" if wr >= 50 else "red"))
        grid.add_row("Avg W/L", Text.assemble(
            (f"${avg_win:+.2f}", "green"), "/", (f"${avg_loss:+.2f}", "red")
        ))
        grid.add_row("PF", f"{pf:.2f}" if pf > 0 else "n/a")
        grid.add_row("DD", f"{dd:.1f}% (max {mdd:.1f}%)")
        grid.add_row("Daily R", f"{st.daily_loss_R:.1f} / {max_R}")
        grid.add_row("Signal", Text.assemble(
            (dir_text, dir_style), " ", (f"{st.last_confidence:.0%}", "yellow")
        ))

        # Position info
        if st.positions:
            sym, pos = next(iter(st.positions.items()))
            side_s = "LONG" if pos.side.value == 1 else "SHORT"
            side_style = "green" if pos.side.value == 1 else "red"
            up = pos.unrealized_pnl
            up_style = "green" if up >= 0 else "red"
            grid.add_row("Position", Text.assemble(
                (side_s, side_style), f" {sym[:8]} ", (f"${up:+.2f}", up_style)
            ))

        return Panel(grid, title="METRICS", border_style="cyan", padding=(0, 0))

    def _build_signals_panel_text(self) -> Panel:
        """Text-based signal display for small screens (no chart)."""
        st = self.state
        scores = st.last_scores
        focused_sym = st.last_symbol or ""

        if not scores:
            return Panel(
                Text("Waiting for signals...", style="dim"),
                title="SIGNALS",
                border_style="magenta",
                padding=(0, 0),
            )

        # Signal pairs with net value
        signal_pairs = [
            ("OBI", "obi_long", "obi_short"),
            ("PRT", "prt_long", "prt_short"),
            ("UMOM", "umom_long", "umom_short"),
            ("LTB", "ltb_long", "ltb_short"),
            ("SWEEP", "sweep_up", "sweep_down"),
            ("ICE", "ice_long", "ice_short"),
            ("VWAP", "vwap_long", "vwap_short"),
            ("REGIME", "regime_long", "regime_short"),
            ("CVD", "cvd_long", "cvd_short"),
            ("TPS", "tps_long", "tps_short"),
            ("LIQ", "liq_long", "liq_short"),
            ("MVR", "mvr_long", "mvr_short"),
            ("ABSORB", "absorb_long", "absorb_short"),
        ]

        lines = []
        row_items = []
        for i, (label, long_key, short_key) in enumerate(signal_pairs):
            net = scores.get(long_key, 0.0) - scores.get(short_key, 0.0)
            if net > 0.05:
                style = "green"
                arrow = "+"
            elif net < -0.05:
                style = "red"
                arrow = "-"
            else:
                style = "dim"
                arrow = " "
            row_items.append(Text.assemble((f"{label[:3]}{arrow}", style)))

            # 4 signals per row
            if len(row_items) == 4 or i == len(signal_pairs) - 1:
                lines.append(Text(" ").join(row_items))
                row_items = []

        # Count agreements
        agree_l = sum(1 for v in scores.values() if v > 0.05)
        agree_s = sum(1 for v in scores.values() if v < -0.05)
        min_sig = self.config.get("min_signals", 3)

        lines.append(Text(""))
        lines.append(Text.assemble(
            ("Agree: ", "dim"),
            (f"{agree_l}L", "green"), "/",
            (f"{agree_s}S", "red"),
            (f" (min {min_sig})", "dim"),
        ))

        content = Text("\n").join(lines)
        title = f"SIG {focused_sym[:6]}" if focused_sym else "SIGNALS"
        return Panel(content, title=title, border_style="magenta", padding=(0, 0))

    def _build_trades_panel_compact(self) -> Panel:
        """Compact trade history for small/tiny screens."""
        completed = list(self.state.completed_trades.get())

        if not completed:
            return Panel(
                Text("No trades yet", style="dim"),
                title="TRADES",
                border_style="yellow",
                padding=(0, 0),
            )

        lines = []
        # Show last 6 trades in compact format
        for t in reversed(completed[-6:]):
            side_s = "L" if t.side.value == 1 else "S"
            side_style = "green" if t.side.value == 1 else "red"
            pnl_style = "green" if t.pnl >= 0 else "red"

            # Duration
            dur_ms = t.duration_ms
            if dur_ms >= 60000:
                dur_s = f"{dur_ms // 60000}m"
            else:
                dur_s = f"{dur_ms // 1000}s"

            lines.append(Text.assemble(
                (side_s, side_style), " ",
                f"{t.symbol[:6]} ",
                (f"${t.pnl:+.2f}", pnl_style), " ",
                (f"{t.pnl_R:+.1f}R", pnl_style), " ",
                (dur_s, "dim"),
            ))

        # Summary line
        total_pnl = sum(t.pnl for t in completed)
        total_style = "green" if total_pnl >= 0 else "red"
        lines.append(Text(""))
        lines.append(Text.assemble(
            ("Total: ", "dim"),
            (f"${total_pnl:+.2f}", total_style),
            f" ({len(completed)} trades)",
        ))

        content = Text("\n").join(lines)
        return Panel(content, title="TRADES", border_style="yellow", padding=(0, 0))

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
            expand=True,
        )
        table.add_column("Side", width=4)
        table.add_column("Symbol", min_width=10)
        table.add_column("R-Mult", justify="right", min_width=7)
        table.add_column("PnL", justify="right", min_width=11)
        table.add_column("Duration", justify="right", min_width=7)
        table.add_column("Exit Reason", min_width=12)

        for t in reversed(completed[-10:]):
            sd = "BUY" if t.side.value == 1 else "SELL"
            sd_style = "green" if t.side.value == 1 else "red"
            sym = t.symbol  # full symbol, no truncation
            r_s = f"{t.pnl_R:+.2f}R"
            p_s = f"${t.pnl:+.4f}"

            # Format duration: show minutes/seconds
            dur_ms = t.duration_ms
            if dur_ms >= 60000:
                dur_s = f"{dur_ms // 60000}m{(dur_ms % 60000) // 1000}s"
            else:
                dur_s = f"{dur_ms // 1000}s"

            # Full exit reason without truncation
            rsn = t.exit_reason or "unknown"
            pnl_style = "green" if t.pnl >= 0 else "red"

            table.add_row(
                Text(sd, style=sd_style),
                sym,
                Text(r_s, style=pnl_style),
                Text(p_s, style=pnl_style),
                Text(dur_s, style="dim"),
                Text(rsn, style="dim"),
            )

        return Panel(table, title="RECENT TRADES", title_align="left",
                     border_style="yellow")

    # ── Candlestick chart panel ───────────────────────────────────

    def _build_candle_panel(self) -> Panel:
        candles = list(self._candles)
        # Include current forming candle (avoid duplicate if already in buffer)
        if self._current_candle is not None:
            if not candles or candles[-1].ts != self._current_candle.ts:
                candles.append(self._current_candle)

        if len(candles) < 2:
            title = "PRICE" if self._is_tiny() else "PRICE (1m candles)"
            return Panel(
                Text("Waiting..." if self._is_tiny() else "  Waiting for price data...", style="dim"),
                title=title,
                title_align="left",
                border_style="blue",
            )

        cols, rows = self._term_cols, self._term_rows

        # Adaptive chart sizing
        if self._is_tiny():
            chart_w = max(20, cols - 4)
            chart_h = max(4, rows - 16)
            max_candles = max(5, chart_w // 3)
        elif self._is_small():
            chart_w = max(30, cols // 2 - 4)
            chart_h = max(6, int(rows * 0.4))
            max_candles = max(10, chart_w // 2)
        else:
            chart_w = max(40, cols * 20 // 34 - 4)
            chart_h = max(5, int(rows * 0.55) - 4)
            max_candles = max(10, chart_w // 2)

        display = candles[-max_candles:]

        plt.clf()
        plt.clt()
        plt.plotsize(chart_w, chart_h)
        plt.theme("dark")

        # Compute Y-axis range from displayed candles ONLY (stable base)
        y_lo_raw = min(c.low for c in display)
        y_hi_raw = max(c.high for c in display)

        # Sanity check: if range is too small (flat price), expand it
        if y_hi_raw - y_lo_raw < y_lo_raw * 0.001:
            y_margin = y_lo_raw * 0.005
        else:
            y_margin = (y_hi_raw - y_lo_raw) * 0.08  # 8% margin for visual clarity

        y_lo_target = y_lo_raw - y_margin
        y_hi_target = y_hi_raw + y_margin

        # Y-axis smoothing: limit change per frame to prevent jumps
        if self._y_lo_smooth == 0.0 or self._y_hi_smooth == 0.0:
            # First frame: initialize directly
            self._y_lo_smooth = y_lo_target
            self._y_hi_smooth = y_hi_target
        else:
            # Smooth transition: move at most 2% toward target per frame
            # But if price is outside current range, expand immediately
            max_change = (self._y_hi_smooth - self._y_lo_smooth) * _Y_AXIS_MAX_CHANGE_PCT

            # Expand immediately if price goes outside, contract slowly
            if y_lo_target < self._y_lo_smooth:
                self._y_lo_smooth = y_lo_target  # expand down immediately
            else:
                self._y_lo_smooth += min(y_lo_target - self._y_lo_smooth, max_change)

            if y_hi_target > self._y_hi_smooth:
                self._y_hi_smooth = y_hi_target  # expand up immediately
            else:
                self._y_hi_smooth -= min(self._y_hi_smooth - y_hi_target, max_change)

        y_lo = self._y_lo_smooth
        y_hi = self._y_hi_smooth

        # Lock Y-axis BEFORE drawing anything (prevents plotext auto-scaling)
        plt.ylim(y_lo, y_hi)

        # Draw OHLC candles: wick (high-low line) + body (open-close bar)
        for i, c in enumerate(display):
            color = "green" if c.close >= c.open else "red"
            # Wick
            plt.plot([i, i], [c.low, c.high], color=color)
            # Body
            body_lo = min(c.open, c.close)
            body_hi = max(c.open, c.close)
            if body_hi == body_lo:
                body_hi = body_lo + y_margin * 0.1  # doji: tiny visible body
            plt.plot([i, i], [body_lo, body_hi], color=color, marker="hd")

        # Position entry/SL/TP lines — ONLY if strictly within candle range
        # (never extend Y-axis for these — they're info overlays, not data)
        for pos in self.state.positions.values():
            # Only show lines that fall within the current candle Y range
            if pos.entry_price > 0 and y_lo <= pos.entry_price <= y_hi:
                plt.hline(pos.entry_price, color="cyan")
            if pos.stop_price > 0 and y_lo <= pos.stop_price <= y_hi:
                plt.hline(pos.stop_price, color="red+")
            if pos.tp_price > 0 and y_lo <= pos.tp_price <= y_hi:
                plt.hline(pos.tp_price, color="green+")

        # Determine symbol + price for title
        sym_label = self.chart_symbol or (
            self.config.get("symbols", [""])[0] if self.config.get("symbols") else ""
        )
        last_price = display[-1].close

        # Adaptive title based on screen size
        if self._is_tiny():
            # Very compact: just symbol and price
            title = f"{sym_label[:4]} ${last_price:,.0f}"
        elif self._is_small():
            title = f"{sym_label} ${last_price:,.2f}"
        else:
            pos_info = ""
            for pos in self.state.positions.values():
                side_s = "L" if pos.side.value == 1 else "S"
                pos_info += (f"  [{side_s}] entry={pos.entry_price:.2f}"
                             f" SL={pos.stop_price:.2f} TP={pos.tp_price:.2f}")
            title = f"PRICE {sym_label} ${last_price:,.2f}"
            if pos_info:
                title += pos_info

        return Panel(Text.from_ansi(plt.build()), title=title,
                     title_align="left", border_style="blue")

    # ── Equity curve panel ────────────────────────────────────────

    def _build_equity_panel(self) -> Panel:
        cols, rows = self._term_cols, self._term_rows

        chart_w = max(30, cols * 10 // 34 - 4)
        chart_h = max(5, int(rows * 0.35) - 4)

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
        cols, rows = self._term_cols, self._term_rows
        st = self.state

        # Adaptive chart sizing
        if self._is_tiny():
            chart_w = max(15, cols - 4)
            chart_h = max(4, rows // 3)
        elif self._is_small():
            chart_w = max(25, cols // 2 - 4)
            chart_h = max(5, int(rows * 0.3))
        else:
            chart_w = max(30, cols * 10 // 34 - 4)
            chart_h = max(5, int(rows * 0.35) - 6)

        plt.clf()
        plt.clt()
        plt.plotsize(chart_w, chart_h)
        plt.theme("dark")

        scores = st.last_scores
        focused_sym = st.last_symbol or ""

        if not scores:
            plt.scatter([0], [0], marker="dot", label="wait")
        else:
            # Signal keys: each has _long/_short — show net (long - short)
            # Use shorter labels on small screens
            if self._is_tiny():
                signal_pairs = [
                    ("OB",  "obi_long",    "obi_short"),
                    ("PR",  "prt_long",    "prt_short"),
                    ("MO",  "umom_long",   "umom_short"),
                    ("LT",  "ltb_long",    "ltb_short"),
                    ("SW",  "sweep_up",    "sweep_down"),
                    ("CV",  "cvd_long",    "cvd_short"),
                ]
            elif self._is_small():
                signal_pairs = [
                    ("OBI",   "obi_long",    "obi_short"),
                    ("PRT",   "prt_long",    "prt_short"),
                    ("MOM",   "umom_long",   "umom_short"),
                    ("LTB",   "ltb_long",    "ltb_short"),
                    ("SWP",   "sweep_up",    "sweep_down"),
                    ("CVD",   "cvd_long",    "cvd_short"),
                    ("LIQ",   "liq_long",    "liq_short"),
                    ("MVR",   "mvr_long",    "mvr_short"),
                ]
            else:
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

        chart_text = Text.from_ansi(plt.build())

        # Build multi-symbol summary if we have multiple symbols tracked (skip on tiny)
        multi_sym = st.scores_by_symbol
        if len(multi_sym) > 1 and not self._is_tiny():
            summary_parts = []
            # Limit symbols shown based on screen size
            max_syms = 3 if self._is_small() else 5
            for sym, data in sorted(multi_sym.items(), key=lambda x: -x[1].get("conf", 0))[:max_syms]:
                conf = data.get("conf", 0)
                direction = data.get("direction", 0)
                dir_s = "L" if direction == 1 else "S" if direction == -1 else "-"
                dir_style = "green" if direction == 1 else "red" if direction == -1 else "dim"
                is_focused = sym == focused_sym
                sym_style = "bold cyan" if is_focused else "white"
                sym_short = sym[:4] if self._is_small() else sym[:8]
                summary_parts.append(Text.assemble(
                    (sym_short, sym_style), " ",
                    (dir_s, dir_style),
                    (f"{conf:.0%}", "yellow" if conf > 0.5 else "dim"),
                    " "
                ))
            summary_line = Text.assemble(*summary_parts) if summary_parts else Text("")
            content = Text.assemble(chart_text, "\n", summary_line)
        else:
            content = chart_text

        # Compact title for small screens
        if self._is_tiny():
            title = f"SIG"
        elif self._is_small():
            title = f"SIG {focused_sym[:4]}" if focused_sym else "SIG"
        else:
            title = f"SIGNALS ({focused_sym})" if focused_sym else "SIGNALS"

        return Panel(content, title=title,
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
