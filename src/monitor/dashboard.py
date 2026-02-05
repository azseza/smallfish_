"""Terminal Dashboard — split-panel visualization for Smallfish.

Layout: vertical split
  Left panel  (text):   Status, session stats, positions, recent trades, 7D perf
  Right panel (charts): 2x2 grid — Price, Equity, Trade PnL, Signal strength

Usage:
    dashboard = TerminalDashboard(state, config)
    dashboard.start()   # launches async refresh loop
    dashboard.stop()
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import plotext as plt

from core.state import RuntimeState
from core.types import TradeResult

log = logging.getLogger(__name__)

# Maximum data points kept in rolling buffers
_MAX_PRICE_POINTS = 300
_MAX_EQUITY_POINTS = 200

# ANSI escape code pattern for visible-length calculation
_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

# Box-drawing characters
_H, _V = "\u2500", "\u2502"
_TL, _TR, _BL, _BR = "\u250c", "\u2510", "\u2514", "\u2518"
_TT, _BT = "\u252c", "\u2534"


def _ansi_len(text: str) -> int:
    """Visible length of text, excluding ANSI escape sequences."""
    return len(_ANSI_RE.sub("", text))


def _ansi_pad(text: str, width: int) -> str:
    """Pad text to *visual* width, accounting for ANSI codes."""
    pad = width - _ansi_len(text)
    if pad > 0:
        return text + " " * pad
    return text


def _c(text: str, code: str) -> str:
    """Wrap *text* with ANSI colour/style *code*."""
    return f"\033[{code}m{text}\033[0m"


def _pnl_col(val: float, text: str) -> str:
    """Green if non-negative, red if negative."""
    return _c(text, "32") if val >= 0 else _c(text, "31")


class TerminalDashboard:
    """Async terminal dashboard with split-panel layout."""

    def __init__(self, state: RuntimeState, config: dict,
                 grid_strategy=None, refresh_s: float = 2.0):
        self.state = state
        self.config = config
        self.grid = grid_strategy
        self.refresh_s = refresh_s
        self.enabled = config.get("dashboard", {}).get("enabled", False)
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Rolling price data for price chart
        self._prices: deque[float] = deque(maxlen=_MAX_PRICE_POINTS)
        self._price_ts: deque[int] = deque(maxlen=_MAX_PRICE_POINTS)

        # Order markers: (index_approx, price)
        self._buy_markers_x: list[float] = []
        self._buy_markers_y: list[float] = []
        self._sell_markers_x: list[float] = []
        self._sell_markers_y: list[float] = []

        # Equity history for equity curve
        self._equity_history: list[float] = []

    # ── Public data-feed methods ──────────────────────────────────

    def add_price_point(self, mid_price: float) -> None:
        """Feed a new mid-price from a book update."""
        self._prices.append(mid_price)
        self._price_ts.append(int(time.time()))

    def add_order_marker(self, side: str, price: float) -> None:
        """Mark an order on the price chart (side = 'BUY' or 'SELL')."""
        idx = float(len(self._prices) - 1) if self._prices else 0.0
        if side.upper() == "BUY":
            self._buy_markers_x.append(idx)
            self._buy_markers_y.append(price)
        else:
            self._sell_markers_x.append(idx)
            self._sell_markers_y.append(price)
        # Keep marker lists bounded
        if len(self._buy_markers_x) > 200:
            self._buy_markers_x = self._buy_markers_x[-100:]
            self._buy_markers_y = self._buy_markers_y[-100:]
        if len(self._sell_markers_x) > 200:
            self._sell_markers_x = self._sell_markers_x[-100:]
            self._sell_markers_y = self._sell_markers_y[-100:]

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the dashboard refresh loop."""
        if not self.enabled:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._refresh_loop())
        log.info("Terminal dashboard started (refresh every %.1fs)", self.refresh_s)

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _refresh_loop(self) -> None:
        """Periodically redraw the dashboard."""
        while self._running:
            try:
                self._record_snapshot()
                self._render()
            except Exception as e:
                log.debug("Dashboard render error: %s", e)
            await asyncio.sleep(self.refresh_s)

    def _record_snapshot(self) -> None:
        """Record current equity for history."""
        self._equity_history.append(self.state.equity)
        if len(self._equity_history) > _MAX_EQUITY_POINTS:
            self._equity_history = self._equity_history[-_MAX_EQUITY_POINTS:]

    # ── Left panel (text summary) ─────────────────────────────────

    def _build_left_lines(self, w: int, h: int) -> list[str]:
        """Build text summary panel as a list of *w*-wide lines.

        Sections: header, today stats, open positions, recent trades,
        7-day performance.
        """
        lines: list[str] = []

        def ln(text: str = ""):
            lines.append(_ansi_pad(text, w))

        def sep(title: str):
            ln()
            bar = _H * max(0, w - len(title) - 4)
            ln(_c(f"{_H}{_H} {title} {bar}", "1;36"))

        st = self.state
        cfg = self.config

        # ── Header ────────────────────────────────────────────────
        profile = cfg.get("profile", {}).get("name", "default")
        status = _c("KILLED", "31") if st.kill_switch else _c("LIVE", "32")
        regime = _c(st.vol_regime or "normal", "33")

        ln(_c(f" SMALLFISH  [{profile.upper()}]", "1"))
        ln(f" Status {status}   Regime {regime}   {_c(time.strftime('%H:%M:%S'), '2')}")
        ln(f" Latency {st.latency_ms}ms  (ema {st.latency_ema:.0f}ms)")

        # ── Today ─────────────────────────────────────────────────
        sep("TODAY")
        daily = st.daily_pnl
        daily_pct = daily / max(st.daily_start_equity, 0.01) * 100
        pnl_s = f"${daily:+.2f} ({daily_pct:+.1f}%)"
        ln(f"  Equity   ${st.equity:.2f}")
        ln(f"  PnL      {_pnl_col(daily, pnl_s)}")

        tc = st.trade_count
        wc, lc = st.win_count, st.loss_count
        wr = wc / max(tc, 1) * 100
        ln(f"  Trades   {tc}  (W:{wc} L:{lc})")
        ln(f"  WinRate  {wr:.1f}%")

        dd = st.drawdown * 100
        mdd = st.max_drawdown * 100
        ln(f"  DD       {dd:.2f}%  (max {mdd:.2f}%)")
        ln(f"  Daily R  {st.daily_loss_R:.1f}R / {cfg.get('max_daily_R', 10)}R")

        # Confidence / direction
        dir_s = {1: _c("LONG", "32"), -1: _c("SHORT", "31")}.get(
            st.last_direction, _c("FLAT", "2"))
        ln(f"  Signal   {dir_s}  conf {st.last_confidence:.2f}")

        # ── Open positions ────────────────────────────────────────
        sep("POSITIONS")
        if st.positions:
            for sym, pos in st.positions.items():
                sd = _c("LONG", "32") if pos.side.value == 1 else _c("SHORT", "31")
                up = pos.unrealized_pnl
                ln(f"  {sd} {sym}")
                ln(f"    qty {pos.quantity:.6f}  @{pos.entry_price:.2f}")
                up_s = f"${up:+.4f}"
                ln(f"    uPnL {_pnl_col(up, up_s)}  "
                   f"SL {pos.stop_price:.2f}  TP {pos.tp_price:.2f}")
        else:
            ln(_c("  No open positions", "2"))

        # ── Recent trades ─────────────────────────────────────────
        sep("RECENT TRADES")
        completed = list(st.completed_trades.get())
        if completed:
            for t in reversed(completed[-8:]):
                sd = "B" if t.side.value == 1 else "S"
                sym = t.symbol[:7].ljust(7)
                r_s = f"{t.pnl_R:+.2f}R".rjust(7)
                p_s = f"${t.pnl:+.4f}".rjust(10)
                rsn = _c((t.exit_reason or "")[:6], "2")
                ln(f"  {sd} {sym} {_pnl_col(t.pnl, r_s)}"
                   f" {_pnl_col(t.pnl, p_s)} {rsn}")
        else:
            ln(_c("  No completed trades", "2"))

        # ── 7-day performance ─────────────────────────────────────
        sep("7D PERFORMANCE")
        perf = self._calc_7d()
        if perf:
            for label, pnl, wins, losses in perf:
                ps = f"${pnl:+.2f}".rjust(9)
                ln(f"  {label}  {_pnl_col(pnl, ps)}  {wins}W/{losses}L")
            tot_p = sum(p for _, p, _, _ in perf)
            tot_w = sum(wi for _, _, wi, _ in perf)
            tot_l = sum(lo for _, _, _, lo in perf)
            ts = f"${tot_p:+.2f}".rjust(9)
            ln(f"  {'TOTAL':<6}{_pnl_col(tot_p, ts)}  {tot_w}W/{tot_l}L")
        else:
            ln(_c("  No trade history", "2"))

        # Pad remaining lines to fill panel height
        while len(lines) < h:
            ln()
        return lines[:h]

    def _calc_7d(self) -> list[tuple[str, float, int, int]]:
        """Group completed trades by calendar day for the last 7 days.

        Returns [(day_label, pnl_sum, wins, losses), ...] in
        chronological order, skipping days with no trades.
        """
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

    # ── Right panel (2x2 chart grid) ──────────────────────────────

    def _build_chart_str(self, width: int, height: int) -> str:
        """Build a 2x2 plotext chart grid and return as a string."""
        plt.clf()
        plt.clt()
        plt.plotsize(width, height)
        plt.subplots(2, 2)

        self._subplot_price()
        self._subplot_equity()
        self._subplot_pnl()
        self._subplot_signals()

        return plt.build()

    def _subplot_price(self) -> None:
        """Top-left: price line with buy/sell scatter + position lines."""
        plt.subplot(1, 1)
        plt.title("Price")
        plt.theme("dark")

        prices = list(self._prices)
        if not prices:
            plt.scatter([0], [0], marker="dot", label="waiting...")
            return

        xs = list(range(len(prices)))
        plt.plot(xs, prices, color="white", label="mid")

        if self._buy_markers_x:
            bx = [x for x in self._buy_markers_x if x < len(prices)]
            by = self._buy_markers_y[-len(bx):]
            if bx:
                plt.scatter(bx, by, marker="dot", color="green", label="buy")

        if self._sell_markers_x:
            sx = [x for x in self._sell_markers_x if x < len(prices)]
            sy = self._sell_markers_y[-len(sx):]
            if sx:
                plt.scatter(sx, sy, marker="dot", color="red", label="sell")

        for pos in self.state.positions.values():
            plt.hline(pos.entry_price, color="cyan")
            plt.hline(pos.stop_price, color="red")
            plt.hline(pos.tp_price, color="green")

    def _subplot_equity(self) -> None:
        """Top-right: equity curve line chart."""
        plt.subplot(1, 2)
        plt.title("Equity")
        plt.theme("dark")

        eq = self._equity_history
        if len(eq) < 2:
            plt.scatter([0], [self.state.equity], marker="dot", label="current")
            return

        xs = list(range(len(eq)))
        color = "green" if eq[-1] >= eq[0] else "red"
        plt.plot(xs, eq, color=color, label=f"${eq[-1]:.2f}")

    def _subplot_pnl(self) -> None:
        """Bottom-left: per-trade PnL bar chart."""
        plt.subplot(2, 1)
        plt.title("Trade P&L")
        plt.theme("dark")

        completed = list(self.state.completed_trades.get())
        if not completed:
            plt.scatter([0], [0], marker="dot", label="no trades")
            return

        recent = completed[-30:]
        pnls = [t.pnl for t in recent]
        colors = ["green" if p >= 0 else "red" for p in pnls]
        plt.bar(list(range(1, len(pnls) + 1)), pnls, color=colors)

    def _subplot_signals(self) -> None:
        """Bottom-right: signal strength horizontal bars."""
        plt.subplot(2, 2)
        plt.title("Signals")
        plt.theme("dark")

        scores = self.state.last_scores
        if not scores:
            plt.scatter([0], [0], marker="dot", label="waiting...")
            return

        signal_names = ["obi", "prt", "umom", "ltb", "sweep",
                        "ice", "vwap_dev", "vol_regime"]
        names = []
        vals = []
        colors = []
        for name in signal_names:
            v = scores.get(name, 0.0)
            names.append(name)
            vals.append(v)
            colors.append("green" if v > 0 else "red" if v < 0 else "gray")

        plt.bar(names, vals, color=colors, orientation="horizontal")

    # ── Main render ───────────────────────────────────────────────

    def _render(self) -> None:
        """Render the full split-panel dashboard to terminal."""
        try:
            cols = os.get_terminal_size().columns
            rows = os.get_terminal_size().lines
        except OSError:
            cols, rows = 160, 50

        left_w = max(38, cols * 2 // 5)
        right_w = cols - left_w - 3          # 3 border chars: | | |
        body_h = max(20, rows - 3)           # top + bottom borders + margin

        # Build both panels
        left = self._build_left_lines(left_w, body_h)

        chart_str = self._build_chart_str(right_w, body_h)
        right = chart_str.split("\n")
        # Pad / truncate right panel to body_h lines
        while len(right) < body_h:
            right.append("")
        right = right[:body_h]

        # Assemble framed output
        out: list[str] = []
        out.append(_TL + _H * left_w + _TT + _H * right_w + _TR)

        for i in range(body_h):
            l_line = _ansi_pad(left[i] if i < len(left) else "", left_w)
            r_line = _ansi_pad(right[i] if i < len(right) else "", right_w)
            out.append(f"{_V}{l_line}{_V}{r_line}{_V}")

        out.append(_BL + _H * left_w + _BT + _H * right_w + _BR)

        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("\n".join(out))
        sys.stdout.flush()

    # ── Static one-shot renders for backtest / reporting ──────────

    @staticmethod
    def render_backtest_trades(trades: List[dict], width: int = 60) -> str:
        """Render backtest trade PnL as a plotext bar chart string."""
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
        """Render equity curve as a plotext line chart string."""
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
        """Render signal weights as a plotext horizontal bar chart string."""
        weights = config.get("weights", {})
        w_vals = weights.get("w", [0.30, 0.15, 0.20])
        v_vals = weights.get("v", [0.12, 0.08, 0.05])
        x_vals = weights.get("x", [0.07, 0.03])

        names = ["OBI", "PRT", "UMOM", "LTB", "SWEEP", "ICE", "VWAP", "REGIME"]
        vals = w_vals + v_vals + x_vals

        plt.clf()
        plt.clt()
        plt.plotsize(50, 15)
        plt.title("Signal Weights")
        plt.theme("dark")

        plt.bar(names, vals, color="blue", orientation="horizontal")
        return plt.build()
