"""Terminal Dashboard — live bar chart visualization for Smallfish.

Renders trade PnL, equity curve, signal weights, and grid status as
terminal bar charts. No external dependencies — pure ANSI terminal
rendering inspired by termgraph.

Designed for headless Raspberry Pi deployments where you SSH in.

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
from typing import Dict, List, Optional, Tuple

from core.state import RuntimeState
from core.types import TradeResult

log = logging.getLogger(__name__)

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"

# Bar characters
FULL_BLOCK = "\u2588"
HALF_BLOCK = "\u2584"
LIGHT_SHADE = "\u2591"
MED_SHADE = "\u2592"
DARK_SHADE = "\u2593"
BAR_H = "\u2501"
BAR_V = "\u2503"
CORNER_TL = "\u250f"
CORNER_TR = "\u2513"
CORNER_BL = "\u2517"
CORNER_BR = "\u251b"
TEE_L = "\u2523"
TEE_R = "\u252b"
TEE_T = "\u2533"
TEE_B = "\u253b"


def _bar(value: float, max_val: float, width: int = 30, color: str = GREEN) -> str:
    """Render a horizontal bar."""
    if max_val <= 0:
        return ""
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    bar_str = FULL_BLOCK * filled + LIGHT_SHADE * (width - filled)
    return f"{color}{bar_str}{RESET}"


def _pnl_bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a PnL bar (green for positive, red for negative)."""
    if max_val <= 0:
        return " " * width
    ratio = min(abs(value) / max_val, 1.0)
    filled = max(1, int(ratio * width))
    color = GREEN if value >= 0 else RED
    char = FULL_BLOCK
    return f"{color}{char * filled}{RESET}{' ' * (width - filled)}"


def _sparkline(values: List[float], width: int = 40) -> str:
    """Render a sparkline from a list of values."""
    if not values:
        return ""
    chars = " " + "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn = min(values)
    mx = max(values)
    rng = mx - mn if mx > mn else 1.0

    # Resample to width
    step = max(1, len(values) // width)
    sampled = values[::step][:width]

    line = ""
    for v in sampled:
        idx = int((v - mn) / rng * 7) + 1
        idx = min(idx, 8)
        line += chars[idx]
    return f"{CYAN}{line}{RESET}"


class TerminalDashboard:
    """Async terminal dashboard with bar chart visualizations."""

    def __init__(self, state: RuntimeState, config: dict,
                 grid_strategy=None, refresh_s: float = 2.0):
        self.state = state
        self.config = config
        self.grid = grid_strategy
        self.refresh_s = refresh_s
        self.enabled = config.get("dashboard", {}).get("enabled", False)
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Equity history for sparkline
        self._equity_history: List[float] = []
        self._pnl_history: List[float] = []

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
        if len(self._equity_history) > 200:
            self._equity_history = self._equity_history[-200:]

    def _render(self) -> None:
        """Render the full dashboard to terminal."""
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 80
        width = min(cols, 100)

        lines = []
        lines.append("")
        lines.append(self._header(width))
        lines.append(self._account_section(width))
        lines.append(self._pnl_section(width))
        lines.append(self._signals_section(width))
        lines.append(self._trades_section(width))
        if self.grid:
            lines.append(self._grid_section(width))
        lines.append(self._equity_sparkline(width))
        lines.append(self._footer(width))

        output = "\n".join(lines)

        # Clear screen and redraw
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(output)
        sys.stdout.flush()

    def _header(self, w: int) -> str:
        ts = time.strftime("%H:%M:%S")
        regime_colors = {
            "low": GREEN, "normal": WHITE,
            "high": YELLOW, "extreme": RED,
        }
        rc = regime_colors.get(self.state.vol_regime, WHITE)
        regime = f"{rc}{self.state.vol_regime}{RESET}"

        kill = f"{RED}KILLED: {self.state.kill_reason}{RESET}" if self.state.kill_switch else f"{GREEN}LIVE{RESET}"

        header = (
            f"{BOLD}{CYAN}"
            f"  ><(((o>  SMALLFISH DASHBOARD  <o)))><"
            f"{RESET}\n"
            f"  {DIM}{ts}{RESET}  |  regime: {regime}  |  {kill}  |  "
            f"latency: {self.state.latency_ms}ms"
        )
        return header

    def _account_section(self, w: int) -> str:
        s = self.state
        dd_color = GREEN if s.drawdown < 0.02 else YELLOW if s.drawdown < 0.05 else RED
        return (
            f"\n  {BOLD}ACCOUNT{RESET}\n"
            f"  Equity:    ${s.equity:>10.2f}  |  "
            f"Peak: ${s.peak_equity:>10.2f}  |  "
            f"DD: {dd_color}{s.drawdown*100:.2f}%{RESET}"
        )

    def _pnl_section(self, w: int) -> str:
        s = self.state
        daily_color = GREEN if s.daily_pnl >= 0 else RED
        total_color = GREEN if s.realized_pnl >= 0 else RED
        wr = s.win_count / max(s.trade_count, 1) * 100

        bar_w = 25
        max_pnl = max(abs(s.daily_pnl), abs(s.realized_pnl), 1.0)

        return (
            f"\n  {BOLD}P&L{RESET}\n"
            f"  Daily:   {daily_color}${s.daily_pnl:>+9.4f}{RESET}  "
            f"{_pnl_bar(s.daily_pnl, max_pnl, bar_w)}\n"
            f"  Total:   {total_color}${s.realized_pnl:>+9.4f}{RESET}  "
            f"{_pnl_bar(s.realized_pnl, max_pnl, bar_w)}\n"
            f"  Trades:  {s.trade_count:>4d}  |  "
            f"W/L: {s.win_count}/{s.loss_count}  |  "
            f"WR: {wr:.1f}%  |  "
            f"Daily R: {s.daily_loss_R:.1f}/{self.config.get('max_daily_R', 10)}"
        )

    def _signals_section(self, w: int) -> str:
        scores = self.state.last_scores
        if not scores:
            return f"\n  {BOLD}SIGNALS{RESET}\n  {DIM}(waiting for data){RESET}"

        signal_names = ["obi", "prt", "umom", "ltb", "sweep", "ice", "vwap_dev", "vol_regime"]
        bar_w = 20
        lines = [f"\n  {BOLD}SIGNALS{RESET}  "
                 f"dir={self.state.last_direction:+d}  "
                 f"conf={self.state.last_confidence:.3f}"]

        for name in signal_names:
            val = scores.get(name, 0.0)
            color = GREEN if val > 0 else RED if val < 0 else DIM
            bar = _bar(abs(val), 1.0, bar_w, color)
            lines.append(f"  {name:>12s}  {color}{val:>+6.3f}{RESET}  {bar}")

        return "\n".join(lines)

    def _trades_section(self, w: int) -> str:
        trades = list(self.state.completed_trades.get())
        if not trades:
            return f"\n  {BOLD}RECENT TRADES{RESET}\n  {DIM}(no trades yet){RESET}"

        recent = trades[-10:]  # last 10
        max_pnl = max(abs(t.pnl) for t in recent) if recent else 1.0
        bar_w = 15

        lines = [f"\n  {BOLD}RECENT TRADES{RESET}"]
        for t in recent:
            color = GREEN if t.pnl >= 0 else RED
            side_str = "L" if t.side == 1 else "S"
            bar = _pnl_bar(t.pnl, max_pnl, bar_w)
            reason = t.exit_reason[:8] if t.exit_reason else "?"
            lines.append(
                f"  {side_str} {t.symbol:<10s} "
                f"{color}${t.pnl:>+8.4f}{RESET} "
                f"({t.pnl_R:>+5.2f}R) "
                f"{bar} "
                f"{DIM}{reason}{RESET}"
            )

        return "\n".join(lines)

    def _grid_section(self, w: int) -> str:
        if not self.grid:
            return ""

        all_status = self.grid.all_status()
        if not any(s.get("active") for s in all_status.values()):
            return f"\n  {BOLD}MULTIGRID{RESET}\n  {DIM}(inactive){RESET}"

        lines = [f"\n  {BOLD}MULTIGRID{RESET}"]
        for sym, gs in all_status.items():
            if not gs.get("active"):
                continue
            lines.append(
                f"  {sym:<10s}  "
                f"center={gs['center']:>10.2f}  "
                f"B:{gs['filled_buys']}/{gs['buy_levels']}  "
                f"S:{gs['filled_sells']}/{gs['sell_levels']}  "
                f"pending={gs['pending_orders']}  "
                f"pnl=${gs['total_pnl']:>+.4f}  "
                f"trips={gs['round_trips']}"
            )

        return "\n".join(lines)

    def _equity_sparkline(self, w: int) -> str:
        if len(self._equity_history) < 2:
            return ""

        spark = _sparkline(self._equity_history, width=min(w - 20, 60))
        start = self._equity_history[0]
        end = self._equity_history[-1]
        change = end - start
        color = GREEN if change >= 0 else RED

        return (
            f"\n  {BOLD}EQUITY{RESET}  "
            f"${start:.2f} -> ${end:.2f} "
            f"({color}{change:+.2f}{RESET})\n"
            f"  {spark}"
        )

    def _footer(self, w: int) -> str:
        positions = list(self.state.positions.values())
        pos_str = ""
        if positions:
            parts = []
            for p in positions:
                side = "L" if p.side == 1 else "S"
                color = GREEN if p.unrealized_pnl >= 0 else RED
                parts.append(
                    f"{side} {p.symbol} "
                    f"qty={p.quantity:.4f} "
                    f"entry={p.entry_price:.2f} "
                    f"{color}uPnL=${p.unrealized_pnl:+.4f}{RESET}"
                )
            pos_str = "\n  ".join(parts)

        if pos_str:
            return f"\n  {BOLD}POSITIONS{RESET}\n  {pos_str}\n"
        return f"\n  {DIM}no open positions{RESET}\n"

    # --- One-shot renders for backtest / reporting ---

    @staticmethod
    def render_backtest_trades(trades: List[dict], width: int = 60) -> str:
        """Render backtest trade PnL as terminal bar chart."""
        if not trades:
            return "  (no trades)"

        pnls = [t["pnl"] for t in trades]
        max_pnl = max(abs(p) for p in pnls) if pnls else 1.0
        bar_w = min(width // 2, 25)

        lines = [f"\n  {BOLD}TRADE P&L CHART{RESET}"]
        # Show last 30 trades
        for t in trades[-30:]:
            color = GREEN if t["pnl"] >= 0 else RED
            side = t.get("side", "?")[0]
            sym = t.get("symbol", "?")[:6]
            bar = _pnl_bar(t["pnl"], max_pnl, bar_w)
            lines.append(
                f"  {side} {sym:<6s} {color}${t['pnl']:>+8.4f}{RESET} {bar}"
            )

        return "\n".join(lines)

    @staticmethod
    def render_equity_curve(equity_curve: List[Tuple[int, float]], width: int = 60) -> str:
        """Render equity curve as sparkline."""
        if not equity_curve:
            return ""
        values = [e for _, e in equity_curve]
        spark = _sparkline(values, width=min(width, 60))
        start = values[0]
        end = values[-1]
        change = end - start
        color = GREEN if change >= 0 else RED

        return (
            f"\n  {BOLD}EQUITY CURVE{RESET}\n"
            f"  ${start:.2f} -> ${end:.2f} "
            f"({color}{change:+.2f}{RESET})\n"
            f"  {spark}\n"
        )

    @staticmethod
    def render_signal_weights(config: dict) -> str:
        """Render signal weights as horizontal bars."""
        weights = config.get("weights", {})
        w_vals = weights.get("w", [0.30, 0.15, 0.20])
        v_vals = weights.get("v", [0.12, 0.08, 0.05])
        x_vals = weights.get("x", [0.07, 0.03])

        names = ["OBI", "PRT", "UMOM", "LTB", "SWEEP", "ICE", "VWAP", "REGIME"]
        vals = w_vals + v_vals + x_vals
        max_w = max(vals) if vals else 1.0

        lines = [f"\n  {BOLD}SIGNAL WEIGHTS{RESET}"]
        for name, val in zip(names, vals):
            bar = _bar(val, max_w, 20, BLUE)
            lines.append(f"  {name:>8s}  {val:.0%}  {bar}")

        return "\n".join(lines)
