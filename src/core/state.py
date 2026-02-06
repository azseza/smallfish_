from __future__ import annotations
import logging
import time
from typing import Optional, Dict, List

from core.types import (
    Position, Order, Execution, TradeResult, Side, OrderStatus,
)
from core.ringbuffer import RingBuffer
from core.utils import time_now_ms, clamp

log = logging.getLogger(__name__)


class RuntimeState:
    """Central mutable state for the trading bot."""

    def __init__(self, config: dict):
        self.config = config
        self.started_at = time_now_ms()

        # --- Account ---
        self.equity: float = config.get("initial_equity", 1000.0)
        self.peak_equity: float = self.equity
        self.daily_start_equity: float = self.equity

        # --- PnL tracking ---
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_loss_R: float = 0.0
        self.trade_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0

        # --- Drawdown ---
        self.drawdown: float = 0.0          # current DD as fraction of peak
        self.max_drawdown: float = 0.0      # worst DD seen today

        # --- Risk ---
        self.slip_breaches: int = 0
        self.kill_switch: bool = False
        self.kill_reason: str = ""
        self.last_loss_time: int = 0        # ms, for cooldown

        # --- Positions & Orders ---
        self.positions: Dict[str, Position] = {}  # symbol → Position
        self.active_orders: Dict[str, Order] = {}  # order_id → Order

        # --- Grid order tracking ---
        self.grid_order_ids: set[str] = set()  # order IDs that belong to grid strategy
        self.grid_harvest_ids: Dict[str, str] = {}  # harvest_order_id → original_level_order_id

        # --- Signal state ---
        self.last_scores: dict = {}
        self.last_raw: float = 0.0
        self.last_confidence: float = 0.0
        self.last_direction: int = 0         # +1 long, -1 short, 0 flat
        self.last_symbol: str = ""           # symbol for last_scores (focused coin)

        # Per-symbol scores for multi-coin mode
        self.scores_by_symbol: Dict[str, dict] = {}  # symbol → {scores, conf, direction, raw, ts}

        # --- Latency ---
        self.latency_ms: int = 0
        self.latency_ema: float = 0.0

        # --- Completed trades (for adaptive weights) ---
        self.completed_trades: RingBuffer[TradeResult] = RingBuffer(500)

        # --- Signal edge tracking ---
        self.signal_pnl: Dict[str, float] = {}  # signal_name → cumulative PnL contribution

        # --- Volatility regime ---
        self.vol_regime: str = "normal"  # "low", "normal", "high", "extreme"

        # --- Per-entry cooldown (matches backtest cooldown_ms) ---
        self.last_entry_ts: Dict[str, int] = {}  # symbol → last entry timestamp ms

        # --- Manual cooldown (from /cooldown command) ---
        self.cooldown_until_ms: int = 0

        # --- Cash out state ---
        self.pending_cashout: Optional[dict] = None  # preview state for confirmation
        self.withdrawal_history: list[dict] = []  # [{ts, amount, tx_id, address}]

    # --- Position helpers ---

    def position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def no_position(self, symbol: str) -> bool:
        return symbol not in self.positions

    def open_position_count(self) -> int:
        return len(self.positions)

    # --- Trading session ---

    def within_session(self) -> bool:
        return not self.kill_switch

    def in_funding_window(self, symbol: str = "") -> bool:
        """Check if we are within N minutes of a funding timestamp.
        Bybit perpetual funding occurs every 8h: 00:00, 08:00, 16:00 UTC.
        """
        window_min = self.config.get("funding_window_minutes", 3)
        now = time.time()
        # seconds since midnight UTC
        sod = now % 86400
        funding_times = [0, 28800, 57600]  # 00:00, 08:00, 16:00
        window_s = window_min * 60
        for ft in funding_times:
            dist = min(abs(sod - ft), 86400 - abs(sod - ft))
            if dist < window_s:
                return True
        return False

    def in_cooldown(self) -> bool:
        if self.last_loss_time == 0:
            return False
        cooldown = self.config.get("cooldown_after_loss_ms", 5000)
        return (time_now_ms() - self.last_loss_time) < cooldown

    def in_manual_cooldown(self) -> bool:
        """Check if manual cooldown (from /cooldown command) is active."""
        if self.cooldown_until_ms == 0:
            return False
        return time_now_ms() < self.cooldown_until_ms

    # --- Kill switch ---

    def trigger_kill_switch(self, reason: str) -> None:
        self.kill_switch = True
        self.kill_reason = reason
        log.critical("KILL SWITCH TRIGGERED: %s", reason)

    def check_daily_limits(self) -> None:
        max_R = self.config.get("max_daily_R", 10)
        if self.daily_loss_R >= max_R:
            self.trigger_kill_switch(f"daily_loss_R={self.daily_loss_R:.1f} >= {max_R}")

    # --- Drawdown sizing ---

    def size_multiplier(self) -> float:
        """Return position size multiplier based on current drawdown tier."""
        tiers = self.config.get("drawdown_tiers", [])
        dd = self.drawdown
        mult = 1.0
        for tier in tiers:
            if dd >= tier["threshold"]:
                mult = tier["multiplier"]
        return mult

    # --- Trade recording ---

    def on_enter(self, symbol: str, side: Side, fill_price: float,
                 quantity: float, confidence: float, scores: dict,
                 stop_price: float = 0.0, tp_price: float = 0.0) -> Position:
        pos = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=fill_price,
            side=side,
            stop_price=stop_price,
            tp_price=tp_price,
            entry_time=time_now_ms(),
            peak_favorable=fill_price,
            worst_adverse=fill_price,
            signals_at_entry=dict(scores),  # capture at entry, not exit
            confidence_at_entry=confidence,
        )
        self.positions[symbol] = pos
        self.last_entry_ts[symbol] = time_now_ms()
        log.info("ENTER %s %s qty=%.6f px=%.2f conf=%.3f",
                 side.name, symbol, quantity, fill_price, confidence)
        return pos

    def on_exit(self, symbol: str, exit_price: float, exit_reason: str,
                slippage_entry: float = 0.0, slippage_exit: float = 0.0) -> Optional[TradeResult]:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        now = time_now_ms()
        raw_pnl = (exit_price - pos.entry_price) * pos.quantity * int(pos.side)
        risk_per_trade = self.config.get("risk_per_trade", 0.005) * self.equity
        risk_per_trade = max(risk_per_trade, 0.01)
        pnl_R = raw_pnl / risk_per_trade

        # Calculate MAE (max adverse excursion) and MFE (max favorable excursion)
        # MAE = worst price seen relative to entry (negative = adverse)
        # MFE = best price seen relative to entry (positive = favorable)
        if pos.side == Side.BUY:
            mae = (pos.worst_adverse - pos.entry_price) * pos.quantity if pos.worst_adverse else 0.0
            mfe = (pos.peak_favorable - pos.entry_price) * pos.quantity if pos.peak_favorable else 0.0
        else:
            mae = (pos.entry_price - pos.worst_adverse) * pos.quantity if pos.worst_adverse else 0.0
            mfe = (pos.entry_price - pos.peak_favorable) * pos.quantity if pos.peak_favorable else 0.0

        result = TradeResult(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=raw_pnl,
            pnl_R=pnl_R,
            entry_time=pos.entry_time,
            exit_time=now,
            duration_ms=now - pos.entry_time,
            slippage_entry=slippage_entry,
            slippage_exit=slippage_exit,
            signals_at_entry=pos.signals_at_entry,  # use stored entry signals
            exit_reason=exit_reason,
            mae=mae,
            mfe=mfe,
        )
        self.completed_trades.append(result)

        # Update PnL
        self.realized_pnl += raw_pnl
        self.daily_pnl += raw_pnl
        self.equity += raw_pnl
        self.trade_count += 1

        if raw_pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            self.daily_loss_R += abs(pnl_R)
            self.last_loss_time = now

        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, self.drawdown)

        self.check_daily_limits()

        log.info("EXIT %s %s reason=%s pnl=%.4f (%.2fR) equity=%.2f dd=%.4f",
                 pos.side.name, symbol, exit_reason, raw_pnl, pnl_R,
                 self.equity, self.drawdown)
        return result

    def update_latency(self, latency_ms: int) -> None:
        self.latency_ms = latency_ms
        alpha = 0.3
        self.latency_ema = alpha * latency_ms + (1 - alpha) * self.latency_ema

    def reset_daily(self) -> None:
        """Reset daily counters (call at 00:00 UTC)."""
        self.daily_start_equity = self.equity
        self.daily_pnl = 0.0
        self.daily_loss_R = 0.0
        self.max_drawdown = 0.0
        self.slip_breaches = 0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.kill_switch = False
        self.kill_reason = ""
        log.info("Daily state reset. Equity=%.2f", self.equity)

    def rolling_win_rate(self, n: int = 20) -> float:
        """Return win rate over the last *n* completed trades.

        Returns 1.0 when fewer than *n* trades have been recorded
        (not enough data to judge).
        """
        trades = self.completed_trades.last(n)
        if len(trades) < n:
            return 1.0  # not enough data — don't restrict
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    def summary(self) -> dict:
        return {
            "equity": round(self.equity, 2),
            "daily_pnl": round(self.daily_pnl, 4),
            "daily_loss_R": round(self.daily_loss_R, 2),
            "drawdown": round(self.drawdown, 4),
            "trades": self.trade_count,
            "wins": self.win_count,
            "losses": self.loss_count,
            "win_rate": round(self.win_count / max(self.trade_count, 1), 3),
            "kill_switch": self.kill_switch,
            "vol_regime": self.vol_regime,
            "latency_ema": round(self.latency_ema, 1),
        }
