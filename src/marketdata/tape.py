from __future__ import annotations
import math
import logging
from typing import List, Optional
from core.types import Trade, Side
from core.ringbuffer import RingBuffer
from core.utils import time_now_ms, safe_div

log = logging.getLogger(__name__)


class TradeTape:
    """Maintains the recent trade stream with rolling analytics."""

    def __init__(self, capacity: int = 5000):
        self.trades: RingBuffer[Trade] = RingBuffer(capacity)

        # Rolling statistics (updated incrementally)
        self._price_sum: float = 0.0
        self._vol_sum: float = 0.0       # sum of qty
        self._notional_sum: float = 0.0   # sum of price*qty (for VWAP)

        # For median estimation we keep a sorted window
        self._recent_sizes: RingBuffer[float] = RingBuffer(200)

        # EMA prices for momentum
        self.ema_fast: float = 0.0   # ~1s half-life
        self.ema_slow: float = 0.0   # ~5s half-life
        self._last_ema_ts: int = 0

        # Price series for realized vol
        self._price_history: RingBuffer[float] = RingBuffer(120)  # ~120s at 1/s
        self._price_ts: RingBuffer[int] = RingBuffer(120)

    def add_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self._recent_sizes.append(trade.quantity)
        self._notional_sum += trade.price * trade.quantity
        self._vol_sum += trade.quantity

        # Update EMAs
        dt_s = (trade.timestamp - self._last_ema_ts) / 1000.0 if self._last_ema_ts > 0 else 0.0
        if self.ema_fast == 0.0:
            self.ema_fast = trade.price
            self.ema_slow = trade.price
        else:
            if dt_s > 0:
                # EMA with time-based decay
                alpha_fast = 1.0 - math.exp(-dt_s / 1.0)   # 1s half-life
                alpha_slow = 1.0 - math.exp(-dt_s / 5.0)   # 5s half-life
                self.ema_fast = alpha_fast * trade.price + (1 - alpha_fast) * self.ema_fast
                self.ema_slow = alpha_slow * trade.price + (1 - alpha_slow) * self.ema_slow
        self._last_ema_ts = trade.timestamp

        # Add to price history (throttled to ~1 per second)
        if not self._price_ts or (trade.timestamp - self._price_ts[-1]) >= 500:
            self._price_history.append(trade.price)
            self._price_ts.append(trade.timestamp)

    def add_trades(self, trades: List[Trade]) -> None:
        for t in trades:
            self.add_trade(t)

    # --- Accessors ---

    def last_trade(self) -> Optional[Trade]:
        return self.trades.peek()

    def last_price(self) -> float:
        t = self.trades.peek()
        return t.price if t else 0.0

    def get_recent(self, n: int) -> List[Trade]:
        return self.trades.last(n)

    # --- Analytics ---

    def trade_count(self, window_ms: int = 1000) -> int:
        now = time_now_ms()
        cutoff = now - window_ms
        count = 0
        for t in reversed(self.trades.get()):
            if t.timestamp < cutoff:
                break
            count += 1
        return count

    def trade_rate(self, window_ms: int = 1000) -> float:
        """Trades per second in the last window."""
        count = self.trade_count(window_ms)
        return count / (window_ms / 1000.0) if window_ms > 0 else 0.0

    def volume_in_window(self, window_ms: int = 1000, side: Optional[Side] = None) -> float:
        now = time_now_ms()
        cutoff = now - window_ms
        vol = 0.0
        for t in reversed(self.trades.get()):
            if t.timestamp < cutoff:
                break
            if side is None or t.side == side:
                vol += t.quantity
        return vol

    def notional_in_window(self, window_ms: int = 1000, side: Optional[Side] = None) -> float:
        now = time_now_ms()
        cutoff = now - window_ms
        notional = 0.0
        for t in reversed(self.trades.get()):
            if t.timestamp < cutoff:
                break
            if side is None or t.side == side:
                notional += t.notional
        return notional

    def buy_sell_ratio(self, window_ms: int = 2000) -> float:
        """Buy volume / total volume. 0.5 = balanced, >0.5 = buy heavy."""
        buy_vol = self.volume_in_window(window_ms, Side.BUY)
        sell_vol = self.volume_in_window(window_ms, Side.SELL)
        return safe_div(buy_vol, buy_vol + sell_vol, 0.5)

    def cvd_in_window(self, window_ms: int = 5000) -> float:
        """Cumulative Volume Delta: buy_vol - sell_vol in window."""
        return (self.volume_in_window(window_ms, Side.BUY)
                - self.volume_in_window(window_ms, Side.SELL))

    def cvd_acceleration(self, window_ms: int = 5000) -> float:
        """CVD recent_half - CVD older_half, normalized by total vol.

        Positive = accelerating buy pressure, negative = accelerating sell pressure.
        """
        now = time_now_ms()
        cutoff = now - window_ms
        midpoint = now - window_ms // 2

        recent_delta = 0.0
        older_delta = 0.0
        total_vol = 0.0

        for t in reversed(self.trades.get()):
            if t.timestamp < cutoff:
                break
            sign = 1.0 if t.side == Side.BUY else -1.0
            total_vol += t.quantity
            if t.timestamp >= midpoint:
                recent_delta += sign * t.quantity
            else:
                older_delta += sign * t.quantity

        return safe_div(recent_delta - older_delta, max(total_vol, 1e-12))

    def median_trade_size(self) -> float:
        sizes = sorted(self._recent_sizes.get())
        if not sizes:
            return 0.0
        n = len(sizes)
        if n % 2 == 1:
            return sizes[n // 2]
        return (sizes[n // 2 - 1] + sizes[n // 2]) / 2.0

    def vwap(self, n_trades: int = 500) -> float:
        """Volume-weighted average price over last N trades."""
        trades = self.trades.last(min(n_trades, len(self.trades)))
        if not trades:
            return 0.0
        notional = sum(t.price * t.quantity for t in trades)
        volume = sum(t.quantity for t in trades)
        return safe_div(notional, volume)

    def realized_vol(self, lookback_s: int = 60) -> float:
        """Realized volatility over last N seconds of price samples."""
        prices = self._price_history.get()
        if len(prices) < 2:
            return 0.0
        # Use only prices within lookback
        timestamps = self._price_ts.get()
        now = time_now_ms()
        cutoff = now - lookback_s * 1000
        filtered = [p for p, ts in zip(prices, timestamps) if ts >= cutoff]
        if len(filtered) < 2:
            return 0.0

        sum_sq = 0.0
        n = 0
        for i in range(1, len(filtered)):
            if filtered[i - 1] <= 0:
                continue
            log_ret = math.log(filtered[i] / filtered[i - 1])
            sum_sq += log_ret * log_ret
            n += 1
        if n == 0:
            return 0.0
        return math.sqrt(sum_sq / n) * math.sqrt(n)  # realized vol for the window

    def large_trades(self, window_ms: int = 2000, multiplier: float = 5.0,
                     min_usd: float = 50000) -> List[Trade]:
        """Return trades that qualify as 'whale' trades."""
        median = self.median_trade_size()
        threshold_qty = max(multiplier * median, 0.001)
        now = time_now_ms()
        cutoff = now - window_ms
        result = []
        for t in reversed(self.trades.get()):
            if t.timestamp < cutoff:
                break
            if t.quantity >= threshold_qty or t.notional >= min_usd:
                result.append(t)
        return result

    def buy_burst(self, window_ms: int = 2000, **kwargs) -> float:
        """Total volume of large buy trades in window."""
        whales = self.large_trades(window_ms, **kwargs)
        return sum(t.quantity for t in whales if t.side == Side.BUY)

    def sell_burst(self, window_ms: int = 2000, **kwargs) -> float:
        """Total volume of large sell trades in window."""
        whales = self.large_trades(window_ms, **kwargs)
        return sum(t.quantity for t in whales if t.side == Side.SELL)
