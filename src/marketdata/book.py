from __future__ import annotations
import logging
from typing import List, Tuple, Optional, Dict
from core.utils import time_now_ms, safe_div
from core.ringbuffer import RingBuffer

log = logging.getLogger(__name__)


class OrderBook:
    """Maintains a local order book with delta processing and analytics."""

    __slots__ = (
        "symbol", "depth", "tick_size",
        "bids", "asks",
        "_bid_map", "_ask_map",
        "last_update_ts", "last_update_seq",
        "_prev_bid_total", "_prev_ask_total",
        "_prev_mid", "_mid_history",
        "_cancel_tracker",
    )

    def __init__(self, symbol: str = "", depth: int = 50, tick_size: float = 0.1):
        self.symbol = symbol
        self.depth = depth
        self.tick_size = tick_size

        self.bids: List[Tuple[float, float]] = []  # [(price, size), ...] desc
        self.asks: List[Tuple[float, float]] = []  # [(price, size), ...] asc

        self._bid_map: Dict[float, float] = {}
        self._ask_map: Dict[float, float] = {}

        self.last_update_ts: int = 0
        self.last_update_seq: int = 0

        self._prev_bid_total: float = 0.0
        self._prev_ask_total: float = 0.0
        self._prev_mid: float = 0.0

        self._mid_history: RingBuffer[Tuple[int, float]] = RingBuffer(200)

        # Cancel tracking: record (timestamp, side, price, old_size)
        self._cancel_tracker: RingBuffer[Tuple[int, str, float, float]] = RingBuffer(500)

    # --- Snapshot & Delta ---

    def on_snapshot(self, bids: list, asks: list, seq: int = 0) -> None:
        """Process a full book snapshot. bids/asks are [[price, size], ...]."""
        self._bid_map.clear()
        self._ask_map.clear()

        for p, s in bids:
            price, size = float(p), float(s)
            if size > 0:
                self._bid_map[price] = size

        for p, s in asks:
            price, size = float(p), float(s)
            if size > 0:
                self._ask_map[price] = size

        self._rebuild_sorted()
        self.last_update_seq = seq
        self.last_update_ts = time_now_ms()
        self._prev_mid = self.mid_price()

    def on_delta(self, bids: list, asks: list, seq: int = 0) -> None:
        """Process an incremental book delta."""
        now = time_now_ms()

        # Track cancellations before applying delta
        for p, s in bids:
            price, size = float(p), float(s)
            old = self._bid_map.get(price, 0.0)
            if size == 0 and old > 0:
                self._cancel_tracker.append((now, "bid", price, old))
            if size > 0:
                self._bid_map[price] = size
            elif price in self._bid_map:
                del self._bid_map[price]

        for p, s in asks:
            price, size = float(p), float(s)
            old = self._ask_map.get(price, 0.0)
            if size == 0 and old > 0:
                self._cancel_tracker.append((now, "ask", price, old))
            if size > 0:
                self._ask_map[price] = size
            elif price in self._ask_map:
                del self._ask_map[price]

        self._rebuild_sorted()
        self.last_update_seq = seq
        self.last_update_ts = now
        mid = self.mid_price()
        self._mid_history.append((now, mid))
        self._prev_mid = mid

    def _rebuild_sorted(self) -> None:
        self.bids = sorted(self._bid_map.items(), key=lambda x: -x[0])[:self.depth]
        self.asks = sorted(self._ask_map.items(), key=lambda x: x[0])[:self.depth]

    # --- Core Accessors ---

    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    def best_bid_size(self) -> float:
        return self.bids[0][1] if self.bids else 0.0

    def best_ask_size(self) -> float:
        return self.asks[0][1] if self.asks else 0.0

    def mid_price(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb <= 0 or ba <= 0:
            return 0.0
        return (bb + ba) / 2.0

    def spread(self) -> float:
        return self.best_ask() - self.best_bid()

    def spread_ticks(self) -> float:
        if self.tick_size <= 0:
            return self.spread()
        return self.spread() / self.tick_size

    def is_fresh(self, max_staleness_ms: int = 250) -> bool:
        if self.last_update_ts == 0:
            return False
        age = time_now_ms() - self.last_update_ts
        return age < max_staleness_ms

    # --- Analytics ---

    def bid_volume(self, levels: int = 5) -> float:
        return sum(s for _, s in self.bids[:levels])

    def ask_volume(self, levels: int = 5) -> float:
        return sum(s for _, s in self.asks[:levels])

    def imbalance(self, levels: int = 5) -> float:
        """OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol).
        +1 = all bid, -1 = all ask.
        """
        bv = self.bid_volume(levels)
        av = self.ask_volume(levels)
        return safe_div(bv - av, bv + av)

    def weighted_mid(self) -> float:
        """Volume-weighted mid price using top-of-book."""
        bb, bs = (self.bids[0] if self.bids else (0, 0))
        ba, a_s = (self.asks[0] if self.asks else (0, 0))
        total = bs + a_s
        if total <= 0 or bb <= 0 or ba <= 0:
            return self.mid_price()
        return (bb * a_s + ba * bs) / total

    def depth_at_price(self, price: float, side: str) -> float:
        mp = self._bid_map if side == "bid" else self._ask_map
        return mp.get(price, 0.0)

    def cumulative_depth(self, side: str, ticks: int) -> float:
        """Total size within N ticks from BBO."""
        if side == "bid":
            threshold = self.best_bid() - ticks * self.tick_size
            return sum(s for p, s in self.bids if p >= threshold)
        else:
            threshold = self.best_ask() + ticks * self.tick_size
            return sum(s for p, s in self.asks if p <= threshold)

    def cancel_rate(self, side: str, window_ms: int = 500) -> float:
        """Fraction of levels cancelled on a side in the last window."""
        now = time_now_ms()
        cutoff = now - window_ms
        cancels = sum(1 for ts, s, _, _ in self._cancel_tracker
                      if ts >= cutoff and s == side)
        total = sum(1 for ts, s, _, _ in self._cancel_tracker
                    if ts >= cutoff and s == side) + max(
            len(self.bids if side == "bid" else self.asks), 1)
        return safe_div(cancels, total)

    def cancel_volume(self, side: str, window_ms: int = 500) -> float:
        """Total volume cancelled on a side in the last window."""
        now = time_now_ms()
        cutoff = now - window_ms
        return sum(sz for ts, s, _, sz in self._cancel_tracker
                   if ts >= cutoff and s == side)

    def mid_change_ticks(self, lookback_ms: int = 500) -> float:
        """How many ticks mid has moved in the last lookback_ms."""
        if len(self._mid_history) < 2 or self.tick_size <= 0:
            return 0.0
        now = time_now_ms()
        cutoff = now - lookback_ms
        # find oldest mid in window
        old_mid = self.mid_price()
        for ts, mid in self._mid_history:
            if ts >= cutoff:
                old_mid = mid
                break
        return (self.mid_price() - old_mid) / self.tick_size

    def levels_cleared_since(self, side: str, window_ms: int = 1000) -> int:
        """Count distinct price levels that were fully cancelled (swept)."""
        now = time_now_ms()
        cutoff = now - window_ms
        prices = set()
        for ts, s, price, _ in self._cancel_tracker:
            if ts >= cutoff and s == side:
                prices.add(price)
        return len(prices)
