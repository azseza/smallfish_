"""Append-only CSV logging for decisions, orders, trades, and snapshots.

Uses buffered writes for performance, with periodic flush to disk.
"""
from __future__ import annotations
import csv
import io
import os
import logging
from typing import List, Dict, Any
from core.utils import time_now_ms

log = logging.getLogger(__name__)

FLUSH_INTERVAL_MS = 5000  # flush every 5 seconds
BUFFER_LIMIT = 100        # or every 100 rows


class CsvLogger:
    """Buffered CSV writer for a single log file."""

    def __init__(self, filepath: str, headers: List[str]):
        self.filepath = filepath
        self.headers = headers
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush: int = 0

        # Create file with headers if it doesn't exist
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= BUFFER_LIMIT:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        try:
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers, extrasaction="ignore")
                for row in self._buffer:
                    writer.writerow(row)
            self._buffer.clear()
            self._last_flush = time_now_ms()
        except Exception as e:
            log.error("Failed to flush %s: %s", self.filepath, e)

    def maybe_flush(self) -> None:
        if self._buffer and (time_now_ms() - self._last_flush > FLUSH_INTERVAL_MS):
            self.flush()


class Persistence:
    """Manages all CSV log files."""

    DECISION_HEADERS = [
        "timestamp", "symbol", "direction", "confidence", "raw_score",
        "obi_long", "obi_short", "prt_long", "prt_short",
        "umom_long", "umom_short", "ltb_long", "ltb_short",
        "sweep_up", "sweep_down", "ice_long", "ice_short",
        "vwap_long", "vwap_short", "vol_regime",
        "spread_ticks", "mid_price", "action",
    ]

    ORDER_HEADERS = [
        "timestamp", "order_id", "symbol", "side", "type",
        "price", "quantity", "status", "time_in_force",
        "filled_qty", "avg_fill_price",
    ]

    TRADE_HEADERS = [
        "timestamp", "symbol", "side", "entry_price", "exit_price",
        "quantity", "pnl", "pnl_R", "duration_ms",
        "slippage_entry", "slippage_exit", "exit_reason",
        "mae", "mfe",
    ]

    SNAPSHOT_HEADERS = [
        "timestamp", "equity", "daily_pnl", "drawdown",
        "daily_loss_R", "trades", "wins", "losses",
        "win_rate", "vol_regime", "latency_ema",
    ]

    def __init__(self, log_dir: str = "data/logs"):
        self.decisions = CsvLogger(
            os.path.join(log_dir, "decisions.csv"), self.DECISION_HEADERS
        )
        self.orders = CsvLogger(
            os.path.join(log_dir, "orders.csv"), self.ORDER_HEADERS
        )
        self.trades = CsvLogger(
            os.path.join(log_dir, "trades.csv"), self.TRADE_HEADERS
        )
        self.snapshots = CsvLogger(
            os.path.join(log_dir, "snapshots.csv"), self.SNAPSHOT_HEADERS
        )
        self._last_snapshot: int = 0

    def log_decision(self, data: dict) -> None:
        data.setdefault("timestamp", time_now_ms())
        self.decisions.write(data)

    def log_order(self, data: dict) -> None:
        data.setdefault("timestamp", time_now_ms())
        self.orders.write(data)

    def log_trade(self, data: dict) -> None:
        data.setdefault("timestamp", time_now_ms())
        self.trades.write(data)

    def log_snapshot(self, state_summary: dict) -> None:
        state_summary.setdefault("timestamp", time_now_ms())
        self.snapshots.write(state_summary)

    def flush_if_needed(self) -> None:
        self.decisions.maybe_flush()
        self.orders.maybe_flush()
        self.trades.maybe_flush()
        self.snapshots.maybe_flush()

    def flush_all(self) -> None:
        self.decisions.flush()
        self.orders.flush()
        self.trades.flush()
        self.snapshots.flush()
