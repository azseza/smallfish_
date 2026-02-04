"""Heartbeat and health monitoring.

Periodically checks system health, logs state snapshots, and
coordinates graceful shutdown.
"""
from __future__ import annotations
import asyncio
import logging
import time

from core.state import RuntimeState
from core.utils import time_now_ms
from gateway.persistence import Persistence
from monitor.metrics import compute_report, edge_per_signal

log = logging.getLogger(__name__)


class Heartbeat:
    """Periodic health checker and state reporter."""

    def __init__(self, state: RuntimeState, persistence: Persistence, config: dict):
        self.state = state
        self.persistence = persistence
        self.config = config
        self._last_heartbeat: int = 0
        self._last_snapshot: int = 0
        self._last_daily_reset: int = 0
        self._heartbeat_interval: int = 60_000      # 60s
        self._snapshot_interval: int = config.get("snapshot_interval_s", 300) * 1000

    def check(self) -> None:
        """Run periodic checks. Call this on every iteration of the main loop."""
        now = time_now_ms()

        # Heartbeat log
        if (now - self._last_heartbeat) >= self._heartbeat_interval:
            self._do_heartbeat(now)
            self._last_heartbeat = now

        # State snapshot
        if (now - self._last_snapshot) >= self._snapshot_interval:
            self._do_snapshot(now)
            self._last_snapshot = now

        # Daily reset check (midnight UTC)
        current_day = now // 86_400_000
        last_day = self._last_daily_reset // 86_400_000 if self._last_daily_reset > 0 else -1
        if current_day != last_day and self._last_daily_reset > 0:
            self.state.reset_daily()
            self._last_daily_reset = now
        elif self._last_daily_reset == 0:
            self._last_daily_reset = now

    def _do_heartbeat(self, now: int) -> None:
        summary = self.state.summary()
        report = compute_report(self.state.completed_trades)
        log.info(
            "HEARTBEAT | equity=%.2f | daily_pnl=%.4f | dd=%.4f | "
            "trades=%d (W:%d L:%d) | wr=%.1f%% | sharpe=%.2f | "
            "latency=%dms | regime=%s | kill=%s",
            summary["equity"],
            summary["daily_pnl"],
            summary["drawdown"],
            summary["trades"],
            summary["wins"],
            summary["losses"],
            summary["win_rate"] * 100,
            report.get("sharpe", 0),
            summary["latency_ema"],
            summary["vol_regime"],
            summary["kill_switch"],
        )

    def _do_snapshot(self, now: int) -> None:
        self.persistence.log_snapshot(self.state.summary())
        self.persistence.flush_all()
        log.debug("State snapshot saved")

    def get_signal_edges(self) -> dict:
        """Compute per-signal edge for adaptive weights."""
        adaptive_cfg = self.config.get("adaptive", {})
        lookback = adaptive_cfg.get("lookback_trades", 100)
        return edge_per_signal(self.state.completed_trades, lookback)
