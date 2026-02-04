"""Email Alerts — critical event notifications for Smallfish.

Sends email alerts for kill switch triggers, large drawdowns,
and daily summaries. Uses async SMTP via aiosmtplib.

Setup:
    Set in .env:
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_USER=your@gmail.com
        SMTP_PASS=your_app_password
        ALERT_EMAIL_TO=your@gmail.com
"""
from __future__ import annotations
import asyncio
import logging
import os
import time
from email.mime.text import MIMEText
from typing import Optional

from core.state import RuntimeState

log = logging.getLogger(__name__)


class EmailAlerter:
    """Async email alert sender for critical Smallfish events."""

    def __init__(self, state: RuntimeState, config: dict):
        self.state = state
        self.config = config

        self.smtp_host = os.environ.get("SMTP_HOST", "")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER", "")
        self.smtp_pass = os.environ.get("SMTP_PASS", "")
        self.to_addr = os.environ.get("ALERT_EMAIL_TO", "")

        self.enabled = bool(self.smtp_host and self.smtp_user and self.to_addr)

        # Cooldowns
        self._last_alert: dict[str, float] = {}
        self._cooldown_s = config.get("email", {}).get("cooldown_s", 600)

    async def send_alert(self, subject: str, body: str, alert_type: str = "generic") -> None:
        """Send an email alert with cooldown."""
        if not self.enabled:
            return

        now = time.time()
        last = self._last_alert.get(alert_type, 0)
        if now - last < self._cooldown_s:
            return
        self._last_alert[alert_type] = now

        try:
            import aiosmtplib

            msg = MIMEText(body, "plain")
            msg["Subject"] = f"[Smallfish] {subject}"
            msg["From"] = self.smtp_user
            msg["To"] = self.to_addr

            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_pass,
                use_tls=False,
                start_tls=True,
            )
            log.info("Email alert sent: %s", subject)
        except ImportError:
            log.debug("aiosmtplib not installed, skipping email alert")
        except Exception as e:
            log.debug("Email alert error: %s", e)

    async def alert_kill_switch(self, reason: str) -> None:
        """Send kill switch alert."""
        s = self.state
        body = (
            f"Kill switch triggered: {reason}\n\n"
            f"Equity: ${s.equity:.2f}\n"
            f"Daily PnL: ${s.daily_pnl:+.4f}\n"
            f"Drawdown: {s.drawdown*100:.2f}%\n"
            f"Trades: {s.trade_count}\n"
            f"Win Rate: {s.win_count/max(s.trade_count,1)*100:.1f}%\n"
        )
        await self.send_alert(f"KILL SWITCH: {reason}", body, "kill_switch")

    async def alert_drawdown(self, dd_pct: float) -> None:
        """Send drawdown alert."""
        s = self.state
        body = (
            f"Drawdown alert: {dd_pct:.2f}%\n\n"
            f"Equity: ${s.equity:.2f}\n"
            f"Peak: ${s.peak_equity:.2f}\n"
            f"Daily PnL: ${s.daily_pnl:+.4f}\n"
        )
        await self.send_alert(f"DRAWDOWN: {dd_pct:.1f}%", body, "drawdown")

    async def send_daily_summary(self) -> None:
        """Send end-of-day summary."""
        s = self.state
        wr = s.win_count / max(s.trade_count, 1) * 100
        body = (
            f"Daily Summary\n"
            f"{'='*40}\n\n"
            f"Equity: ${s.equity:.2f}\n"
            f"Daily PnL: ${s.daily_pnl:+.4f}\n"
            f"Total PnL: ${s.realized_pnl:+.4f}\n"
            f"Trades: {s.trade_count}\n"
            f"Win Rate: {wr:.1f}%\n"
            f"Drawdown: {s.drawdown*100:.2f}%\n"
            f"Max Drawdown: {s.max_drawdown*100:.2f}%\n"
            f"Vol Regime: {s.vol_regime}\n"
        )
        # Reset cooldown for daily summary
        self._last_alert.pop("daily_summary", None)
        await self.send_alert("Daily Summary", body, "daily_summary")
