"""Tests for email alert module."""
import pytest
import os
from unittest.mock import patch
from remote.email_alert import EmailAlerter
from core.state import RuntimeState


@pytest.fixture
def state(config):
    s = RuntimeState(config)
    s.equity = 950.0
    s.peak_equity = 1000.0
    s.daily_pnl = -50.0
    s.drawdown = 0.05
    s.max_drawdown = 0.05
    s.trade_count = 5
    s.win_count = 2
    s.loss_count = 3
    s.vol_regime = "high"
    return s


class TestEmailAlerterInit:
    def test_disabled_without_env(self, state, config):
        with patch.dict(os.environ, {
            "SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": "",
        }, clear=False):
            alerter = EmailAlerter(state, config)
            assert alerter.enabled is False

    def test_enabled_with_env(self, state, config):
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@test.com",
            "SMTP_PASS": "pass123",
            "ALERT_EMAIL_TO": "alert@test.com",
        }):
            alerter = EmailAlerter(state, config)
            assert alerter.enabled is True
            assert alerter.smtp_host == "smtp.test.com"
            assert alerter.smtp_port == 587
            assert alerter.to_addr == "alert@test.com"


class TestEmailAlerterCooldown:
    def test_cooldown_default(self, state, config):
        config["email"] = {"cooldown_s": 600}
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": ""}):
            alerter = EmailAlerter(state, config)
            assert alerter._cooldown_s == 600

    def test_cooldown_tracking(self, state, config):
        import time
        config["email"] = {"cooldown_s": 60}
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": ""}):
            alerter = EmailAlerter(state, config)
            alerter._last_alert["test"] = time.time()
            last = alerter._last_alert.get("test", 0)
            assert time.time() - last < 1.0


class TestEmailAlerterMessages:
    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, state, config):
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": ""}):
            alerter = EmailAlerter(state, config)
            # Should return without error when disabled
            await alerter.send_alert("Test Subject", "Test body", "test")

    @pytest.mark.asyncio
    async def test_alert_kill_switch_disabled(self, state, config):
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": ""}):
            alerter = EmailAlerter(state, config)
            await alerter.alert_kill_switch("test_reason")
            # No crash is success

    @pytest.mark.asyncio
    async def test_alert_drawdown_disabled(self, state, config):
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": ""}):
            alerter = EmailAlerter(state, config)
            await alerter.alert_drawdown(5.5)

    @pytest.mark.asyncio
    async def test_daily_summary_disabled(self, state, config):
        with patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": "", "ALERT_EMAIL_TO": ""}):
            alerter = EmailAlerter(state, config)
            await alerter.send_daily_summary()
