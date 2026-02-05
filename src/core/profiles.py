"""Shared risk profiles for Smallfish.

Profiles define risk parameters used by both the backtest engine and the
live trading loop.  ``apply_profile`` maps profile keys to the flat config
dict consumed by the runtime.
"""
from __future__ import annotations

from typing import Dict, Any

PROFILES: Dict[str, Dict[str, Any]] = {
    # ── Profiles tuned from 7D backtest + corrected param sweep ──────
    #
    # Validated findings (BacktestEngine-confirmed):
    #   - partial_tp=False     (always — cuts avg win in half, kills edge)
    #   - breakeven_R=999      (disabled — early BE chops winners via noise)
    #   - sl_range_mult=0.50   (0.30 is too tight for 1m candles, gets whipsawed)
    #   - tp_range_mult≥1.30   (wider TP lets winners run, improves avg_win/avg_loss)
    #   - trail_pct=0.20-0.30  (tight trailing, never disabled)

    "conservative": {
        "risk_pct": 0.005,       # 0.5% risk per trade
        "max_risk_usd": 5.0,
        "equity_cap_mult": 2,    # cap compounding at 2x initial
        "sl_range_mult": 0.50,   # proven SL distance for 1m candles
        "tp_range_mult": 1.30,   # R:R = 2.6:1
        "trail_pct": 0.30,       # moderate trail
        "cooldown_ms": 60_000,   # 1 minute between entries
        "max_hold": 15,          # 15 candle max hold
        "max_daily_R": 10,       # daily loss limit
        "max_positions": 2,      # max concurrent positions
        "C_enter": 0.58,
        "C_exit": 0.35,
        "alpha": 4,
        "conf_scale": False,     # no confidence-based size scaling
        "breakeven_R": 999,      # disabled — let trail do the work
        "partial_tp": False,     # NEVER — kills edge
        "min_signals": 5,        # strict filter
    },
    "balanced": {
        "risk_pct": 0.015,       # 1.5% risk per trade
        "max_risk_usd": 20.0,
        "equity_cap_mult": 3,    # moderate compounding
        "sl_range_mult": 0.50,   # proven SL distance
        "tp_range_mult": 1.60,   # R:R = 3.2:1
        "trail_pct": 0.25,       # tighter trail
        "cooldown_ms": 45_000,   # 45s cooldown
        "max_hold": 12,
        "max_daily_R": 15,
        "max_positions": 3,
        "C_enter": 0.55,
        "C_exit": 0.33,
        "alpha": 4,
        "conf_scale": True,
        "breakeven_R": 999,      # disabled
        "partial_tp": False,     # NEVER
        "min_signals": 4,        # moderate filter
    },
    "aggressive": {
        "risk_pct": 0.025,       # 2.5% risk per trade
        "max_risk_usd": 40.0,
        "equity_cap_mult": 4,    # compound up to 4x
        "sl_range_mult": 0.50,   # proven SL — R:R = 3.2:1
        "tp_range_mult": 1.60,   # wider TP than before (was 1.20)
        "trail_pct": 0.22,       # tight trail to lock profit
        "cooldown_ms": 30_000,   # 30s cooldown
        "max_hold": 12,
        "max_daily_R": 20,
        "max_positions": 3,
        "C_enter": 0.55,
        "C_exit": 0.30,
        "alpha": 4,
        "conf_scale": True,
        "breakeven_R": 999,      # disabled — early BE chops winners
        "partial_tp": False,     # NEVER — was killing edge (-$2.57 avg loss vs $1.74 avg win)
        "min_signals": 3,        # wide entry
    },
    "ultra": {
        "risk_pct": 0.035,       # 3.5% risk per trade
        "max_risk_usd": 75.0,
        "equity_cap_mult": 6,    # heavy compounding
        "sl_range_mult": 0.50,   # proven SL — R:R = 4:1
        "tp_range_mult": 2.00,   # widest TP
        "trail_pct": 0.20,       # tightest trail
        "cooldown_ms": 20_000,   # 20s cooldown
        "max_hold": 12,
        "max_daily_R": 30,
        "max_positions": 4,
        "C_enter": 0.52,
        "C_exit": 0.28,
        "alpha": 4,
        "conf_scale": True,
        "breakeven_R": 999,      # disabled
        "partial_tp": False,     # NEVER
        "min_signals": 3,        # wide entry
    },
}


def apply_profile(config: dict, profile_name: str) -> None:
    """Apply a risk profile to the runtime *config* dict **in-place**.

    Keys that map directly to top-level config entries are set there;
    remaining profile-specific keys are stored under ``config["profile"]``.
    """
    p = PROFILES.get(profile_name)
    if p is None:
        raise ValueError(f"Unknown profile: {profile_name!r}. "
                         f"Choose from {list(PROFILES)}")

    # Direct mappings to live config keys
    config["risk_per_trade"] = p["risk_pct"]
    config["max_risk_dollars"] = p["max_risk_usd"]
    config["C_enter"] = p["C_enter"]
    config["C_exit"] = p["C_exit"]
    config["alpha"] = p["alpha"]
    config["max_daily_R"] = p["max_daily_R"]
    config["cooldown_after_loss_ms"] = p["cooldown_ms"]
    config["max_open_orders"] = p["max_positions"]

    config["min_signals"] = p.get("min_signals", 2)

    # Store remaining profile-specific keys for the engine to read
    config["profile"] = {
        "name": profile_name,
        "trail_pct": p["trail_pct"],
        "breakeven_R": p["breakeven_R"],
        "partial_tp": p["partial_tp"],
        "conf_scale": p["conf_scale"],
        "sl_range_mult": p["sl_range_mult"],
        "tp_range_mult": p["tp_range_mult"],
        "equity_cap_mult": p["equity_cap_mult"],
        "max_hold": p["max_hold"],
    }
