"""Shared risk profiles for Smallfish.

Profiles define risk parameters used by both the backtest engine and the
live trading loop.  ``apply_profile`` maps profile keys to the flat config
dict consumed by the runtime.
"""
from __future__ import annotations

from typing import Dict, Any

PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "risk_pct": 0.005,       # 0.5% risk per trade
        "max_risk_usd": 5.0,
        "equity_cap_mult": 2,    # cap compounding at 2x initial
        "sl_range_mult": 0.50,   # SL = 0.5x avg candle range
        "tp_range_mult": 0.70,   # TP = 0.7x avg range -> R:R = 1.4:1
        "trail_pct": 0.30,       # trail 30% of range behind peak
        "cooldown_ms": 60_000,   # 1 minute between entries
        "max_hold": 15,          # 15 candle max hold
        "max_daily_R": 10,       # daily loss limit
        "max_positions": 2,      # max concurrent positions
        "C_enter": 0.55,
        "C_exit": 0.35,
        "alpha": 4,
        "conf_scale": False,     # no confidence-based size scaling
        "breakeven_R": 999,      # never move to breakeven
        "partial_tp": False,     # no partial exits
        "min_signals": 3,        # require 3 signal categories to agree
    },
    "balanced": {
        "risk_pct": 0.015,       # 1.5% risk per trade
        "max_risk_usd": 20.0,
        "equity_cap_mult": 3,    # moderate compounding
        "sl_range_mult": 0.50,   # same SL -- don't tighten, it works
        "tp_range_mult": 1.00,   # wider TP -> R:R = 2:1
        "trail_pct": 0.25,       # tighter trail
        "cooldown_ms": 45_000,   # 45s cooldown
        "max_hold": 12,
        "max_daily_R": 15,       # higher limit for more risk
        "max_positions": 3,      # max concurrent positions
        "C_enter": 0.53,
        "C_exit": 0.33,
        "alpha": 4,
        "conf_scale": True,      # scale size with confidence
        "breakeven_R": 0.8,      # move to BE after 0.8R profit
        "partial_tp": False,
        "min_signals": 3,        # require 3 signal categories to agree
    },
    "aggressive": {
        "risk_pct": 0.025,       # 2.5% risk per trade
        "max_risk_usd": 40.0,
        "equity_cap_mult": 4,    # compound up to 4x
        "sl_range_mult": 0.50,   # keep SL same -- proven to work
        "tp_range_mult": 1.20,   # wide TP -> R:R = 2.4:1
        "trail_pct": 0.22,       # tighter trail to lock profit
        "cooldown_ms": 30_000,   # 30s cooldown
        "max_hold": 12,
        "max_daily_R": 20,       # high daily allowance
        "max_positions": 3,      # max concurrent positions
        "C_enter": 0.52,
        "C_exit": 0.30,
        "alpha": 4,
        "conf_scale": True,
        "breakeven_R": 0.7,      # move to BE after 0.7R
        "partial_tp": True,      # take half at 1R
        "min_signals": 2,        # require 2 signal categories to agree
    },
    "ultra": {
        "risk_pct": 0.035,       # 3.5% risk per trade
        "max_risk_usd": 75.0,
        "equity_cap_mult": 6,    # heavy compounding
        "sl_range_mult": 0.50,   # KEEP SL the same -- no tighter!
        "tp_range_mult": 1.40,   # very wide TP -> R:R = 2.8:1
        "trail_pct": 0.20,       # tight trail
        "cooldown_ms": 20_000,   # 20s cooldown
        "max_hold": 12,
        "max_daily_R": 30,       # very high daily allowance
        "max_positions": 4,      # max concurrent positions
        "C_enter": 0.50,
        "C_exit": 0.28,
        "alpha": 3.5,
        "conf_scale": True,
        "breakeven_R": 0.6,
        "partial_tp": True,
        "min_signals": 2,        # require 2 signal categories to agree
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
