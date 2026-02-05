"""Feature extraction from market data structures.

Each function returns a dict of named features consumed by signal modules.
Features are pure functions of OrderBook / TradeTape — no side effects.
"""
from __future__ import annotations
import math
from typing import Dict
from marketdata.book import OrderBook
from marketdata.tape import TradeTape
from core.utils import safe_div


def obi(book: OrderBook, levels: int = 5) -> Dict[str, float]:
    """Order-book imbalance features."""
    imb = book.imbalance(levels)

    # Rate of change: compare current imbalance to ~250ms-old state
    # We approximate via prev totals stored on the book
    bv = book.bid_volume(levels)
    av = book.ask_volume(levels)
    prev_imb = safe_div(
        book._prev_bid_total - book._prev_ask_total,
        book._prev_bid_total + book._prev_ask_total,
    )
    d_obi_dt = imb - prev_imb

    # Store for next delta
    book._prev_bid_total = bv
    book._prev_ask_total = av

    spread_penalty = max(book.spread_ticks() - 1.0, 0.0)  # penalty above 1 tick

    # Weighted mid vs mid — micro-pressure indicator
    wmid = book.weighted_mid()
    mid = book.mid_price()
    micro_pressure = safe_div(wmid - mid, book.tick_size) if book.tick_size > 0 else 0.0

    return {
        "obi": imb,
        "d_obi_dt": d_obi_dt,
        "spread_penalty": spread_penalty,
        "spread_ticks": book.spread_ticks(),
        "micro_pressure": micro_pressure,
        "bid_volume": bv,
        "ask_volume": av,
    }


def umom(tape: TradeTape, book: OrderBook, vol_lookback_s: int = 60) -> Dict[str, float]:
    """Micro-momentum features from the trade tape."""
    return {
        "ema_fast": tape.ema_fast,
        "ema_slow": tape.ema_slow,
        "momentum": tape.ema_fast - tape.ema_slow,
        "rv_10s": tape.realized_vol(lookback_s=10),
        "rv_60s": tape.realized_vol(lookback_s=vol_lookback_s),
        "trade_rate": tape.trade_rate(1000),
        "buy_sell_ratio": tape.buy_sell_ratio(2000),
        "spread_ticks": book.spread_ticks(),
    }


def prt(book: OrderBook, config: dict) -> Dict[str, float]:
    """Pull-&-Replace Trap features — spoofing detection."""
    prt_cfg = config.get("prt", {})
    window_ms = prt_cfg.get("cancel_window_ms", 500)

    cancel_rate_ask = book.cancel_rate("ask", window_ms)
    cancel_rate_bid = book.cancel_rate("bid", window_ms)
    cancel_vol_ask = book.cancel_volume("ask", window_ms)
    cancel_vol_bid = book.cancel_volume("bid", window_ms)
    mid_chg = book.mid_change_ticks(window_ms)

    return {
        "cancel_rate_ask": cancel_rate_ask,
        "cancel_rate_bid": cancel_rate_bid,
        "cancel_vol_ask": cancel_vol_ask,
        "cancel_vol_bid": cancel_vol_bid,
        "mid_change_ticks": mid_chg,
    }


def whales(tape: TradeTape, book: OrderBook, config: dict) -> Dict[str, float]:
    """Whale activity features: Large-Trade-Burst, Sweep, Iceberg."""
    whale_cfg = config.get("whale", {})
    burst_window = whale_cfg.get("burst_window_ms", 2000)
    multiplier = whale_cfg.get("large_trade_multiplier", 5.0)
    min_usd = whale_cfg.get("large_trade_min_usd", 50000)
    sweep_levels_threshold = whale_cfg.get("sweep_levels", 3)
    ice_window = whale_cfg.get("ice_window_ms", 5000)

    # Large Trade Burst
    bb = tape.buy_burst(burst_window, multiplier=multiplier, min_usd=min_usd)
    sb = tape.sell_burst(burst_window, multiplier=multiplier, min_usd=min_usd)

    # Sweep detection: levels cleared on one side
    ask_levels_cleared = book.levels_cleared_since("ask", burst_window)
    bid_levels_cleared = book.levels_cleared_since("bid", burst_window)
    sweep_up = 1.0 if ask_levels_cleared >= sweep_levels_threshold else ask_levels_cleared / max(sweep_levels_threshold, 1)
    sweep_down = 1.0 if bid_levels_cleared >= sweep_levels_threshold else bid_levels_cleared / max(sweep_levels_threshold, 1)

    # Iceberg detection: repeated size replenishment at a single price
    # Approximate: if a bid level has high cancel volume but price hasn't moved,
    # it's likely being replenished (iceberg absorbing)
    ice_absorb_bid = book.cancel_volume("bid", ice_window) if abs(book.mid_change_ticks(ice_window)) < 1 else 0.0
    ice_absorb_ask = book.cancel_volume("ask", ice_window) if abs(book.mid_change_ticks(ice_window)) < 1 else 0.0
    # Positive = bids absorbing sells (bullish), negative = asks absorbing buys (bearish)
    ice_signal = safe_div(ice_absorb_bid - ice_absorb_ask, ice_absorb_bid + ice_absorb_ask + 1e-9)

    return {
        "buy_burst": bb,
        "sell_burst": sb,
        "sweep_up": sweep_up,
        "sweep_down": sweep_down,
        "ice_absorb": ice_signal,
        "ice_absorb_bid": ice_absorb_bid,
        "ice_absorb_ask": ice_absorb_ask,
    }


def vwap_features(tape: TradeTape, book: OrderBook, config: dict) -> Dict[str, float]:
    """VWAP deviation features — mean-reversion signal."""
    vwap_cfg = config.get("vwap", {})
    n_trades = vwap_cfg.get("lookback_trades", 500)

    vwap_price = tape.vwap(n_trades)
    current_mid = book.mid_price()
    tick_size = book.tick_size if book.tick_size > 0 else 1.0

    deviation = (current_mid - vwap_price) / tick_size if vwap_price > 0 else 0.0

    return {
        "vwap": vwap_price,
        "vwap_deviation_ticks": deviation,
        "price_vs_vwap": safe_div(current_mid - vwap_price, vwap_price),
    }


def vol_regime(tape: TradeTape, config: dict) -> Dict[str, float]:
    """Volatility regime classification."""
    vol_cfg = config.get("vol_regime", {})
    lookback = vol_cfg.get("lookback_s", 60)
    low_thresh = vol_cfg.get("low_vol_threshold", 0.3)
    high_thresh = vol_cfg.get("high_vol_threshold", 1.2)
    extreme_thresh = vol_cfg.get("extreme_vol_threshold", 3.0)

    rv = tape.realized_vol(lookback)

    if rv < low_thresh:
        regime = 0.0       # mean-revert
        regime_name = "low"
    elif rv < high_thresh:
        regime = 0.5       # normal
        regime_name = "normal"
    elif rv < extreme_thresh:
        regime = 1.0       # momentum
        regime_name = "high"
    else:
        regime = -1.0      # too volatile, stay out
        regime_name = "extreme"

    return {
        "rv_regime": rv,
        "vol_regime": regime,
        "vol_regime_name": regime_name,
    }


def cvd_features(tape: TradeTape, book: OrderBook, config: dict) -> Dict[str, float]:
    """CVD (Cumulative Volume Delta) features — flow confirmation + divergence."""
    micro_cfg = config.get("microstructure", {})
    window_ms = micro_cfg.get("cvd_window_ms", 5000)

    cvd = tape.cvd_in_window(window_ms)
    total_vol = tape.volume_in_window(window_ms)
    cvd_norm = safe_div(cvd, max(total_vol, 1e-12))
    cvd_accel = tape.cvd_acceleration(window_ms)

    # CVD divergence: price going one way, CVD going the other
    # +1 = bullish divergence (price down but CVD up)
    # -1 = bearish divergence (price up but CVD down)
    mid_change = book.mid_change_ticks(window_ms)
    if mid_change < -0.5 and cvd_norm > 0.1:
        cvd_divergence = 1.0   # bullish: price down but buyers dominating
    elif mid_change > 0.5 and cvd_norm < -0.1:
        cvd_divergence = -1.0  # bearish: price up but sellers dominating
    else:
        cvd_divergence = 0.0

    return {
        "cvd": cvd,
        "cvd_norm": cvd_norm,
        "cvd_accel": cvd_accel,
        "cvd_divergence": cvd_divergence,
    }


def microstructure_features(tape: TradeTape, book: OrderBook, config: dict) -> Dict[str, float]:
    """Microstructure features: trade intensity, liquidity, micro vol ratio, absorption."""
    micro_cfg = config.get("microstructure", {})

    # TPS: trade intensity ratio (short window / baseline)
    tps_window = micro_cfg.get("tps_window_ms", 1000)
    tps_baseline = micro_cfg.get("tps_baseline_ms", 30000)
    rate_short = tape.trade_rate(tps_window)
    rate_long = tape.trade_rate(tps_baseline)
    tps_ratio = safe_div(rate_short, max(rate_long, 1e-6), 1.0)

    # LIQ: liquidity depth within N ticks of BBO
    liq_ticks = micro_cfg.get("liq_depth_ticks", 10)
    liq_bid = book.cumulative_depth("bid", liq_ticks)
    liq_ask = book.cumulative_depth("ask", liq_ticks)
    liq_total = liq_bid + liq_ask
    liq_thinness = safe_div(1.0, max(liq_total, 1e-12))
    liq_asymmetry = safe_div(liq_bid - liq_ask, max(liq_total, 1e-12))

    # MVR: micro volatility ratio (RV_short / RV_long)
    mvr_short_s = micro_cfg.get("mvr_short_s", 2)
    mvr_long_s = micro_cfg.get("mvr_long_s", 30)
    rv_short = tape.realized_vol(lookback_s=mvr_short_s)
    rv_long = tape.realized_vol(lookback_s=mvr_long_s)
    mvr = safe_div(rv_short, max(rv_long, 1e-9), 1.0)

    # ABS: absorption — aggressive volume that didn't move price
    abs_window = micro_cfg.get("absorption_window_ms", 3000)
    total_aggressive_vol = tape.volume_in_window(abs_window)
    mid_change_abs = abs(book.mid_change_ticks(abs_window))
    absorption = safe_div(total_aggressive_vol, max(mid_change_abs, 1e-6))

    return {
        "tps_ratio": tps_ratio,
        "liq_bid": liq_bid,
        "liq_ask": liq_ask,
        "liq_thinness": liq_thinness,
        "liq_asymmetry": liq_asymmetry,
        "mvr": mvr,
        "absorption": absorption,
    }


def compute_all(book: OrderBook, tape: TradeTape, config: dict) -> Dict[str, float]:
    """Compute all features in a single pass."""
    f: Dict[str, float] = {}
    f.update(obi(book))
    f.update(umom(tape, book, config.get("vol_regime", {}).get("lookback_s", 60)))
    f.update(prt(book, config))
    f.update(whales(tape, book, config))
    f.update(vwap_features(tape, book, config))
    f.update(vol_regime(tape, config))
    f.update(cvd_features(tape, book, config))
    f.update(microstructure_features(tape, book, config))
    return f
