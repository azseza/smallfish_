"""Backtesting Engine for Bybit Scalper.

Downloads historical kline + trade data from Bybit public API,
simulates the full signal pipeline, and reports projected performance.

Usage:
    python src/backtest.py --equity 50 --days 30 --symbol BTCUSDT
    python src/backtest.py --equity 50 --days 30 --multi --mode aggressive
    python src/backtest.py --equity 50 --days 30 --sweep
"""
from __future__ import annotations
import asyncio
import argparse
import copy
import logging
import math
import os
import sys
import time
from typing import List, Dict, Tuple, Optional

import aiohttp
import orjson
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from core.types import Trade, Side, Position, TradeResult
from core.state import RuntimeState
from core.ringbuffer import RingBuffer
from core.utils import (
    sigmoid, clamp, ema_update, ema_alpha_seconds, safe_div, tick_round, qty_round, time_now_ms
)
from marketdata.book import OrderBook
from marketdata.tape import TradeTape
import marketdata.features as features
import signals.fuse as fuse
import exec.risk as risk

log = logging.getLogger("backtest")


# ─── Optimization Profiles ──────────────────────────────────────────

PROFILES = {
    "conservative": {
        "risk_pct": 0.005,       # 0.5% risk per trade
        "max_risk_usd": 5.0,
        "equity_cap_mult": 3,    # cap compounding at 3x initial
        "sl_range_mult": 0.50,   # SL = 0.5x avg candle range
        "tp_range_mult": 0.70,   # TP = 0.7x avg range → R:R = 1.4:1
        "trail_pct": 0.30,       # trail 30% of range behind peak
        "cooldown_ms": 60_000,   # 1 minute between entries
        "max_hold": 15,          # 15 candle max hold
        "max_daily_R": 10,       # daily loss limit
        "C_enter": 0.55,
        "C_exit": 0.35,
        "alpha": 4,
        "conf_scale": False,     # no confidence-based size scaling
        "breakeven_R": 999,      # never move to breakeven
        "partial_tp": False,     # no partial exits
    },
    "balanced": {
        "risk_pct": 0.015,       # 1.5% risk per trade
        "max_risk_usd": 20.0,
        "equity_cap_mult": 5,    # moderate compounding
        "sl_range_mult": 0.50,   # same SL — don't tighten, it works
        "tp_range_mult": 1.00,   # wider TP → R:R = 2:1
        "trail_pct": 0.25,       # tighter trail
        "cooldown_ms": 45_000,   # 45s cooldown
        "max_hold": 12,
        "max_daily_R": 15,       # higher limit for more risk
        "C_enter": 0.53,
        "C_exit": 0.33,
        "alpha": 4,
        "conf_scale": True,      # scale size with confidence
        "breakeven_R": 0.8,      # move to BE after 0.8R profit
        "partial_tp": False,
    },
    "aggressive": {
        "risk_pct": 0.025,       # 2.5% risk per trade
        "max_risk_usd": 40.0,
        "equity_cap_mult": 8,    # compound up to 8x
        "sl_range_mult": 0.50,   # keep SL same — proven to work
        "tp_range_mult": 1.20,   # wide TP → R:R = 2.4:1
        "trail_pct": 0.22,       # tighter trail to lock profit
        "cooldown_ms": 30_000,   # 30s cooldown
        "max_hold": 12,
        "max_daily_R": 20,       # high daily allowance
        "C_enter": 0.52,
        "C_exit": 0.30,
        "alpha": 4,
        "conf_scale": True,
        "breakeven_R": 0.7,      # move to BE after 0.7R
        "partial_tp": True,      # take half at 1R
    },
    "ultra": {
        "risk_pct": 0.035,       # 3.5% risk per trade
        "max_risk_usd": 75.0,
        "equity_cap_mult": 15,   # heavy compounding
        "sl_range_mult": 0.50,   # KEEP SL the same — no tighter!
        "tp_range_mult": 1.40,   # very wide TP → R:R = 2.8:1
        "trail_pct": 0.20,       # tight trail
        "cooldown_ms": 20_000,   # 20s cooldown
        "max_hold": 12,
        "max_daily_R": 30,       # very high daily allowance
        "C_enter": 0.50,
        "C_exit": 0.28,
        "alpha": 3.5,
        "conf_scale": True,
        "breakeven_R": 0.6,
        "partial_tp": True,
    },
}


# ─── Historical Data Download ─────────────────────────────────────────

async def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[dict]:
    """Download kline data from Bybit public API.
    Bybit returns klines in REVERSE order (newest first), so we paginate backwards.
    """
    url = "https://api.bybit.com/v5/market/kline"
    all_klines = []
    cursor_end = end_ms

    async with aiohttp.ClientSession() as session:
        while cursor_end > start_ms:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": cursor_end,
                "limit": 1000,
            }
            async with session.get(url, params=params) as resp:
                data = await resp.json(content_type=None)
                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break

                for k in klines:
                    all_klines.append({
                        "ts": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "turnover": float(k[6]),
                    })

                # Oldest candle in this batch (last element since reverse order)
                oldest_ts = int(klines[-1][0])
                if oldest_ts >= cursor_end:
                    break  # no progress
                cursor_end = oldest_ts - 1

                log.info("  ... downloaded to %s (%d candles so far)",
                         time.strftime("%Y-%m-%d", time.gmtime(oldest_ts / 1000)),
                         len(all_klines))
                await asyncio.sleep(0.05)

    # Deduplicate and sort ascending
    seen = set()
    unique = []
    for k in all_klines:
        if k["ts"] not in seen:
            seen.add(k["ts"])
            unique.append(k)
    unique.sort(key=lambda x: x["ts"])
    return unique


# ─── Synthetic Book from Kline ─────────────────────────────────────────

def build_synthetic_book(kline: dict, prev_kline: dict, tick_size: float) -> OrderBook:
    """Build a synthetic order book from kline data."""
    book = OrderBook(symbol="", depth=10, tick_size=tick_size)

    close = kline["close"]
    volume = kline["volume"]
    high = kline["high"]
    low = kline["low"]
    opn = kline["open"]
    rng = high - low if high > low else tick_size

    spread = max(tick_size, min(rng * 0.01, tick_size * 3))

    mid = close
    bb = tick_round(mid - spread / 2, tick_size)
    ba = tick_round(mid + spread / 2, tick_size)

    base_size = max(volume / 15, 0.01)

    body = close - opn
    body_ratio = body / rng if rng > 0 else 0
    upper_wick = (high - max(close, opn)) / rng if rng > 0 else 0
    lower_wick = (min(close, opn) - low) / rng if rng > 0 else 0

    imbalance = clamp(body_ratio * 1.5, -0.8, 0.8)

    bids = []
    asks = []
    for i in range(5):
        bid_px = tick_round(bb - i * tick_size, tick_size)
        ask_px = tick_round(ba + i * tick_size, tick_size)
        decay = (5 - i) / 5
        bid_size = base_size * decay * (1.0 + imbalance)
        ask_size = base_size * decay * (1.0 - imbalance)
        if i < 2:
            bid_size *= (1.0 + lower_wick * 2)
            ask_size *= (1.0 + upper_wick * 2)
        bids.append([bid_px, max(bid_size, 0.001)])
        asks.append([ask_px, max(ask_size, 0.001)])

    book.on_snapshot(bids, asks, seq=1)
    return book


# ─── Synthetic Trades from Kline ─────────────────────────────────────────

def generate_synthetic_trades(kline: dict, prev_close: float, symbol: str) -> List[Trade]:
    """Generate synthetic trades from a 1-minute candle."""
    trades = []
    ts_start = kline["ts"]
    volume = kline["volume"]
    o, h, l, c = kline["open"], kline["high"], kline["low"], kline["close"]

    if volume <= 0:
        return trades

    n_trades = max(5, min(50, int(volume * 2)))
    vol_per_trade = volume / n_trades

    np.random.seed(int(ts_start) % (2**31))
    prices = []

    if c >= o:
        path = [o, l, h, c]
    else:
        path = [o, h, l, c]

    for i in range(n_trades):
        frac = i / max(n_trades - 1, 1)
        if frac < 0.25:
            px = path[0] + (path[1] - path[0]) * (frac / 0.25)
        elif frac < 0.5:
            px = path[1] + (path[2] - path[1]) * ((frac - 0.25) / 0.25)
        elif frac < 0.75:
            px = path[2] + (path[3] - path[2]) * ((frac - 0.5) / 0.25)
        else:
            px = path[3] + np.random.normal(0, (h - l) * 0.01)

        px += np.random.normal(0, (h - l) * 0.005)
        px = max(l * 0.999, min(h * 1.001, px))

        if len(prices) > 0:
            side = Side.BUY if px >= prices[-1] else Side.SELL
        else:
            side = Side.BUY if c >= o else Side.SELL

        prices.append(px)
        ts = ts_start + int(i * 60000 / n_trades)
        qty = vol_per_trade * (0.5 + np.random.random())

        trades.append(Trade(
            trade_id=f"bt_{ts}_{i}",
            symbol=symbol,
            price=px,
            quantity=qty,
            side=side,
            timestamp=ts,
        ))

    return trades


# ─── Backtest Engine ─────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, config: dict, initial_equity: float = 50.0,
                 profile: str = "conservative"):
        config["initial_equity"] = initial_equity
        self.config = config
        self.state = RuntimeState(config)
        self.state.equity = initial_equity
        self.state.peak_equity = initial_equity
        self.state.daily_start_equity = initial_equity

        self.books: Dict[str, OrderBook] = {}
        self.tapes: Dict[str, TradeTape] = {}

        self.equity_curve: List[Tuple[int, float]] = []
        self.trade_log: List[dict] = []
        self.signal_log: List[dict] = []

        # Load profile
        self.profile_name = profile
        p = PROFILES.get(profile, PROFILES["conservative"])
        self._risk_pct = p["risk_pct"]
        self._max_risk_usd = p["max_risk_usd"]
        self._equity_cap_mult = p["equity_cap_mult"]
        self._sl_range_mult = p["sl_range_mult"]
        self._tp_range_mult = p["tp_range_mult"]
        self._trail_pct = p["trail_pct"]
        self._min_entry_gap_ms = p["cooldown_ms"]
        self._max_hold_candles = p["max_hold"]
        self._conf_scale = p["conf_scale"]
        self._breakeven_R = p["breakeven_R"]
        self._partial_tp = p["partial_tp"]

        self.config["C_enter"] = p["C_enter"]
        self.config["C_exit"] = p["C_exit"]
        self.config["alpha"] = p["alpha"]
        self.config["risk_per_trade"] = p["risk_pct"]  # sync for R calculations
        self.config["max_daily_R"] = p["max_daily_R"]  # daily loss limit per profile

        # Rate limiter
        self._last_entry_ts: Dict[str, int] = {}

        # Fixed sizing base
        self._fixed_equity = initial_equity

        # Hold counters
        self._hold_counter: Dict[str, int] = {}

        # Track if partial TP was taken
        self._partial_taken: Dict[str, bool] = {}

        # Per-symbol range tracker
        self._recent_ranges: Dict[str, RingBuffer] = {}

        # Streak tracking for momentum sizing
        self._win_streak = 0
        self._loss_streak = 0

    def init_symbol(self, symbol: str):
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)
        self.books[symbol] = OrderBook(symbol=symbol, depth=10, tick_size=tick_size)
        self.tapes[symbol] = TradeTape(capacity=5000)
        self._last_entry_ts[symbol] = 0
        self._recent_ranges[symbol] = RingBuffer(20)

    def process_candle(self, symbol: str, kline: dict, prev_kline: dict) -> None:
        """Process one 1-minute candle through the full signal pipeline."""
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)
        book = self.books[symbol]
        tape = self.tapes[symbol]

        # Apply delta for cancel tracking
        if prev_kline and book.bids:
            old_bids = list(book.bids[:3])
            old_asks = list(book.asks[:3])

        # Build synthetic book
        new_book = build_synthetic_book(kline, prev_kline, tick_size)

        if prev_kline and book.bids:
            delta_bids = [[p, s] for p, s in new_book.bids]
            delta_asks = [[p, s] for p, s in new_book.asks]
            new_bid_prices = {p for p, _ in new_book.bids}
            new_ask_prices = {p for p, _ in new_book.asks}
            for p, s in old_bids:
                if p not in new_bid_prices:
                    delta_bids.append([p, 0])
            for p, s in old_asks:
                if p not in new_ask_prices:
                    delta_asks.append([p, 0])
            book.on_delta(delta_bids, delta_asks, seq=book.last_update_seq + 1)
        else:
            book.on_snapshot(
                [[p, s] for p, s in new_book.bids],
                [[p, s] for p, s in new_book.asks],
                seq=book.last_update_seq + 1,
            )

        book.last_update_ts = kline["ts"]

        # Track per-symbol candle ranges
        candle_range = kline["high"] - kline["low"]
        if candle_range > 0:
            self._recent_ranges[symbol].append(candle_range)

        # Generate and process synthetic trades
        prev_close = prev_kline["close"] if prev_kline else kline["open"]
        trades = generate_synthetic_trades(kline, prev_close, symbol)
        for t in trades:
            tape.add_trade(t)

        # Manage existing position for this symbol
        if self.state.has_position(symbol):
            self._manage_position(symbol, kline)
            return  # don't look for new entries on same symbol

        if self.state.kill_switch:
            return

        # Rate limit per symbol
        ts = kline["ts"]
        if ts - self._last_entry_ts.get(symbol, 0) < self._min_entry_gap_ms:
            return

        # Compute features
        f = features.compute_all(book, tape, self.config)

        vol_name = f.get("vol_regime_name", "normal")
        self.state.vol_regime = vol_name
        if vol_name == "extreme":
            return

        quality_gate = 1.0

        scores = fuse.score_all(f, self.config)
        self.state.last_scores = scores

        weights = fuse.get_weights(self.config, {}, self.config.get("adaptive", {}))
        direction, conf, raw = fuse.decide(scores, weights, self.config.get("alpha", 5), quality_gate)

        if direction == 0 or conf < self.config.get("C_enter", 0.65):
            return

        self._try_enter(symbol, direction, conf, kline, scores)
        self._last_entry_ts[symbol] = ts

    def _try_enter(self, symbol: str, direction: int, confidence: float,
                   kline: dict, scores: dict) -> None:
        side = Side.BUY if direction == 1 else Side.SELL
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)

        # Per-symbol volatility-adapted stops
        rng_buf = self._recent_ranges.get(symbol)
        avg_range = np.mean(rng_buf.get()) if rng_buf and len(rng_buf) > 0 else kline["high"] - kline["low"]
        avg_range = max(avg_range, tick_size * 5)

        stop_distance = avg_range * self._sl_range_mult
        tp_distance = avg_range * self._tp_range_mult

        entry_price = kline["close"]

        if direction == 1:
            stop_price = entry_price - stop_distance
            tp_price = entry_price + tp_distance
        else:
            stop_price = entry_price + stop_distance
            tp_price = entry_price - tp_distance

        # Position sizing with compounding
        use_equity = min(self.state.equity, self._fixed_equity * self._equity_cap_mult)
        risk_dollars = min(use_equity * self._risk_pct, self._max_risk_usd)

        # Confidence-scaled sizing: higher conf = bigger size (up to 1.5x)
        if self._conf_scale and confidence > 0.6:
            conf_mult = 1.0 + (confidence - 0.6) * 1.25  # 0.6→1.0x, 0.8→1.25x, 1.0→1.5x
            risk_dollars *= min(conf_mult, 1.5)

        # Win streak bonus: after 3 consecutive wins, add 20% per extra win
        if self._win_streak >= 3:
            streak_mult = 1.0 + min(self._win_streak - 2, 3) * 0.2  # max 1.6x
            risk_dollars *= streak_mult

        if stop_distance <= 0:
            return
        qty = risk_dollars / stop_distance
        qty_step = self.config.get("qty_step", {}).get(symbol, 0.001)
        min_qty = self.config.get("min_qty", {}).get(symbol, 0.001)
        qty = qty_round(qty, qty_step)
        if qty < min_qty:
            return

        # Slippage: half a tick adverse
        entry_price += direction * tick_size * 0.5

        pos = self.state.on_enter(
            symbol=symbol,
            side=side,
            fill_price=entry_price,
            quantity=qty,
            confidence=confidence,
            scores=scores,
            stop_price=stop_price,
            tp_price=tp_price,
        )
        self._hold_counter[symbol] = 0
        self._partial_taken[symbol] = False

        self.signal_log.append({
            "ts": kline["ts"],
            "symbol": symbol,
            "direction": direction,
            "confidence": round(confidence, 4),
            "entry_price": entry_price,
            "stop": stop_price,
            "tp": tp_price,
            "qty": qty,
        })

    def _manage_position(self, symbol: str, kline: dict) -> None:
        pos = self.state.position(symbol)
        if not pos:
            return

        high = kline["high"]
        low = kline["low"]
        close = kline["close"]
        tick_size = self.config.get("tick_sizes", {}).get(symbol, 0.01)

        self._hold_counter[symbol] = self._hold_counter.get(symbol, 0) + 1

        # Max hold time exceeded — force exit at close
        if self._hold_counter.get(symbol, 0) >= self._max_hold_candles:
            self._exit_position(symbol, close, "max_hold", kline["ts"])
            return

        # Check stop loss hit
        if pos.side == Side.BUY:
            if low <= pos.stop_price:
                self._exit_position(symbol, pos.stop_price, "sl_hit", kline["ts"])
                return
            # Partial TP at 1R (half position)
            r_distance = pos.entry_price - pos.stop_price  # risk distance
            if self._partial_tp and not self._partial_taken.get(symbol, False):
                partial_target = pos.entry_price + r_distance  # 1R target
                if high >= partial_target:
                    self._take_partial(symbol, partial_target, kline["ts"])
            if high >= pos.tp_price:
                self._exit_position(symbol, pos.tp_price, "tp_hit", kline["ts"])
                return
            if high > pos.peak_favorable:
                pos.peak_favorable = high
        else:  # SHORT
            if high >= pos.stop_price:
                self._exit_position(symbol, pos.stop_price, "sl_hit", kline["ts"])
                return
            r_distance = pos.stop_price - pos.entry_price
            if self._partial_tp and not self._partial_taken.get(symbol, False):
                partial_target = pos.entry_price - r_distance
                if low <= partial_target:
                    self._take_partial(symbol, partial_target, kline["ts"])
            if low <= pos.tp_price:
                self._exit_position(symbol, pos.tp_price, "tp_hit", kline["ts"])
                return
            if low < pos.peak_favorable or pos.peak_favorable == 0:
                pos.peak_favorable = low

        # Breakeven stop: move SL to entry after reaching breakeven_R profit
        if pos.side == Side.BUY:
            r_distance = pos.entry_price - pos.stop_price if pos.stop_price < pos.entry_price else 0
            current_R = (pos.peak_favorable - pos.entry_price) / r_distance if r_distance > 0 else 0
            if current_R >= self._breakeven_R and pos.stop_price < pos.entry_price:
                pos.stop_price = pos.entry_price + tick_size  # breakeven + 1 tick
        else:
            r_distance = pos.stop_price - pos.entry_price if pos.stop_price > pos.entry_price else 0
            current_R = (pos.entry_price - pos.peak_favorable) / r_distance if r_distance > 0 else 0
            if current_R >= self._breakeven_R and pos.stop_price > pos.entry_price:
                pos.stop_price = pos.entry_price - tick_size

        # Trailing stop
        rng_buf = self._recent_ranges.get(symbol)
        avg_range = np.mean(rng_buf.get()) if rng_buf and len(rng_buf) > 0 else 100
        trail_distance = avg_range * self._trail_pct

        if pos.side == Side.BUY:
            new_stop = pos.peak_favorable - trail_distance
            if new_stop > pos.stop_price:
                pos.stop_price = new_stop
        else:
            new_stop = pos.peak_favorable + trail_distance
            if new_stop < pos.stop_price:
                pos.stop_price = new_stop

    def _take_partial(self, symbol: str, price: float, ts: int) -> None:
        """Take profit on half the position at 1R."""
        pos = self.state.position(symbol)
        if not pos:
            return

        qty_step = self.config.get("qty_step", {}).get(symbol, 0.001)
        min_qty = self.config.get("min_qty", {}).get(symbol, 0.001)
        half_qty = qty_round(pos.quantity / 2, qty_step)
        if half_qty < min_qty:
            return

        # Record partial PnL
        raw_pnl = (price - pos.entry_price) * half_qty * int(pos.side)
        self.state.equity += raw_pnl
        self.state.realized_pnl += raw_pnl
        self.state.daily_pnl += raw_pnl
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        # Reduce position size
        pos.quantity -= half_qty
        self._partial_taken[symbol] = True

        self.trade_log.append({
            "ts": ts,
            "symbol": symbol,
            "side": pos.side.name,
            "entry": pos.entry_price,
            "exit": price,
            "qty": half_qty,
            "pnl": raw_pnl,
            "pnl_R": 1.0,  # partial at 1R by definition
            "duration_ms": 0,
            "reason": "partial_tp",
        })
        self.equity_curve.append((ts, self.state.equity))

    def _exit_position(self, symbol: str, exit_price: float, reason: str, ts: int) -> None:
        result = self.state.on_exit(symbol, exit_price, reason)
        if result:
            self.trade_log.append({
                "ts": ts,
                "symbol": symbol,
                "side": result.side.name,
                "entry": result.entry_price,
                "exit": exit_price,
                "qty": result.quantity,
                "pnl": result.pnl,
                "pnl_R": result.pnl_R,
                "duration_ms": result.duration_ms,
                "reason": reason,
            })

            # Update streaks
            if result.pnl > 0:
                self._win_streak += 1
                self._loss_streak = 0
            else:
                self._loss_streak += 1
                self._win_streak = 0

        self.equity_curve.append((ts, self.state.equity))
        self._partial_taken.pop(symbol, None)

    def force_close_all(self, kline: dict) -> None:
        for symbol in list(self.state.positions.keys()):
            self._exit_position(symbol, kline["close"], "backtest_end", kline["ts"])

    def report(self) -> dict:
        trades = self.trade_log
        if not trades:
            return {"error": "No trades executed"}

        pnls = [t["pnl"] for t in trades]
        pnl_Rs = [t["pnl_R"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_count = len(wins)
        loss_count = len(losses)
        n = len(trades)
        win_rate = win_count / n if n > 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_pnl = np.mean(pnls)
        avg_R = np.mean(pnl_Rs)

        if len(pnl_Rs) > 1:
            sharpe = np.mean(pnl_Rs) / (np.std(pnl_Rs) + 1e-9)
        else:
            sharpe = 0

        max_dd = 0
        peak = self.state.daily_start_equity
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        start_eq = self.config.get("initial_equity", 50)
        end_eq = self.state.equity
        total_return = (end_eq - start_eq) / start_eq * 100

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t["reason"]
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0.0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t["pnl"]

        return {
            "initial_equity": start_eq,
            "final_equity": round(end_eq, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": n,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "avg_pnl_per_trade": round(avg_pnl, 4),
            "avg_R": round(avg_R, 3),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "trades_per_day": round(n / max(len(set(t["ts"] // 86400000 for t in trades)), 1), 1),
            "exit_reasons": reasons,
        }


# ─── Main ─────────────────────────────────────────────────────────────────

async def run_backtest(symbols: List[str], days: int = 30, equity: float = 50.0,
                       profile: str = "conservative"):
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sym_str = "+".join(symbols)
    log.info("=" * 60)
    log.info("  BACKTEST: %s | %d days | $%.2f | profile=%s",
             sym_str, days, equity, profile)
    log.info("=" * 60)

    # Download data for all symbols
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    all_klines: Dict[str, List[dict]] = {}
    for sym in symbols:
        log.info("Downloading %d days of 1-minute klines for %s...", days, sym)
        klines = await fetch_klines(sym, "1", start_ms, end_ms)
        log.info("Downloaded %d candles for %s (%.1f days)", len(klines), sym,
                 len(klines) / 1440 if klines else 0)
        if len(klines) >= 100:
            all_klines[sym] = klines
        else:
            log.warning("Not enough data for %s, skipping", sym)

    if not all_klines:
        log.error("No symbols with sufficient data")
        return

    # Initialize engine
    engine = BacktestEngine(config, initial_equity=equity, profile=profile)
    for sym in all_klines:
        engine.init_symbol(sym)

    # Warm up
    warmup = 60
    log.info("Warming up indicators with %d candles...", warmup)
    for sym, klines in all_klines.items():
        tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
        for i in range(min(warmup, len(klines))):
            prev = klines[i - 1] if i > 0 else None
            book = build_synthetic_book(klines[i], prev, tick_size)
            engine.books[sym].on_snapshot(
                [[p, s] for p, s in book.bids],
                [[p, s] for p, s in book.asks],
            )
            trades = generate_synthetic_trades(
                klines[i], prev["close"] if prev else klines[i]["open"], sym)
            for t in trades:
                engine.tapes[sym].add_trade(t)

    # Build unified timeline: merge all candles sorted by timestamp
    timeline = []
    for sym, klines in all_klines.items():
        for i in range(warmup, len(klines)):
            timeline.append((klines[i]["ts"], sym, klines[i], klines[i - 1]))
    timeline.sort(key=lambda x: x[0])

    log.info("Running backtest on %d candle events across %d symbols...",
             len(timeline), len(all_klines))

    last_day = 0
    for ts, sym, kline, prev in timeline:
        engine.process_candle(sym, kline, prev)

        day = ts // 86400000
        if day != last_day:
            engine.equity_curve.append((ts, engine.state.equity))
            last_day = day
            if engine.state.kill_switch:
                engine.state.reset_daily()

    # Close any open positions
    for sym, klines in all_klines.items():
        if engine.state.has_position(sym):
            engine.force_close_all(klines[-1])

    report = engine.report()
    print_report(report, symbols, days, equity, profile)
    return report


def print_report(report: dict, symbols: list, days: int, equity: float, profile: str):
    from datetime import datetime, timezone
    sym_str = "+".join(symbols)

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS — [{profile.upper()}]")
    print("=" * 60)
    print(f"  Symbols:             {sym_str}")
    print(f"  Period:              {days} days")
    print(f"  Profile:             {profile}")
    print(f"  Initial Equity:      ${report.get('initial_equity', 0):.2f}")
    print(f"  Final Equity:        ${report.get('final_equity', 0):.2f}")
    print(f"  Total P&L:           ${report.get('total_pnl', 0):.2f}")
    print(f"  Total Return:        {report.get('total_return_pct', 0):.2f}%")
    print("-" * 60)
    print(f"  Total Trades:        {report.get('total_trades', 0)}")
    print(f"  Wins / Losses:       {report.get('wins', 0)} / {report.get('losses', 0)}")
    print(f"  Win Rate:            {report.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:       {report.get('profit_factor', 0):.2f}")
    print(f"  Avg Win:             ${report.get('avg_win', 0):.4f}")
    print(f"  Avg Loss:            ${report.get('avg_loss', 0):.4f}")
    print(f"  Avg P&L/Trade:       ${report.get('avg_pnl_per_trade', 0):.4f}")
    print(f"  Avg R-Multiple:      {report.get('avg_R', 0):.3f}R")
    print(f"  Sharpe Ratio:        {report.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:        {report.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Trades/Day:          {report.get('trades_per_day', 0):.1f}")

    # Exit reason breakdown
    reasons = report.get("exit_reasons", {})
    if reasons:
        print("-" * 60)
        print("  EXIT REASONS:")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"    {reason:15s}  {data['count']:5d} trades  P&L=${data['pnl']:+.2f}")

    print("=" * 60)

    # Profile comparison hint
    p = PROFILES.get(profile, {})
    rr = p.get("tp_range_mult", 0) / p.get("sl_range_mult", 1) if p.get("sl_range_mult", 0) > 0 else 0
    print(f"\n  PROFILE PARAMETERS:")
    print(f"    Risk/Trade:  {p.get('risk_pct', 0)*100:.1f}%")
    print(f"    R:R Ratio:   {rr:.1f}:1")
    print(f"    Equity Cap:  {p.get('equity_cap_mult', 1)}x initial")
    print(f"    Cooldown:    {p.get('cooldown_ms', 0)//1000}s")
    print(f"    Max Hold:    {p.get('max_hold', 0)} candles")
    print(f"    Trail:       {p.get('trail_pct', 0)*100:.0f}% of range")
    print(f"    Breakeven:   {p.get('breakeven_R', 999):.1f}R")
    print(f"    Partial TP:  {'Yes (50% at 1R)' if p.get('partial_tp') else 'No'}")
    print(f"    Conf Scale:  {'Yes' if p.get('conf_scale') else 'No'}")


async def run_sweep(symbols: List[str], days: int = 30, equity: float = 50.0):
    """Run all profiles and compare results."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    # Download data once
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    all_klines: Dict[str, List[dict]] = {}
    for sym in symbols:
        log.info("Downloading %d days of 1-minute klines for %s...", days, sym)
        klines = await fetch_klines(sym, "1", start_ms, end_ms)
        log.info("Downloaded %d candles for %s", len(klines), sym)
        if len(klines) >= 100:
            all_klines[sym] = klines

    if not all_klines:
        log.error("No data")
        return

    results = {}
    for profile_name in ["conservative", "balanced", "aggressive", "ultra"]:
        config = copy.deepcopy(base_config)
        engine = BacktestEngine(config, initial_equity=equity, profile=profile_name)
        for sym in all_klines:
            engine.init_symbol(sym)

        # Warm up
        warmup = 60
        for sym, klines in all_klines.items():
            tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
            for i in range(min(warmup, len(klines))):
                prev = klines[i - 1] if i > 0 else None
                book = build_synthetic_book(klines[i], prev, tick_size)
                engine.books[sym].on_snapshot(
                    [[p, s] for p, s in book.bids],
                    [[p, s] for p, s in book.asks],
                )
                trades = generate_synthetic_trades(
                    klines[i], prev["close"] if prev else klines[i]["open"], sym)
                for t in trades:
                    engine.tapes[sym].add_trade(t)

        # Build timeline
        timeline = []
        for sym, klines in all_klines.items():
            for i in range(warmup, len(klines)):
                timeline.append((klines[i]["ts"], sym, klines[i], klines[i - 1]))
        timeline.sort(key=lambda x: x[0])

        last_day = 0
        for ts, sym, kline, prev in timeline:
            engine.process_candle(sym, kline, prev)
            day = ts // 86400000
            if day != last_day:
                engine.equity_curve.append((ts, engine.state.equity))
                last_day = day
                if engine.state.kill_switch:
                    engine.state.reset_daily()

        for sym, klines in all_klines.items():
            if engine.state.has_position(sym):
                engine.force_close_all(klines[-1])

        report = engine.report()
        results[profile_name] = report
        log.info("Profile %s: return=%.1f%% trades=%d dd=%.1f%%",
                 profile_name,
                 report.get("total_return_pct", 0),
                 report.get("total_trades", 0),
                 report.get("max_drawdown_pct", 0))

    # Print comparison
    print("\n" + "=" * 80)
    print("  PROFILE COMPARISON SWEEP")
    print("=" * 80)
    sym_str = "+".join(symbols)
    print(f"  Symbols: {sym_str} | Period: {days} days | Start: ${equity:.2f}")
    print("-" * 80)
    print(f"  {'Profile':<14} {'Return':>8} {'Final $':>9} {'Trades':>7} {'WR':>6} {'PF':>6} "
          f"{'Sharpe':>7} {'MaxDD':>7} {'Avg R':>7}")
    print("-" * 80)

    for name in ["conservative", "balanced", "aggressive", "ultra"]:
        r = results[name]
        print(f"  {name:<14} {r.get('total_return_pct',0):>+7.1f}% "
              f"${r.get('final_equity',0):>7.2f} "
              f"{r.get('total_trades',0):>7d} "
              f"{r.get('win_rate',0):>5.1f}% "
              f"{r.get('profit_factor',0):>5.2f} "
              f"{r.get('sharpe_ratio',0):>7.3f} "
              f"{r.get('max_drawdown_pct',0):>6.1f}% "
              f"{r.get('avg_R',0):>6.3f}R")

    print("=" * 80)

    # Recommend best profile
    best = max(results.items(), key=lambda x: x[1].get("total_return_pct", -999))
    safest = min(results.items(), key=lambda x: x[1].get("max_drawdown_pct", 999))
    best_risk_adj = max(results.items(),
                        key=lambda x: x[1].get("total_return_pct", 0) / max(x[1].get("max_drawdown_pct", 1), 0.1))

    print(f"\n  RECOMMENDATIONS:")
    print(f"    Highest Return:      {best[0]} ({best[1].get('total_return_pct',0):+.1f}%)")
    print(f"    Lowest Drawdown:     {safest[0]} ({safest[1].get('max_drawdown_pct',0):.1f}%)")
    print(f"    Best Risk-Adjusted:  {best_risk_adj[0]} "
          f"(return/DD = {best_risk_adj[1].get('total_return_pct',0)/max(best_risk_adj[1].get('max_drawdown_pct',1),0.1):.2f})")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Bybit Scalper Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--multi", action="store_true", help="Trade both BTCUSDT + ETHUSDT")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--equity", type=float, default=50.0, help="Initial equity in USD")
    parser.add_argument("--mode", default="conservative",
                        choices=["conservative", "balanced", "aggressive", "ultra"],
                        help="Risk profile")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all profiles and compare")
    args = parser.parse_args()

    symbols = ["BTCUSDT", "ETHUSDT"] if args.multi else [args.symbol]

    if args.sweep:
        asyncio.run(run_sweep(symbols, args.days, args.equity))
    else:
        asyncio.run(run_backtest(symbols, args.days, args.equity, args.mode))
