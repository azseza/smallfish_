"""Parameter Sweep for Smallfish.

Precomputes signals once (position-independent), then replays a lightweight
trade engine across a grid of profile parameters.  Results are ranked by
multiple metrics and saved to CSV.

Usage:
    python src/param_sweep.py --symbol BTCUSDT --days 7 --equity 50
    python src/param_sweep.py --symbols BTCUSDT ETHUSDT --days 7 --equity 50 --workers 4
    python src/param_sweep.py --symbol BTCUSDT --days 30 --equity 50 --top 30
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import logging
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from core.utils import (
    sigmoid, clamp, set_sim_time, clear_sim_time, tick_round, qty_round,
)
from core.ringbuffer import RingBuffer
from core.types import Side
from marketdata.book import OrderBook
from marketdata.tape import TradeTape
import marketdata.features as features
import signals.fuse as fuse
from backtest import fetch_klines, build_synthetic_book, generate_synthetic_trades
from gateway.factory import create_rest

log = logging.getLogger("param_sweep")

# ─── Parameter Grid ──────────────────────────────────────────────────────

GRID = {
    "sl_range_mult":     [0.30, 0.40, 0.50, 0.65, 0.80],
    "tp_range_mult":     [0.80, 1.00, 1.30, 1.60, 2.00],
    "trail_pct":         [0.20, 0.30, 0.45, 999.0],
    "breakeven_R":       [0.7, 1.0, 1.5, 999.0],
    "partial_tp":        [True, False],
    "min_signals":       [3, 4, 5],
    "C_enter":           [0.50, 0.55, 0.60],
    "rolling_wr_filter": [True, False],
}

# Fixed params (from aggressive profile baseline)
FIXED = {
    "risk_pct":        0.025,
    "max_risk_usd":    40.0,
    "alpha":           4,
    "max_hold":        12,
    "cooldown_ms":     60_000,
    "max_positions":   3,
    "equity_cap_mult": 4,
    "conf_scale":      True,
    "max_daily_R":     20,
}


def grid_combos() -> List[dict]:
    """Generate all parameter combinations."""
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))
        combo.update(FIXED)
        combos.append(combo)
    return combos


# ─── Precomputed Candle Data ─────────────────────────────────────────────

@dataclass(slots=True)
class CandleData:
    ts: int
    sym: str
    close: float
    high: float
    low: float
    open: float
    meta_long: float
    meta_short: float
    agreement_long: int
    agreement_short: int
    vol_regime: str
    candle_range: float


def precompute_signals(
    all_klines: Dict[str, List[dict]],
    config: dict,
    warmup: int = 60,
) -> List[CandleData]:
    """Run the full signal pipeline once for every candle.

    Returns a list of CandleData sorted by timestamp, ready for replay.
    """
    books: Dict[str, OrderBook] = {}
    tapes: Dict[str, TradeTape] = {}

    # Init per-symbol structures
    for sym in all_klines:
        tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
        books[sym] = OrderBook(symbol=sym, depth=10, tick_size=tick_size)
        tapes[sym] = TradeTape(capacity=5000)

    # Warm up
    for sym, klines in all_klines.items():
        tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
        for i in range(min(warmup, len(klines))):
            set_sim_time(klines[i]["ts"] + 59_999)
            prev = klines[i - 1] if i > 0 else None
            book = build_synthetic_book(klines[i], prev, tick_size)
            books[sym].on_snapshot(
                [[p, s] for p, s in book.bids],
                [[p, s] for p, s in book.asks],
            )
            trades = generate_synthetic_trades(
                klines[i], prev["close"] if prev else klines[i]["open"], sym,
            )
            for t in trades:
                tapes[sym].add_trade(t)

    # Build timeline
    timeline: List[Tuple[int, str, dict, Optional[dict]]] = []
    for sym, klines in all_klines.items():
        for i in range(warmup, len(klines)):
            timeline.append((klines[i]["ts"], sym, klines[i], klines[i - 1]))
    timeline.sort(key=lambda x: x[0])

    # Get weights once (fixed for sweep — we only sweep trade-management params)
    weights = fuse.get_weights(config, {}, config.get("adaptive", {}))

    precomputed: List[CandleData] = []

    for ts, sym, kline, prev in timeline:
        set_sim_time(kline["ts"] + 59_999)

        tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
        book = books[sym]
        tape = tapes[sym]

        # Apply delta / snapshot
        if prev and book.bids:
            old_bids = list(book.bids[:3])
            old_asks = list(book.asks[:3])
            new_book = build_synthetic_book(kline, prev, tick_size)
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
            new_book = build_synthetic_book(kline, prev, tick_size)
            book.on_snapshot(
                [[p, s] for p, s in new_book.bids],
                [[p, s] for p, s in new_book.asks],
                seq=book.last_update_seq + 1,
            )
        book.last_update_ts = kline["ts"]

        # Generate trades
        prev_close = prev["close"] if prev else kline["open"]
        syn_trades = generate_synthetic_trades(kline, prev_close, sym)
        for t in syn_trades:
            tape.add_trade(t)

        # Compute features + scores
        f = features.compute_all(book, tape, config)
        vol_name = f.get("vol_regime_name", "normal")

        scores = fuse.score_all(f, config)
        ml, ms = fuse.meta(scores, weights)
        al = fuse.count_agreement(scores, 1)
        as_ = fuse.count_agreement(scores, -1)

        candle_range = kline["high"] - kline["low"]

        precomputed.append(CandleData(
            ts=kline["ts"],
            sym=sym,
            close=kline["close"],
            high=kline["high"],
            low=kline["low"],
            open=kline["open"],
            meta_long=ml,
            meta_short=ms,
            agreement_long=al,
            agreement_short=as_,
            vol_regime=vol_name,
            candle_range=candle_range,
        ))

    clear_sim_time()
    return precomputed


# ─── Fast Replay Engine ─────────────────────────────────────────────────

def _replay_one(args: Tuple[dict, float, list, dict]) -> dict:
    """Entry point for multiprocessing — unpacks and runs FastReplay."""
    params, equity, candles_raw, sym_tick_sizes = args
    candles = [CandleData(*c) for c in candles_raw]
    return FastReplay(params, equity, candles, sym_tick_sizes).run()


class FastReplay:
    """Lightweight position simulator for parameter sweep.

    No logging, no state objects, no imports during the hot loop.
    Mirrors BacktestEngine logic but stripped to essentials.
    """

    __slots__ = (
        "p", "equity", "candles", "sym_tick_sizes",
        "positions", "last_entry_ts", "range_bufs",
        "trades", "peak_equity", "max_dd",
        "partial_taken", "hold_counter", "recent_results",
        "daily_loss_R", "last_day", "kill_switch",
    )

    def __init__(self, params: dict, equity: float,
                 candles: List[CandleData], sym_tick_sizes: dict):
        self.p = params
        self.equity = equity
        self.candles = candles
        self.sym_tick_sizes = sym_tick_sizes

        self.positions: Dict[str, dict] = {}
        self.last_entry_ts: Dict[str, int] = {}
        self.range_bufs: Dict[str, deque] = {}
        self.hold_counter: Dict[str, int] = {}
        self.partial_taken: Dict[str, bool] = {}
        self.recent_results: deque = deque(maxlen=20)

        self.trades: List[dict] = []
        self.peak_equity = equity
        self.max_dd = 0.0

        # Daily loss tracking (matches BacktestEngine kill switch)
        self.daily_loss_R = 0.0
        self.last_day = 0
        self.kill_switch = False

    def run(self) -> dict:
        for c in self.candles:
            self._process(c)
        self._close_all()
        return self._report()

    def _rolling_wr(self) -> float:
        if len(self.recent_results) < 20:
            return 1.0
        return sum(1 for r in self.recent_results if r > 0) / len(self.recent_results)

    def _avg_range(self, sym: str, fallback: float) -> float:
        buf = self.range_bufs.get(sym)
        if buf and len(buf) > 0:
            return max(sum(buf) / len(buf), fallback)
        return max(fallback, 0.0001)

    def _process(self, c: CandleData) -> None:
        sym = c.sym

        # Daily reset (matches BacktestEngine daily boundary logic)
        day = c.ts // 86400000
        if day != self.last_day:
            self.last_day = day
            if self.kill_switch:
                self.kill_switch = False
                self.daily_loss_R = 0.0

        # Update rolling range buffer
        if c.candle_range > 0:
            buf = self.range_bufs.get(sym)
            if buf is None:
                buf = deque(maxlen=20)
                self.range_bufs[sym] = buf
            buf.append(c.candle_range)

        # Manage existing position
        if sym in self.positions:
            self._manage(sym, c)
            return

        # Kill switch active — skip entries
        if self.kill_switch:
            return

        # Skip extreme vol
        if c.vol_regime == "extreme":
            return

        # Max concurrent positions
        if len(self.positions) >= self.p["max_positions"]:
            return

        # Cooldown
        if c.ts - self.last_entry_ts.get(sym, 0) < self.p["cooldown_ms"]:
            return

        # Direction from precomputed meta
        raw = c.meta_long - c.meta_short
        if abs(raw) < 1e-6:
            return
        direction = 1 if raw > 0 else -1

        # Agreement gate
        agreement = c.agreement_long if direction == 1 else c.agreement_short
        if agreement < self.p["min_signals"]:
            return

        # Confidence gate
        conf = sigmoid(self.p["alpha"] * abs(raw))
        if conf < self.p["C_enter"]:
            return

        # Mandatory rolling WR check (matches BacktestEngine._try_enter)
        wr = self._rolling_wr()
        if wr < 0.20:
            return

        # Optional stricter WR filter
        if self.p.get("rolling_wr_filter", False) and wr < 0.30:
            return  # halve sizing handled below, but also gate entry

        # Entry
        tick_size = self.sym_tick_sizes.get(sym, 0.01)
        avg_range = self._avg_range(sym, tick_size * 5)

        stop_dist = avg_range * self.p["sl_range_mult"]
        tp_dist = avg_range * self.p["tp_range_mult"]
        entry_px = c.close + direction * tick_size * 0.5  # slippage

        if direction == 1:
            stop_px = entry_px - stop_dist
            tp_px = entry_px + tp_dist
        else:
            stop_px = entry_px + stop_dist
            tp_px = entry_px - tp_dist

        # Position sizing
        fixed_eq = self.p.get("_initial_equity", self.equity)
        if self.equity <= 0:
            return  # blown up
        use_eq = fixed_eq * min(
            math.sqrt(max(self.equity / max(fixed_eq, 0.01), 0.0)),
            self.p["equity_cap_mult"],
        )
        risk_dollars = min(use_eq * self.p["risk_pct"], self.p["max_risk_usd"])

        # Confidence scaling
        if self.p["conf_scale"] and conf > 0.6:
            conf_mult = 1.0 + (conf - 0.6) * 0.5
            risk_dollars *= min(conf_mult, 1.2)

        # Rolling edge multiplier (always applied, matches BacktestEngine)
        wr = self._rolling_wr()
        if wr < 0.30:
            risk_dollars *= 0.5

        if stop_dist <= 0:
            return
        qty = risk_dollars / stop_dist
        qty_step = self.sym_tick_sizes.get(sym + "_step", 0.001)
        min_qty = self.sym_tick_sizes.get(sym + "_min", 0.001)
        qty = math.floor(qty / qty_step) * qty_step
        if qty < min_qty:
            return

        self.positions[sym] = {
            "entry": entry_px,
            "stop": stop_px,
            "tp": tp_px,
            "side": direction,
            "qty": qty,
            "peak": entry_px,
            "partial": False,
        }
        self.hold_counter[sym] = 0
        self.partial_taken[sym] = False
        self.last_entry_ts[sym] = c.ts

    def _manage(self, sym: str, c: CandleData) -> None:
        pos = self.positions[sym]
        side = pos["side"]
        high, low, close = c.high, c.low, c.close
        tick_size = self.sym_tick_sizes.get(sym, 0.01)

        self.hold_counter[sym] = self.hold_counter.get(sym, 0) + 1

        # Max hold
        if self.hold_counter[sym] >= self.p["max_hold"]:
            self._exit(sym, close, c.ts)
            return

        # Stop loss
        if side == 1:
            if low <= pos["stop"]:
                self._exit(sym, pos["stop"], c.ts)
                return
        else:
            if high >= pos["stop"]:
                self._exit(sym, pos["stop"], c.ts)
                return

        # Partial TP at 1R
        if self.p["partial_tp"] and not self.partial_taken.get(sym, False):
            r_dist = abs(pos["entry"] - pos["stop"])
            if r_dist > 0:
                if side == 1:
                    partial_target = pos["entry"] + r_dist
                    if high >= partial_target:
                        self._take_partial(sym, partial_target, c.ts)
                else:
                    partial_target = pos["entry"] - r_dist
                    if low <= partial_target:
                        self._take_partial(sym, partial_target, c.ts)

        # Take profit
        if side == 1:
            if high >= pos["tp"]:
                self._exit(sym, pos["tp"], c.ts)
                return
        else:
            if low <= pos["tp"]:
                self._exit(sym, pos["tp"], c.ts)
                return

        # Update peak
        if side == 1:
            if high > pos["peak"]:
                pos["peak"] = high
        else:
            if low < pos["peak"] or pos["peak"] == pos["entry"]:
                pos["peak"] = low

        # Breakeven stop
        be_R = self.p["breakeven_R"]
        if be_R < 900:
            r_dist = abs(pos["entry"] - pos["stop"])
            if r_dist > 0:
                if side == 1:
                    current_R = (pos["peak"] - pos["entry"]) / r_dist
                    if current_R >= be_R and pos["stop"] < pos["entry"]:
                        pos["stop"] = pos["entry"] + tick_size
                else:
                    current_R = (pos["entry"] - pos["peak"]) / r_dist
                    if current_R >= be_R and pos["stop"] > pos["entry"]:
                        pos["stop"] = pos["entry"] - tick_size

        # Trailing stop
        trail = self.p["trail_pct"]
        if trail < 900:
            avg_range = self._avg_range(sym, tick_size * 5)
            trail_dist = avg_range * trail
            if side == 1:
                new_stop = pos["peak"] - trail_dist
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop
            else:
                new_stop = pos["peak"] + trail_dist
                if new_stop < pos["stop"]:
                    pos["stop"] = new_stop

    def _take_partial(self, sym: str, price: float, ts: int) -> None:
        pos = self.positions[sym]
        half_qty = pos["qty"] / 2
        qty_step = self.sym_tick_sizes.get(sym + "_step", 0.001)
        min_qty = self.sym_tick_sizes.get(sym + "_min", 0.001)
        half_qty = math.floor(half_qty / qty_step) * qty_step
        if half_qty < min_qty:
            return

        pnl = (price - pos["entry"]) * half_qty * pos["side"]
        self.equity += pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_dd = max(self.max_dd, dd)

        pos["qty"] -= half_qty
        self.partial_taken[sym] = True
        self.trades.append({"pnl": pnl, "ts": ts})
        self.recent_results.append(pnl)

    def _exit(self, sym: str, exit_px: float, ts: int) -> None:
        pos = self.positions.pop(sym)
        pnl = (exit_px - pos["entry"]) * pos["qty"] * pos["side"]
        self.equity += pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_dd = max(self.max_dd, dd)

        self.trades.append({"pnl": pnl, "ts": ts})
        self.recent_results.append(pnl)
        self.hold_counter.pop(sym, None)
        self.partial_taken.pop(sym, None)

        # Track daily loss R and trigger kill switch (matches BacktestEngine)
        if pnl < 0:
            initial = self.p.get("_initial_equity", 50.0)
            risk_per_trade = self.p["risk_pct"] * max(self.equity, initial * 0.1)
            loss_R = abs(pnl) / max(risk_per_trade, 0.01)
            self.daily_loss_R += loss_R
            max_daily_R = self.p.get("max_daily_R", 20)
            if self.daily_loss_R >= max_daily_R:
                self.kill_switch = True

    def _close_all(self) -> None:
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            # Use last candle close as proxy
            last_close = self.candles[-1].close if self.candles else pos["entry"]
            self._exit(sym, last_close, self.candles[-1].ts if self.candles else 0)

    def _report(self) -> dict:
        pnls = [t["pnl"] for t in self.trades]
        n = len(pnls)
        if n == 0:
            return {
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "max_dd_pct": 0.0,
                "trade_count": 0,
                "params": self.p,
            }

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        initial = self.p.get("_initial_equity", 50.0)
        total_return = total_pnl / initial * 100

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else (
            999.0 if gross_profit > 0 else 0.0)

        wr = len(wins) / n * 100

        # Sharpe on per-trade PnL
        if n > 1:
            arr = np.array(pnls)
            sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9))
        else:
            sharpe = 0.0

        return {
            "total_return_pct": round(total_return, 3),
            "sharpe": round(sharpe, 4),
            "profit_factor": round(pf, 3),
            "win_rate": round(wr, 2),
            "max_dd_pct": round(self.max_dd * 100, 3),
            "trade_count": n,
            "final_equity": round(self.equity, 4),
            "params": self.p,
        }


# ─── Sweep Runner ────────────────────────────────────────────────────────

def run_sweep_sync(
    candles: List[CandleData],
    equity: float,
    sym_tick_sizes: dict,
    workers: int = 4,
) -> List[dict]:
    """Run all parameter combos, optionally in parallel."""
    combos = grid_combos()
    n_combos = len(combos)

    # Tag each combo with initial equity for sizing
    for c in combos:
        c["_initial_equity"] = equity

    # Serialize CandleData to tuples for pickling
    candles_raw = [
        (c.ts, c.sym, c.close, c.high, c.low, c.open,
         c.meta_long, c.meta_short, c.agreement_long, c.agreement_short,
         c.vol_regime, c.candle_range)
        for c in candles
    ]

    args_list = [(combo, equity, candles_raw, sym_tick_sizes) for combo in combos]

    results: List[dict] = []

    if workers <= 1:
        for i, args in enumerate(args_list):
            results.append(_replay_one(args))
            if (i + 1) % 1000 == 0:
                print(f"    ... {i + 1}/{n_combos} done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(_replay_one, args_list, chunksize=64)):
                results.append(result)
                if (i + 1) % 2000 == 0:
                    print(f"    ... {i + 1}/{n_combos} done", flush=True)

    return results


# ─── Output ──────────────────────────────────────────────────────────────

def _param_str(p: dict) -> str:
    trail = "off" if p.get("trail_pct", 999) >= 900 else f"{p['trail_pct']:.2f}"
    be = "off" if p.get("breakeven_R", 999) >= 900 else f"{p['breakeven_R']:.1f}"
    ptl = "Yes" if p.get("partial_tp") else "No"
    wrflt = "Yes" if p.get("rolling_wr_filter") else "No"
    return (f"SL={p['sl_range_mult']:.2f}  TP={p['tp_range_mult']:.2f}  "
            f"Trail={trail}  BE={be}  Ptl={ptl}  "
            f"MinS={p['min_signals']}  C={p['C_enter']:.2f}  WRF={wrflt}")


def _format_row(i: int, r: dict) -> str:
    p = r["params"]
    trail = "off" if p.get("trail_pct", 999) >= 900 else f"{p['trail_pct']:.2f}"
    be = "off" if p.get("breakeven_R", 999) >= 900 else f"{p['breakeven_R']:.1f}"
    ptl = "Y" if p.get("partial_tp") else "N"
    wrflt = "Y" if p.get("rolling_wr_filter") else "N"
    return (
        f"  {i:>3} | {r['total_return_pct']:>+7.1f}% | {r['sharpe']:>6.3f} | "
        f"{r['profit_factor']:>5.2f} | {r['win_rate']:>5.1f}% | {r['max_dd_pct']:>5.1f}% | "
        f"{r['trade_count']:>5d} | "
        f"{p['sl_range_mult']:.2f} | {p['tp_range_mult']:.2f} | "
        f"{trail:>4s} | {be:>4s} | {ptl:>1s} | {p['min_signals']:>1d} | "
        f"{p['C_enter']:.2f} | {wrflt:>1s}"
    )


HEADER = (
    "    # |  Return | Sharpe |    PF |    WR |  MaxDD | Trades | "
    "  SL |   TP | Trl  |  BE  | P | S |   C  | W"
)
SEP = "  " + "-" * 108


def print_results(results: List[dict], symbols: List[str], days: int,
                  equity: float, top_n: int = 20) -> None:
    n_combos = len(results)
    sym_str = "+".join(symbols)

    print()
    print("=" * 112)
    print(f"  PARAMETER SWEEP — {sym_str} | {days} days | ${equity:.2f} | "
          f"{n_combos:,} combos")
    print("=" * 112)

    # Filter out zero-trade results
    active = [r for r in results if r["trade_count"] > 0]
    if not active:
        print("\n  No trades were generated for any parameter combination.")
        print("=" * 112)
        return

    # --- Top by return ---
    by_return = sorted(active, key=lambda x: x["total_return_pct"], reverse=True)
    print(f"\n  TOP {min(top_n, len(by_return))} BY TOTAL RETURN:")
    print(SEP)
    print(HEADER)
    print(SEP)
    for i, r in enumerate(by_return[:top_n], 1):
        print(_format_row(i, r))

    # --- Top by Sharpe ---
    by_sharpe = sorted(active, key=lambda x: x["sharpe"], reverse=True)
    print(f"\n  TOP {min(10, len(by_sharpe))} BY SHARPE RATIO:")
    print(SEP)
    print(HEADER)
    print(SEP)
    for i, r in enumerate(by_sharpe[:10], 1):
        print(_format_row(i, r))

    # --- Top by return / max DD ---
    by_risk_adj = sorted(
        [r for r in active if r["max_dd_pct"] > 0.1 and r["total_return_pct"] > 0],
        key=lambda x: x["total_return_pct"] / max(x["max_dd_pct"], 0.1),
        reverse=True,
    )
    if by_risk_adj:
        print(f"\n  TOP {min(10, len(by_risk_adj))} BY RETURN / MAX_DRAWDOWN:")
        print(SEP)
        print(HEADER)
        print(SEP)
        for i, r in enumerate(by_risk_adj[:10], 1):
            ratio = r["total_return_pct"] / max(r["max_dd_pct"], 0.1)
            print(_format_row(i, r) + f"  R/DD={ratio:.2f}")

    # --- Consensus pick ---
    top_return_set = set(id(r) for r in by_return[:10])
    top_sharpe_set = set(id(r) for r in by_sharpe[:10])
    top_risk_set = set(id(r) for r in by_risk_adj[:10]) if by_risk_adj else set()

    consensus = [
        r for r in active
        if sum([
            id(r) in top_return_set,
            id(r) in top_sharpe_set,
            id(r) in top_risk_set,
        ]) >= 2
    ]
    if consensus:
        best = max(consensus, key=lambda x: x["total_return_pct"])
        print(f"\n  CONSENSUS PICK (appears in 2+ top-10 lists):")
        print(f"    {_param_str(best['params'])}")
        print(f"    Return: {best['total_return_pct']:+.1f}%  "
              f"Sharpe: {best['sharpe']:.3f}  "
              f"PF: {best['profit_factor']:.2f}  "
              f"MaxDD: {best['max_dd_pct']:.1f}%  "
              f"Trades: {best['trade_count']}")

    print()
    print("=" * 112)


def save_csv(results: List[dict], path: str) -> None:
    """Write all results to CSV."""
    if not results:
        return

    fieldnames = [
        "total_return_pct", "sharpe", "profit_factor", "win_rate",
        "max_dd_pct", "trade_count", "final_equity",
        "sl_range_mult", "tp_range_mult", "trail_pct", "breakeven_R",
        "partial_tp", "min_signals", "C_enter", "rolling_wr_filter",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "total_return_pct": r["total_return_pct"],
                "sharpe": r["sharpe"],
                "profit_factor": r["profit_factor"],
                "win_rate": r["win_rate"],
                "max_dd_pct": r["max_dd_pct"],
                "trade_count": r["trade_count"],
                "final_equity": r.get("final_equity", ""),
                "sl_range_mult": r["params"]["sl_range_mult"],
                "tp_range_mult": r["params"]["tp_range_mult"],
                "trail_pct": r["params"]["trail_pct"],
                "breakeven_R": r["params"]["breakeven_R"],
                "partial_tp": r["params"]["partial_tp"],
                "min_signals": r["params"]["min_signals"],
                "C_enter": r["params"]["C_enter"],
                "rolling_wr_filter": r["params"].get("rolling_wr_filter", False),
            }
            writer.writerow(row)
    print(f"  Results saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────

async def download_data(
    symbols: List[str], days: int, config: dict, exchange: str,
) -> Dict[str, List[dict]]:
    """Download kline data and populate config with tick sizes."""
    # Fetch instrument specs
    rest = create_rest(exchange, api_key="", api_secret="")
    try:
        specs_list = await rest.get_instruments()
    finally:
        await rest.close()
    specs = {s.symbol: s for s in specs_list}
    for sym in symbols:
        if sym in specs:
            config.setdefault("tick_sizes", {})[sym] = specs[sym].tick_size
            config.setdefault("min_qty", {})[sym] = specs[sym].min_qty
            config.setdefault("qty_step", {})[sym] = specs[sym].qty_step

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    all_klines: Dict[str, List[dict]] = {}
    for sym in symbols:
        print(f"  Downloading {days}d klines for {sym}...", flush=True)
        klines = await fetch_klines(sym, "1", start_ms, end_ms, exchange=exchange)
        print(f"    {len(klines)} candles ({len(klines)/1440:.1f} days)")
        if len(klines) >= 100:
            all_klines[sym] = klines
        else:
            print(f"    SKIP: not enough data for {sym}")
    return all_klines


async def main():
    parser = argparse.ArgumentParser(description="Smallfish Parameter Sweep")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--equity", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=20, help="Show top N results")
    parser.add_argument("--exchange", default="bybit", choices=["bybit", "binance", "mexc", "dydx"])
    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    exchange = config.get("exchange", args.exchange)

    # 1. Download data
    all_klines = await download_data(symbols, args.days, config, exchange)
    if not all_klines:
        print("  No data available. Exiting.")
        return

    # Build tick size lookup for replay
    sym_tick_sizes = {}
    for sym in all_klines:
        sym_tick_sizes[sym] = config.get("tick_sizes", {}).get(sym, 0.01)
        sym_tick_sizes[sym + "_step"] = config.get("qty_step", {}).get(sym, 0.001)
        sym_tick_sizes[sym + "_min"] = config.get("min_qty", {}).get(sym, 0.001)

    # 2. Precompute signals
    n_combos = 1
    for v in GRID.values():
        n_combos *= len(v)

    total_candles = sum(max(len(k) - 60, 0) for k in all_klines.values())
    print(f"\n  Precomputing signals... ", end="", flush=True)
    t0 = time.time()
    candles = precompute_signals(all_klines, config)
    dt_pre = time.time() - t0
    print(f"{len(candles)} candles ({total_candles/1440:.1f} days) in {dt_pre:.1f}s")

    # 3. Run sweep
    print(f"  Running {n_combos:,} parameter combinations on {args.workers} workers... ",
          end="", flush=True)
    t0 = time.time()
    results = run_sweep_sync(candles, args.equity, sym_tick_sizes, workers=args.workers)
    dt_sweep = time.time() - t0
    print(f"done in {dt_sweep:.1f}s")

    # 4. Print results
    print_results(results, symbols, args.days, args.equity, top_n=args.top)

    # 5. Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sweep_results.csv")
    save_csv(results, csv_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
