"""Smallfish — Main Application.

Open-source signal-fusion scalping engine for Bybit perpetual futures.
Async event-driven architecture processing orderbook and trade stream data.

  ><(((o>  smallfish  <o)))><
"""
from __future__ import annotations
import argparse
import asyncio
import datetime
import logging
import os
import signal
import sys
import time

import yaml
from dotenv import load_dotenv

from core.state import RuntimeState
from core.types import (
    EventType, Side, Trade, WsEvent, OrderStatus, Position,
    NormalizedExecution, NormalizedOrderUpdate, NormalizedPositionUpdate,
)
from core.utils import time_now_ms
from core.profiles import PROFILES, apply_profile

from marketdata.book import OrderBook
from marketdata.tape import TradeTape
import marketdata.features as features

import signals.fuse as fuse

from exec.router import OrderRouter
from exec.oco import OcoManager
import exec.risk as risk

from gateway.factory import create_rest, create_ws
from gateway.persistence import Persistence

from monitor.heartbeat import Heartbeat
from monitor import metrics as metrics_mod

log = logging.getLogger("smallfish")


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smallfish — microstructure tools for the small fish"
    )
    parser.add_argument("--mode",
                        choices=list(PROFILES),
                        default=None,
                        help="Risk profile (conservative/balanced/aggressive/ultra)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Enable terminal dashboard with live bar charts")
    parser.add_argument("--multigrid", action="store_true",
                        help="Enable multigrid strategy layer")
    parser.add_argument("--telegram", action="store_true",
                        help="Enable Telegram bot for remote control")
    parser.add_argument("--email-alerts", action="store_true",
                        help="Enable email alerts for critical events")
    parser.add_argument("--backfill", action="store_true",
                        help="Pre-fill 7D dashboard performance with backtested data")
    parser.add_argument("--equity", type=float, default=50.0,
                        help="Equity for backtest backfill (default: $50)")
    return parser.parse_args()


def _setup_signal_handlers() -> None:
    """Install SIGTERM handler for clean systemd/Docker shutdown on RPi."""
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt("SIGTERM received — shutting down cleanly")
    signal.signal(signal.SIGTERM, _sigterm_handler)


async def main() -> None:
    _setup_signal_handlers()
    load_dotenv()
    args = parse_args()
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))

    # Apply risk profile if specified
    if args.mode:
        apply_profile(config, args.mode)

    profile_label = args.mode.upper() if args.mode else "DEFAULT"

    exchange = config.get("exchange", "bybit").lower()

    if exchange == "binance":
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        testnet = os.environ.get("BINANCE_TESTNET", "false").lower() == "true"
        env_label = "BINANCE_API_KEY / BINANCE_API_SECRET"
    elif exchange == "mexc":
        api_key = os.environ.get("MEXC_API_KEY", "")
        api_secret = os.environ.get("MEXC_API_SECRET", "")
        testnet = False  # MEXC has no futures testnet
        env_label = "MEXC_API_KEY / MEXC_API_SECRET"
    elif exchange == "dydx":
        api_key = os.environ.get("DYDX_ADDRESS", "")
        api_secret = os.environ.get("DYDX_MNEMONIC", "")
        testnet = os.environ.get("DYDX_TESTNET", "false").lower() == "true"
        env_label = "DYDX_ADDRESS / DYDX_MNEMONIC"
    else:
        api_key = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")
        testnet = os.environ.get("BYBIT_TESTNET", "false").lower() == "true"
        env_label = "BYBIT_API_KEY / BYBIT_API_SECRET"

    log.info("=" * 60)
    log.info("  ><(((o>  SMALLFISH v2.0  <o)))><")
    log.info("  microstructure tools for the small fish")
    log.info("  exchange: %s%s  profile: %s",
             exchange.upper(), " (testnet)" if testnet else "", profile_label)
    log.info("=" * 60)

    if not api_key or not api_secret:
        log.error("Missing %s in environment", env_label)
        return

    symbols = config.get("symbols", ["BTCUSDT"])

    # --- CLI flag overrides ---
    if args.dashboard:
        config.setdefault("dashboard", {})["enabled"] = True
    if args.multigrid:
        config.setdefault("multigrid", {})["enabled"] = True
    if args.telegram:
        config.setdefault("telegram", {})["enabled"] = True
    if args.email_alerts:
        config.setdefault("email", {})["enabled"] = True

    # --- Initialize Components ---
    # If --equity is provided, use it as the effective trading capital
    # This caps position sizing even if wallet has more funds
    if args.equity:
        config["initial_equity"] = float(args.equity)
        config["equity_override"] = float(args.equity)
        log.info("Equity override from CLI: $%.2f", args.equity)

    state = RuntimeState(config)
    persistence = Persistence()
    rest = create_rest(exchange, api_key, api_secret, testnet=testnet)

    # --- Dynamic Symbol Selection ---
    auto_symbols = int(os.environ.get("AUTO_SYMBOLS", "0"))
    if auto_symbols > 0:
        from marketdata.scanner import scan_top_symbols, apply_specs_to_config
        log.info("Scanning for top %d symbols on %s...", auto_symbols, exchange.upper())
        top = await scan_top_symbols(rest, n=auto_symbols)
        if top:
            apply_specs_to_config(config, top)
            symbols = config["symbols"]
            for s in top:
                log.info("  %s  vol=$%.0fM  chg=%+.1f%%",
                         s["symbol"], s["volume_24h_usd"]/1e6, s["change_24h_pct"])
        else:
            log.warning("Scanner returned no symbols, using config defaults")

    # One order book and tape per symbol
    books: dict[str, OrderBook] = {}
    tapes: dict[str, TradeTape] = {}
    for sym in symbols:
        tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
        books[sym] = OrderBook(
            symbol=sym,
            depth=config.get("ws", {}).get("orderbook_depth", 50),
            tick_size=tick_size,
        )
        tapes[sym] = TradeTape(capacity=5000)
    router = OrderRouter(rest, config)
    oco = OcoManager(rest, config)
    heartbeat = Heartbeat(state, persistence, config)

    ws = create_ws(
        exchange=exchange,
        symbols=symbols,
        config=config,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    # --- Startup: fetch account info and set leverage ---
    try:
        wallet = await rest.get_wallet_balance()
        if wallet.equity > 0:
            # Respect --equity override: use min(wallet, override) for risk management
            equity_override = config.get("equity_override")
            if equity_override and equity_override > 0:
                effective_equity = min(wallet.equity, equity_override)
                log.info("Wallet: $%.2f USDT | Using: $%.2f (--equity cap)",
                         wallet.equity, effective_equity)
            else:
                effective_equity = wallet.equity
                log.info("Account equity: $%.2f USDT", wallet.equity)

            state.equity = effective_equity
            state.peak_equity = effective_equity
            state.daily_start_equity = effective_equity
            config["initial_equity"] = effective_equity  # for sqrt-compounding
        else:
            log.warning("Wallet returned zero equity — check API keys / account type")

        leverage = config.get("leverage", 10)
        for sym in symbols:
            result = await rest.set_leverage(sym, leverage)
            if result.success:
                log.info("Leverage set: %s × %d", sym, leverage)
            else:
                log.info("Leverage OK: %s × %d (already set or error: %s)",
                         sym, leverage, result.error_msg)

        # Latency calibration
        server_time = await rest.get_server_time()
        latency = abs(time_now_ms() - server_time)
        state.update_latency(latency)
        log.info("Initial latency: %dms", latency)

    except Exception as e:
        log.error("Startup error (equity may be wrong!): %s", e)

    # --- Position recovery: detect orphaned positions from prior crash ---
    try:
        existing_positions = await rest.get_positions()
        if existing_positions:
            recovered = 0
            for raw_pos in existing_positions:
                # Normalize symbol from exchange format
                raw_sym = raw_pos.get("symbol", "")
                if hasattr(rest, '_from_exchange_symbol'):
                    sym = rest._from_exchange_symbol(raw_sym)
                else:
                    from gateway.symbol_map import from_mexc, from_binance
                    if exchange == "mexc":
                        sym = from_mexc(raw_sym)
                    elif exchange == "binance":
                        sym = from_binance(raw_sym) if 'from_binance' in dir() else raw_sym
                    else:
                        sym = raw_sym

                if sym not in symbols:
                    continue

                hold_vol = float(raw_pos.get("holdVol", raw_pos.get("positionAmt", 0)))
                if hold_vol <= 0:
                    continue

                # Determine side
                pos_type = raw_pos.get("positionType", 0)
                if pos_type == 1:
                    side = Side.BUY
                elif pos_type == 2:
                    side = Side.SELL
                else:
                    # Bybit/Binance format
                    side_str = raw_pos.get("side", "")
                    side = Side.BUY if side_str in ("Buy", "LONG") else Side.SELL

                entry_price = float(raw_pos.get("openAvgPrice",
                                   raw_pos.get("entryPrice", 0)))

                if entry_price <= 0 or state.has_position(sym):
                    continue

                # Recover into state (no TP/SL — will be managed on next tick)
                tick_size = config.get("tick_sizes", {}).get(sym, 0.01)
                avg_range = tick_size * 5  # rough estimate
                sl_mult = config.get("profile", {}).get("sl_range_mult", 0.80)
                tp_mult = config.get("profile", {}).get("tp_range_mult", 2.00)
                stop_dist = avg_range * sl_mult
                if side == Side.BUY:
                    stop_price = entry_price - stop_dist
                    tp_price = entry_price + stop_dist * tp_mult / sl_mult
                else:
                    stop_price = entry_price + stop_dist
                    tp_price = entry_price - stop_dist * tp_mult / sl_mult

                state.on_enter(
                    symbol=sym, side=side, fill_price=entry_price,
                    quantity=hold_vol, confidence=0.5, scores={},
                    stop_price=stop_price, tp_price=tp_price,
                )
                recovered += 1
                log.warning("RECOVERED position: %s %s qty=%.4f entry=%.4f",
                            side.name, sym, hold_vol, entry_price)

            if recovered > 0:
                log.warning("Recovered %d orphaned position(s) from exchange", recovered)
                # Re-attach TP/SL brackets for recovered positions
                for sym in list(state.positions.keys()):
                    pos = state.position(sym)
                    if pos:
                        oco_ok = await oco.attach(pos)
                        if oco_ok:
                            log.info("Re-attached TP/SL for recovered %s", sym)
                        else:
                            log.warning("Failed to re-attach TP/SL for %s — will manage on next tick", sym)
    except Exception as e:
        log.warning("Position recovery check failed (non-fatal): %s", e)

    # Equity sync interval (re-fetch wallet balance periodically)
    _equity_sync_interval_s = config.get("equity_sync_interval_s", 30)
    _last_equity_sync = time_now_ms()

    # --- Optional: Backfill 7D performance ---
    bt_report = None
    if args.backfill:
        from backtest import backfill_trades, print_report
        bt_profile = args.mode or "aggressive"
        # Use --equity arg for backfill (simulates starting with that amount)
        # IMPORTANT: args.equity is the backtest starting capital, NOT wallet balance
        bt_equity = float(args.equity)  # explicit float conversion
        log.info("Backfilling 7-day performance (profile=%s, equity=$%.2f)...",
                 bt_profile, bt_equity)
        bt_trades, bt_report = await backfill_trades(
            symbols, config, days=7,
            equity=bt_equity, profile=bt_profile,
        )
        # Verify the report shows correct initial equity
        if bt_report and bt_report.get("initial_equity") != bt_equity:
            log.warning("Backfill equity mismatch: expected $%.2f, got $%.2f",
                        bt_equity, bt_report.get("initial_equity", 0))
        for t in bt_trades:
            state.completed_trades.append(t)
        log.info("Backfilled %d trades into dashboard 7D history", len(bt_trades))

        # Print explicit backtest results before dashboard takes over stdout
        if bt_report and "error" not in bt_report:
            print_report(bt_report, symbols, 7, bt_equity, bt_profile)
            input("\n  Press Enter to launch dashboard...")

    # --- Optional: Multigrid Strategy ---
    grid = None
    if config.get("multigrid", {}).get("enabled", False):
        from strategies.multigrid import MultigridStrategy
        grid = MultigridStrategy(config)
        for sym in symbols:
            grid.init_symbol(sym)
        log.info("Multigrid strategy enabled (%d levels, %.2f%% spacing)",
                 grid.num_levels, grid.spacing_pct * 100)

    # --- Optional: Terminal Dashboard ---
    dashboard = None
    if config.get("dashboard", {}).get("enabled", False):
        from monitor.dashboard import TerminalDashboard
        dashboard = TerminalDashboard(state, config, grid_strategy=grid)
        dashboard.enabled = True

        # Pre-fill candle chart with recent klines (2h of 1m candles)
        try:
            from backtest import fetch_klines
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - 2 * 3600 * 1000  # last 2 hours
            primary_sym = symbols[0] if symbols else "BTCUSDT"
            recent_klines = await fetch_klines(primary_sym, "1", start_ms, end_ms,
                                                exchange=exchange)
            if recent_klines:
                dashboard.load_historical_candles(recent_klines)
                log.info("Dashboard: loaded %d historical candles for %s",
                         len(recent_klines), primary_sym)
        except Exception as e:
            log.warning("Dashboard: could not load historical candles: %s", e)

        # Pass 7D backtest ROI to dashboard
        if bt_report and "error" not in bt_report:
            dashboard.set_backfill_roi(
                roi_pct=bt_report.get("total_return_pct", 0),
                trade_count=bt_report.get("total_trades", 0),
                win_rate=bt_report.get("win_rate", 0),
            )

        log.info("Terminal dashboard enabled")

    # --- Optional: Telegram Bot ---
    telegram = None
    if config.get("telegram", {}).get("enabled", False):
        from remote.telegram_bot import TelegramBot
        telegram = TelegramBot(state, config, grid_strategy=grid)
        log.info("Telegram bot enabled")

    # --- Optional: Email Alerts ---
    email_alerter = None
    if config.get("email", {}).get("enabled", False):
        from remote.email_alert import EmailAlerter
        email_alerter = EmailAlerter(state, config)
        log.info("Email alerts enabled")

    # --- Launch background services ---
    ws_task = asyncio.create_task(ws.start())

    if telegram:
        await telegram.start()

    if dashboard:
        dashboard.start()

    log.info("Waiting for WebSocket connection...")
    await asyncio.sleep(2)  # Give WS time to connect and receive initial data

    log.info("Entering main event loop for symbols: %s", symbols)

    # Track calendar day for daily reset
    _last_day = datetime.datetime.now(datetime.timezone.utc).date()

    # Per-symbol signal throttle: only compute signals once per N ms per symbol
    _signal_throttle_ms = config.get("signal_throttle_ms", 100)
    _last_signal_ts: dict[str, int] = {sym: 0 for sym in symbols}

    # Periodic REST book refresh: heal accumulated delta drift
    _book_refresh_interval_ms = 30_000  # refresh each book every 30s via REST
    _last_book_refresh: dict[str, int] = {sym: 0 for sym in symbols}

    # --- Main Event Loop ---
    try:
        while not state.kill_switch:
            # 1. FAST PATH: Drain all pending events — only update book/tape data.
            #    No signal computation here. This keeps up with the WS event rate
            #    even with 5+ symbols (~200-500 events/sec).
            _dirty_syms: set[str] = set()
            _drain_count = 0
            while _drain_count < 500:
                evt = await ws.next_event(timeout_s=0.005 if _drain_count > 0 else 0.02)
                if evt is None:
                    break
                _drain_count += 1
                updated = ingest_event(evt, books, tapes, dashboard=dashboard)
                if updated:
                    _dirty_syms.add(updated)

            # 2. SLOW PATH: Signal computation — only for dirty symbols,
            #    throttled to once per signal_throttle_ms per symbol.
            now_ms = time_now_ms()
            for sym in _dirty_syms:
                if now_ms - _last_signal_ts.get(sym, 0) >= _signal_throttle_ms:
                    _last_signal_ts[sym] = now_ms
                    await evaluate_symbol(
                        sym, books, tapes, state, config,
                        router, oco, persistence, heartbeat,
                        dashboard=dashboard,
                    )

            # 3. Process private events (order fills, position updates)
            for prv_evt in ws.drain_private():
                await process_private_event(prv_evt, state, config, books,
                                            oco, persistence, rest,
                                            telegram=telegram, email_alerter=email_alerter,
                                            grid=grid)

            # 4. Manage open positions
            for sym in symbols:
                if state.has_position(sym):
                    await manage_position(sym, state, books[sym], tapes[sym],
                                          config, oco, persistence)

            # 5. Multigrid: place/manage grid orders
            if grid and grid.enabled:
                for sym in symbols:
                    book = books[sym]
                    if book.is_fresh():
                        mid = book.mid_price()
                        grid.recalculate(
                            sym, mid,
                            state.last_direction,
                            state.last_confidence,
                            state.equity,
                            state.vol_regime,
                            time_now_ms(),
                        )
                        # Place pending grid orders (entry levels)
                        for level in grid.pending_orders(sym):
                            order_id = await router.place_limit(
                                sym, level.side, level.quantity, level.price)
                            if order_id:
                                grid.mark_order(sym, level.price, level.side, order_id)
                                state.grid_order_ids.add(order_id)

                        # Place harvest orders (take-profit for filled levels)
                        for level in grid.harvest_orders(sym):
                            # Harvest = opposite side at pair_price
                            harvest_side = Side.SELL if level.side == Side.BUY else Side.BUY
                            harvest_price = level.pair_price
                            harvest_id = await router.place_limit(
                                sym, harvest_side, level.quantity, harvest_price,
                                reduce_only=False)  # not reduce_only - grid accumulates
                            if harvest_id:
                                state.grid_order_ids.add(harvest_id)
                                state.grid_harvest_ids[harvest_id] = level.order_id
                                # Clear pair_price to prevent re-placing
                                level.pair_price = 0.0
                                log.info("GRID HARVEST ORDER: %s %s @ %.2f for level %s",
                                         harvest_side.name, sym, harvest_price, level.order_id)

            # 6. Housekeeping
            state.update_latency(ws.latency_estimate_ms())
            heartbeat.check()
            persistence.flush_if_needed()

            # 6a. Book health: REST snapshot refresh
            #     - Immediate: if book.needs_snapshot (crossed/seq gap), fetch NOW
            #     - Periodic: refresh every 30s to heal silent drift
            _now_bk = time_now_ms()
            for sym in symbols:
                b = books[sym]
                need_immediate = b.needs_snapshot
                need_periodic = (_now_bk - _last_book_refresh.get(sym, 0)) >= _book_refresh_interval_ms
                if need_immediate or need_periodic:
                    try:
                        ob = await rest.get_orderbook(sym, b.depth)
                        if ob and ob.get("b") and ob.get("a"):
                            b.on_snapshot(ob["b"], ob["a"], ob.get("seq", 0))
                            _last_book_refresh[sym] = _now_bk
                            if need_immediate:
                                log.info("Book healed via REST for %s", sym)
                        else:
                            # REST failed, fall back to WS resubscribe
                            await ws.resubscribe_book(sym)
                    except Exception as e:
                        log.debug("REST book refresh failed for %s: %s", sym, e)
                        if need_immediate:
                            await ws.resubscribe_book(sym)

            # 6b. Daily reset at midnight UTC
            today = datetime.datetime.now(datetime.timezone.utc).date()
            if today != _last_day:
                state.reset_daily()
                _last_day = today
                log.info("Daily reset at midnight UTC")

            # 6c. Periodic equity sync from exchange
            now_ms = time_now_ms()
            if now_ms - _last_equity_sync > _equity_sync_interval_s * 1000:
                _last_equity_sync = now_ms
                try:
                    wallet = await rest.get_wallet_balance()
                    if wallet.equity > 0:
                        old_eq = state.equity
                        # Respect --equity override during sync
                        equity_override = config.get("equity_override")
                        if equity_override and equity_override > 0:
                            # Don't sync above the override cap
                            # But DO track actual PnL changes within the cap
                            new_equity = min(wallet.equity, equity_override)
                        else:
                            new_equity = wallet.equity

                        state.equity = new_equity
                        if new_equity > state.peak_equity:
                            state.peak_equity = new_equity
                        drift = new_equity - old_eq
                        if abs(drift) > 0.001:
                            log.debug("Equity synced: $%.2f -> $%.2f (drift $%+.4f)",
                                      old_eq, new_equity, drift)
                except Exception:
                    pass  # non-critical; will retry next interval

            # 7. Remote alerts on kill switch
            if state.kill_switch:
                if telegram:
                    await telegram.alert_kill_switch(state.kill_reason)
                if email_alerter:
                    await email_alerter.alert_kill_switch(state.kill_reason)

    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
    except Exception as e:
        log.exception("Fatal error in main loop: %s", e)
    finally:
        log.info("Shutting down...")

        # Cancel all orders, flatten positions
        for sym in symbols:
            try:
                await router.cancel_all(sym)
                if grid:
                    for oid in grid.cancel_all(sym):
                        try:
                            await rest.cancel_order(sym, oid)
                        except Exception:
                            pass
            except Exception:
                pass

        if state.positions:
            log.warning("Flattening %d open positions...", len(state.positions))
            failed = await oco.flatten_all(state.positions)
            if failed:
                log.critical("CRITICAL: Failed to flatten positions on shutdown: %s - CHECK EXCHANGE MANUALLY!", failed)
                if telegram:
                    await telegram.send_message(f"⚠️ CRITICAL: Failed to close positions on shutdown: {failed}")

        if dashboard:
            dashboard.stop()
        if telegram:
            await telegram.stop()
        if email_alerter:
            await email_alerter.send_daily_summary()

        await ws.stop()
        await rest.close()
        persistence.flush_all()
        ws_task.cancel()

        log.info("Final state: %s", state.summary())
        log.info("Shutdown complete.")


def ingest_event(
    evt: WsEvent,
    books: dict[str, OrderBook],
    tapes: dict[str, TradeTape],
    dashboard=None,
) -> str | None:
    """Fast path: update book/tape data. No signal computation.

    Returns the symbol name if data was ingested, None if skipped.
    """
    sym = evt.symbol
    book = books.get(sym)
    tape = tapes.get(sym)
    if not book or not tape:
        return None

    if evt.event_type == EventType.BOOK_SNAPSHOT:
        data = evt.data
        book.on_snapshot(
            data.get("b", []),
            data.get("a", []),
            data.get("seq", 0),
        )
        return sym

    elif evt.event_type == EventType.BOOK_DELTA:
        data = evt.data
        book.on_delta(
            data.get("b", []),
            data.get("a", []),
            data.get("seq", 0),
        )
        return sym

    elif evt.event_type == EventType.TRADE:
        trade: Trade = evt.data
        tape.add_trade(trade)
        if dashboard and sym == dashboard.chart_symbol:
            dashboard.add_price_point(trade.price, trade.quantity)
        return sym

    return None


async def evaluate_symbol(
    sym: str,
    books: dict[str, OrderBook],
    tapes: dict[str, TradeTape],
    state: RuntimeState,
    config: dict,
    router: OrderRouter,
    oco: OcoManager,
    persistence: Persistence,
    heartbeat: Heartbeat,
    dashboard=None,
) -> None:
    """Slow path: signal computation + trade decision for one symbol.

    Called at most once per signal_throttle_ms per symbol.
    """
    book = books[sym]
    tape = tapes[sym]

    if not book.is_fresh():
        return

    # Compute features
    f = features.compute_all(book, tape, config)

    # Update vol regime on state
    vol_name = f.get("vol_regime_name", "normal")
    state.vol_regime = vol_name

    # Don't trade in extreme vol
    if vol_name == "extreme":
        return

    # Compute quality gate
    can, reason = risk.can_trade(state, book, sym)
    if not can:
        log.debug("GATE BLOCKED %s: %s", sym, reason)

    # Compute signal scores
    scores = fuse.score_all(f, config)
    state.last_scores = scores

    # Get weights (with adaptive adjustment)
    signal_edges = heartbeat.get_signal_edges()
    weights = fuse.get_weights(config, signal_edges, config.get("adaptive", {}))

    # Fuse into decision — compute RAW confidence first (for dashboard display)
    # Then apply quality gate only for trading decision
    direction_raw, conf_raw, raw = fuse.decide(
        scores, weights, config.get("alpha", 5), quality_gate=1.0,
        min_signals=config.get("min_signals", 0),
    )

    # Store raw values for dashboard (shows signal strength even when can't trade)
    state.last_direction = direction_raw
    state.last_confidence = conf_raw
    state.last_raw = raw
    state.last_symbol = sym

    # Store per-symbol scores for multi-coin dashboard
    state.scores_by_symbol[sym] = {
        "scores": dict(scores),
        "conf": conf_raw,
        "direction": direction_raw,
        "raw": raw,
        "ts": time_now_ms(),
    }

    # Apply quality gate for actual trading decision
    if not can:
        conf_gated = 0.0
        direction = 0
    else:
        conf_gated = conf_raw
        direction = direction_raw

    # Attempt entry
    if can and state.no_position(sym) and direction != 0 and conf_gated >= config.get("C_enter", 0.65):
        await try_enter(sym, direction, conf_gated, book, tape, state, config,
                        router, oco, scores, persistence,
                        dashboard=dashboard)

    # Optional decision logging (throttled along with signals — no spam)
    if config.get("log_decisions") and abs(raw) > 0.01:
        decision_data = {
            "symbol": sym,
            "direction": direction,
            "confidence": round(conf_raw, 4),
            "raw_score": round(raw, 4),
            "spread_ticks": round(book.spread_ticks(), 2),
            "mid_price": round(book.mid_price(), 2),
            "vol_regime": vol_name,
            "gate_reason": reason if not can else "",
            "action": "enter" if state.has_position(sym) else "none",
        }
        decision_data.update({k: round(v, 4) for k, v in scores.items()})
        persistence.log_decision(decision_data)


async def try_enter(
    symbol: str,
    direction: int,
    confidence: float,
    book: OrderBook,
    tape: TradeTape,
    state: RuntimeState,
    config: dict,
    router: OrderRouter,
    oco: OcoManager,
    scores: dict,
    persistence: Persistence,
    dashboard=None,
) -> None:
    """Attempt to enter a trade."""
    side = Side.BUY if direction == 1 else Side.SELL
    tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)

    # Compute volatility-adapted stops (matches backtest engine)
    avg_range = risk.estimate_avg_range(tape, tick_size)
    stop_price, tp_price, stop_distance = risk.compute_stops(
        book, direction, config, symbol, avg_range=avg_range)

    # Position size using price-based stop distance + confidence scaling
    qty = risk.position_size(state, book, stop_distance, symbol,
                             confidence=confidence)
    if qty <= 0:
        wr = state.rolling_win_rate(20)
        log.info("ENTRY BLOCKED %s: position_size=0 (wr=%.2f, equity=%.2f, stop_dist=%.4f)",
                 symbol, wr, state.equity, stop_distance)
        return

    log.info("ENTRY SIGNAL: %s %s conf=%.3f raw=%.4f qty=%.6f stop=%.2f tp=%.2f range=%.2f",
             side.name, symbol, confidence, state.last_raw, qty, stop_price, tp_price, avg_range)

    # Try post-only limit
    order_id = await router.enter(symbol, side, qty, book)

    if not order_id:
        # Fall back to IOC if probability is high enough
        ioc_prob = router.fill_probability(side, book, qty)
        if ioc_prob >= config.get("ioc_pmin", 0.65):
            order_id = await router.ioc_fallback(symbol, side, qty, book)

    if order_id:
        # Record entry in state (actual fill price will come via WS execution update)
        estimated_price = book.best_bid() if side == Side.BUY else book.best_ask()
        pos = state.on_enter(
            symbol=symbol,
            side=side,
            fill_price=estimated_price,
            quantity=qty,
            confidence=confidence,
            scores=scores,
            stop_price=stop_price,
            tp_price=tp_price,
        )

        # Attach TP/SL - CRITICAL: if this fails, close position immediately
        oco_success = await oco.attach(pos)
        if not oco_success:
            log.error("OCO attach failed for %s - closing position to avoid naked exposure", symbol)
            close_success = await oco.market_flatten(pos)
            if close_success:
                state.on_exit(symbol, estimated_price, "oco_failed")
            else:
                log.critical("CRITICAL: Failed to close position %s after OCO failure - MANUAL INTERVENTION REQUIRED", symbol)
            return

        # Mark on dashboard
        if dashboard:
            dashboard.add_order_marker(side.name, estimated_price)

        # Log order
        persistence.log_order({
            "order_id": order_id,
            "symbol": symbol,
            "side": side.name,
            "type": "ENTRY",
            "price": estimated_price,
            "quantity": qty,
            "status": "SENT",
        })


async def process_private_event(
    evt: WsEvent,
    state: RuntimeState,
    config: dict,
    books: dict[str, OrderBook],
    oco: OcoManager,
    persistence: Persistence,
    rest=None,
    telegram=None,
    email_alerter=None,
    grid=None,
) -> None:
    """Handle private WS events: order fills, position updates.

    WS adapters emit NormalizedExecution / NormalizedOrderUpdate /
    NormalizedPositionUpdate dataclasses — access via attribute, not .get().
    """
    if evt.event_type == EventType.EXECUTION:
        data: NormalizedExecution = evt.data
        symbol = data.symbol
        exec_price = float(data.price)
        exec_qty = float(data.quantity)
        side_str = data.side.name if isinstance(data.side, Side) else str(data.side)
        order_type = data.order_type
        order_id = data.order_id

        # --- Grid fill handling ---
        if grid and order_id in state.grid_order_ids:
            # Check if this is a harvest order fill
            if order_id in state.grid_harvest_ids:
                # Find the original level and record harvest PnL
                orig_order_id = state.grid_harvest_ids.pop(order_id)
                state.grid_order_ids.discard(order_id)
                # Find the level by original order_id
                gs = grid.grids.get(symbol)
                if gs:
                    for lv in gs.levels:
                        if lv.order_id == orig_order_id and lv.is_filled:
                            pnl = grid.on_harvest(symbol, lv, exec_price)
                            state.equity += pnl  # credit grid PnL
                            log.info("GRID HARVEST COMPLETE: %s pnl=%.4f equity=%.2f",
                                     symbol, pnl, state.equity)
                            persistence.log_trade({
                                "symbol": symbol,
                                "side": lv.side.name,
                                "entry_price": lv.fill_price,
                                "exit_price": exec_price,
                                "quantity": lv.quantity,
                                "pnl": round(pnl, 6),
                                "pnl_R": 0.0,  # grid trades don't use R-multiple
                                "duration_ms": 0,
                                "duration_s": 0.0,
                                "datetime": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                "exit_reason": "grid_harvest",
                            })
                            break
            else:
                # Initial grid level fill
                state.grid_order_ids.discard(order_id)
                filled_level = grid.on_fill(symbol, exec_price, data.side, order_id)
                if filled_level:
                    log.info("GRID LEVEL FILLED: %s %s @ %.2f -> harvest target %.2f",
                             data.side.name, symbol, exec_price, filled_level.pair_price)
            return  # don't process as regular order

        # Check for slippage on entry fills
        pos = state.position(symbol)
        if pos:
            tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)
            max_slip = config.get("max_slippage_ticks", 2)
            slip, is_breach = risk.check_slippage(
                pos.entry_price, exec_price, int(pos.side), tick_size, max_slip
            )
            if is_breach:
                state.slip_breaches += 1
                log.warning("Slippage breach #%d: %.1f ticks on %s",
                            state.slip_breaches, slip, symbol)
                if state.slip_breaches >= config.get("kill_switch_slip_breaches", 3):
                    state.trigger_kill_switch("slippage_breaches")
                    failed = await oco.flatten_all(state.positions)
                    if failed:
                        log.critical("CRITICAL: Kill switch failed to flatten: %s", failed)

        persistence.log_order({
            "order_id": order_id,
            "symbol": symbol,
            "side": side_str,
            "type": order_type,
            "price": exec_price,
            "quantity": exec_qty,
            "status": "FILLED",
        })

    elif evt.event_type == EventType.ORDER:
        data: NormalizedOrderUpdate = evt.data
        status = data.status
        symbol = data.symbol

        # Handle TP/SL fills (position closed)
        if status == "Filled" and data.reduce_only:
            fill_price = float(data.avg_price)
            reason = "tp_hit" if data.stop_order_type == "TakeProfit" else "sl_hit"
            result = state.on_exit(symbol, fill_price, reason)
            if result:
                # Binance bracket cleanup: cancel the paired TP/SL order
                if rest and hasattr(rest, 'cleanup_bracket'):
                    try:
                        await rest.cleanup_bracket(symbol, data.order_id)
                    except Exception as e:
                        log.debug("Bracket cleanup: %s", e)

                persistence.log_trade({
                    "symbol": symbol,
                    "side": result.side.name,
                    "entry_price": result.entry_price,
                    "exit_price": result.exit_price,
                    "quantity": result.quantity,
                    "pnl": round(result.pnl, 6),
                    "pnl_R": round(result.pnl_R, 4),
                    "duration_ms": result.duration_ms,
                    "duration_s": round(result.duration_ms / 1000, 2),
                    "datetime": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_reason": reason,
                })
                if telegram:
                    await telegram.alert_trade(result)

    elif evt.event_type == EventType.POSITION:
        data: NormalizedPositionUpdate = evt.data
        symbol = data.symbol
        size = float(data.size)

        # Position closed externally (stop hit, liquidation, manual)
        if size == 0 and state.has_position(symbol):
            pos = state.position(symbol)
            book = books.get(symbol)
            exit_price = book.mid_price() if book else pos.entry_price
            result = state.on_exit(symbol, exit_price, "position_closed")
            if result:
                persistence.log_trade({
                    "symbol": symbol,
                    "side": result.side.name,
                    "entry_price": result.entry_price,
                    "exit_price": result.exit_price,
                    "quantity": result.quantity,
                    "pnl": round(result.pnl, 6),
                    "pnl_R": round(result.pnl_R, 4),
                    "duration_ms": result.duration_ms,
                    "duration_s": round(result.duration_ms / 1000, 2),
                    "datetime": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_reason": "position_closed",
                })
                if telegram:
                    await telegram.alert_trade(result)

    # Check drawdown alerts
    if state.drawdown > 0.05:
        dd_pct = state.drawdown * 100
        if telegram:
            await telegram.alert_drawdown(dd_pct)
        if email_alerter:
            await email_alerter.alert_drawdown(dd_pct)


async def manage_position(
    symbol: str,
    state: RuntimeState,
    book: OrderBook,
    tape: TradeTape,
    config: dict,
    oco: OcoManager,
    persistence: Persistence,
) -> None:
    """Manage an open position: max_hold, trail stop, check confidence exit.

    Matches backtest engine logic: volatility-adapted trailing, max_hold exit.
    """
    pos = state.position(symbol)
    if not pos:
        return

    current_price = book.mid_price()
    if current_price <= 0:
        return

    # Mark to market
    pos.mark_to_market(current_price)

    # Max hold enforcement (matches backtest _manage_position)
    profile = config.get("profile")
    if profile:
        max_hold_ms = profile.get("max_hold", 12) * 60_000  # candles → ms
        hold_duration = time_now_ms() - pos.entry_time
        if hold_duration >= max_hold_ms:
            log.info("Max hold reached for %s (%d min), flattening",
                     symbol, hold_duration // 60_000)
            await oco.market_flatten(pos)
            result = state.on_exit(symbol, current_price, "max_hold")
            if result:
                persistence.log_trade({
                    "symbol": symbol,
                    "side": result.side.name,
                    "entry_price": result.entry_price,
                    "exit_price": result.exit_price,
                    "quantity": result.quantity,
                    "pnl": round(result.pnl, 6),
                    "pnl_R": round(result.pnl_R, 4),
                    "duration_ms": result.duration_ms,
                    "duration_s": round(result.duration_ms / 1000, 2),
                    "datetime": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_reason": "max_hold",
                })
            return

    # Use per-symbol confidence, not the global last_confidence which could
    # belong to a different symbol (overwritten by whichever symbol processed last)
    sym_data = state.scores_by_symbol.get(symbol, {})
    sym_confidence = sym_data.get("conf", 0.0) if sym_data else 0.0

    # Check if we have fresh signal data for this symbol
    # Without it, confidence defaults to 0.0 which would wrongly trigger exit
    has_fresh_scores = bool(sym_data) and (time_now_ms() - sym_data.get("ts", 0)) < 30_000

    # Volatility-adapted trailing (matches backtest trailing logic)
    tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)
    avg_range = risk.estimate_avg_range(tape, tick_size)
    await oco.update_trailing(pos, current_price, sym_confidence,
                              avg_range=avg_range)

    # Check confidence exit — only if we have fresh signal data
    # Otherwise conf=0.0 default would falsely trigger exit
    should_exit = False
    if has_fresh_scores:
        should_exit = await oco.tighten_on_confidence_drop(pos, sym_confidence)
    if should_exit:
        await oco.market_flatten(pos)
        result = state.on_exit(symbol, current_price, "confidence_exit")
        if result:
            persistence.log_trade({
                "symbol": symbol,
                "side": result.side.name,
                "entry_price": result.entry_price,
                "exit_price": result.exit_price,
                "quantity": result.quantity,
                "pnl": round(result.pnl, 6),
                "pnl_R": round(result.pnl_R, 4),
                "duration_ms": result.duration_ms,
                "duration_s": round(result.duration_ms / 1000, 2),
                "datetime": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "exit_reason": "confidence_exit",
            })


if __name__ == "__main__":
    asyncio.run(main())
