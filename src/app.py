"""Smallfish — Main Application.

Open-source signal-fusion scalping engine for Bybit perpetual futures.
Async event-driven architecture processing orderbook and trade stream data.

  ><(((o>  smallfish  <o)))><
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
import time

import yaml
from dotenv import load_dotenv

from core.state import RuntimeState
from core.types import (
    EventType, Side, Trade, WsEvent, OrderStatus, Position,
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


async def main() -> None:
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
            state.equity = wallet.equity
            state.peak_equity = wallet.equity
            state.daily_start_equity = wallet.equity
            log.info("Account equity: $%.2f USDT", wallet.equity)
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

    # Equity sync interval (re-fetch wallet balance periodically)
    _equity_sync_interval_s = config.get("equity_sync_interval_s", 30)
    _last_equity_sync = time_now_ms()

    # --- Optional: Backfill 7D performance ---
    bt_report = None
    if args.backfill:
        from backtest import backfill_trades, print_report
        bt_profile = args.mode or "aggressive"
        # Use the real wallet equity (fetched above), not --equity arg
        bt_equity = state.equity if state.equity > 1 else args.equity
        log.info("Backfilling 7-day performance (profile=%s, equity=$%.2f)...",
                 bt_profile, bt_equity)
        bt_trades, bt_report = await backfill_trades(
            symbols, config, days=7,
            equity=bt_equity, profile=bt_profile,
        )
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

    # --- Main Event Loop ---
    try:
        while not state.kill_switch:
            # 1. Process public events (book + trade updates)
            evt = await ws.next_event(timeout_s=0.02)
            if evt is not None:
                await process_public_event(evt, books, tapes, state, config,
                                           router, oco, persistence, heartbeat,
                                           grid=grid, dashboard=dashboard)

            # 2. Process private events (order fills, position updates)
            for prv_evt in ws.drain_private():
                await process_private_event(prv_evt, state, config, books,
                                            oco, persistence,
                                            telegram=telegram, email_alerter=email_alerter)

            # 3. Manage open positions
            for sym in symbols:
                if state.has_position(sym):
                    await manage_position(sym, state, books[sym], tapes[sym],
                                          config, oco, persistence)

            # 4. Multigrid: place/manage grid orders
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
                        for level in grid.pending_orders(sym):
                            order_id = await router.enter(
                                sym, level.side, level.quantity, book)
                            if order_id:
                                grid.mark_order(sym, level.price, level.side, order_id)

            # 5. Housekeeping
            state.update_latency(ws.latency_estimate_ms())
            heartbeat.check()
            persistence.flush_if_needed()

            # 5b. Periodic equity sync from exchange
            now_ms = time_now_ms()
            if now_ms - _last_equity_sync > _equity_sync_interval_s * 1000:
                _last_equity_sync = now_ms
                try:
                    wallet = await rest.get_wallet_balance()
                    if wallet.equity > 0:
                        old_eq = state.equity
                        state.equity = wallet.equity
                        if wallet.equity > state.peak_equity:
                            state.peak_equity = wallet.equity
                        drift = wallet.equity - old_eq
                        if abs(drift) > 0.001:
                            log.debug("Equity synced: $%.2f -> $%.2f (drift $%+.4f)",
                                      old_eq, wallet.equity, drift)
                except Exception:
                    pass  # non-critical; will retry next interval

            # 6. Remote alerts on kill switch
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
            await oco.flatten_all(state.positions)

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


async def process_public_event(
    evt: WsEvent,
    books: dict[str, OrderBook],
    tapes: dict[str, TradeTape],
    state: RuntimeState,
    config: dict,
    router: OrderRouter,
    oco: OcoManager,
    persistence: Persistence,
    heartbeat: Heartbeat,
    grid=None,
    dashboard=None,
) -> None:
    """Handle a public WebSocket event: book update or trade."""
    sym = evt.symbol
    book = books.get(sym)
    tape = tapes.get(sym)
    if not book or not tape:
        return

    # Update market data
    if evt.event_type == EventType.BOOK_SNAPSHOT:
        data = evt.data
        book.on_snapshot(
            data.get("b", []),
            data.get("a", []),
            data.get("seq", 0),
        )
        return  # Don't trade on snapshots, wait for deltas

    elif evt.event_type == EventType.BOOK_DELTA:
        data = evt.data
        book.on_delta(
            data.get("b", []),
            data.get("a", []),
            data.get("seq", 0),
        )

    elif evt.event_type == EventType.TRADE:
        trade: Trade = evt.data
        tape.add_trade(trade)
        # Feed trade price to dashboard (builds real OHLCV candles)
        # Only feed the primary symbol to avoid mixing prices from
        # different symbols (e.g. BTC 98k + DOGE 0.25 → broken Y-axis)
        if dashboard and sym == dashboard.chart_symbol:
            dashboard.add_price_point(trade.price, trade.quantity)

    # --- Signal Pipeline (runs on every book delta and trade) ---
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
    quality_gate = 1.0 if can else 0.0

    # Compute signal scores
    scores = fuse.score_all(f, config)
    state.last_scores = scores

    # Get weights (with adaptive adjustment)
    signal_edges = heartbeat.get_signal_edges()
    weights = fuse.get_weights(config, signal_edges, config.get("adaptive", {}))

    # Fuse into decision
    direction, conf, raw = fuse.decide(
        scores, weights, config.get("alpha", 5), quality_gate,
        min_signals=config.get("min_signals", 0),
    )
    state.last_direction = direction
    state.last_confidence = conf
    state.last_raw = raw

    # Log decision
    if config.get("log_decisions") and abs(raw) > 0.01:
        decision_data = {
            "symbol": sym,
            "direction": direction,
            "confidence": round(conf, 4),
            "raw_score": round(raw, 4),
            "spread_ticks": round(book.spread_ticks(), 2),
            "mid_price": round(book.mid_price(), 2),
            "vol_regime": vol_name,
            "action": "none",
        }
        decision_data.update({k: round(v, 4) for k, v in scores.items()})

        # Attempt entry if conditions met
        if can and state.no_position(sym) and direction != 0 and conf >= config.get("C_enter", 0.65):
            decision_data["action"] = "enter"
            await try_enter(sym, direction, conf, book, tape, state, config,
                            router, oco, scores, persistence,
                            dashboard=dashboard)

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
        log.debug("Position size too small, skipping entry")
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

        # Attach TP/SL
        await oco.attach(pos)

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
    telegram=None,
    email_alerter=None,
) -> None:
    """Handle private WS events: order fills, position updates."""
    if evt.event_type == EventType.EXECUTION:
        data = evt.data
        symbol = data.get("symbol", "")
        exec_price = float(data.get("execPrice", 0))
        exec_qty = float(data.get("execQty", 0))
        side_str = data.get("side", "")
        is_maker = data.get("isMaker", False)
        order_type = data.get("orderType", "")

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
                    await oco.flatten_all(state.positions)

        persistence.log_order({
            "order_id": data.get("orderId", ""),
            "symbol": symbol,
            "side": side_str,
            "type": order_type,
            "price": exec_price,
            "quantity": exec_qty,
            "status": "FILLED",
        })

    elif evt.event_type == EventType.ORDER:
        data = evt.data
        status = data.get("orderStatus", "")
        symbol = data.get("symbol", "")

        # Handle TP/SL fills (position closed)
        if status == "Filled" and data.get("reduceOnly", False):
            fill_price = float(data.get("avgPrice", 0))
            reason = "tp_hit" if data.get("stopOrderType") == "TakeProfit" else "sl_hit"
            result = state.on_exit(symbol, fill_price, reason)
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
                    "exit_reason": reason,
                })
                if telegram:
                    await telegram.alert_trade(result)

    elif evt.event_type == EventType.POSITION:
        data = evt.data
        symbol = data.get("symbol", "")
        size = float(data.get("size", 0))

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
                    "exit_reason": "max_hold",
                })
            return

    # Volatility-adapted trailing (matches backtest trailing logic)
    tick_size = config.get("tick_sizes", {}).get(symbol, 0.01)
    avg_range = risk.estimate_avg_range(tape, tick_size)
    await oco.update_trailing(pos, current_price, state.last_confidence,
                              avg_range=avg_range)

    # Check confidence exit
    should_exit = await oco.tighten_on_confidence_drop(pos, state.last_confidence)
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
                "exit_reason": "confidence_exit",
            })


if __name__ == "__main__":
    asyncio.run(main())
