"""Bybit Scalper — Main Application.

High-frequency multi-signal fusion scalping bot for Bybit perpetual futures.
Async event-driven architecture processing orderbook and trade stream data.
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

from core.state import RuntimeState
from core.types import (
    EventType, Side, Trade, WsEvent, OrderStatus, Position,
)
from core.utils import time_now_ms

from marketdata.book import OrderBook
from marketdata.tape import TradeTape
import marketdata.features as features

import signals.fuse as fuse

from exec.router import OrderRouter
from exec.oco import OcoManager
import exec.risk as risk

from gateway.bybit_ws import BybitWS
from gateway.rest import BybitREST
from gateway.persistence import Persistence

from monitor.heartbeat import Heartbeat
from monitor import metrics as metrics_mod

log = logging.getLogger("scalper")


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


async def main() -> None:
    load_dotenv()
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))

    log.info("=" * 60)
    log.info("  BYBIT SCALPER v1.0 — Starting up")
    log.info("=" * 60)

    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet = os.environ.get("BYBIT_TESTNET", "false").lower() == "true"

    if not api_key or not api_secret:
        log.error("Missing BYBIT_API_KEY or BYBIT_API_SECRET in environment")
        return

    symbols = config.get("symbols", ["BTCUSDT"])

    # --- Dynamic Symbol Selection ---
    auto_symbols = int(os.environ.get("AUTO_SYMBOLS", "0"))
    if auto_symbols > 0:
        from marketdata.scanner import scan_top_symbols, apply_specs_to_config
        log.info("Scanning for top %d symbols to trade...", auto_symbols)
        top = await scan_top_symbols(n=auto_symbols, testnet=testnet)
        if top:
            apply_specs_to_config(config, top)
            symbols = config["symbols"]
            for s in top:
                log.info("  %s  vol=$%.0fM  chg=%+.1f%%",
                         s["symbol"], s["volume_24h_usd"]/1e6, s["change_24h_pct"])
        else:
            log.warning("Scanner returned no symbols, using config defaults")

    # --- Initialize Components ---
    state = RuntimeState(config)
    persistence = Persistence()

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

    rest = BybitREST(api_key, api_secret, testnet=testnet)
    router = OrderRouter(rest, config)
    oco = OcoManager(rest, config)
    heartbeat = Heartbeat(state, persistence, config)

    ws = BybitWS(
        symbols=symbols,
        config=config,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    # --- Startup: fetch account info and set leverage ---
    try:
        wallet = await rest.get_wallet_balance()
        coins = wallet.get("result", {}).get("list", [{}])[0].get("coin", [])
        for coin in coins:
            if coin.get("coin") == "USDT":
                equity = float(coin.get("equity", config.get("initial_equity", 1000)))
                state.equity = equity
                state.peak_equity = equity
                state.daily_start_equity = equity
                log.info("Account equity: $%.2f USDT", equity)
                break

        leverage = config.get("leverage", 10)
        for sym in symbols:
            await rest.set_leverage(sym, leverage)
            log.info("Leverage set: %s × %d", sym, leverage)

        # Latency calibration
        server_time = await rest.get_server_time()
        latency = abs(time_now_ms() - server_time)
        state.update_latency(latency)
        log.info("Initial latency: %dms", latency)

    except Exception as e:
        log.error("Startup error: %s", e)

    # --- Launch WebSocket in background ---
    ws_task = asyncio.create_task(ws.start())

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
                                           router, oco, persistence, heartbeat)

            # 2. Process private events (order fills, position updates)
            for prv_evt in ws.drain_private():
                await process_private_event(prv_evt, state, config, books,
                                            oco, persistence)

            # 3. Manage open positions
            for sym in symbols:
                if state.has_position(sym):
                    await manage_position(sym, state, books[sym], tapes[sym],
                                          config, oco, persistence)

            # 4. Housekeeping
            state.update_latency(ws.latency_estimate_ms())
            heartbeat.check()
            persistence.flush_if_needed()

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
            except Exception:
                pass

        if state.positions:
            log.warning("Flattening %d open positions...", len(state.positions))
            await oco.flatten_all(state.positions)

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
    direction, conf, raw = fuse.decide(scores, weights, config.get("alpha", 5), quality_gate)
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
                            router, oco, scores, persistence)

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
) -> None:
    """Attempt to enter a trade."""
    side = Side.BUY if direction == 1 else Side.SELL

    # Compute stops
    stop_price, tp_price, stop_ticks = risk.compute_stops(book, direction, config, symbol)

    # Position size
    qty = risk.position_size(state, book, stop_ticks, symbol)
    if qty <= 0:
        log.debug("Position size too small, skipping entry")
        return

    log.info("ENTRY SIGNAL: %s %s conf=%.3f raw=%.4f qty=%.6f stop=%.2f tp=%.2f",
             side.name, symbol, confidence, state.last_raw, qty, stop_price, tp_price)

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


async def manage_position(
    symbol: str,
    state: RuntimeState,
    book: OrderBook,
    tape: TradeTape,
    config: dict,
    oco: OcoManager,
    persistence: Persistence,
) -> None:
    """Manage an open position: trail stop, check confidence exit."""
    pos = state.position(symbol)
    if not pos:
        return

    current_price = book.mid_price()
    if current_price <= 0:
        return

    # Mark to market
    pos.mark_to_market(current_price)

    # Trail stop if confidence is high
    await oco.update_trailing(pos, current_price, state.last_confidence)

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
