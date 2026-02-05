# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Smallfish

Signal-fusion scalping engine for **Bybit and Binance** USDT-M perpetual futures. Reads order book microstructure (spoofing, icebergs, sweeps, whale bursts, CVD flow, trade intensity, liquidity thinness, micro-volatility, absorption) across 13 signals, fuses them into confidence-weighted decisions, and executes with smart order routing and strict risk management. Exchange selection via `exchange: "bybit"` or `exchange: "binance"` in config. Runs async on Python 3.10+.

## Commands

```bash
# Install dependencies (Poetry)
poetry install

# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_signals.py -v

# Run a single test
python -m pytest tests/test_signals.py::test_obi_signal -v

# Live trading — Bybit (always start on testnet)
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# Live trading — Binance
BINANCE_TESTNET=true python src/app.py --mode aggressive --dashboard

# Full stack: multigrid + dashboard + Telegram
python src/app.py --mode aggressive --dashboard --multigrid --telegram

# Auto-select top N symbols by volume+volatility
AUTO_SYMBOLS=5 BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# Backtest single symbol (default: Bybit)
python src/backtest.py --symbol BTCUSDT --days 30 --mode aggressive --equity 50

# Backtest on Binance
python src/backtest.py --symbol BTCUSDT --days 30 --mode aggressive --equity 50 --exchange binance

# Backtest with profile sweep (compare all 4 profiles)
python src/backtest.py --symbols BTCUSDT ETHUSDT --days 30 --sweep --equity 50

# Auto-scan top N and backtest
python src/backtest.py --auto 5 --days 7 --mode aggressive --equity 50 --chart
```

## Architecture

### Data flow

```
Exchange WebSocket (L2 book + trades)     ← ExchangeWS ABC (Bybit or Binance)
  → OrderBook / TradeTape (src/marketdata/)
    → Feature extraction (src/marketdata/features.py)
      → 13 signal scorers (src/signals/)
        → Signal fusion + quality gate (src/signals/fuse.py)
          → Risk sizing + pre-trade gates (src/exec/risk.py)
            → Smart order router (src/exec/router.py)  ← ExchangeREST ABC
              → OCO brackets + trailing stop (src/exec/oco.py)
```

### Module map

- **`src/core/`** — Shared foundation: data classes (`types.py`), mutable runtime state (`state.py`), risk profiles (`profiles.py`), math utils (`utils.py`), ring buffer (`ringbuffer.py`).
- **`src/marketdata/`** — L2 order book (`book.py`), trade tape (`tape.py`), derived feature computation (`features.py`), symbol scanner (`scanner.py`).
- **`src/signals/`** — 13 independent signal scorers in 4 weight groups plus fusion engine (`fuse.py`) with adaptive weight adjustment:
  - **Core (w):** OBI (`obi.py`), PRT (`prt.py`), UMOM (`umom.py`)
  - **Whale (v):** LTB/SWEEP/ICE (`whale.py`)
  - **Meta (x):** VWAP (`vwap.py`), Regime (`regime.py`)
  - **Microstructure (t):** CVD (`cvd.py`), TPS (`tps.py`), LIQ (`liq.py`), MVR (`mvr.py`), Absorption (`absorb.py`)
- **`src/exec/`** — Order routing with post-only→reprice→IOC fallback (`router.py`), OCO bracket + trailing stop management (`oco.py`), position sizing + drawdown tiers (`risk.py`).
- **`src/gateway/`** — Exchange-agnostic gateway layer:
  - Abstract interfaces (`base.py`): `ExchangeREST`, `ExchangeWS` ABCs + normalized types (`OrderResponse`, `WalletInfo`, `InstrumentSpec`, `TickerInfo`)
  - Factory (`factory.py`): `create_rest()` / `create_ws()` dispatching by exchange name
  - Bybit adapter: REST (`rest.py`) + WebSocket (`bybit_ws.py`)
  - Binance adapter: REST (`binance_rest.py`) + WebSocket (`binance_ws.py`)
  - CSV persistence (`persistence.py`)
- **`src/strategies/`** — Optional multigrid strategy (`multigrid.py`) for layered grid trading with signal-biased level shifting.
- **`src/monitor/`** — Performance metrics (`metrics.py`), heartbeat snapshots (`heartbeat.py`), terminal 2x2 dashboard (`dashboard.py`).
- **`src/remote/`** — Telegram bot for remote control (`telegram_bot.py`), email alerts (`email_alert.py`).

### Key types (`src/core/types.py`)

`Side`, `Order`, `Position`, `Trade`, `Execution`, `TradeResult` (completed round-trip with PnL, R-multiple, MAE/MFE), `WsEvent`, `NormalizedExecution`, `NormalizedOrderUpdate`, `NormalizedPositionUpdate`.

### Gateway types (`src/gateway/base.py`)

`OrderResponse(success, order_id, error_code, error_msg, raw)`, `WalletInfo(equity, raw)`, `InstrumentSpec(symbol, tick_size, min_qty, qty_step, min_notional)`, `TickerInfo(symbol, last_price, turnover_24h, price_change_pct)`.

Consumers (router, oco) use `ExchangeREST` ABC and `OrderResponse` — never raw exchange dicts. Use `result.success` and `result.order_id` instead of `result.get("retCode")`.

### RuntimeState (`src/core/state.py`)

Central mutable state: equity, drawdown, positions by symbol, orders by ID, signal scores, kill switch, daily limits. Key methods: `on_enter()`, `on_exit()`, `trigger_kill_switch()`, `check_daily_limits()`, `size_multiplier()`.

### Signal weights (default)

4 weight groups summing to 1.0:
- **Core (w):** OBI 22%, PRT 10%, UMOM 15%
- **Whale (v):** LTB 8%, SWEEP 5%, ICE 3%
- **Meta (x):** VWAP 5%, Regime 2%
- **Microstructure (t):** CVD 7%, TPS 5%, LIQ 6%, MVR 6%, ABSORB 6%

Weights adapt based on per-signal edge tracking when `adaptive_weights.enabled` is true in config.

### New microstructure signals

| Signal | File | Role | Key feature inputs |
|--------|------|------|--------------------|
| **CVD** | `cvd.py` | Flow confirmation + divergence | `cvd_accel`, `cvd_norm`, `cvd_divergence` |
| **TPS** | `tps.py` | Urgency gate/filter | `tps_ratio`, `buy_sell_ratio` |
| **LIQ** | `liq.py` | Liquidity thinness + direction | `liq_thinness`, `liq_asymmetry`, `obi` |
| **MVR** | `mvr.py` | Breakout vs chop regime filter | `mvr` (RV_short/RV_long), `momentum` |
| **ABS** | `absorb.py` | Absorption/level defense | `absorption`, `ice_absorb_bid/ask`, `mid_change_ticks` |

### Risk profiles (`src/core/profiles.py`)

4 profiles (conservative/balanced/aggressive/ultra) controlling risk%, R:R ratio, equity cap, partial TP, cooldowns, confidence thresholds. Selected via `--mode` CLI arg.

| Profile | min_signals | C_enter | Notes |
|---------|-------------|---------|-------|
| conservative | 5/13 | 0.58 | Strictest filter, fewest trades |
| balanced | 4/13 | 0.55 | Moderate filter |
| aggressive | 3/13 | 0.55 | Wider entry, partial TP at 1R |
| ultra | 3/13 | 0.52 | Widest entry, heavy compounding |

### Configuration

Master config: `config/default.yaml` (signal weights, gate thresholds, WS params, grid settings, microstructure params). Secrets: `.env` for API keys + Telegram/SMTP credentials.

### Backtesting & Simulated Time

The backtest engine (`src/backtest.py`) uses `set_sim_time()` / `clear_sim_time()` from `core.utils` to override `time_now_ms()` with the current candle's timestamp. This is **critical** — without it, all time-windowed analytics (tape volume windows, book cancel tracking, realized vol, etc.) would see zero data because historical trade timestamps don't match wall-clock time. Always call `set_sim_time(kline["ts"])` before processing each candle in any backtest-like code.

## Conventions

- Fully async (`asyncio` + `aiohttp` + `websockets`). Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- `src/` is on the Python path for imports (configured in `pyproject.toml`). Import as `from core.types import Side`, not `from src.core.types`.
- All prices/quantities go through tick-size rounding (`utils.round_to_tick`). Never use raw floats for order prices.
- Pre-trade gates (spread, latency, funding window, kill switch) must pass before any entry. See `risk.py`.
- CSV logs written to `data/logs/` (decisions, orders, trades, snapshots). Don't break the persistence schema without updating column headers in `persistence.py`.
- `time_now_ms()` supports simulated time for backtesting. Never use `time.time()` directly for timestamps — always go through `time_now_ms()`.
