# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Smallfish

Signal-fusion scalping engine for Bybit perpetual futures. Reads order book microstructure (spoofing, icebergs, sweeps, whale bursts) across 8 signals, fuses them into confidence-weighted decisions, and executes with smart order routing and strict risk management. Runs async on Python 3.10+.

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

# Live trading (always start on testnet)
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# Full stack: multigrid + dashboard + Telegram
python src/app.py --mode aggressive --dashboard --multigrid --telegram

# Auto-select top N symbols by volume+volatility
AUTO_SYMBOLS=5 BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# Backtest single symbol
python src/backtest.py --symbol BTCUSDT --days 30 --mode aggressive --equity 50

# Backtest with profile sweep (compare all 4 profiles)
python src/backtest.py --symbols BTCUSDT ETHUSDT --days 30 --sweep --equity 50

# Auto-scan top N and backtest
python src/backtest.py --auto 5 --days 7 --mode aggressive --equity 50 --chart
```

## Architecture

### Data flow

```
Bybit WebSocket (L2 book + trades)
  → OrderBook / TradeTape (src/marketdata/)
    → Feature extraction (src/marketdata/features.py)
      → 8 signal scorers (src/signals/)
        → Signal fusion + quality gate (src/signals/fuse.py)
          → Risk sizing + pre-trade gates (src/exec/risk.py)
            → Smart order router (src/exec/router.py)
              → OCO brackets + trailing stop (src/exec/oco.py)
```

### Module map

- **`src/core/`** — Shared foundation: data classes (`types.py`), mutable runtime state (`state.py`), risk profiles (`profiles.py`), math utils (`utils.py`), ring buffer (`ringbuffer.py`).
- **`src/marketdata/`** — L2 order book (`book.py`), trade tape (`tape.py`), derived feature computation (`features.py`), symbol scanner (`scanner.py`).
- **`src/signals/`** — 8 independent signal scorers (obi, prt, umom, whale, vwap, regime) plus fusion engine (`fuse.py`) with adaptive weight adjustment.
- **`src/exec/`** — Order routing with post-only→reprice→IOC fallback (`router.py`), OCO bracket + trailing stop management (`oco.py`), position sizing + drawdown tiers (`risk.py`).
- **`src/gateway/`** — Bybit V5 async WebSocket (`bybit_ws.py`) and REST client with HMAC auth (`rest.py`), CSV persistence (`persistence.py`).
- **`src/strategies/`** — Optional multigrid strategy (`multigrid.py`) for layered grid trading with signal-biased level shifting.
- **`src/monitor/`** — Performance metrics (`metrics.py`), heartbeat snapshots (`heartbeat.py`), terminal 2x2 dashboard (`dashboard.py`).
- **`src/remote/`** — Telegram bot for remote control (`telegram_bot.py`), email alerts (`email_alert.py`).

### Key types (`src/core/types.py`)

`Side`, `Order`, `Position`, `Trade`, `Execution`, `TradeResult` (completed round-trip with PnL, R-multiple, MAE/MFE), `WsEvent`.

### RuntimeState (`src/core/state.py`)

Central mutable state: equity, drawdown, positions by symbol, orders by ID, signal scores, kill switch, daily limits. Key methods: `on_enter()`, `on_exit()`, `trigger_kill_switch()`, `check_daily_limits()`, `size_multiplier()`.

### Signal weights (default)

OBI 30%, UMOM 20%, PRT 15%, LTB 12%, SWEEP 8%, VWAP 7%, ICE 5%, Regime 3%. Weights adapt based on per-signal edge tracking when `adaptive_weights.enabled` is true in config.

### Risk profiles (`src/core/profiles.py`)

4 profiles (conservative/balanced/aggressive/ultra) controlling risk%, R:R ratio, equity cap, partial TP, cooldowns, confidence thresholds. Selected via `--mode` CLI arg.

### Configuration

Master config: `config/default.yaml` (signal weights, gate thresholds, WS params, grid settings). Secrets: `.env` for API keys + Telegram/SMTP credentials.

## Conventions

- Fully async (`asyncio` + `aiohttp` + `websockets`). Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- `src/` is on the Python path for imports (configured in `pyproject.toml`). Import as `from core.types import Side`, not `from src.core.types`.
- All prices/quantities go through tick-size rounding (`utils.round_to_tick`). Never use raw floats for order prices.
- Pre-trade gates (spread, latency, funding window, kill switch) must pass before any entry. See `risk.py`.
- CSV logs written to `data/logs/` (decisions, orders, trades, snapshots). Don't break the persistence schema without updating column headers in `persistence.py`.
