# Smallfish

> *"He robs from the rich and gives to the poor."*

The markets aren't free. They never were. Institutional players, market makers, and algorithmic
hedge funds operate with information asymmetry, co-located servers, and order flow privileges
that retail traders will never have. They see your stops. They hunt your liquidations. They
front-run your orders. The game is rigged and the house always wins — unless you learn to
read what the house is doing.

**Smallfish** is an open-source, signal-fusion scalping engine that reads the same
microstructure footprints the big players leave behind — spoofing patterns, iceberg orders,
order book sweeps, whale bursts, CVD flow, trade intensity, liquidity thinning, micro-volatility
spikes, and absorption walls — and uses them to swim in the wake of the whales instead
of being swallowed by them.

This isn't financial advice. This is a tool. A crowbar for the locked door.

```
  ><(((o>  smallfish  <o)))><
```

## Philosophy

The name says it all. We are the small fish. We don't move markets. We don't have Bloomberg
terminals or dark pool access. But we have pattern recognition, we have code, and we have
each other. By open-sourcing this engine, we ensure that the tools of the powerful are
available to everyone.

- **Transparency over secrecy.** Every signal, every weight, every risk gate is visible and auditable.
- **Community over profit.** This project is Apache 2.0 licensed. Fork it. Improve it. Share it.
- **Survival over greed.** The risk management is designed to keep you alive, not to make you rich overnight. The kill switch exists because the market will always be there tomorrow.

## Architecture

```
Exchange WebSocket (L2 book + trades)     ← ExchangeWS ABC (Bybit or Binance)
  → OrderBook / TradeTape (src/marketdata/)
    → Feature Extraction (src/marketdata/features.py)
      → 13 Signal Scorers (src/signals/)
        → Signal Fusion + Quality Gate (src/signals/fuse.py)
          → Risk Sizing + Pre-Trade Gates (src/exec/risk.py)
            → Smart Order Router (src/exec/router.py)  ← ExchangeREST ABC
              → OCO Brackets + Trailing Stop (src/exec/oco.py)
```

### Multi-Exchange Gateway

Smallfish uses an exchange-agnostic gateway layer. Both Bybit and Binance USDT-M perpetual
futures are supported through abstract interfaces:

```
gateway/
├── base.py           # ExchangeREST + ExchangeWS ABCs, normalized types
├── factory.py        # create_rest() / create_ws() — dispatch by exchange name
├── rest.py           # Bybit REST adapter
├── bybit_ws.py       # Bybit WebSocket adapter
├── binance_rest.py   # Binance REST adapter
├── binance_ws.py     # Binance WebSocket adapter
└── persistence.py    # CSV logging
```

Switch between exchanges with a single config line:
```yaml
exchange: "bybit"   # or "binance"
```

## Signals (13 total)

Smallfish fuses 13 independent microstructure signals across 4 weight groups.
Each signal scorer returns `(long_score, short_score)` in [0, 1].

### Core Signals (w) — 47% weight

| Signal | Weight | File | Description |
|--------|--------|------|-------------|
| **OBI** (Order Book Imbalance) | 22% | `obi.py` | Bid/ask volume imbalance + rate of change + micro-pressure. Measures whether buyers or sellers are stacking the book harder. |
| **PRT** (Pull & Replace Trap) | 10% | `prt.py` | Spoofing detection — when one side's cancel rate spikes while the mid-price doesn't move, someone is faking pressure. Fade the fake side. |
| **UMOM** (Micro-Momentum) | 15% | `umom.py` | 1s vs 5s EMA crossover with volatility gate + trade flow confirmation. Captures short-term momentum shifts validated by actual trade direction. |

### Whale Lens Signals (v) — 16% weight

| Signal | Weight | File | Description |
|--------|--------|------|-------------|
| **LTB** (Large Trade Burst) | 8% | `whale.py` | Detects institutional-size trade bursts (>5x median or >$50K) within a 2s window. Big money moving fast = directional intent. |
| **SWEEP** (Order Book Sweep) | 5% | `whale.py` | Multi-level aggressive order clearing 3+ price levels. Someone is paying spread to get filled NOW — that's conviction. |
| **ICE** (Iceberg Refill) | 3% | `whale.py` | Hidden order detection via repeated refills at the same price level (4+ refills in 5s). Icebergs = accumulation/distribution by large players. |

### Meta Signals (x) — 7% weight

| Signal | Weight | File | Description |
|--------|--------|------|-------------|
| **VWAP** (VWAP Deviation) | 5% | `vwap.py` | Mean-reversion signal based on deviation from 500-trade rolling VWAP. Works best in low-volatility regimes where price tends to revert to fair value. |
| **REGIME** (Volatility Regime) | 2% | `regime.py` | Classifies current market state (low/normal/high/extreme volatility). Low vol = mean-revert mode, high vol = momentum mode, extreme = stop trading. |

### Microstructure Signals (t) — 30% weight

| Signal | Weight | File | Description |
|--------|--------|------|-------------|
| **CVD** (Cumulative Volume Delta) | 7% | `cvd.py` | Net buy vs sell flow over a 5s window. Detects flow acceleration and price-flow divergence (price up but CVD down = exhaustion). |
| **TPS** (Trades Per Second) | 5% | `tps.py` | Trade intensity ratio (1s vs 30s baseline). Sudden spikes in trade rate signal urgency. Combined with buy/sell ratio for direction. |
| **LIQ** (Liquidity Thinness) | 6% | `liq.py` | Measures how thin the order book is within 10 ticks + directional asymmetry. Thin book on one side = vulnerability to a move. Combined with OBI for confirmation. |
| **MVR** (Micro-Volatility Ratio) | 6% | `mvr.py` | Short-term (2s) vs long-term (30s) realized volatility ratio. MVR > 1 = breakout regime, MVR < 1 = mean-revert. Combined with momentum for direction. |
| **ABSORB** (Absorption Detection) | 6% | `absorb.py` | Detects large volume absorbed at a price level without the price moving. Someone is defending that level — walls that actually hold signal institutional intent. |

### Signal Fusion

All 13 signals are weighted-summed into a single raw score per direction (long/short).
The raw difference goes through a sigmoid with steepness `alpha` to produce a confidence
score in [0, 1]. Entry requires:

1. Confidence >= `C_enter` threshold
2. At least `min_signals` signals agreeing (score > 0.05 in the same direction)
3. All quality gates passing (spread, latency, funding window, drawdown, kill switch)

Weights adapt in real-time based on per-signal edge tracking when `adaptive.enabled: true`.

## Trading Modes

### Signal Fusion (default)
The core scalping engine. Fuses 13 microstructure signals into a single directional
confidence score, enters via post-only limit orders, manages positions with OCO
brackets and trailing stops.

### Multigrid (optional)
A multi-level grid strategy that places layered buy/sell orders around a dynamic
center price (VWAP or mid). The grid bias shifts based on the signal fusion
confidence — bullish signals shift the grid upward, bearish signals shift it down.
Grid profits are harvested automatically as price oscillates through levels.

Activate with `--multigrid` or set `multigrid.enabled: true` in config.

### Terminal Dashboard
Live split-panel dashboard rendered directly in your terminal using Rich + py-candlestick-chart:

- **Candlestick chart** — real 1-minute OHLCV candles with position entry/SL/TP markers
- **Equity curve** — running equity line (green when up, red when down)
- **Signal strength** — horizontal bars showing each of the 13 signal scores
- **System info** — CPU usage, temperature, memory, uptime (Raspberry Pi friendly)
- **Live metrics** — PnL, win rate, profit factor, avg win/loss, drawdown, daily R usage
- **Position tracker** — open positions with uPnL, SL, TP levels
- **7-day history** — daily PnL breakdown with win/loss counts
- **Recent trades** — last 10 completed trades with R-multiple and exit reason

Designed for headless Raspberry Pi deployments where you SSH in to check on things.

Activate with `--dashboard` or set `dashboard.enabled: true` in config.

## Risk Management

- **Per-trade risk**: 0.5-3.5% of equity (configurable per profile)
- **Drawdown tiers**: Automatically reduces position size as drawdown increases (full → half → quarter → stop)
- **Daily loss limit**: Configurable R-multiples → kill switch (10R conservative, 30R ultra)
- **Rolling win-rate gate**: Blocks entries when rolling WR(20) < 20%, halves size when WR < 30%
- **Slippage monitoring**: Kill switch after 3 slippage breaches
- **Spread gate**: Won't trade if spread > 2 ticks
- **Latency gate**: Won't trade if latency > 80ms
- **Funding avoidance**: Stays flat 3 minutes around funding times
- **Cooldown**: Configurable pause after a losing trade (20s-60s by profile)
- **Adaptive weights**: Per-signal edge tracking adjusts weights in real-time

## Risk Profiles

4 profiles tuned from 7-day parameter sweeps and backtest validation:

| Profile | Risk/Trade | SL | TP | Trail | R:R | Partial TP | Min Signals | C_enter |
|---------|-----------|-----|-----|-------|-----|------------|-------------|---------|
| **conservative** | 0.5% | 0.50x | 1.30x | 30% | 2.6:1 | No | 5/13 | 0.58 |
| **balanced** | 1.5% | 0.50x | 1.60x | 25% | 3.2:1 | No | 4/13 | 0.55 |
| **aggressive** | 2.5% | 0.50x | 1.60x | 22% | 3.2:1 | No | 3/13 | 0.55 |
| **ultra** | 3.5% | 0.50x | 2.00x | 20% | 4.0:1 | No | 3/13 | 0.52 |

Key findings from parameter sweeps:
- `partial_tp=False` always — partial TP cuts avg win in half, destroying the edge
- `breakeven_R=disabled` always — early breakeven move chops winners via noise
- `sl_range_mult=0.50` — 0.30 is too tight for 1m candles, gets whipsawed
- `tp_range_mult >= 1.30` — wider TP lets winners run, improves avg_win/avg_loss ratio
- `trail_pct=0.20-0.30` — tight trailing, never disabled

## Remote Control

### Telegram Bot
Control and monitor your Smallfish instance from anywhere via Telegram.

Commands:
- `/status` — Equity, positions, PnL, drawdown, vol regime
- `/trades` — Recent trade history with PnL bars
- `/equity` — Equity curve chart
- `/grid` — Multigrid status and active levels
- `/kill` — Trigger kill switch remotely
- `/resume` — Reset kill switch and resume trading
- `/config` — View current profile and risk parameters

### Email Alerts
Receive email notifications for critical events:
- Kill switch triggered
- Drawdown exceeds threshold
- Daily summary report

## Getting Started

### Requirements
- Python 3.10+
- Bybit or Binance API key with trading permissions

### Installation

```bash
# With Poetry (recommended)
poetry install

# Or with pip
pip install pyyaml websockets aiohttp numpy orjson python-dotenv plotext rich
pip install candlestick-chart psutil                      # dashboard: candle charts + system info
pip install python-telegram-bot aiosmtplib                # optional: remote control
```

### Exchange Setup

#### Bybit

1. Go to [bybit.com](https://www.bybit.com) → API Management → Create New Key
2. Enable **Contract Trading** permissions (read + write)
3. For testnet: use [testnet.bybit.com](https://testnet.bybit.com) and create a separate API key
4. Add to your `.env`:
```bash
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=true    # remove or set to false for live trading
```

#### Binance

1. Go to [binance.com](https://www.binance.com) → API Management → Create API
2. Select **System Generated** key type
3. Enable **Futures** permissions (read + write) — do NOT enable withdrawal
4. Restrict to your IP address (recommended for live trading)
5. For testnet: go to [testnet.binancefuture.com](https://testnet.binancefuture.com), log in with GitHub, and create API keys
6. Add to your `.env`:
```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true  # remove or set to false for live trading
```

7. Set the exchange in `config/default.yaml`:
```yaml
exchange: "binance"   # or "bybit"
```

#### Raspberry Pi Deployment

```bash
# 1. Install Python 3.10+ and dependencies
sudo apt install python3 python3-pip python3-venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # or: poetry install

# 2. Configure .env with your API keys (see above)

# 3. Run on testnet first (always!)
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard --backfill

# 4. Run headless with Telegram monitoring
nohup python src/app.py --mode aggressive --telegram > smallfish.log 2>&1 &

# 5. Check on it
tail -f smallfish.log          # or SSH and check dashboard
# Or send /status to your Telegram bot

# 6. When confident, switch to live
# Edit .env: remove BYBIT_TESTNET=true (or BINANCE_TESTNET=true)
# Start with conservative profile first!
python src/app.py --mode conservative --telegram --dashboard
```

### Configuration

```bash
# Review trading parameters
# Edit config/default.yaml

# Set up Telegram bot (optional)
# Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env

# Set up email alerts (optional)
# Add SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO to your .env
```

### Run

```bash
# --- Live Trading (Bybit) ---

# Testnet — always start here
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# With backfilled 7D performance history
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard --backfill

# Dashboard + multigrid + Telegram — the full stack
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard --multigrid --telegram

# Auto-select top 5 symbols by volume+volatility
AUTO_SYMBOLS=5 BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# --- Live Trading (Binance) ---

BINANCE_TESTNET=true python src/app.py --mode aggressive --dashboard

# --- Backtesting ---

# Single symbol
python src/backtest.py --symbol BTCUSDT --days 30 --mode aggressive --equity 50

# On Binance data
python src/backtest.py --symbol BTCUSDT --days 30 --mode aggressive --equity 50 --exchange binance

# Multiple symbols
python src/backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 14 --mode aggressive --equity 50

# Auto-scan top N symbols and backtest them
python src/backtest.py --auto 5 --days 7 --mode aggressive --equity 50

# With terminal charts (trade PnL + equity curve + signal weights)
python src/backtest.py --auto 5 --days 7 --mode aggressive --equity 50 --chart

# Sweep all 4 profiles and compare
python src/backtest.py --symbols BTCUSDT ETHUSDT --days 30 --sweep --equity 50

# --- Parameter Sweep ---

# Find optimal profile parameters
python src/param_sweep.py --symbol BTCUSDT --days 7 --equity 50

# Multi-symbol with parallel workers
python src/param_sweep.py --symbols BTCUSDT ETHUSDT --days 7 --equity 50 --workers 4
```

### Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
src/
├── app.py                 # Main async event loop (multi-exchange)
├── backtest.py            # Backtesting engine with 4 profiles
├── param_sweep.py         # Parameter sweep for optimal profile tuning
├── core/
│   ├── types.py           # Data types (Order, Position, Trade, etc.)
│   ├── state.py           # Runtime state management
│   ├── profiles.py        # Risk profiles (conservative → ultra) + apply_profile()
│   ├── ringbuffer.py      # Fixed-size circular buffer
│   └── utils.py           # Math utilities (sigmoid, EMA, simulated time, etc.)
├── marketdata/
│   ├── book.py            # L2 order book with delta processing
│   ├── tape.py            # Trade tape with rolling analytics
│   ├── features.py        # Feature extraction pipeline
│   └── scanner.py         # Symbol scanner for top performers (exchange-agnostic)
├── signals/
│   ├── obi.py             # Order Book Imbalance signal
│   ├── prt.py             # Pull & Replace Trap signal
│   ├── umom.py            # Micro-Momentum signal
│   ├── whale.py           # Whale activity signals (LTB, SWEEP, ICE)
│   ├── vwap.py            # VWAP deviation signal
│   ├── regime.py          # Volatility regime signal
│   ├── cvd.py             # Cumulative Volume Delta signal
│   ├── tps.py             # Trades Per Second signal
│   ├── liq.py             # Liquidity Thinness signal
│   ├── mvr.py             # Micro-Volatility Ratio signal
│   ├── absorb.py          # Absorption Detection signal
│   └── fuse.py            # Signal fusion + adaptive weights
├── strategies/
│   └── multigrid.py       # Multi-level grid trading strategy
├── exec/
│   ├── router.py          # Smart order routing (PostOnly → reprice → IOC)
│   ├── oco.py             # OCO bracket + trailing stop management
│   └── risk.py            # Position sizing + pre-trade gates
├── gateway/
│   ├── base.py            # ExchangeREST + ExchangeWS ABCs + normalized types
│   ├── factory.py         # Exchange factory: create_rest() / create_ws()
│   ├── rest.py            # Bybit REST adapter
│   ├── bybit_ws.py        # Bybit WebSocket adapter
│   ├── binance_rest.py    # Binance REST adapter
│   ├── binance_ws.py      # Binance WebSocket adapter
│   └── persistence.py     # CSV logging (decisions, orders, trades)
├── monitor/
│   ├── metrics.py         # Performance analytics
│   ├── heartbeat.py       # Health monitoring + state snapshots
│   └── dashboard.py       # Terminal dashboard (Rich + candlestick-chart + plotext)
└── remote/
    ├── telegram_bot.py    # Telegram bot for remote control
    └── email_alert.py     # Email notifications for critical events
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Free as in freedom. Use it, fork it, improve it, share it.

## Disclaimer

This is a trading bot. Cryptocurrency trading involves substantial risk of loss. Markets
are volatile, leveraged, and unforgiving. Past performance — including backtests — is not
indicative of future results. The authors and contributors accept no liability for financial
losses incurred through the use of this software.

Start with testnet. Start with small amounts. Respect the kill switch.

> *"The market can stay irrational longer than you can stay solvent."*
> *— but the market can't stay irrational longer than code can stay patient.*
