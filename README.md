# Smallfish

> *"He robs from the rich and gives to the poor."*

The markets aren't free. They never were. Institutional players, market makers, and algorithmic
hedge funds operate with information asymmetry, co-located servers, and order flow privileges
that retail traders will never have. They see your stops. They hunt your liquidations. They
front-run your orders. The game is rigged and the house always wins — unless you learn to
read what the house is doing.

**Smallfish** is an open-source, signal-fusion scalping engine that reads the same
microstructure footprints the big players leave behind — spoofing patterns, iceberg orders,
order book sweeps, whale bursts — and uses them to swim in the wake of the whales instead
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
WebSocket (L2 book + trades)
    |
    v
+----------------------------------+
|  Feature Extraction              |
|  OBI . UMOM . PRT . Whale . VWAP |
+----------------+-----------------+
                 v
+----------------------------------+
|  8 Signal Scores                 |
|  -> Weighted Fusion              |
|  -> Adaptive Weights (per-signal |
|     edge tracking)               |
|  -> Confidence Score             |
+----------------+-----------------+
                 v
+----------------------------------+
|  Quality Gates                   |
|  Spread . Latency . Funding      |
|  Drawdown . Kill Switch          |
+----------------+-----------------+
                 v
+----------------------------------+     +------------------+
|  Smart Order Router              |     |  Multigrid       |
|  Post-Only -> Reprice -> IOC     |<--->|  Strategy Layer  |
|  OCO Brackets (TP/SL)           |     |  (optional mode) |
|  Trailing Stop                   |     +------------------+
+----------------+-----------------+
                 v
         Live Trading
                 |
    +------------+------------+
    |                         |
    v                         v
+----------+          +---------------+
| Terminal |          | Remote Control|
| Dashboard|          | Telegram / Email
| (plotext) |         +---------------+
+----------+
```

## Signals

| Signal | Weight | Description |
|--------|--------|-------------|
| **OBI** (Order Book Imbalance) | 30% | Bid/ask volume imbalance + rate of change + micro-pressure |
| **UMOM** (Micro-Momentum) | 20% | 1s vs 5s EMA crossover with volatility gate + trade flow confirmation |
| **PRT** (Pull & Replace Trap) | 15% | Spoofing detection — fade fake pressure |
| **LTB** (Large Trade Burst) | 12% | Institutional-size trade burst detection |
| **SWEEP** (Order Book Sweep) | 8% | Multi-level aggressive order detection |
| **ICE** (Iceberg Refill) | 5% | Hidden order absorption detection |
| **VWAP** (VWAP Deviation) | 7% | Mean-reversion to volume-weighted average price |
| **Regime** (Volatility Regime) | 3% | Adapts strategy to low/normal/high/extreme volatility |

## Trading Modes

### Signal Fusion (default)
The core scalping engine. Fuses 8 microstructure signals into a single directional
confidence score, enters via post-only limit orders, manages positions with OCO
brackets and trailing stops.

### Multigrid (optional)
A multi-level grid strategy that places layered buy/sell orders around a dynamic
center price (VWAP or mid). The grid bias shifts based on the signal fusion
confidence — bullish signals shift the grid upward, bearish signals shift it down.
Grid profits are harvested automatically as price oscillates through levels.

Activate with `--multigrid` or set `multigrid.enabled: true` in config.

### Terminal Dashboard (optional)
Live 2x2 plotext chart dashboard rendered directly in your terminal:

- **Price chart** — live mid-price line with buy/sell scatter markers and position entry/SL/TP lines
- **Equity curve** — running equity line (green when up, red when down)
- **Trade P&L** — per-trade bar chart (green wins, red losses)
- **Signal strength** — horizontal bars showing each signal's current score

Designed for headless Raspberry Pi deployments where you SSH in to check on things.

Activate with `--dashboard` or set `dashboard.enabled: true` in config.

## Risk Management

- **Per-trade risk**: 0.5% of equity (configurable per profile)
- **Drawdown tiers**: Automatically reduces position size as drawdown increases
- **Daily loss limit**: 10 R-multiples -> kill switch
- **Slippage monitoring**: Kill switch after 3 slippage breaches
- **Spread gate**: Won't trade if spread > 2 ticks
- **Latency gate**: Won't trade if latency > 80ms
- **Funding avoidance**: Stays flat 3 minutes around funding times
- **Cooldown**: 5-second pause after a losing trade
- **Adaptive weights**: Per-signal edge tracking adjusts weights in real-time

## Optimization Profiles

| Profile | Risk/Trade | R:R | Equity Cap | Partial TP | Best For |
|---------|-----------|-----|------------|------------|----------|
| conservative | 0.5% | 1.4:1 | 3x | No | Learning, small accounts |
| balanced | 1.5% | 2:1 | 5x | No | Steady growth |
| **aggressive** | **2.5%** | **2.4:1** | **8x** | **Yes (50% at 1R)** | **Proven edge, compounding** |
| ultra | 3.5% | 2.8:1 | 15x | Yes | High conviction only |

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
- Bybit API key with trading permissions

### Installation

```bash
# With Poetry (recommended)
poetry install

# Or with pip
pip install pyyaml websockets aiohttp numpy orjson python-dotenv plotext
pip install python-telegram-bot aiosmtplib  # optional: remote control
```

### Configuration

```bash
# Set up API credentials
cp config/secrets.template.yaml config/secrets.yaml
# Edit with your Bybit API key and secret

# Set up Telegram bot (optional)
# Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env

# Review trading parameters
# Edit config/default.yaml
```

### Run

```bash
# --- Live Trading ---

# Default config
python src/app.py

# With a risk profile (conservative / balanced / aggressive / ultra)
python src/app.py --mode aggressive

# With the plotext terminal dashboard
python src/app.py --dashboard

# Dashboard + auto-scan top 5 high-volume symbols + aggressive profile
AUTO_SYMBOLS=5 python src/app.py --mode aggressive --dashboard

# Dashboard + multigrid + Telegram — the full stack
python src/app.py --mode aggressive --dashboard --multigrid --telegram

# Testnet (always start here)
BYBIT_TESTNET=true python src/app.py --mode aggressive --dashboard

# Testnet + auto-select the top 3 movers
BYBIT_TESTNET=true AUTO_SYMBOLS=3 python src/app.py --mode aggressive --dashboard

# --- Backtesting ---

# Single symbol
python src/backtest.py --symbol BTCUSDT --days 30 --mode aggressive --equity 50

# Multiple symbols
python src/backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 14 --mode aggressive --equity 50

# Auto-scan top N symbols and backtest them
python src/backtest.py --auto 5 --days 7 --mode aggressive --equity 50

# With terminal charts (trade PnL + equity curve + signal weights)
python src/backtest.py --auto 5 --days 7 --mode aggressive --equity 50 --chart

# Sweep all 4 profiles and compare
python src/backtest.py --symbols BTCUSDT ETHUSDT --days 30 --sweep --equity 50
```

### Tests

```bash
python -m pytest tests/ -v
```

## Raspberry Pi Deployment

Smallfish is designed to run headless on a Raspberry Pi for 24/7 live trading:

```bash
# Auto-pick the hottest symbols, aggressive mode, full monitoring
BYBIT_TESTNET=true AUTO_SYMBOLS=5 \
  nohup python src/app.py --mode aggressive --telegram --dashboard > /dev/null 2>&1 &

# Check on it via Telegram: send /status to your bot
# Or SSH in and attach to the dashboard
```

## Project Structure

```
src/
|-- app.py                 # Main async event loop
|-- backtest.py            # Backtesting engine with 4 profiles
|-- core/
|   |-- types.py           # Data types (Order, Position, Trade, etc.)
|   |-- state.py           # Runtime state management
|   |-- profiles.py        # Risk profiles (conservative → ultra) + apply_profile()
|   |-- ringbuffer.py      # Fixed-size circular buffer
|   +-- utils.py           # Math utilities (sigmoid, EMA, etc.)
|-- marketdata/
|   |-- book.py            # L2 order book with delta processing
|   |-- tape.py            # Trade tape with rolling analytics
|   |-- features.py        # Feature extraction pipeline
|   +-- scanner.py         # Symbol scanner for top performers
|-- signals/
|   |-- obi.py             # Order Book Imbalance signal
|   |-- prt.py             # Pull & Replace Trap signal
|   |-- umom.py            # Micro-Momentum signal
|   |-- whale.py           # Whale activity signals (LTB, SWEEP, ICE)
|   |-- vwap.py            # VWAP deviation signal
|   |-- regime.py          # Volatility regime signal
|   +-- fuse.py            # Signal fusion + adaptive weights
|-- strategies/
|   +-- multigrid.py       # Multi-level grid trading strategy
|-- exec/
|   |-- router.py          # Smart order routing
|   |-- oco.py             # OCO bracket + trailing stop management
|   +-- risk.py            # Position sizing + pre-trade gates
|-- gateway/
|   |-- bybit_ws.py        # Async WebSocket client (public + private)
|   |-- rest.py            # Async REST client with HMAC auth
|   +-- persistence.py     # CSV logging (decisions, orders, trades)
|-- monitor/
|   |-- metrics.py         # Performance analytics
|   |-- heartbeat.py       # Health monitoring + state snapshots
|   +-- dashboard.py       # Terminal 2x2 chart dashboard (plotext)
+-- remote/
    |-- telegram_bot.py    # Telegram bot for remote control
    +-- email_alert.py     # Email notifications for critical events
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
