# Bybit Scalper

A production-grade, high-frequency multi-signal fusion scalping bot for Bybit USDT perpetual futures.

## Architecture

```
WebSocket (L2 book + trades)
    │
    ▼
┌─────────────────────────────────┐
│  Feature Extraction             │
│  OBI · UMOM · PRT · Whale · VWAP│
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  8 Signal Scores                │
│  → Weighted Fusion              │
│  → Adaptive Weights (per-signal │
│    edge tracking)               │
│  → Confidence Score             │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Quality Gates                  │
│  Spread · Latency · Funding     │
│  Drawdown · Kill Switch         │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Smart Order Router             │
│  Post-Only → Reprice → IOC      │
│  OCO Brackets (TP/SL)          │
│  Trailing Stop                  │
└──────────────┬──────────────────┘
               ▼
         Live Trading
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

## Risk Management

- **Per-trade risk**: 0.5% of equity (configurable)
- **Drawdown tiers**: Automatically reduces position size as drawdown increases
- **Daily loss limit**: 10 R-multiples → kill switch
- **Slippage monitoring**: Kill switch after 3 slippage breaches
- **Spread gate**: Won't trade if spread > 2 ticks
- **Latency gate**: Won't trade if latency > 80ms
- **Funding avoidance**: Stays flat 3 minutes around funding times
- **Cooldown**: 5-second pause after a losing trade
- **Adaptive weights**: Per-signal edge tracking adjusts weights in real-time

## Getting Started

### Requirements
- Python 3.10+
- Bybit API key with trading permissions

### Installation

```bash
# Install dependencies
pip install pyyaml websockets aiohttp numpy orjson python-dotenv

# Or with Poetry
poetry install
```

### Configuration

```bash
# Set up API credentials
cp .env.example .env
# Edit .env with your Bybit API key and secret

# Review trading parameters
# Edit config/default.yaml
```

### Run

```bash
# Production (mainnet)
python src/app.py

# Testnet (set BYBIT_TESTNET=true in .env)
BYBIT_TESTNET=true python src/app.py
```

### Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
src/
├── app.py                 # Main async event loop
├── core/
│   ├── types.py           # Data types (Order, Position, Trade, etc.)
│   ├── state.py           # Runtime state management
│   ├── ringbuffer.py      # Fixed-size circular buffer
│   └── utils.py           # Math utilities (sigmoid, EMA, etc.)
├── marketdata/
│   ├── book.py            # L2 order book with delta processing
│   ├── tape.py            # Trade tape with rolling analytics
│   └── features.py        # Feature extraction pipeline
├── signals/
│   ├── obi.py             # Order Book Imbalance signal
│   ├── prt.py             # Pull & Replace Trap signal
│   ├── umom.py            # Micro-Momentum signal
│   ├── whale.py           # Whale activity signals (LTB, SWEEP, ICE)
│   ├── vwap.py            # VWAP deviation signal
│   ├── regime.py          # Volatility regime signal
│   └── fuse.py            # Signal fusion + adaptive weights
├── exec/
│   ├── router.py          # Smart order routing
│   ├── oco.py             # OCO bracket + trailing stop management
│   └── risk.py            # Position sizing + pre-trade gates
├── io/
│   ├── bybit_ws.py        # Async WebSocket client (public + private)
│   ├── rest.py            # Async REST client with HMAC auth
│   └── persistence.py     # CSV logging (decisions, orders, trades)
└── monitor/
    ├── metrics.py          # Performance analytics (Sharpe, fill ratio, etc.)
    └── heartbeat.py        # Health monitoring + state snapshots
```

## Disclaimer

This is a trading bot. Cryptocurrency trading involves substantial risk of loss. Past performance is not indicative of future results. Use at your own risk. Always start with testnet and small amounts.
