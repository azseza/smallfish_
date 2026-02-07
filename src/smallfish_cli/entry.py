"""Console entry points for the smallfish package.

After ``pip install .`` or ``poetry install``, two commands are available:

    smallfish            – live trading (wraps src/app.py)
    smallfish-backtest   – backtesting  (wraps src/backtest.py)

Both commands accept the same CLI arguments as running the scripts directly.
"""
from __future__ import annotations

import asyncio
import sys


def main_live() -> None:
    """Entry point for ``smallfish`` console command (live trading)."""
    # app.py already parses sys.argv via argparse
    from app import main  # noqa: E402 — deferred to allow clean packaging
    asyncio.run(main())


def main_backtest() -> None:
    """Entry point for ``smallfish-backtest`` console command."""
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from backtest import run_backtest, run_sweep  # noqa: E402

    parser = argparse.ArgumentParser(description="Smallfish Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--symbols", nargs="+",
                        help="Multiple symbols: --symbols BTCUSDT ETHUSDT SOLUSDT")
    parser.add_argument("--auto", type=int, default=0, metavar="N",
                        help="Auto-select top N symbols by volume+volatility")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to backtest")
    parser.add_argument("--equity", type=float, default=50.0,
                        help="Initial equity in USD")
    parser.add_argument("--mode", default="aggressive",
                        choices=["conservative", "balanced", "aggressive",
                                 "ultra", "starter_50"],
                        help="Risk profile")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all profiles and compare")
    parser.add_argument("--chart", action="store_true",
                        help="Show terminal charts")
    parser.add_argument("--exchange", default="bybit",
                        choices=["bybit", "binance", "mexc", "dydx"],
                        help="Exchange to use for data download")
    parser.add_argument("--fees", action="store_true",
                        help="Enable realistic fee modeling (maker 0.02%%, taker 0.055%%)")
    parser.add_argument("--interval", default="1",
                        choices=["1", "3", "5", "15", "30", "60"],
                        help="Candle interval in minutes (default: 1)")
    parser.add_argument("--maker-fee", type=float, default=None, metavar="PCT",
                        help="Override maker fee rate (e.g. 0.0002 for 0.02%%)")
    parser.add_argument("--taker-fee", type=float, default=None, metavar="PCT",
                        help="Override taker fee rate (e.g. 0.00055 for 0.055%%)")
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.auto > 0:
        symbols = []
    else:
        symbols = [args.symbol]

    fee_override = {}
    if args.fees:
        fee_override["backtest_fees"] = True
    if args.maker_fee is not None:
        fee_override["backtest_maker_fee"] = args.maker_fee
        fee_override["backtest_fees"] = True
    if args.taker_fee is not None:
        fee_override["backtest_taker_fee"] = args.taker_fee
        fee_override["backtest_fees"] = True

    if args.sweep:
        asyncio.run(run_sweep(symbols, args.days, args.equity,
                              exchange=args.exchange,
                              fee_override=fee_override,
                              interval=args.interval))
    else:
        asyncio.run(run_backtest(symbols, args.days, args.equity, args.mode,
                                 auto_symbols=args.auto,
                                 show_chart=args.chart,
                                 exchange=args.exchange,
                                 fee_override=fee_override,
                                 interval=args.interval))
