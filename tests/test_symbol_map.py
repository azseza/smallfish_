"""Tests for gateway.symbol_map — symbol format conversions."""
import pytest

from gateway.symbol_map import to_mexc, from_mexc, to_dydx, from_dydx


# ── MEXC ──────────────────────────────────────────────────────────────

class TestMexcSymbolMap:
    def test_to_mexc_btc(self):
        assert to_mexc("BTCUSDT") == "BTC_USDT"

    def test_to_mexc_eth(self):
        assert to_mexc("ETHUSDT") == "ETH_USDT"

    def test_to_mexc_ada(self):
        assert to_mexc("ADAUSDT") == "ADA_USDT"

    def test_to_mexc_doge(self):
        assert to_mexc("DOGEUSDT") == "DOGE_USDT"

    def test_to_mexc_sui(self):
        assert to_mexc("SUIUSDT") == "SUI_USDT"

    def test_to_mexc_1000pepe(self):
        assert to_mexc("1000PEPEUSDT") == "1000PEPE_USDT"

    def test_to_mexc_usdc_quote(self):
        assert to_mexc("BTCUSDC") == "BTC_USDC"

    def test_from_mexc_btc(self):
        assert from_mexc("BTC_USDT") == "BTCUSDT"

    def test_from_mexc_eth(self):
        assert from_mexc("ETH_USDT") == "ETHUSDT"

    def test_from_mexc_ada(self):
        assert from_mexc("ADA_USDT") == "ADAUSDT"

    def test_roundtrip_mexc(self):
        for sym in ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
                     "SOLUSDT", "SUIUSDT", "1000PEPEUSDT"]:
            assert from_mexc(to_mexc(sym)) == sym


# ── dYdX ──────────────────────────────────────────────────────────────

class TestDydxSymbolMap:
    def test_to_dydx_btc(self):
        assert to_dydx("BTCUSDT") == "BTC-USD"

    def test_to_dydx_eth(self):
        assert to_dydx("ETHUSDT") == "ETH-USD"

    def test_to_dydx_ada(self):
        assert to_dydx("ADAUSDT") == "ADA-USD"

    def test_to_dydx_doge(self):
        assert to_dydx("DOGEUSDT") == "DOGE-USD"

    def test_to_dydx_sol(self):
        assert to_dydx("SOLUSDT") == "SOL-USD"

    def test_from_dydx_btc(self):
        assert from_dydx("BTC-USD") == "BTCUSDT"

    def test_from_dydx_eth(self):
        assert from_dydx("ETH-USD") == "ETHUSDT"

    def test_from_dydx_ada(self):
        assert from_dydx("ADA-USD") == "ADAUSDT"

    def test_roundtrip_dydx(self):
        for sym in ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]:
            assert from_dydx(to_dydx(sym)) == sym
