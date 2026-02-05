"""Tests for exchange gateway abstraction layer."""
import pytest
from gateway.base import ExchangeREST, ExchangeWS, OrderResponse, WalletInfo, InstrumentSpec, TickerInfo
from gateway.factory import create_rest, create_ws
from gateway.rest import BybitREST
from gateway.bybit_ws import BybitWS
from gateway.binance_rest import BinanceREST
from gateway.binance_ws import BinanceWS


class TestOrderResponse:
    def test_success(self):
        r = OrderResponse(success=True, order_id="123")
        assert r.success
        assert r.order_id == "123"
        assert r.error_code == 0

    def test_failure(self):
        r = OrderResponse(success=False, error_code=110001, error_msg="bad")
        assert not r.success
        assert r.order_id == ""
        assert r.error_msg == "bad"


class TestWalletInfo:
    def test_defaults(self):
        w = WalletInfo()
        assert w.equity == 0.0

    def test_with_equity(self):
        w = WalletInfo(equity=1000.0)
        assert w.equity == 1000.0


class TestInstrumentSpec:
    def test_defaults(self):
        s = InstrumentSpec(symbol="BTCUSDT")
        assert s.tick_size == 0.01
        assert s.min_qty == 0.001


class TestTickerInfo:
    def test_creation(self):
        t = TickerInfo(symbol="BTCUSDT", last_price=50000.0, turnover_24h=1e9)
        assert t.symbol == "BTCUSDT"
        assert t.last_price == 50000.0


class TestInterfaceConformance:
    """Verify both adapters implement all abstract methods."""

    def test_bybit_rest_is_exchange_rest(self):
        r = BybitREST("key", "secret")
        assert isinstance(r, ExchangeREST)

    def test_binance_rest_is_exchange_rest(self):
        r = BinanceREST("key", "secret")
        assert isinstance(r, ExchangeREST)

    def test_bybit_ws_is_exchange_ws(self):
        ws = BybitWS(["BTCUSDT"], {})
        assert isinstance(ws, ExchangeWS)

    def test_binance_ws_is_exchange_ws(self):
        ws = BinanceWS(["BTCUSDT"], {})
        assert isinstance(ws, ExchangeWS)

    def test_bybit_rest_has_all_methods(self):
        r = BybitREST("k", "s")
        for method in [
            "place_order", "cancel_order", "cancel_all_orders", "amend_order",
            "set_trading_stop", "set_leverage", "get_wallet_balance",
            "get_positions", "get_open_orders", "get_instruments", "get_tickers",
            "get_server_time", "get_klines", "cleanup_bracket", "close",
        ]:
            assert hasattr(r, method), f"BybitREST missing {method}"

    def test_binance_rest_has_all_methods(self):
        r = BinanceREST("k", "s")
        for method in [
            "place_order", "cancel_order", "cancel_all_orders", "amend_order",
            "set_trading_stop", "set_leverage", "get_wallet_balance",
            "get_positions", "get_open_orders", "get_instruments", "get_tickers",
            "get_server_time", "get_klines", "cleanup_bracket", "close",
        ]:
            assert hasattr(r, method), f"BinanceREST missing {method}"

    def test_bybit_ws_has_all_methods(self):
        ws = BybitWS(["BTCUSDT"], {})
        for method in [
            "start", "stop", "next_event", "drain_private",
            "latency_estimate_ms", "is_connected",
        ]:
            assert hasattr(ws, method), f"BybitWS missing {method}"

    def test_binance_ws_has_all_methods(self):
        ws = BinanceWS(["BTCUSDT"], {})
        for method in [
            "start", "stop", "next_event", "drain_private",
            "latency_estimate_ms", "is_connected",
        ]:
            assert hasattr(ws, method), f"BinanceWS missing {method}"


class TestFactory:
    def test_create_rest_bybit(self):
        r = create_rest("bybit", "key", "secret")
        assert isinstance(r, BybitREST)

    def test_create_rest_binance(self):
        r = create_rest("binance", "key", "secret")
        assert isinstance(r, BinanceREST)

    def test_create_rest_case_insensitive(self):
        r = create_rest("BYBIT", "key", "secret")
        assert isinstance(r, BybitREST)

    def test_create_rest_invalid(self):
        with pytest.raises(ValueError, match="Unsupported exchange"):
            create_rest("kraken", "key", "secret")

    def test_create_ws_bybit(self):
        ws = create_ws("bybit", ["BTCUSDT"], {})
        assert isinstance(ws, BybitWS)

    def test_create_ws_binance(self):
        ws = create_ws("binance", ["BTCUSDT"], {})
        assert isinstance(ws, BinanceWS)

    def test_create_ws_invalid(self):
        with pytest.raises(ValueError, match="Unsupported exchange"):
            create_ws("kraken", ["BTCUSDT"], {})

    def test_create_rest_testnet(self):
        r = create_rest("bybit", "key", "secret", testnet=True)
        assert r.testnet is True

    def test_create_rest_binance_testnet(self):
        r = create_rest("binance", "key", "secret", testnet=True)
        assert r.testnet is True
