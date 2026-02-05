"""Tests for Binance adapter specifics — side inversion, TIF mapping,
TP/SL emulation, bracket cleanup, WS parsing, kline parsing.
"""
import pytest
from core.types import (
    Side, EventType, Trade,
    NormalizedExecution, NormalizedOrderUpdate, NormalizedPositionUpdate,
)
from gateway.binance_rest import BinanceREST, _tif_to_binance, _otype_to_binance, _fmt
from gateway.binance_ws import BinanceWS, _STATUS_MAP, _STOP_TYPE_MAP
from core.types import TimeInForce, OrderType


# ── REST helpers ────────────────────────────────────────────────────────

class TestBinanceRestHelpers:
    def test_tif_post_only_maps_to_gtx(self):
        assert _tif_to_binance(TimeInForce.POST_ONLY) == "GTX"

    def test_tif_gtc_unchanged(self):
        assert _tif_to_binance(TimeInForce.GTC) == "GTC"

    def test_tif_ioc_unchanged(self):
        assert _tif_to_binance(TimeInForce.IOC) == "IOC"

    def test_otype_limit(self):
        assert _otype_to_binance(OrderType.LIMIT) == "LIMIT"

    def test_otype_market(self):
        assert _otype_to_binance(OrderType.MARKET) == "MARKET"

    def test_fmt_integer(self):
        assert _fmt(8333.0) == "8333"

    def test_fmt_small_decimal(self):
        assert _fmt(0.00001) == "0.00001"

    def test_fmt_trailing_zeros(self):
        assert _fmt(1.50) == "1.5"


class TestBinanceRestResponseParsing:
    def test_success_response(self):
        r = BinanceREST("k", "s")
        resp = r._to_order_response({"orderId": 12345, "status": "NEW"})
        assert resp.success
        assert resp.order_id == "12345"

    def test_error_response(self):
        r = BinanceREST("k", "s")
        resp = r._to_order_response({"code": -1021, "msg": "Timestamp for this request is outside of the recvWindow."})
        assert not resp.success
        assert resp.error_code == -1021
        assert "recvWindow" in resp.error_msg

    def test_empty_response(self):
        r = BinanceREST("k", "s")
        resp = r._to_order_response({})
        assert resp.success
        assert resp.order_id == ""


class TestBinanceTPSLTracking:
    def test_bracket_tracking_init(self):
        r = BinanceREST("k", "s")
        assert r._tp_sl_orders == {}

    def test_bracket_cleanup_no_bracket(self):
        """cleanup_bracket with no tracked bracket does nothing."""
        import asyncio
        r = BinanceREST("k", "s")
        # Should not raise
        asyncio.get_event_loop().run_until_complete(r.cleanup_bracket("BTCUSDT", "123"))


class TestBinanceAuth:
    def test_sign_deterministic(self):
        r = BinanceREST("key", "secret")
        sig1 = r._sign({"symbol": "BTCUSDT", "timestamp": "1000"})
        sig2 = r._sign({"symbol": "BTCUSDT", "timestamp": "1000"})
        assert sig1 == sig2
        assert len(sig1) == 64  # SHA256 hex

    def test_auth_headers(self):
        r = BinanceREST("mykey", "mysecret")
        h = r._auth_headers()
        assert h["X-MBX-APIKEY"] == "mykey"

    def test_signed_params_adds_timestamp(self):
        r = BinanceREST("k", "s")
        params = r._signed_params({"symbol": "BTCUSDT"})
        assert "timestamp" in params
        assert "signature" in params
        assert "recvWindow" in params

    def test_testnet_url(self):
        r = BinanceREST("k", "s", testnet=True)
        assert "demo-fapi" in r.base_url or "testnet" in r.base_url

    def test_mainnet_url(self):
        r = BinanceREST("k", "s", testnet=False)
        assert r.base_url == "https://fapi.binance.com"


# ── WebSocket parsing ───────────────────────────────────────────────────

class TestBinanceWsParsing:
    def _make_ws(self) -> BinanceWS:
        return BinanceWS(["BTCUSDT"], {})

    def test_aggtrade_side_inversion_sell(self):
        """m=true means buyer is maker → trade was a SELL (taker sold)."""
        ws = self._make_ws()
        events = ws._parse_public("btcusdt@aggTrade", {
            "a": 123456,
            "p": "50000.5",
            "q": "0.01",
            "T": 1700000000000,
            "m": True,
        }, 1700000000000)
        assert len(events) == 1
        trade = events[0].data
        assert isinstance(trade, Trade)
        assert trade.side == Side.SELL

    def test_aggtrade_side_buy(self):
        """m=false means buyer is taker → trade was a BUY."""
        ws = self._make_ws()
        events = ws._parse_public("btcusdt@aggTrade", {
            "a": 123457,
            "p": "50001.0",
            "q": "0.02",
            "T": 1700000000001,
            "m": False,
        }, 1700000000001)
        trade = events[0].data
        assert trade.side == Side.BUY

    def test_aggtrade_symbol_uppercase(self):
        """Symbols are lowercased in streams but should be uppercase in events."""
        ws = self._make_ws()
        events = ws._parse_public("ethusdt@aggTrade", {
            "a": 1, "p": "3000", "q": "1", "T": 100, "m": False
        }, 100)
        assert events[0].symbol == "ETHUSDT"

    def test_depth_first_is_snapshot(self):
        """First depth message for a symbol should be BOOK_SNAPSHOT."""
        ws = self._make_ws()
        events = ws._parse_public("btcusdt@depth20@100ms", {
            "b": [["50000.0", "1.0"]], "a": [["50001.0", "1.0"]]
        }, 100)
        assert events[0].event_type == EventType.BOOK_SNAPSHOT

    def test_depth_second_is_delta(self):
        """Second depth message should be BOOK_DELTA."""
        ws = self._make_ws()
        ws._parse_public("btcusdt@depth20@100ms", {
            "b": [["50000.0", "1.0"]], "a": [["50001.0", "1.0"]]
        }, 100)
        events = ws._parse_public("btcusdt@depth20@100ms", {
            "b": [["50000.0", "1.1"]], "a": [["50001.0", "0.9"]]
        }, 200)
        assert events[0].event_type == EventType.BOOK_DELTA

    def test_depth_different_symbols_independent(self):
        """First message per symbol is snapshot, independent of other symbols."""
        ws = self._make_ws()
        ws._parse_public("btcusdt@depth20@100ms", {
            "b": [["50000.0", "1.0"]], "a": [["50001.0", "1.0"]]
        }, 100)
        events = ws._parse_public("ethusdt@depth20@100ms", {
            "b": [["3000.0", "1.0"]], "a": [["3001.0", "1.0"]]
        }, 200)
        assert events[0].event_type == EventType.BOOK_SNAPSHOT


class TestBinanceWsPrivateParsing:
    def _make_ws(self) -> BinanceWS:
        return BinanceWS(["BTCUSDT"], {})

    def test_order_trade_update_with_fill(self):
        """ORDER_TRADE_UPDATE with fill emits both execution and order update."""
        ws = self._make_ws()
        events = ws._parse_private({
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "BTCUSDT",
                "S": "BUY",
                "X": "FILLED",
                "i": 100001,
                "t": 500001,
                "L": "50000.5",
                "l": "0.01",
                "ap": "50000.5",
                "z": "0.01",
                "m": True,
                "o": "LIMIT",
                "ot": "LIMIT",
                "R": False,
            }
        }, 1700000000000)

        assert len(events) == 2
        exec_evt = events[0]
        order_evt = events[1]

        assert exec_evt.event_type == EventType.EXECUTION
        ex = exec_evt.data
        assert isinstance(ex, NormalizedExecution)
        assert ex.side == Side.BUY
        assert ex.price == 50000.5
        assert ex.quantity == 0.01
        assert ex.is_maker is True

        assert order_evt.event_type == EventType.ORDER
        ou = order_evt.data
        assert isinstance(ou, NormalizedOrderUpdate)
        assert ou.status == "Filled"
        assert ou.filled_qty == 0.01

    def test_order_trade_update_no_fill(self):
        """ORDER_TRADE_UPDATE with no fill emits only order update."""
        ws = self._make_ws()
        events = ws._parse_private({
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "X": "NEW",
                "i": 100002,
                "t": 0,
                "L": "0",
                "l": "0",
                "ap": "0",
                "z": "0",
                "m": False,
                "o": "LIMIT",
                "ot": "LIMIT",
                "R": False,
            }
        }, 1700000000000)

        assert len(events) == 1
        assert events[0].event_type == EventType.ORDER

    def test_stop_order_type_mapping(self):
        """TAKE_PROFIT_MARKET → TakeProfit, STOP_MARKET → StopLoss."""
        ws = self._make_ws()
        events = ws._parse_private({
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "X": "FILLED",
                "i": 100003,
                "t": 500002,
                "L": "51000",
                "l": "0.01",
                "ap": "51000",
                "z": "0.01",
                "m": False,
                "o": "TAKE_PROFIT_MARKET",
                "ot": "TAKE_PROFIT_MARKET",
                "R": True,
            }
        }, 1700000000000)

        order_evt = events[1]
        ou = order_evt.data
        assert ou.stop_order_type == "TakeProfit"
        assert ou.reduce_only is True

    def test_account_update_long_position(self):
        """ACCOUNT_UPDATE with positive size → Buy side."""
        ws = self._make_ws()
        events = ws._parse_private({
            "e": "ACCOUNT_UPDATE",
            "a": {
                "P": [{
                    "s": "BTCUSDT",
                    "pa": "0.01",
                    "ep": "50000.0",
                }]
            }
        }, 1700000000000)

        assert len(events) == 1
        pos = events[0].data
        assert isinstance(pos, NormalizedPositionUpdate)
        assert pos.side == "Buy"
        assert pos.size == 0.01
        assert pos.entry_price == 50000.0

    def test_account_update_short_position(self):
        """ACCOUNT_UPDATE with negative size → Sell side."""
        ws = self._make_ws()
        events = ws._parse_private({
            "e": "ACCOUNT_UPDATE",
            "a": {
                "P": [{
                    "s": "ETHUSDT",
                    "pa": "-0.5",
                    "ep": "3000.0",
                }]
            }
        }, 1700000000000)

        pos = events[0].data
        assert pos.side == "Sell"
        assert pos.size == 0.5

    def test_account_update_flat(self):
        """ACCOUNT_UPDATE with zero size → empty side."""
        ws = self._make_ws()
        events = ws._parse_private({
            "e": "ACCOUNT_UPDATE",
            "a": {
                "P": [{
                    "s": "BTCUSDT",
                    "pa": "0",
                    "ep": "0",
                }]
            }
        }, 1700000000000)

        pos = events[0].data
        assert pos.side == ""
        assert pos.size == 0.0


class TestStatusMapping:
    def test_all_binance_statuses_mapped(self):
        for binance_status in ["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
            assert binance_status in _STATUS_MAP

    def test_stop_types_mapped(self):
        assert _STOP_TYPE_MAP["TAKE_PROFIT_MARKET"] == "TakeProfit"
        assert _STOP_TYPE_MAP["STOP_MARKET"] == "StopLoss"
        assert _STOP_TYPE_MAP["TAKE_PROFIT"] == "TakeProfit"
        assert _STOP_TYPE_MAP["STOP"] == "StopLoss"


class TestBinanceWsUrls:
    def test_public_url_building(self):
        ws = BinanceWS(["BTCUSDT", "ETHUSDT"], {})
        url = ws._build_public_url()
        assert "btcusdt@depth20@100ms" in url
        assert "ethusdt@aggTrade" in url
        assert "fstream.binance.com" in url

    def test_testnet_url(self):
        ws = BinanceWS(["BTCUSDT"], {}, testnet=True)
        url = ws._build_public_url()
        assert "binancefuture.com" in url

    def test_latency_default(self):
        ws = BinanceWS(["BTCUSDT"], {})
        assert ws.latency_estimate_ms() == 999

    def test_not_connected_initially(self):
        ws = BinanceWS(["BTCUSDT"], {})
        assert ws.is_connected is False

    def test_drain_private_empty(self):
        ws = BinanceWS(["BTCUSDT"], {})
        assert ws.drain_private() == []
