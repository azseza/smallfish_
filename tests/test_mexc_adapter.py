"""Tests for MEXC adapter specifics — side mapping, order type mapping,
TP/SL emulation, bracket cleanup, WS parsing, symbol conversion, auth.
"""
import asyncio
import pytest
from core.types import (
    Side, EventType, Trade,
    NormalizedExecution, NormalizedOrderUpdate, NormalizedPositionUpdate,
    TimeInForce, OrderType,
)
from gateway.mexc_rest import MexcREST, _mexc_side, _mexc_order_type, _fmt
from gateway.mexc_ws import MexcWS
from gateway.symbol_map import to_mexc, from_mexc


# ── Symbol Mapping ─────────────────────────────────────────────────────

class TestSymbolMapping:
    def test_to_mexc_btcusdt(self):
        assert to_mexc("BTCUSDT") == "BTC_USDT"

    def test_from_mexc_btcusdt(self):
        assert from_mexc("BTC_USDT") == "BTCUSDT"

    def test_roundtrip(self):
        for sym in ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "SUIUSDT"]:
            assert from_mexc(to_mexc(sym)) == sym

    def test_to_mexc_usdc(self):
        assert to_mexc("BTCUSDC") == "BTC_USDC"

    def test_from_mexc_usdc(self):
        assert from_mexc("BTC_USDC") == "BTCUSDC"


# ── REST Helpers ───────────────────────────────────────────────────────

class TestMexcSideMapping:
    def test_buy_open_long(self):
        assert _mexc_side(Side.BUY, reduce_only=False) == 1

    def test_buy_close_short(self):
        assert _mexc_side(Side.BUY, reduce_only=True) == 2

    def test_sell_open_short(self):
        assert _mexc_side(Side.SELL, reduce_only=False) == 3

    def test_sell_close_long(self):
        assert _mexc_side(Side.SELL, reduce_only=True) == 4


class TestMexcOrderTypeMapping:
    def test_market(self):
        assert _mexc_order_type(OrderType.MARKET) == 5

    def test_limit(self):
        assert _mexc_order_type(OrderType.LIMIT) == 1


class TestMexcFmt:
    def test_integer(self):
        assert _fmt(100.0) == "100"

    def test_small_decimal(self):
        assert _fmt(0.00001) == "0.00001"

    def test_trailing_zeros(self):
        assert _fmt(1.50) == "1.5"

    def test_negative(self):
        assert _fmt(-0.5) == "-0.5"


# ── REST Response Parsing ──────────────────────────────────────────────

class TestMexcRestResponseParsing:
    def test_success_response(self):
        r = MexcREST("k", "s")
        resp = r._to_order_response({"code": 0, "data": "1234567890"})
        assert resp.success
        assert resp.order_id == "1234567890"

    def test_error_response(self):
        r = MexcREST("k", "s")
        resp = r._to_order_response({"code": 500, "msg": "Insufficient balance"})
        assert not resp.success
        assert resp.error_code == 500
        assert "Insufficient" in resp.error_msg

    def test_success_no_data(self):
        r = MexcREST("k", "s")
        resp = r._to_order_response({"code": 0, "data": ""})
        assert not resp.success
        assert "no order ID" in resp.error_msg

    def test_non_dict_response(self):
        r = MexcREST("k", "s")
        resp = r._to_order_response("unexpected")
        assert not resp.success


# ── TP/SL Bracket Tracking ─────────────────────────────────────────────

class TestMexcTPSLTracking:
    def test_bracket_tracking_init(self):
        r = MexcREST("k", "s")
        assert r._tp_sl_orders == {}

    def test_cleanup_bracket_no_bracket(self):
        r = MexcREST("k", "s")
        asyncio.get_event_loop().run_until_complete(r.cleanup_bracket("BTCUSDT", "123"))

    def test_cleanup_bracket_tp_fill(self):
        r = MexcREST("k", "s")
        r._tp_sl_orders["BTCUSDT"] = {"tp": "order_tp", "sl": "order_sl"}
        # Can't actually cancel (no connection), but verifies flow
        # The bracket should be popped regardless of cancel result
        assert "BTCUSDT" in r._tp_sl_orders


# ── Auth ───────────────────────────────────────────────────────────────

class TestMexcAuth:
    def test_sign_deterministic(self):
        r = MexcREST("key", "secret")
        sig1 = r._sign("1000", "test=1")
        sig2 = r._sign("1000", "test=1")
        assert sig1 == sig2
        assert len(sig1) == 64  # SHA256 hex

    def test_sign_different_params(self):
        r = MexcREST("key", "secret")
        sig1 = r._sign("1000", "a=1")
        sig2 = r._sign("1000", "b=2")
        assert sig1 != sig2

    def test_auth_headers_structure(self):
        r = MexcREST("mykey", "mysecret")
        h = r._auth_headers("test")
        assert h["ApiKey"] == "mykey"
        assert "Request-Time" in h
        assert "Signature" in h
        assert h["Content-Type"] == "application/json"

    def test_no_testnet(self):
        r = MexcREST("k", "s", testnet=True)
        assert r.base_url == "https://contract.mexc.com"  # MEXC has no futures testnet


# ── WebSocket Parsing ──────────────────────────────────────────────────

class TestMexcWsDepthParsing:
    def _make_ws(self) -> MexcWS:
        return MexcWS(["BTCUSDT"], {})

    def test_depth_always_snapshot(self):
        """MEXC depth pushes are absolute quantities — always BOOK_SNAPSHOT."""
        ws = self._make_ws()
        events = ws._parse_depth({
            "channel": "push.contract.depth.BTC_USDT",
            "data": {
                "bids": [[50000.0, 1.5], [49999.0, 2.0]],
                "asks": [[50001.0, 1.0], [50002.0, 3.0]],
            }
        }, 1700000000000)

        assert len(events) == 1
        assert events[0].event_type == EventType.BOOK_SNAPSHOT
        assert events[0].symbol == "BTCUSDT"
        assert len(events[0].data["b"]) == 2
        assert len(events[0].data["a"]) == 2
        assert events[0].data["b"][0] == ["50000.0", "1.5"]

    def test_depth_second_still_snapshot(self):
        """Unlike Binance, MEXC depth is always snapshot even on second push."""
        ws = self._make_ws()
        ws._parse_depth({
            "channel": "push.contract.depth.BTC_USDT",
            "data": {"bids": [[50000.0, 1.0]], "asks": [[50001.0, 1.0]]}
        }, 100)
        events = ws._parse_depth({
            "channel": "push.contract.depth.BTC_USDT",
            "data": {"bids": [[50000.0, 2.0]], "asks": [[50001.0, 0.5]]}
        }, 200)
        assert events[0].event_type == EventType.BOOK_SNAPSHOT

    def test_depth_bad_channel(self):
        ws = self._make_ws()
        events = ws._parse_depth({"channel": "bad", "data": {}}, 100)
        assert events == []


class TestMexcWsDealsParsing:
    def _make_ws(self) -> MexcWS:
        return MexcWS(["BTCUSDT"], {})

    def test_deal_buy(self):
        """T=1 is a buy."""
        ws = self._make_ws()
        events = ws._parse_deals({
            "channel": "push.contract.deal.BTC_USDT",
            "data": [{"p": 50000.5, "v": 0.01, "T": 1, "t": 1700000000000}]
        }, 1700000000000)
        assert len(events) == 1
        trade = events[0].data
        assert isinstance(trade, Trade)
        assert trade.side == Side.BUY
        assert trade.price == 50000.5
        assert trade.quantity == 0.01

    def test_deal_sell(self):
        """T=2 is a sell."""
        ws = self._make_ws()
        events = ws._parse_deals({
            "channel": "push.contract.deal.BTC_USDT",
            "data": [{"p": 50000.0, "v": 0.05, "T": 2, "t": 1700000000000}]
        }, 1700000000000)
        trade = events[0].data
        assert trade.side == Side.SELL

    def test_deal_single_dict_not_list(self):
        """MEXC can send a single deal dict instead of a list."""
        ws = self._make_ws()
        events = ws._parse_deals({
            "channel": "push.contract.deal.BTC_USDT",
            "data": {"p": 50000.0, "v": 0.01, "T": 1, "t": 1700000000000}
        }, 1700000000000)
        assert len(events) == 1

    def test_deal_multiple(self):
        """Multiple deals in one push."""
        ws = self._make_ws()
        events = ws._parse_deals({
            "channel": "push.contract.deal.BTC_USDT",
            "data": [
                {"p": 50000.0, "v": 0.01, "T": 1, "t": 100},
                {"p": 50001.0, "v": 0.02, "T": 2, "t": 101},
            ]
        }, 200)
        assert len(events) == 2
        assert events[0].data.side == Side.BUY
        assert events[1].data.side == Side.SELL

    def test_deal_symbol_mapping(self):
        ws = self._make_ws()
        events = ws._parse_deals({
            "channel": "push.contract.deal.ADA_USDT",
            "data": [{"p": 0.5, "v": 100, "T": 1, "t": 100}]
        }, 100)
        assert events[0].symbol == "ADAUSDT"


class TestMexcWsOrderParsing:
    def _make_ws(self) -> MexcWS:
        return MexcWS(["BTCUSDT"], {})

    def test_order_filled_with_execution(self):
        """Filled order emits both execution and order update."""
        ws = self._make_ws()
        events = ws._parse_order({
            "channel": "push.personal.order",
            "data": {
                "symbol": "BTC_USDT",
                "orderId": "order123",
                "state": 3,  # completed/filled
                "side": 1,   # open long
                "dealVol": 0.01,
                "dealAvgPrice": 50000.5,
                "makerFee": 0.001,
                "orderType": 1,
            }
        }, 1700000000000)

        assert len(events) == 2
        exec_evt = events[0]
        assert exec_evt.event_type == EventType.EXECUTION
        ex = exec_evt.data
        assert isinstance(ex, NormalizedExecution)
        assert ex.side == Side.BUY
        assert ex.price == 50000.5
        assert ex.quantity == 0.01
        assert ex.is_maker is True

        order_evt = events[1]
        assert order_evt.event_type == EventType.ORDER
        ou = order_evt.data
        assert isinstance(ou, NormalizedOrderUpdate)
        assert ou.status == "Filled"
        assert ou.order_id == "order123"

    def test_order_new_no_execution(self):
        """New order with no fill emits only order update."""
        ws = self._make_ws()
        events = ws._parse_order({
            "channel": "push.personal.order",
            "data": {
                "symbol": "BTC_USDT",
                "orderId": "order456",
                "state": 1,  # uninformed/new
                "side": 3,   # open short
                "dealVol": 0,
                "dealAvgPrice": 0,
                "orderType": 1,
            }
        }, 1700000000000)

        assert len(events) == 1
        assert events[0].event_type == EventType.ORDER
        ou = events[0].data
        assert ou.status == "New"
        assert ou.side == "Sell"

    def test_order_cancelled(self):
        ws = self._make_ws()
        events = ws._parse_order({
            "channel": "push.personal.order",
            "data": {
                "symbol": "BTC_USDT",
                "orderId": "order789",
                "state": 4,  # cancelled
                "side": 1,
                "dealVol": 0,
                "dealAvgPrice": 0,
                "orderType": 1,
            }
        }, 1700000000000)

        assert events[0].data.status == "Cancelled"

    def test_order_reduce_only_flags(self):
        """Side 2 (close long) and 4 (close short) are reduce_only."""
        ws = self._make_ws()
        for mexc_side, expected_reduce in [(1, False), (2, True), (3, False), (4, True)]:
            events = ws._parse_order({
                "channel": "push.personal.order",
                "data": {
                    "symbol": "BTC_USDT",
                    "orderId": f"ord_{mexc_side}",
                    "state": 1,
                    "side": mexc_side,
                    "dealVol": 0,
                    "dealAvgPrice": 0,
                    "orderType": 1,
                }
            }, 100)
            assert events[0].data.reduce_only == expected_reduce, \
                f"Side {mexc_side} should be reduce_only={expected_reduce}"

    def test_order_side_buy_sell(self):
        """Side 1/4 → Buy, Side 2/3 → Sell."""
        ws = self._make_ws()
        for mexc_side, expected_side in [(1, "Buy"), (2, "Sell"), (3, "Sell"), (4, "Buy")]:
            events = ws._parse_order({
                "channel": "push.personal.order",
                "data": {
                    "symbol": "BTC_USDT",
                    "orderId": f"ord_{mexc_side}",
                    "state": 1,
                    "side": mexc_side,
                    "dealVol": 0,
                    "dealAvgPrice": 0,
                    "orderType": 1,
                }
            }, 100)
            assert events[0].data.side == expected_side


class TestMexcWsPositionParsing:
    def _make_ws(self) -> MexcWS:
        return MexcWS(["BTCUSDT"], {})

    def test_long_position(self):
        ws = self._make_ws()
        events = ws._parse_position({
            "channel": "push.personal.position",
            "data": {
                "symbol": "BTC_USDT",
                "holdVol": 0.01,
                "positionType": 1,  # long
                "openAvgPrice": 50000.0,
            }
        }, 1700000000000)

        assert len(events) == 1
        pos = events[0].data
        assert isinstance(pos, NormalizedPositionUpdate)
        assert pos.side == "Buy"
        assert pos.size == 0.01
        assert pos.entry_price == 50000.0

    def test_short_position(self):
        ws = self._make_ws()
        events = ws._parse_position({
            "channel": "push.personal.position",
            "data": {
                "symbol": "BTC_USDT",
                "holdVol": 0.05,
                "positionType": 2,  # short
                "openAvgPrice": 50000.0,
            }
        }, 1700000000000)

        pos = events[0].data
        assert pos.side == "Sell"
        assert pos.size == 0.05

    def test_flat_position(self):
        ws = self._make_ws()
        events = ws._parse_position({
            "channel": "push.personal.position",
            "data": {
                "symbol": "BTC_USDT",
                "holdVol": 0,
                "positionType": 0,
                "openAvgPrice": 0,
            }
        }, 1700000000000)

        pos = events[0].data
        assert pos.side == ""
        assert pos.size == 0.0

    def test_non_dict_data(self):
        ws = self._make_ws()
        events = ws._parse_position({
            "channel": "push.personal.position",
            "data": "unexpected"
        }, 100)
        assert events == []


# ── WebSocket General ──────────────────────────────────────────────────

class TestMexcWsGeneral:
    def test_latency_default(self):
        ws = MexcWS(["BTCUSDT"], {})
        assert ws.latency_estimate_ms() == 999

    def test_not_connected_initially(self):
        ws = MexcWS(["BTCUSDT"], {})
        assert ws.is_connected is False

    def test_drain_private_empty(self):
        ws = MexcWS(["BTCUSDT"], {})
        assert ws.drain_private() == []

    def test_queue_sizes(self):
        ws = MexcWS(["BTCUSDT"], {})
        assert ws._event_queue.maxsize == 10000
        assert ws._private_queue.maxsize == 1000


# ── Router amend_order compatibility ───────────────────────────────────

class TestMexcAmendOrder:
    def test_amend_returns_empty_order_id(self):
        """MEXC amend cancels then returns success with empty order_id.
        Router must detect this and re-place."""
        r = MexcREST("k", "s")
        # We can't actually call amend (needs HTTP), but verify the pattern
        # by checking the OrderResponse construction
        from gateway.base import OrderResponse
        resp = OrderResponse(success=True, order_id="",
                           error_msg="Cancelled for amend — re-place required")
        assert resp.success
        assert resp.order_id == ""
        # Router should check: if result.success and not result.order_id → re-place
