"""Tests for market data structures: OrderBook and TradeTape."""
import pytest
from marketdata.book import OrderBook
from marketdata.tape import TradeTape
from core.types import Trade, Side


class TestOrderBook:
    def test_snapshot(self, sample_book):
        assert sample_book.best_bid() == 50000.0
        assert sample_book.best_ask() == 50000.1
        assert abs(sample_book.mid_price() - 50000.05) < 0.01

    def test_spread_ticks(self, sample_book):
        # spread = 0.1, tick_size = 0.10 → 1 tick
        assert abs(sample_book.spread_ticks() - 1.0) < 0.01

    def test_delta_update(self, sample_book):
        # Update: new bid at 50000.0 with larger size
        sample_book.on_delta([[50000.0, 3.0]], [], seq=2)
        assert sample_book.best_bid_size() == 3.0

    def test_delta_remove_level(self, sample_book):
        # Remove best ask
        sample_book.on_delta([], [[50000.1, 0]], seq=2)
        assert sample_book.best_ask() == 50000.2

    def test_bid_ask_volume(self, sample_book):
        bv = sample_book.bid_volume(3)
        assert bv == 1.5 + 2.0 + 3.0  # first 3 levels

        av = sample_book.ask_volume(3)
        assert av == 1.2 + 2.5 + 1.8

    def test_imbalance(self, sample_book):
        imb = sample_book.imbalance(5)
        # Bids = 1.5+2+3+1+0.5 = 8.0, Asks = 1.2+2.5+1.8+0.8+0.3 = 6.6
        assert imb > 0  # bids heavier

    def test_weighted_mid(self, sample_book):
        wmid = sample_book.weighted_mid()
        mid = sample_book.mid_price()
        # With more bid size, wmid should be slightly above mid
        assert wmid > 0

    def test_is_fresh(self, sample_book):
        assert sample_book.is_fresh()

    def test_stale_book(self):
        book = OrderBook(symbol="TEST", depth=5, tick_size=0.01)
        assert not book.is_fresh()  # no data yet

    def test_empty_book(self):
        book = OrderBook()
        assert book.best_bid() == 0.0
        assert book.best_ask() == 0.0
        assert book.mid_price() == 0.0
        # Empty book: spread=0-0=0, tick_size=0.1 → 0/0.1=0
        assert book.spread_ticks() == 0.0

    def test_crossed_book_resets(self):
        """A crossed book (bid >= ask) should trigger reset."""
        book = OrderBook(symbol="TEST", depth=5, tick_size=0.01)
        book.on_snapshot([[100, 1]], [[101, 1]], seq=1)
        assert book.is_fresh()
        # Force a crossed book via delta: bid jumps above ask
        book.on_delta([[102, 1]], [], seq=2)
        # Book should have been reset
        assert book.last_update_ts == 0
        assert book.needs_snapshot is True
        assert not book.is_fresh()

    def test_seq_gap_resets(self):
        """A sequence gap should trigger reset."""
        book = OrderBook(symbol="TEST", depth=5, tick_size=0.01)
        book.on_snapshot([[100, 1]], [[101, 1]], seq=10)
        assert book.is_fresh()
        # Delta with gap: expected seq=11 but got 15
        book.on_delta([[99, 1]], [], seq=15)
        assert book.last_update_ts == 0
        assert book.needs_snapshot is True

    def test_warmup_after_reset(self):
        """After reset + resnapshot, need WARMUP_DELTAS valid deltas."""
        book = OrderBook(symbol="TEST", depth=5, tick_size=0.01)
        book.on_snapshot([[100, 1]], [[101, 1]], seq=1)
        assert book.is_fresh()
        # Force crossed → reset
        book.on_delta([[102, 1]], [], seq=2)
        assert not book.is_fresh()
        # Recovery snapshot — warmup_count = 0
        book.on_snapshot([[100, 1]], [[101, 1]], seq=20)
        assert not book.is_fresh()  # warmup not done
        # Feed valid deltas
        for i in range(OrderBook.WARMUP_DELTAS):
            book.on_delta([[99.5 - i * 0.01, 0.5]], [], seq=21 + i)
        assert book.is_fresh()  # now warmed up

    def test_deltas_discarded_before_snapshot(self):
        """After reset, deltas without a snapshot should be discarded."""
        book = OrderBook(symbol="TEST", depth=5, tick_size=0.01)
        book.on_snapshot([[100, 1]], [[101, 1]], seq=1)
        book.on_delta([[102, 1]], [], seq=2)  # crosses → reset
        assert book.needs_snapshot
        # Deltas before snapshot should be discarded
        book.on_delta([[99, 1]], [], seq=3)
        assert book.last_update_ts == 0  # still reset

    def test_cancel_tracking(self, sample_book):
        # Remove a bid level — should be tracked as cancel
        sample_book.on_delta([[49999.9, 0]], [], seq=2)
        cancel_vol = sample_book.cancel_volume("bid", 1000)
        assert cancel_vol > 0


class TestTradeTape:
    def test_add_trade(self):
        tape = TradeTape()
        trade = Trade("1", "BTCUSDT", 50000.0, 0.01, Side.BUY, 1700000000000)
        tape.add_trade(trade)
        assert len(tape.trades) == 1
        assert tape.last_price() == 50000.0

    def test_ema_initialization(self):
        tape = TradeTape()
        trade = Trade("1", "BTCUSDT", 50000.0, 0.01, Side.BUY, 1700000000000)
        tape.add_trade(trade)
        assert tape.ema_fast == 50000.0
        assert tape.ema_slow == 50000.0

    def test_ema_tracks_price(self):
        tape = TradeTape()
        for i in range(20):
            price = 50000.0 + i * 10  # steadily increasing
            trade = Trade(str(i), "BTCUSDT", price, 0.01, Side.BUY, 1700000000000 + i * 1000)
            tape.add_trade(trade)
        # Fast EMA should be closer to recent price
        assert tape.ema_fast > tape.ema_slow

    def test_median_trade_size(self, sample_tape):
        median = sample_tape.median_trade_size()
        assert median > 0

    def test_vwap(self, sample_tape):
        vwap = sample_tape.vwap(50)
        assert vwap > 0

    def test_buy_sell_ratio(self, sample_tape):
        ratio = sample_tape.buy_sell_ratio(10000)
        assert 0.0 <= ratio <= 1.0

    def test_large_trades_empty_when_small(self):
        tape = TradeTape()
        for i in range(10):
            trade = Trade(str(i), "BTCUSDT", 50000.0, 0.001, Side.BUY, 1700000000000 + i * 100)
            tape.add_trade(trade)
        # All trades are tiny, none should be "large"
        whales = tape.large_trades(10000, multiplier=5.0, min_usd=50000)
        assert len(whales) == 0

    def test_last_trade(self, sample_tape):
        last = sample_tape.last_trade()
        assert last is not None
        assert last.trade_id == "99"
