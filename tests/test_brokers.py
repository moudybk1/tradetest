"""Order-building tests for the real-money brokers, against fake exchange APIs."""

import sys
from types import SimpleNamespace

import pytest

from xau_agent.brokers.base import OrderError
from xau_agent.brokers.hyperliquid import HyperliquidBroker
from xau_agent.config import HyperliquidConfig, MT5Config

OK_FILL = {"status": "ok", "response": {"type": "order", "data": {"statuses": [
    {"filled": {"totalSz": "0.85", "avgPx": "4012.3", "oid": 1}}]}}}
OK_RESTING = {"status": "ok", "response": {"type": "order", "data": {"statuses": [
    {"resting": {"oid": 2}}, {"resting": {"oid": 3}}]}}}


class FakeExchange:
    def __init__(self, tpsl_response=OK_RESTING):
        self.tpsl_response = tpsl_response
        self.market_opens, self.bulk, self.closes, self.cancels = [], [], [], []

    def market_open(self, coin, is_buy, sz, px, slippage):
        self.market_opens.append((coin, is_buy, sz))
        return OK_FILL

    def bulk_orders(self, orders):
        self.bulk.append(orders)
        return self.tpsl_response

    def market_close(self, coin):
        self.closes.append(coin)
        return OK_FILL

    def bulk_cancel(self, reqs):
        self.cancels.extend(reqs)


class FakeInfo:
    def __init__(self, szi="0", orders=()):
        self.szi = szi
        self.orders = list(orders)

    def user_state(self, address, dex):
        return {"marginSummary": {"accountValue": "10000"},
                "assetPositions": [{"position": {"coin": "PAXG", "szi": self.szi, "entryPx": "4012.3"}}]}

    def open_orders(self, address, dex):
        return self.orders


def hl(exchange=None, info=None, sz_decimals=3):
    b = object.__new__(HyperliquidBroker)
    b.cfg = HyperliquidConfig(slippage=0.003)
    b.coin, b.dex, b.address = "PAXG", "", "0xabc"
    b.sz_decimals = sz_decimals
    b.exchange = exchange
    b.info = info or FakeInfo()
    return b


def test_hl_size_rounds_down_and_respects_min_notional():
    b = hl()
    assert b.size_for_risk(50, 4012.3, 4006.4) == 8.474  # 50 / 5.9 = 8.4745... floored
    assert b.size_for_risk(0.01, 4012.3, 4006.4) == 0.0  # below $10 notional


def test_hl_price_rounding_5_significant_figures():
    assert hl(sz_decimals=3).round_px(4012.3456) == 4012.3
    assert hl(sz_decimals=2).round_px(0.123456) == 0.1235


def test_hl_bracket_places_stop_and_target_reduce_only():
    ex = FakeExchange()
    fill = hl(ex).open_with_bracket("long", 0.85, 4000.0, 4036.0)
    assert fill.size == 0.85 and fill.price == 4012.3
    sl, tp = ex.bulk[0]
    assert sl["order_type"]["trigger"] == {"triggerPx": 4000.0, "isMarket": True, "tpsl": "sl"}
    assert tp["order_type"]["trigger"]["tpsl"] == "tp"
    assert sl["reduce_only"] and tp["reduce_only"]
    assert not sl["is_buy"] and sl["sz"] == 0.85
    assert sl["limit_px"] < 4000.0  # worst acceptable sell price is below the trigger


def test_hl_failed_stop_closes_position():
    bad = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "invalid price"}]}}}
    ex = FakeExchange(tpsl_response=bad)
    with pytest.raises(OrderError):
        hl(ex).open_with_bracket("short", 0.85, 4020.0, 3990.0)
    assert ex.closes == ["PAXG"]  # never left without a stop


def test_hl_read_only_broker_refuses_orders():
    with pytest.raises(OrderError, match="read-only"):
        hl(exchange=None).open_with_bracket("long", 1, 1, 2)


def test_hl_sync_cancels_leftover_orders_when_flat():
    ex = FakeExchange()
    b = hl(ex, FakeInfo(szi="0", orders=[{"coin": "PAXG", "oid": 9}, {"coin": "BTC", "oid": 10}]))
    assert b.sync() == ["cancelled 1 leftover PAXG order(s)"]
    assert ex.cancels == [{"coin": "PAXG", "oid": 9}]


def test_hl_sync_keeps_orders_while_in_position():
    ex = FakeExchange()
    b = hl(ex, FakeInfo(szi="0.85", orders=[{"coin": "PAXG", "oid": 9}]))
    assert b.sync() == [] and ex.cancels == []
    assert b.position().side == "long"


# ---- MT5 ---------------------------------------------------------------------------------------
class FakeMT5(SimpleNamespace):
    def __init__(self):
        super().__init__(
            TIMEFRAME_M15=15, ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, POSITION_TYPE_BUY=0,
            TRADE_ACTION_DEAL=1, ORDER_TIME_GTC=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1,
            ORDER_FILLING_RETURN=2, TRADE_RETCODE_DONE=10009,
        )
        self.sent = []

    def initialize(self, *args, **kwargs):
        return True

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info(self, symbol):
        return SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, volume_step=0.01, volume_min=0.01,
                               volume_max=50, trade_contract_size=100, trade_stops_level=10, point=0.01,
                               digits=2, filling_mode=2)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=4012.0, ask=4012.3, time=0)

    def order_calc_profit(self, order_type, symbol, volume, open_px, close_px):
        return (close_px - open_px) * 100 * volume if order_type == 0 else (open_px - close_px) * 100 * volume

    def order_send(self, request):
        self.sent.append(request)
        return SimpleNamespace(retcode=10009, volume=request["volume"], price=request["price"], comment="done")

    def last_error(self):
        return (0, "ok")


@pytest.fixture
def mt5_broker(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    from xau_agent.brokers.mt5 import MT5Broker

    return MT5Broker(MT5Config(server_utc_offset_hours=2)), fake


def test_mt5_size_uses_exact_loss_per_lot(mt5_broker):
    broker, _ = mt5_broker
    # 100 oz per lot, 6.3 stop -> $630 per lot; $50 risk -> 0.079 -> 0.07 lots
    assert broker.size_for_risk(50, 4012.3, 4006.0) == 0.07


def test_mt5_order_carries_sl_tp_and_magic(mt5_broker):
    broker, fake = mt5_broker
    fill = broker.open_with_bracket("long", 0.07, 4006.0, 4024.9)
    req = fake.sent[0]
    assert req["sl"] == 4006.0 and req["tp"] == 4024.9 and req["magic"] == 240925
    assert req["price"] == 4012.3 and req["type"] == 0
    assert req["type_filling"] == fake.ORDER_FILLING_IOC  # symbol supports IOC only
    assert fill.size == 0.07


def test_mt5_rejects_stop_inside_broker_minimum(mt5_broker):
    broker, _ = mt5_broker
    with pytest.raises(OrderError, match="broker minimum"):
        broker.open_with_bracket("long", 0.07, 4012.25, 4030)
