from datetime import datetime, timezone

import pandas as pd
import pytest

from xau_agent.agent import Decision
from xau_agent.bot import Bot
from xau_agent.brokers.base import Quote

from .conftest import FakeBroker, breakout_day, flat_5m, history, ifvg_hammer_day, ny_orb_day

DAY = pd.Timestamp("2026-10-21", tz="UTC")
BREAKOUT_CLOSE = pd.Timestamp("2026-10-21 07:30", tz="UTC")  # 07:15 candle has just closed


class StubAgent:
    def __init__(self, decision: Decision):
        self.decision = decision
        self.contexts = []

    def news_summary(self, now):
        return "no news"

    def review(self, context):
        self.contexts.append(context)
        return self.decision


class StubCalendar:
    def __init__(self, blackout=False):
        self._blackout = blackout

    def blackout(self, now, before, after):
        return self._blackout, []

    def upcoming(self, now, hours=24):
        return []


class SilentNotifier:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)


def make_bot(cfg, decision=Decision(True, 0.8, "looks good"), blackout=False, now=BREAKOUT_CLOSE):
    candles = pd.concat([history(60), breakout_day(DAY)])
    broker = FakeBroker(candles, Quote(2006.0, 2006.1))
    broker.now = now
    agent = StubAgent(decision)
    bot = Bot(cfg, broker, agent=agent, calendar=StubCalendar(blackout), notifier=SilentNotifier())
    return bot, broker, agent


def step(bot, now=BREAKOUT_CLOSE):
    bot.step(now.to_pydatetime().replace(tzinfo=timezone.utc))


def test_signal_accepted_places_bracket_order(cfg):
    bot, broker, agent = make_bot(cfg)
    step(bot)
    assert len(broker.orders) == 1
    fill = broker.orders[0]
    assert fill.side == "long" and fill.stop == 2000
    # Re-anchored on the live ask (2006.1) with 2R kept: 2006.1 + 2 * 6.1
    assert abs(fill.take_profit - 2018.3) < 1e-9
    # 0.5% of 10,000 = $50 risk over a 6.1 stop
    assert abs(fill.size - round(50 / 6.1, 3)) < 1e-9
    assert agent.contexts and agent.contexts[0]["news_summary"] == "no news"
    assert bot.journal.count_today("entry", datetime.now(timezone.utc).date()) == 1


def test_agent_veto_means_no_order(cfg):
    bot, broker, _ = make_bot(cfg, decision=Decision(False, 0.9, "CPI in 30 minutes"))
    step(bot)
    assert broker.orders == []
    assert any("SKIPPED by agent" in m for m in bot.notifier.messages)


def test_low_confidence_take_is_skipped(cfg):
    bot, broker, _ = make_bot(cfg, decision=Decision(True, 0.4, "meh"))
    step(bot)
    assert broker.orders == []


def test_news_blackout_blocks_before_agent_is_asked(cfg):
    bot, broker, agent = make_bot(cfg, blackout=True)
    step(bot)
    assert broker.orders == [] and agent.contexts == []


def test_stale_breakout_is_not_chased(cfg):
    later = pd.Timestamp("2026-10-21 08:00", tz="UTC")
    bot, broker, agent = make_bot(cfg, now=later)
    step(bot, later)
    assert broker.orders == [] and agent.contexts == []


def test_same_candle_processed_once(cfg):
    bot, broker, _ = make_bot(cfg)
    step(bot)
    broker._position = None  # even if flat again, the same candle must not re-trigger
    step(bot)
    assert len(broker.orders) == 1


def test_flatten_after_session_end(cfg):
    late = pd.Timestamp("2026-10-21 20:45", tz="UTC")
    bot, broker, _ = make_bot(cfg, now=late)
    broker._position = object()
    step(bot, late)
    assert broker.closed == 1


def test_rules_only_mode(cfg):
    cfg.agent.enabled = False
    candles = pd.concat([history(60), breakout_day(DAY)])
    broker = FakeBroker(candles, Quote(2006.0, 2006.1))
    broker.now = BREAKOUT_CLOSE
    bot = Bot(cfg, broker, calendar=StubCalendar(), notifier=SilentNotifier())
    step(bot)
    assert len(broker.orders) == 1


# ---- M5 scalping setup: ny_orb, fixed 25/50 pips, 0.1 lot -------------------------------------------
SIGNAL_CLOSE = pd.Timestamp("2026-10-21 14:10", tz="UTC")  # 14:05 breakout candle just closed


def make_scalp_bot(scalp_cfg, quote=Quote(2001.5, 2001.8), equity=10_000.0, decision=Decision(True, 0.8, "ok")):
    days = pd.bdate_range(end=DAY - pd.Timedelta(days=1), periods=12, tz="UTC")
    candles = pd.concat([flat_5m(d) for d in days] + [ny_orb_day(DAY)])
    broker = FakeBroker(candles, quote, equity=equity)
    broker.now = SIGNAL_CLOSE
    bot = Bot(scalp_cfg, broker, agent=StubAgent(decision), calendar=StubCalendar(), notifier=SilentNotifier())
    return bot, broker


def test_scalp_fixed_lot_entry_with_25_50_pips(scalp_cfg):
    bot, broker = make_scalp_bot(scalp_cfg)
    step(bot, SIGNAL_CLOSE)
    [fill] = broker.orders
    assert fill.side == "long" and fill.size == 10.0  # 0.1 lot = 10 oz on an ounce-based broker
    assert fill.price == 2001.8
    assert (fill.stop, fill.take_profit) == (pytest.approx(1999.3), pytest.approx(2006.8))
    assert any("risk $25.00" in m for m in bot.notifier.messages)


def test_scalp_skips_when_price_ran_away(scalp_cfg):
    # Ask is 0.9 above the 2001.5 signal close: more than 30% of a 2.5 stop.
    bot, broker = make_scalp_bot(scalp_cfg, quote=Quote(2002.3, 2002.4))
    step(bot, SIGNAL_CLOSE)
    assert broker.orders == []
    assert any("price moved" in m for m in bot.notifier.messages)


def test_scalp_fixed_lot_too_big_for_account(scalp_cfg):
    # $25 risk on a $1,000 account = 2.5% > max_risk_per_trade_pct 2%.
    bot, broker = make_scalp_bot(scalp_cfg, equity=1_000.0)
    step(bot, SIGNAL_CLOSE)
    assert broker.orders == []
    assert any("would risk 2.50% of equity" in m for m in bot.notifier.messages)


def test_scalp_second_trade_allowed_same_day(scalp_cfg):
    bot, broker = make_scalp_bot(scalp_cfg)
    bot.journal.log("entry", note="earlier trade today")
    step(bot, SIGNAL_CLOSE)
    assert len(broker.orders) == 1  # max_trades_per_day is 2


def test_rr_bot_entry_risk_varies_with_the_hammer(rr_cfg):
    days = pd.bdate_range(end=DAY - pd.Timedelta(days=1), periods=12, tz="UTC")
    candles = pd.concat([flat_5m(d, base=2002.0, half_range=0.2) for d in days] + [ifvg_hammer_day(DAY)])
    broker = FakeBroker(candles, Quote(2001.8, 2001.8))
    broker.now = pd.Timestamp("2026-10-21 08:30", tz="UTC")  # 08:25 IFVG candle just closed
    bot = Bot(rr_cfg, broker, agent=StubAgent(Decision(True, 0.8, "ok")), calendar=StubCalendar(),
              notifier=SilentNotifier())
    step(bot, broker.now)
    [fill] = broker.orders
    assert fill.size == 10.0 and fill.stop == pytest.approx(1999.6)
    assert fill.take_profit == pytest.approx(2006.2)
    assert any("risk $22.00" in m for m in bot.notifier.messages)  # 22 pips x 10 oz
