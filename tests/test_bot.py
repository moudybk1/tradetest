from datetime import datetime, timezone

import pandas as pd

from xau_agent.agent import Decision
from xau_agent.bot import Bot
from xau_agent.brokers.base import Quote

from .conftest import FakeBroker, breakout_day, history

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
