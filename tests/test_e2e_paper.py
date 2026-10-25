"""Walk a full trading day through the real Bot + PaperBroker, as the live loop would."""

from datetime import timezone

import pandas as pd
import pytest

from xau_agent.agent import Decision
from xau_agent.bot import Bot
from xau_agent.brokers.base import Quote
from xau_agent.brokers.paper import PaperBroker

from .conftest import FakeBroker, breakout_day, flat_5m, history, ny_orb_day
from .test_bot import SilentNotifier, StubAgent, StubCalendar

DAY = pd.Timestamp("2026-10-21", tz="UTC")


def test_full_day_entry_then_take_profit(cfg):
    candles = pd.concat([history(60), breakout_day(DAY)])
    data = FakeBroker(candles, Quote(2006.0, 2006.0))
    paper = PaperBroker(data, cfg.data_path / "paper.json", 10_000, fee_bps=0, clock=lambda: data.now)
    bot = Bot(cfg, paper, agent=StubAgent(Decision(True, 0.9, "ok")), calendar=StubCalendar(),
              notifier=SilentNotifier())

    for now in pd.date_range(DAY + pd.Timedelta("06:00:00"), DAY + pd.Timedelta("21:00:00"), freq="5min"):
        data.now = now
        bot.step(now.to_pydatetime().replace(tzinfo=timezone.utc))

    messages = "\n".join(bot.notifier.messages)
    assert "ENTERED LONG" in messages
    assert "take profit" in messages
    assert paper.position() is None
    # 0.5% risk = $50, target is 2R -> +$100 (no fees in this test)
    assert paper.equity() == pytest.approx(10_100, abs=0.5)
    assert bot.journal.count_today("entry", pd.Timestamp.now(tz="UTC").date()) == 1


def test_full_day_ny_orb_fixed_lot_take_profit(scalp_cfg):
    days = pd.bdate_range(end=DAY - pd.Timedelta(days=1), periods=12, tz="UTC")
    candles = pd.concat([flat_5m(d) for d in days] + [ny_orb_day(DAY)])
    data = FakeBroker(candles, Quote(2001.5, 2001.5))
    paper = PaperBroker(data, scalp_cfg.data_path / "paper.json", 10_000, fee_bps=0, clock=lambda: data.now)
    bot = Bot(scalp_cfg, paper, agent=StubAgent(Decision(True, 0.9, "ok")), calendar=StubCalendar(),
              notifier=SilentNotifier())

    for now in pd.date_range(DAY + pd.Timedelta("13:00:00"), DAY + pd.Timedelta("21:00:00"), freq="1min"):
        data.now = now
        bot.step(now.to_pydatetime().replace(tzinfo=timezone.utc))

    messages = "\n".join(bot.notifier.messages)
    assert "ENTERED LONG 10.0 @ 2001.50" in messages
    assert "take profit" in messages
    # 0.1 lot (10 oz) x 50 pips ($5.00) = +$50
    assert paper.equity() == pytest.approx(10_050, abs=0.01)
