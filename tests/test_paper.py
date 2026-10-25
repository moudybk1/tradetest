import pandas as pd
import pytest

from xau_agent.brokers.base import Quote
from xau_agent.brokers.paper import PaperBroker

from .conftest import FakeBroker


def minute_bars(start: str, prices: list[tuple[float, float]]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": [h for h, _ in prices], "high": [h for h, _ in prices],
         "low": [low for _, low in prices], "close": [low for _, low in prices]},
        index=idx,
    )


def make_paper(tmp_path, fee_bps=0.0):
    data = FakeBroker(minute_bars("2026-10-21 07:00", [(2006, 2005)] * 5), Quote(2005.9, 2006.0))
    return data, PaperBroker(data, tmp_path / "paper.json", 10_000, fee_bps)


def test_stop_loss_costs_the_planned_risk(tmp_path):
    data, paper = make_paper(tmp_path)
    size = paper.size_for_risk(50, 2006, 2000)
    paper.open_with_bracket("long", size, 2000, 2018)
    opened = pd.Timestamp(paper.state["position"]["opened_at"]).ceil("1min")
    data._candles = minute_bars(str(opened), [(2006, 2003), (2004, 1999)])
    data.now = opened + pd.Timedelta(minutes=5)

    msgs = paper.sync()
    assert "stop loss" in msgs[0]
    assert paper.position() is None
    assert paper.equity() == pytest.approx(10_000 - 50, abs=0.1)


def test_take_profit(tmp_path):
    data, paper = make_paper(tmp_path)
    paper.open_with_bracket("long", 1.0, 2000, 2018)
    opened = pd.Timestamp(paper.state["position"]["opened_at"]).ceil("1min")
    data._candles = minute_bars(str(opened), [(2010, 2005), (2019, 2012)])
    data.now = opened + pd.Timedelta(minutes=5)

    assert "take profit" in paper.sync()[0]
    assert paper.equity() == pytest.approx(10_000 + 12)


def test_bar_touching_both_assumes_stop(tmp_path):
    data, paper = make_paper(tmp_path)
    paper.open_with_bracket("long", 1.0, 2000, 2018)
    opened = pd.Timestamp(paper.state["position"]["opened_at"]).ceil("1min")
    data._candles = minute_bars(str(opened), [(2019, 1999)])
    data.now = opened + pd.Timedelta(minutes=5)
    assert "stop loss" in paper.sync()[0]


def test_fees_and_state_persist(tmp_path):
    data, paper = make_paper(tmp_path, fee_bps=5)
    paper.open_with_bracket("short", 1.0, 2012, 1982)
    assert paper.state["cash"] == pytest.approx(10_000 - 2005.9 * 0.0005)
    reloaded = PaperBroker(data, tmp_path / "paper.json", 10_000, 5)
    assert reloaded.position().side == "short"
