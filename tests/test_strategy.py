import pandas as pd
import pytest

from xau_agent.config import StrategyConfig
from xau_agent.indicators import daily_atr
from xau_agent.strategy import find_breakout

from .conftest import breakout_day, flat_day, history

DAY = pd.Timestamp("2026-10-21", tz="UTC")  # a Wednesday


def test_daily_atr_uses_only_completed_weekdays():
    candles = pd.concat([history(), breakout_day(DAY)])
    assert daily_atr(candles, 14, DAY) == pytest.approx(20.0)


@pytest.mark.parametrize("side", ["long", "short"])
def test_breakout_signal(side):
    signal = find_breakout(breakout_day(DAY, side), DAY, 20.0, StrategyConfig())
    assert signal is not None
    assert signal.side == side
    assert signal.candle_time == pd.Timestamp("2026-10-21 07:15", tz="UTC")
    if side == "long":
        assert (signal.entry, signal.stop, signal.take_profit) == (2006, 2000, 2018)
    else:
        assert (signal.entry, signal.stop, signal.take_profit) == (1994, 2000, 1982)


def test_opposite_stop_mode():
    cfg = StrategyConfig(stop_mode="opposite")
    signal = find_breakout(breakout_day(DAY), DAY, 20.0, cfg)
    assert signal.stop == 1996
    assert signal.take_profit == 2006 + 2 * 10


def test_no_signal_when_range_too_wide():
    # Range width 8 vs ATR 5 -> 1.6 ATR > max 1.2: the move already happened.
    assert find_breakout(breakout_day(DAY), DAY, 5.0, StrategyConfig()) is None


def test_no_signal_when_range_too_narrow():
    # Range width 8 vs ATR 100 -> 0.08 ATR < min 0.15.
    assert find_breakout(breakout_day(DAY), DAY, 100.0, StrategyConfig()) is None


def test_no_signal_on_flat_day():
    assert find_breakout(flat_day(DAY, half_range=4), DAY, 20.0, StrategyConfig()) is None


def test_no_signal_with_missing_asian_data():
    df = breakout_day(DAY)
    df = df[(df.index.hour >= 4)]  # lost the first 4 hours of the Asian session
    assert find_breakout(df, DAY, 20.0, StrategyConfig()) is None


def test_only_first_breakout_counts():
    df = breakout_day(DAY, "short")
    # A later bullish close above the range must not replace the earlier short signal.
    df.loc[pd.Timestamp("2026-10-21 10:00", tz="UTC"), "close"] = 2010
    assert find_breakout(df, DAY, 20.0, StrategyConfig()).side == "short"


def test_no_entry_after_entry_window():
    df = flat_day(DAY, half_range=4)
    df.loc[pd.Timestamp("2026-10-21 16:15", tz="UTC"), "close"] = 2010
    assert find_breakout(df, DAY, 20.0, StrategyConfig()) is None


def test_min_stop_distance_enforced():
    cfg = StrategyConfig(min_stop_atr=0.5)  # 10 points minimum with ATR 20
    signal = find_breakout(breakout_day(DAY), DAY, 20.0, cfg)
    assert signal.entry - signal.stop == pytest.approx(10)


def test_with_entry_keeps_stop_and_reward_risk():
    signal = find_breakout(breakout_day(DAY), DAY, 20.0, StrategyConfig()).with_entry(2007, 2.0)
    assert signal.stop == 2000
    assert signal.take_profit == 2007 + 14
