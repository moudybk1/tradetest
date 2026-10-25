import numpy as np
import pandas as pd
import pytest

from xau_agent.indicators import daily_atr, interval_delta
from xau_agent.strategy import (
    _at_ny, asian_breakout, asian_sweep, fixed_exits, ny_orb, prepare_pullback, trend_pullback,
)

from .conftest import breakout_day, flat_5m, flat_day, history, ny_orb_day, set_bar

DAY = pd.Timestamp("2026-10-21", tz="UTC")  # a Wednesday, New York on daylight time (UTC-4)


# ---- Asian breakout (original strategy, ATR exits) -----------------------------------------------
def test_daily_atr_uses_only_completed_weekdays():
    candles = pd.concat([history(), breakout_day(DAY)])
    assert daily_atr(candles, 14, DAY) == pytest.approx(20.0)


@pytest.mark.parametrize("side", ["long", "short"])
def test_asian_breakout_signal(cfg, side):
    [signal] = asian_breakout(breakout_day(DAY, side), DAY, 20.0, cfg)
    assert signal.side == side
    assert signal.candle_time == pd.Timestamp("2026-10-21 07:15", tz="UTC")
    if side == "long":
        assert (signal.entry, signal.stop, signal.take_profit) == (2006, 2000, 2018)
    else:
        assert (signal.entry, signal.stop, signal.take_profit) == (1994, 2000, 1982)


def test_asian_breakout_opposite_stop_mode(cfg):
    cfg.asian_breakout.stop_mode = "opposite"
    [signal] = asian_breakout(breakout_day(DAY), DAY, 20.0, cfg)
    assert signal.stop == 1996 and signal.take_profit == 2006 + 2 * 10


def test_asian_breakout_range_filters(cfg):
    assert asian_breakout(breakout_day(DAY), DAY, 5.0, cfg) == []    # range 1.6 ATR: too wide
    assert asian_breakout(breakout_day(DAY), DAY, 100.0, cfg) == []  # range 0.08 ATR: too narrow


def test_asian_breakout_no_signal_on_flat_day_or_missing_data(cfg):
    assert asian_breakout(flat_day(DAY, half_range=4), DAY, 20.0, cfg) == []
    df = breakout_day(DAY)
    assert asian_breakout(df[df.index.hour >= 4], DAY, 20.0, cfg) == []


def test_asian_breakout_only_first_breakout_counts(cfg):
    df = breakout_day(DAY, "short")
    df.loc[pd.Timestamp("2026-10-21 10:00", tz="UTC"), "close"] = 2010
    [signal] = asian_breakout(df, DAY, 20.0, cfg)
    assert signal.side == "short"


def test_asian_breakout_min_stop_distance(cfg):
    cfg.strategy.min_stop_atr = 0.5
    [signal] = asian_breakout(breakout_day(DAY), DAY, 20.0, cfg)
    assert signal.entry - signal.stop == pytest.approx(10)


def test_asian_breakout_with_fixed_pips(cfg):
    cfg.strategy.exit_mode = "fixed_pips"
    [signal] = asian_breakout(breakout_day(DAY), DAY, 20.0, cfg)
    assert (signal.stop, signal.take_profit) == pytest.approx((2003.5, 2011.0))


# ---- exits ---------------------------------------------------------------------------------------
def test_fixed_exits_25_50_pips(scalp_cfg):
    assert fixed_exits("long", 4012.30, scalp_cfg.strategy) == pytest.approx((4009.80, 4017.30))
    assert fixed_exits("short", 4012.30, scalp_cfg.strategy) == pytest.approx((4014.80, 4007.30))


def test_with_entry_fixed_pips_moves_sl_and_tp(scalp_cfg):
    [signal] = ny_orb(ny_orb_day(DAY), DAY, None, scalp_cfg)
    moved = signal.with_entry(2001.7, scalp_cfg.strategy)
    assert (moved.stop, moved.take_profit) == pytest.approx((1999.2, 2006.7))


def test_with_entry_atr_keeps_stop(cfg):
    [signal] = asian_breakout(breakout_day(DAY), DAY, 20.0, cfg)
    moved = signal.with_entry(2007, cfg.strategy)
    assert moved.stop == 2000 and moved.take_profit == 2007 + 14


# ---- New York opening range ----------------------------------------------------------------------
@pytest.mark.parametrize("side", ["long", "short"])
def test_ny_orb_signal(scalp_cfg, side):
    [signal] = ny_orb(ny_orb_day(DAY, side=side), DAY, None, scalp_cfg)
    assert signal.side == side
    assert signal.candle_time == pd.Timestamp("2026-10-21 14:05", tz="UTC")
    assert signal.stop_distance == pytest.approx(2.5)
    assert abs(signal.take_profit - signal.entry) == pytest.approx(5.0)


def test_ny_open_follows_daylight_saving():
    assert _at_ny(DAY, pd.Timestamp("09:30").time()) == pd.Timestamp("2026-10-21 13:30", tz="UTC")
    winter = pd.Timestamp("2026-12-16", tz="UTC")
    assert _at_ny(winter, pd.Timestamp("09:30").time()) == pd.Timestamp("2026-12-16 14:30", tz="UTC")


def test_ny_orb_in_winter_uses_14_30_utc(scalp_cfg):
    winter = pd.Timestamp("2026-12-16", tz="UTC")
    [signal] = ny_orb(ny_orb_day(winter, open_utc="14:30"), winter, None, scalp_cfg)
    assert signal.candle_time == pd.Timestamp("2026-12-16 15:05", tz="UTC")
    # The summer-time bars don't form a valid range in winter.
    assert ny_orb(ny_orb_day(winter, open_utc="13:30"), winter, None, scalp_cfg) == []


def test_ny_orb_skips_wide_range(scalp_cfg):
    scalp_cfg.ny_orb.max_range_pips = 15  # range is 20 pips
    assert ny_orb(ny_orb_day(DAY), DAY, None, scalp_cfg) == []


# ---- Asian sweep ---------------------------------------------------------------------------------
def sweep_day(wick: float) -> pd.DataFrame:
    """Asian range 1998-2002. 07:30 wicks above to `wick` and closes above, 07:35 closes back inside."""
    df = flat_5m(DAY, half_range=2.0)
    set_bar(df, "2026-10-21 07:30", 2001.5, wick, 2001.4, 2002.3)
    set_bar(df, "2026-10-21 07:35", 2002.3, 2002.4, 2001.3, 2001.5)
    after = df.index > pd.Timestamp("2026-10-21 07:35", tz="UTC")
    df.loc[after, ["open", "close"]] = 2001.5
    df.loc[after, ["high", "low"]] = [2001.8, 2001.2]
    return df


def test_asian_sweep_short_after_high_is_taken(scalp_cfg):
    [signal] = asian_sweep(sweep_day(wick=2002.8), DAY, 5.0, scalp_cfg)
    assert signal.side == "short"
    assert signal.candle_time == pd.Timestamp("2026-10-21 07:35", tz="UTC")
    assert signal.entry == 2001.5 and signal.stop == pytest.approx(2004.0)
    assert signal.context["sweep_extreme"] == 2002.8


def test_asian_sweep_skips_when_wick_is_beyond_the_stop(scalp_cfg):
    # Wick at 2004.5 is above entry + 25 pips (2004.0): a 25-pip stop would sit inside the wick.
    assert asian_sweep(sweep_day(wick=2004.5), DAY, 5.0, scalp_cfg) == []


def test_asian_sweep_needs_close_back_inside(scalp_cfg):
    df = sweep_day(wick=2002.8)
    df.loc[df.index >= pd.Timestamp("2026-10-21 07:35", tz="UTC"), ["open", "high", "low", "close"]] = 2002.5
    assert asian_sweep(df, DAY, 5.0, scalp_cfg) == []


# ---- Trend pullback ------------------------------------------------------------------------------
def test_pullback_trend_has_no_lookahead(scalp_cfg):
    rng = np.random.default_rng(1)
    hours = pd.date_range("2026-10-01", periods=400, freq="1h", tz="UTC")
    close = 2000 + np.cumsum(rng.normal(0, 3, len(hours)))
    hourly = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=hours)
    bars = pd.concat([flat_5m(d) for d in pd.date_range("2026-10-13", periods=5, tz="UTC")])
    out = prepare_pullback(bars, hourly, scalp_cfg)

    fast = hourly["close"].ewm(span=50, adjust=False).mean()
    slow = hourly["close"].ewm(span=200, adjust=False).mean()
    closes = hourly.index + pd.Timedelta(hours=1)
    for ts, value in out["trend"].iloc[::37].items():
        bar_close = ts + interval_delta("5m")
        known = closes <= bar_close  # H1 candles finished by the time this bar closes
        expected = 1.0 if fast[known].iloc[-1] > slow[known].iloc[-1] else -1.0
        if known.sum() <= 200:
            assert np.isnan(value)
        else:
            assert value == expected


def pullback_bars(trend: float) -> pd.DataFrame:
    df = flat_5m(DAY, half_range=0.3)
    df["ema"] = 2000.0
    df["swing_low"], df["swing_high"] = 1999.0, 2001.0
    df["trend"] = trend
    return df


def test_pullback_long_in_uptrend(scalp_cfg):
    df = pullback_bars(1.0)
    # 08:00 dips to the EMA (low 1999.9) and closes bullish above it.
    set_bar(df, "2026-10-21 08:00", 2000.0, 2000.8, 1999.9, 2000.6)
    [signal] = trend_pullback(df, DAY, None, scalp_cfg)
    assert signal.side == "long" and signal.entry == 2000.6
    assert signal.stop == pytest.approx(1998.1)  # 25 pips, below the 1999.0 swing low


def test_pullback_ignored_against_trend_or_outside_session(scalp_cfg):
    df = pullback_bars(-1.0)
    set_bar(df, "2026-10-21 08:00", 2000.0, 2000.8, 1999.9, 2000.6)
    assert trend_pullback(df, DAY, None, scalp_cfg) == []
    df = pullback_bars(1.0)
    set_bar(df, "2026-10-21 12:00", 2000.0, 2000.8, 1999.9, 2000.6)  # 11:00-13:00 is not traded
    assert trend_pullback(df, DAY, None, scalp_cfg) == []


def test_pullback_skips_when_swing_is_beyond_the_stop(scalp_cfg):
    df = pullback_bars(1.0)
    df["swing_low"] = 1997.0  # more than 25 pips under a 2000.6 entry
    set_bar(df, "2026-10-21 08:00", 2000.0, 2000.8, 1999.9, 2000.6)
    assert trend_pullback(df, DAY, None, scalp_cfg) == []


def test_pullback_cooldown(scalp_cfg):
    df = pullback_bars(1.0)
    for ts in ("08:00", "08:05", "08:40"):
        set_bar(df, f"2026-10-21 {ts}", 2000.0, 2000.8, 1999.9, 2000.6)
    signals = trend_pullback(df, DAY, None, scalp_cfg)
    # 08:05 falls inside the 6-bar cooldown after 08:00; 08:40 is after it.
    assert [s.candle_time.strftime("%H:%M") for s in signals] == ["08:00", "08:40"]
