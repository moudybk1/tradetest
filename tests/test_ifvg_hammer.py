import numpy as np
import pandas as pd
import pytest

from xau_agent.backtest import Costs, run_backtest
from xau_agent.strategy import candle_shapes, ifvg_hammer, prepare_ifvg_hammer

from .conftest import flat_5m, ifvg_hammer_day, mirror, set_bar, to_1m

DAY = pd.Timestamp("2026-10-21", tz="UTC")


def signals(df, cfg):
    return ifvg_hammer(prepare_ifvg_hammer(df, df.iloc[:0], cfg), DAY, None, cfg)


def one_candle(o, h, low, c):
    return pd.DataFrame({"open": [o], "high": [h], "low": [low], "close": [c]},
                        index=pd.DatetimeIndex([DAY]))


@pytest.mark.parametrize("candle, is_hammer, is_star", [
    ((2000.4, 2000.55, 1999.9, 2000.5), True, False),   # long lower wick, tiny upper wick
    ((2000.1, 2000.6, 1999.95, 2000.0), False, True),   # shooting star: long upper wick
    ((2000.0, 2000.6, 1999.9, 2000.5), False, False),   # body too big for the wick
    ((2000.4, 2000.9, 1999.9, 2000.5), False, False),   # upper wick too long for a hammer
    ((2000.10, 2000.13, 2000.0, 2000.12), False, False),  # too small (1.3 pips)
])
def test_candle_shapes(scalp_cfg, candle, is_hammer, is_star):
    hammer, star = candle_shapes(one_candle(*candle), scalp_cfg)
    assert (bool(hammer.iloc[0]), bool(star.iloc[0])) == (is_hammer, is_star)


def test_hammer_must_be_a_swing_low(scalp_cfg):
    df = ifvg_hammer_day(DAY)
    df.loc[pd.Timestamp("2026-10-21 08:10", tz="UTC"), "low"] = 1999.0  # a lower low just before
    hammer, _ = candle_shapes(df, scalp_cfg)
    assert not hammer[pd.Timestamp("2026-10-21 08:15", tz="UTC")]
    scalp_cfg.ifvg_hammer.hammer_swing_bars = 0
    hammer, _ = candle_shapes(df, scalp_cfg)
    assert hammer[pd.Timestamp("2026-10-21 08:15", tz="UTC")]


def test_fvg_inversion_marked_on_the_closing_candle(scalp_cfg):
    out = prepare_ifvg_hammer(ifvg_hammer_day(DAY), None, scalp_cfg)
    marked = out.index[out["ifvg_long"]]
    assert list(marked) == [pd.Timestamp("2026-10-21 08:25", tz="UTC")]
    row = out.loc[marked[0]]
    assert (row["ifvg_low"], row["ifvg_high"]) == pytest.approx((2001.0, 2001.6))
    assert row["hammer_low"] == pytest.approx(1999.9)


def test_long_entry_on_ifvg_after_hammer(scalp_cfg):
    [sig] = signals(ifvg_hammer_day(DAY), scalp_cfg)
    assert sig.side == "long" and sig.strategy == "ifvg_hammer"
    assert sig.candle_time == pd.Timestamp("2026-10-21 08:25", tz="UTC")
    assert sig.entry == 2001.8
    assert (sig.stop, sig.take_profit) == pytest.approx((1999.3, 2006.8))  # 25 / 50 pips
    assert sig.context["hammer_low"] == pytest.approx(1999.9)


def test_short_entry_on_mirrored_setup(scalp_cfg):
    [sig] = signals(mirror(ifvg_hammer_day(DAY)), scalp_cfg)
    assert sig.side == "short" and sig.entry == pytest.approx(2002.2)
    assert (sig.stop, sig.take_profit) == pytest.approx((2004.7, 1997.2))


def test_no_hammer_no_trade(scalp_cfg):
    df = ifvg_hammer_day(DAY)
    set_bar(df, "2026-10-21 08:15", 2000.4, 2000.6, 2000.3, 2000.55)  # ordinary candle
    assert signals(df, scalp_cfg) == []


def test_ifvg_too_long_after_hammer(scalp_cfg):
    scalp_cfg.ifvg_hammer.hammer_max_bars_before = 1  # hammer 08:15, IFVG 08:25 = 2 candles later
    assert signals(ifvg_hammer_day(DAY), scalp_cfg) == []


def test_hammer_low_broken_before_ifvg(scalp_cfg):
    df = ifvg_hammer_day(DAY)
    df.loc[pd.Timestamp("2026-10-21 08:20", tz="UTC"), "low"] = 1999.8  # trades under the hammer
    assert signals(df, scalp_cfg) == []


def test_hammer_low_beyond_25_pip_stop(scalp_cfg):
    df = ifvg_hammer_day(DAY, hammer_low=1999.2)  # entry 2001.8 - 25 pips = 1999.3 > 1999.2
    assert signals(df, scalp_cfg) == []
    scalp_cfg.ifvg_hammer.require_stop_beyond_hammer = False
    assert len(signals(df, scalp_cfg)) == 1


def test_small_gaps_and_old_gaps_ignored(scalp_cfg):
    scalp_cfg.ifvg_hammer.min_fvg_pips = 7  # the gap is 6 pips
    assert signals(ifvg_hammer_day(DAY), scalp_cfg) == []
    scalp_cfg.ifvg_hammer.min_fvg_pips = 3
    scalp_cfg.ifvg_hammer.fvg_lookback_bars = 2  # gap formed at 08:10, inverted 3 candles later
    assert signals(ifvg_hammer_day(DAY), scalp_cfg) == []


def test_outside_session(scalp_cfg):
    scalp_cfg.ifvg_hammer.sessions_utc = "13:00-17:00"
    assert signals(ifvg_hammer_day(DAY), scalp_cfg) == []


def test_backtest_trade_hits_target(scalp_cfg):
    scalp_cfg.strategy.name = "ifvg_hammer"
    df = ifvg_hammer_day(DAY)
    later = df.index > pd.Timestamp("2026-10-21 08:25", tz="UTC")
    df.loc[later, ["open", "close"]] = 2002.5
    df.loc[later, ["high", "low"]] = [2002.7, 2002.3]
    df.loc[pd.Timestamp("2026-10-21 09:30", tz="UTC"), "high"] = 2007.0
    prev = flat_5m(pd.Timestamp("2026-10-20", tz="UTC"), base=2002.0, half_range=0.2)
    trades, stats = run_backtest(to_1m(pd.concat([prev, df])), scalp_cfg, Costs(0, 0))
    assert stats["trades"] == 1
    assert trades["reason"].iloc[0] == "target" and trades["usd"].iloc[0] == pytest.approx(50.0)


def test_prepare_uses_only_the_past(scalp_cfg):
    """Adding future candles must never change what was marked on earlier candles."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-10-21", periods=600, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(rng.normal(0, 0.8, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + rng.uniform(0, 0.6, len(idx)),
                       "low": np.minimum(open_, close) - rng.uniform(0, 0.6, len(idx)), "close": close}, index=idx)
    full = prepare_ifvg_hammer(df, None, scalp_cfg)
    part = prepare_ifvg_hammer(df.iloc[:400], None, scalp_cfg)
    cols = ["ifvg_long", "ifvg_short", "hammer_low", "star_high"]
    pd.testing.assert_frame_equal(full[cols].iloc[:400], part[cols])
    assert full["ifvg_long"].any() and full["ifvg_short"].any()
