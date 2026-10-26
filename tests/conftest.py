from __future__ import annotations

import pandas as pd
import pytest

from xau_agent.brokers.base import Broker, Fill, Position, Quote
from xau_agent.config import Config, RiskConfig, StrategyConfig
from xau_agent.indicators import interval_delta, resample_ohlc

BASE = 2000.0


def flat_day(day: pd.Timestamp, base: float = BASE, half_range: float = 10.0) -> pd.DataFrame:
    """96 x 15m bars that never close away from `base`: daily range 20, no breakout."""
    idx = pd.date_range(day, periods=96, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": base, "high": base + half_range, "low": base - half_range, "close": base}, index=idx
    )


def breakout_day(day: pd.Timestamp, side: str = "long", base: float = BASE) -> pd.DataFrame:
    """Asian range 1996-2004, breakout candle at 07:15 closing at 2006 (or 1994), then runs to target."""
    df = flat_day(day, base, half_range=4.0)
    sign = 1 if side == "long" else -1
    t = lambda hm: pd.Timestamp(f"{day.date()} {hm}", tz="UTC")  # noqa: E731
    if side == "long":
        df.loc[t("07:15"), ["open", "high", "low", "close"]] = [base, base + 6.5, base, base + 6]
    else:
        df.loc[t("07:15"), ["open", "high", "low", "close"]] = [base, base, base - 6.5, base - 6]
    after = df.index > t("07:15")
    df.loc[after, "close"] = base + sign * 6
    df.loc[after, "open"] = base + sign * 6
    df.loc[after, "high"] = base + sign * 6 + 1
    df.loc[after, "low"] = base + sign * 6 - 1
    # 09:00: price reaches the 2R target (entry 2006, stop 2000 -> target 2018).
    df.loc[t("09:00"), "high" if side == "long" else "low"] = base + sign * 19
    return df


def history(days: int = 30, end: str = "2026-10-21") -> pd.DataFrame:
    """`days` weekdays of flat 15m bars ending the day before `end`."""
    end_day = pd.Timestamp(end, tz="UTC")
    all_days = pd.bdate_range(end=end_day - pd.Timedelta(days=1), periods=days, tz="UTC")
    return pd.concat([flat_day(d) for d in all_days])


class FakeBroker(Broker):
    """In-memory broker for tests. Candles are sliced to what is 'closed' at self.now."""

    name = "fake"

    def __init__(self, candles: pd.DataFrame, quote: Quote, equity: float = 10_000.0):
        self._candles = candles
        self._quote = quote
        self._equity = equity
        self._position: Position | None = None
        self.orders: list[Fill] = []
        self.closed = 0
        self.now = candles.index[-1] + pd.Timedelta(minutes=15)

    def candles(self, interval: str, limit: int) -> pd.DataFrame:
        step = interval_delta(interval)
        df = self._candles
        if interval != "15m":
            df = resample_ohlc(df, step)
        return df[df.index + step <= self.now].tail(limit)

    def quote(self) -> Quote:
        return self._quote

    def equity(self) -> float:
        return self._equity

    def position(self):
        return self._position

    def size_for_risk(self, risk_amount, entry, stop):
        return round(risk_amount / abs(entry - stop), 3)

    def notional(self, size, price):
        return size * price

    def open_with_bracket(self, side, size, stop, take_profit):
        fill = Fill(side, size, self._quote.ask if side == "long" else self._quote.bid, stop, take_profit)
        self.orders.append(fill)
        self._position = Position(side, size, fill.price)
        return fill

    def close_all(self):
        if self._position:
            self.closed += 1
        self._position = None


@pytest.fixture
def rr_cfg(tmp_path) -> Config:
    """The default config: ifvg_hammer, RR 1:2 with the stop beyond the hammer, 0.1 lot."""
    c = Config()
    c.data_dir = str(tmp_path / "data")
    return c


def legacy_config(data_dir) -> Config:
    """The original setup: Asian breakout on 15m, ATR stops, 0.5% risk sizing."""
    c = Config()
    c.data_dir = str(data_dir)
    c.strategy = StrategyConfig(name="asian_breakout", timeframe="15m", exit_mode="rr", stop_buffer_pips=0,
                                min_stop_pips=0, min_stop_atr=0.1, max_stop_pips=None)
    c.risk = RiskConfig(sizing="percent", max_trades_per_day=1, max_leverage=5, max_spread_pips=None)
    return c


@pytest.fixture
def cfg(tmp_path) -> Config:
    return legacy_config(tmp_path / "data")


@pytest.fixture
def scalp_cfg(tmp_path) -> Config:
    """The M5 setup: fixed 25/50 pip exits, 0.1 lot."""
    c = Config()
    c.data_dir = str(tmp_path / "data")
    c.strategy.name, c.strategy.exit_mode = "ny_orb", "fixed_pips"
    c.ny_orb.minutes = 30  # the ny_orb_day fixture builds a 30-minute opening range
    c.ny_orb.max_range_pips = 25
    return c


def flat_5m(day: pd.Timestamp, base: float = BASE, half_range: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range(day, periods=288, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": base, "high": base + half_range, "low": base - half_range, "close": base}, index=idx
    )


def set_bar(df: pd.DataFrame, ts: str, o: float, h: float, low: float, c: float) -> None:
    df.loc[pd.Timestamp(ts, tz="UTC"), ["open", "high", "low", "close"]] = [o, h, low, c]


def ny_orb_day(day: pd.Timestamp, open_utc: str = "13:30", side: str = "long") -> pd.DataFrame:
    """Opening range 1999-2001 (20 pips) from `open_utc` for 30 min, breakout close 2001.5 (or 1998.5),
    then price runs to the 50-pip target (2006.5 / 1993.5) about an hour later."""
    df = flat_5m(day, half_range=0.5)
    start = pd.Timestamp(f"{day.date()} {open_utc}", tz="UTC")
    rng = (df.index >= start) & (df.index < start + pd.Timedelta(minutes=30))
    df.loc[rng, "high"], df.loc[rng, "low"] = BASE + 1, BASE - 1
    sign = 1 if side == "long" else -1
    brk = start + pd.Timedelta(minutes=35)
    entry = BASE + sign * 1.5
    set_bar(df, str(brk), BASE, max(BASE, entry) + 0.1, min(BASE, entry) - 0.1, entry)
    after = df.index > brk
    df.loc[after, ["open", "close"]] = entry
    df.loc[after, "high"], df.loc[after, "low"] = entry + 0.3, entry - 0.3
    hit = brk + pd.Timedelta(minutes=60)
    if side == "long":
        df.loc[hit, "high"] = entry + 5.2
    else:
        df.loc[hit, "low"] = entry - 5.2
    return df


def to_1m(df5: pd.DataFrame) -> pd.DataFrame:
    """Split each 5m bar into five 1m bars with the same OHLC (good enough for exit tests)."""
    parts = [df5.set_axis(df5.index + pd.Timedelta(minutes=i)) for i in range(5)]
    return pd.concat(parts).sort_index()


def ifvg_hammer_day(day: pd.Timestamp, hammer_low: float = 1999.9) -> pd.DataFrame:
    """Drop that leaves a bearish FVG (2001.0-2001.6), hammer at the low, then a close above the gap.

    08:00-08:10 drop (FVG between the 08:00 low and the 08:10 high), 08:15 hammer,
    08:20 back into the gap, 08:25 closes at 2001.8 above the gap top -> bullish IFVG -> long.
    """
    df = flat_5m(day, base=2002.0, half_range=0.2)
    t = f"{day.date()}"
    set_bar(df, f"{t} 08:00", 2002.0, 2002.1, 2001.6, 2001.7)
    set_bar(df, f"{t} 08:05", 2001.7, 2001.7, 2000.4, 2000.5)
    set_bar(df, f"{t} 08:10", 2000.5, 2001.0, 2000.3, 2000.6)
    set_bar(df, f"{t} 08:15", 2000.4, 2000.55, hammer_low, 2000.5)  # hammer
    set_bar(df, f"{t} 08:20", 2000.5, 2001.3, 2000.45, 2001.2)
    set_bar(df, f"{t} 08:25", 2001.2, 2001.9, 2001.15, 2001.8)      # IFVG close
    return df


def mirror(df: pd.DataFrame, around: float = 2002.0) -> pd.DataFrame:
    """Flip prices upside down around `around`: a long setup becomes the matching short setup."""
    out = df.copy()
    out["open"], out["close"] = 2 * around - df["open"], 2 * around - df["close"]
    out["high"], out["low"] = 2 * around - df["low"], 2 * around - df["high"]
    return out
