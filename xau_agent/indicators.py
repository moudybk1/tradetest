"""Small indicator helpers. All inputs are OHLC DataFrames indexed by UTC candle open time."""

from __future__ import annotations

import pandas as pd

INTERVALS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def interval_delta(interval: str) -> pd.Timedelta:
    try:
        return INTERVALS[interval]
    except KeyError:
        raise ValueError(f"Unsupported interval '{interval}'. Use one of {list(INTERVALS)}") from None


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return out.dropna()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return true_range(df).rolling(period).mean()


def daily_atr(
    intraday: pd.DataFrame, period: int, before: pd.Timestamp, weekdays_only: bool = True
) -> float | None:
    """ATR of completed UTC days strictly before `before` (normally today's midnight).

    Weekend days are dropped by default: PAXG trades 24/7 but spot gold does not, and the
    thin weekend bars would drag the ATR down.
    """
    daily = resample_ohlc(intraday[intraday.index < before], "1D")
    if weekdays_only:
        daily = daily[daily.index.day_of_week < 5]
    series = atr(daily, period).dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)
