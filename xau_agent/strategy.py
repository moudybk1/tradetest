"""Asian range breakout - the trigger.

Pure functions so the live bot and the backtest run exactly the same rules.

Rules (all times UTC, see config.yaml -> strategy):
1. Mark the high/low of the Asian session (00:00-07:00).
2. Skip the day if that range is too wide or too narrow compared with the daily ATR.
3. From 07:00 until entry_end, the FIRST candle that closes beyond the range (+ a small buffer)
   is the signal: close above -> long, close below -> short. One signal per day at most.
4. Stop at the range middle (or opposite side), never closer than min_stop_atr.
   Take profit at reward_risk x the stop distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd

from .config import StrategyConfig
from .indicators import interval_delta


@dataclass(frozen=True)
class Signal:
    side: str  # "long" | "short"
    entry: float
    stop: float
    take_profit: float
    candle_time: pd.Timestamp  # open time of the breakout candle
    range_high: float
    range_low: float
    daily_atr: float

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)

    def with_entry(self, entry: float, reward_risk: float) -> "Signal":
        """Re-anchor on the real entry price, keeping the stop and the reward:risk ratio."""
        risk = entry - self.stop if self.side == "long" else self.stop - entry
        tp = entry + reward_risk * risk if self.side == "long" else entry - reward_risk * risk
        return Signal(
            self.side, entry, self.stop, tp, self.candle_time, self.range_high, self.range_low, self.daily_atr
        )


def _at(day: pd.Timestamp, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day.date(), t), tz="UTC")


def day_candles(candles: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start = day.normalize()
    return candles[(candles.index >= start) & (candles.index < start + pd.Timedelta(days=1))]


def asian_range(candles: pd.DataFrame, day: pd.Timestamp, cfg: StrategyConfig) -> tuple[float, float] | None:
    start, end = _at(day, cfg.asian_start_utc), _at(day, cfg.asian_end_utc)
    window = candles[(candles.index >= start) & (candles.index < end)]
    expected = (end - start) / interval_delta(cfg.timeframe)
    # Missing data (feed gap, market closed) -> no reliable range, no trade.
    if len(window) < 0.8 * expected:
        return None
    return float(window["high"].max()), float(window["low"].min())


def find_breakout(candles: pd.DataFrame, day: pd.Timestamp, atr: float, cfg: StrategyConfig) -> Signal | None:
    """Return the first breakout signal of `day`, using only CLOSED candles in `candles`."""
    if atr is None or atr <= 0:
        return None
    rng = asian_range(candles, day, cfg)
    if rng is None:
        return None
    high, low = rng
    width = high - low
    if width > cfg.max_range_atr * atr or width < cfg.min_range_atr * atr:
        return None

    buffer = cfg.breakout_buffer_atr * atr
    start, end = _at(day, cfg.asian_end_utc), _at(day, cfg.entry_end_utc)
    window = candles[(candles.index >= start) & (candles.index < end)]
    min_stop = cfg.min_stop_atr * atr
    mid = (high + low) / 2

    for ts, row in window.iterrows():
        close = float(row["close"])
        if close > high + buffer:
            side = "long"
            stop = mid if cfg.stop_mode == "mid" else low
            stop = min(stop, close - min_stop)
            tp = close + cfg.reward_risk * (close - stop)
        elif close < low - buffer:
            side = "short"
            stop = mid if cfg.stop_mode == "mid" else high
            stop = max(stop, close + min_stop)
            tp = close - cfg.reward_risk * (stop - close)
        else:
            continue
        return Signal(side, close, stop, tp, ts, high, low, atr)
    return None
