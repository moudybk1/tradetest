"""Entry triggers. Pure functions, shared by the live bot and the backtest so both run the same rules.

Every strategy has two steps:
  prepare(bars, hourly, cfg)        add indicator columns. Must be causal (row t uses data up to t only).
  day_signals(day_bars, day, atr, cfg)  every signal of one UTC day, in time order, from CLOSED bars.

The bot/backtest then decides which signals can actually be traded (one position at a time,
max trades per day, risk rules). Strategies:

  asian_breakout  first close beyond the Asian (00:00-07:00 UTC) range
  ny_orb          first close beyond the New York opening range (09:30-10:00 New York time)
  asian_sweep     London takes out the Asian high/low, then closes back inside -> trade the reversal
  trend_pullback  H1 EMA trend + M5 pullback to the EMA that closes back in the trend direction
  ifvg_hammer     hammer (or shooting star), then a candle closes through a fair value gap -> enter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config, StrategyConfig
from .indicators import ema, interval_delta

NEW_YORK = "America/New_York"


@dataclass(frozen=True)
class Signal:
    strategy: str
    side: str  # "long" | "short"
    entry: float
    stop: float
    take_profit: float
    candle_time: pd.Timestamp  # open time of the signal candle; entry is at its close
    context: dict = field(default_factory=dict, compare=False)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)

    def with_entry(self, entry: float, cfg: StrategyConfig) -> "Signal":
        """Re-anchor on the real entry price.

        fixed_pips: SL and TP move with the entry (always exactly sl_pips / tp_pips away).
        atr: the stop stays at its chart level, the target keeps the reward:risk ratio.
        """
        if cfg.exit_mode == "fixed_pips":
            stop, tp = fixed_exits(self.side, entry, cfg)
        else:
            stop = self.stop
            risk = entry - stop if self.side == "long" else stop - entry
            tp = entry + cfg.reward_risk * risk if self.side == "long" else entry - cfg.reward_risk * risk
        return Signal(self.strategy, self.side, entry, stop, tp, self.candle_time, self.context)


# ---- exits ---------------------------------------------------------------------------------------
def fixed_exits(side: str, entry: float, cfg: StrategyConfig) -> tuple[float, float]:
    sl, tp = cfg.sl_pips * cfg.pip_size, cfg.tp_pips * cfg.pip_size
    return (entry - sl, entry + tp) if side == "long" else (entry + sl, entry - tp)


def make_signal(name: str, side: str, entry: float, chart_stop: float, ts: pd.Timestamp,
                atr: float | None, cfg: StrategyConfig, **context) -> Signal:
    """Build a signal with the configured exit mode. `chart_stop` is the setup's natural stop level."""
    if cfg.exit_mode == "fixed_pips":
        stop, tp = fixed_exits(side, entry, cfg)
    else:
        min_stop = cfg.min_stop_atr * atr if atr else 0.0
        if side == "long":
            stop = min(chart_stop, entry - min_stop)
            tp = entry + cfg.reward_risk * (entry - stop)
        else:
            stop = max(chart_stop, entry + min_stop)
            tp = entry - cfg.reward_risk * (stop - entry)
    return Signal(name, side, float(entry), float(stop), float(tp), ts, context)


# ---- helpers ------------------------------------------------------------------------------------
def _at(day: pd.Timestamp, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day.date(), t), tz="UTC")


def _at_ny(day: pd.Timestamp, t: time) -> pd.Timestamp:
    """`t` New York time on `day`, in UTC. Handles US daylight saving automatically."""
    return pd.Timestamp(datetime.combine(day.date(), t)).tz_localize(NEW_YORK).tz_convert("UTC")


def day_candles(candles: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start = day.normalize()
    return candles[(candles.index >= start) & (candles.index < start + pd.Timedelta(days=1))]


def _window(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return bars[(bars.index >= start) & (bars.index < end)]


def session_range(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> tuple[float, float] | None:
    """High/low of [start, end). None if more than 20% of the bars are missing (feed gap, market closed)."""
    window = _window(bars, start, end)
    expected = (end - start) / interval_delta(timeframe)
    if len(window) < 0.8 * expected:
        return None
    return float(window["high"].max()), float(window["low"].min())


def asian_range(bars: pd.DataFrame, day: pd.Timestamp, start: time, end: time, timeframe: str):
    return session_range(bars, _at(day, start), _at(day, end), timeframe)


def _no_prepare(bars: pd.DataFrame, hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return bars


# ---- 1. Asian range breakout --------------------------------------------------------------------
def asian_breakout(day_bars: pd.DataFrame, day: pd.Timestamp, atr: float | None, cfg: Config) -> list[Signal]:
    p, s = cfg.asian_breakout, cfg.strategy
    if not atr or atr <= 0:
        return []
    rng = asian_range(day_bars, day, p.asian_start_utc, p.asian_end_utc, s.timeframe)
    if rng is None:
        return []
    high, low = rng
    width = high - low
    if width > p.max_range_atr * atr or width < p.min_range_atr * atr:
        return []
    buffer = p.breakout_buffer_atr * atr
    mid = (high + low) / 2
    for ts, row in _window(day_bars, _at(day, p.asian_end_utc), _at(day, p.entry_end_utc)).iterrows():
        close = float(row["close"])
        if close > high + buffer:
            stop = mid if p.stop_mode == "mid" else low
            return [make_signal("asian_breakout", "long", close, stop, ts, atr, s,
                                range_high=high, range_low=low, daily_atr=atr)]
        if close < low - buffer:
            stop = mid if p.stop_mode == "mid" else high
            return [make_signal("asian_breakout", "short", close, stop, ts, atr, s,
                                range_high=high, range_low=low, daily_atr=atr)]
    return []


# ---- 2. New York opening range breakout ---------------------------------------------------------
def ny_orb(day_bars: pd.DataFrame, day: pd.Timestamp, atr: float | None, cfg: Config) -> list[Signal]:
    p, s = cfg.ny_orb, cfg.strategy
    open_at = _at_ny(day, p.start_ny)
    range_end = open_at + pd.Timedelta(minutes=p.minutes)
    rng = session_range(day_bars, open_at, range_end, s.timeframe)
    if rng is None:
        return []
    high, low = rng
    if (high - low) > p.max_range_pips * s.pip_size:
        return []  # opening range too wide for the fixed stop
    for ts, row in _window(day_bars, range_end, _at_ny(day, p.entry_end_ny)).iterrows():
        close = float(row["close"])
        if close > high:
            return [make_signal("ny_orb", "long", close, low, ts, atr, s, range_high=high, range_low=low)]
        if close < low:
            return [make_signal("ny_orb", "short", close, high, ts, atr, s, range_high=high, range_low=low)]
    return []


# ---- 3. Asian high/low sweep and reversal -------------------------------------------------------
def asian_sweep(day_bars: pd.DataFrame, day: pd.Timestamp, atr: float | None, cfg: Config) -> list[Signal]:
    p, s = cfg.asian_sweep, cfg.strategy
    rng = asian_range(day_bars, day, p.asian_start_utc, p.asian_end_utc, s.timeframe)
    if rng is None:
        return []
    high, low = rng
    if atr and ((high - low) > p.max_range_atr * atr or (high - low) < p.min_range_atr * atr):
        return []
    sl_distance = s.sl_pips * s.pip_size
    signals = []
    swept_high = swept_low = None  # most extreme price beyond the range so far
    done_short = done_long = False
    for ts, row in _window(day_bars, _at(day, p.asian_end_utc), _at(day, p.end_utc)).iterrows():
        h, lo, close = float(row["high"]), float(row["low"]), float(row["close"])
        if h > high:
            swept_high = max(swept_high or h, h)
        if lo < low:
            swept_low = min(swept_low or lo, lo)
        if not done_short and swept_high is not None and close < high:
            done_short = True
            # The stop must sit beyond the sweep wick, otherwise the setup is already invalid.
            if s.exit_mode == "atr" or swept_high < close + sl_distance:
                signals.append(make_signal("asian_sweep", "short", close, swept_high, ts, atr, s,
                                           range_high=high, range_low=low, sweep_extreme=swept_high))
        if not done_long and swept_low is not None and close > low:
            done_long = True
            if s.exit_mode == "atr" or swept_low > close - sl_distance:
                signals.append(make_signal("asian_sweep", "long", close, swept_low, ts, atr, s,
                                           range_high=high, range_low=low, sweep_extreme=swept_low))
    return signals


# ---- 4. Trend pullback --------------------------------------------------------------------------
def parse_sessions(text: str) -> list[tuple[time, time]]:
    sessions = []
    for part in text.split(","):
        start, end = part.strip().split("-")
        sessions.append((time.fromisoformat(start), time.fromisoformat(end)))
    return sessions


def prepare_pullback(bars: pd.DataFrame, hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = cfg.trend_pullback
    out = bars.copy()
    out["ema"] = ema(out["close"], p.ema)
    out["swing_low"] = out["low"].rolling(p.swing_bars).min()
    out["swing_high"] = out["high"].rolling(p.swing_bars).max()
    fast, slow = ema(hourly["close"], p.trend_fast), ema(hourly["close"], p.trend_slow)
    trend = pd.Series(np.where(fast > slow, 1.0, -1.0), index=hourly.index + pd.Timedelta(hours=1))
    trend.iloc[: p.trend_slow] = np.nan  # EMA warm-up
    # Trend known when each bar closes = last H1 candle that had CLOSED by then (no look-ahead).
    bar_close = out.index + interval_delta(cfg.strategy.timeframe)
    out["trend"] = trend.reindex(trend.index.union(bar_close)).ffill().reindex(bar_close).to_numpy()
    return out


def trend_pullback(day_bars: pd.DataFrame, day: pd.Timestamp, atr: float | None, cfg: Config) -> list[Signal]:
    p, s = cfg.trend_pullback, cfg.strategy
    sl_distance = s.sl_pips * s.pip_size
    signals = []
    for start, end in parse_sessions(p.sessions_utc):
        cooldown = 0
        for ts, row in _window(day_bars, _at(day, start), _at(day, end)).iterrows():
            if cooldown:
                cooldown -= 1
                continue
            trend = row.get("trend")
            if trend is None or np.isnan(trend):
                continue
            close, op, ema_v = float(row["close"]), float(row["open"]), float(row["ema"])
            if trend > 0 and row["low"] <= ema_v < close and close > op:
                swing = float(row["swing_low"])
                if s.exit_mode == "atr" or swing > close - sl_distance:
                    signals.append(make_signal("trend_pullback", "long", close, swing, ts, atr, s,
                                               ema=ema_v, swing_low=swing, h1_trend="up"))
                    cooldown = p.cooldown_bars
            elif trend < 0 and row["high"] >= ema_v > close and close < op:
                swing = float(row["swing_high"])
                if s.exit_mode == "atr" or swing < close + sl_distance:
                    signals.append(make_signal("trend_pullback", "short", close, swing, ts, atr, s,
                                               ema=ema_v, swing_high=swing, h1_trend="down"))
                    cooldown = p.cooldown_bars
    return sorted(signals, key=lambda sig: sig.candle_time)


# ---- 5. Hammer + IFVG ---------------------------------------------------------------------------
def candle_shapes(bars: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """(hammer, shooting_star) flags. A shooting star is the hammer turned upside down (for shorts)."""
    p, pip = cfg.ifvg_hammer, cfg.strategy.pip_size
    o, h, low, c = bars["open"], bars["high"], bars["low"], bars["close"]
    rng = h - low
    body = (c - o).abs()
    lower = pd.concat([o, c], axis=1).min(axis=1) - low
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    big_enough = rng >= p.hammer_min_range_pips * pip
    hammer = big_enough & (lower >= p.hammer_wick_ratio * body) & (upper <= p.hammer_max_opposite_wick * rng)
    star = big_enough & (upper >= p.hammer_wick_ratio * body) & (lower <= p.hammer_max_opposite_wick * rng)
    if p.hammer_swing_bars > 0:
        hammer &= low <= low.rolling(p.hammer_swing_bars, min_periods=1).min()
        star &= h >= h.rolling(p.hammer_swing_bars, min_periods=1).max()
    return hammer, star


def prepare_ifvg_hammer(bars: pd.DataFrame, hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mark, bar by bar and using only the past:

    ifvg_long   this candle CLOSED above a bearish FVG (gap left by a drop) -> it is now a bullish IFVG
    ifvg_short  this candle CLOSED below a bullish FVG -> bearish IFVG
    hammer_low  low of the most recent hammer in the last N candles, NaN if none or if price has
                traded below it since (a hammer whose low breaks has failed). star_high: the mirror.
    """
    p, pip = cfg.ifvg_hammer, cfg.strategy.pip_size
    out = bars.copy()
    hammer, star = candle_shapes(out, cfg)
    o, h, low, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(out)
    ifvg_long, ifvg_short = np.zeros(n, bool), np.zeros(n, bool)
    zone_lo, zone_hi = np.full(n, np.nan), np.full(n, np.nan)
    hammer_low, star_high = np.full(n, np.nan), np.full(n, np.nan)
    bear_fvgs: list[tuple[int, float, float]] = []  # (formed at, bottom, top)
    bull_fvgs: list[tuple[int, float, float]] = []
    min_gap = p.min_fvg_pips * pip
    last_h = last_s = None  # (index, level)
    hammer_arr, star_arr = hammer.to_numpy(), star.to_numpy()

    for i in range(n):
        # 1) Inversions: this candle's close goes through an FVG that formed earlier.
        bear_fvgs = [g for g in bear_fvgs if i - g[0] <= p.fvg_lookback_bars]
        bull_fvgs = [g for g in bull_fvgs if i - g[0] <= p.fvg_lookback_bars]
        inverted = [g for g in bear_fvgs if c[i] > g[2]]
        if inverted:
            ifvg_long[i] = True
            zone_lo[i], zone_hi[i] = inverted[-1][1], inverted[-1][2]
            bear_fvgs = [g for g in bear_fvgs if g not in inverted]
        inverted = [g for g in bull_fvgs if c[i] < g[1]]
        if inverted:
            ifvg_short[i] = True
            zone_lo[i], zone_hi[i] = inverted[-1][1], inverted[-1][2]
            bull_fvgs = [g for g in bull_fvgs if g not in inverted]
        # 2) New FVG completed by this candle (it is candle 3 of the pattern).
        if i >= 2:
            if low[i - 2] - h[i] >= min_gap:
                bear_fvgs.append((i, h[i], low[i - 2]))
            if low[i] - h[i - 2] >= min_gap:
                bull_fvgs.append((i, h[i - 2], low[i]))
        # 3) Most recent hammer / shooting star, invalidated once its extreme is taken out.
        if hammer_arr[i]:
            last_h = (i, low[i])
        elif last_h and low[i] < last_h[1]:
            last_h = None
        if star_arr[i]:
            last_s = (i, h[i])
        elif last_s and h[i] > last_s[1]:
            last_s = None
        if last_h and i - last_h[0] <= p.hammer_max_bars_before:
            hammer_low[i] = last_h[1]
        if last_s and i - last_s[0] <= p.hammer_max_bars_before:
            star_high[i] = last_s[1]

    out["hammer"], out["star"] = hammer_arr, star_arr
    out["ifvg_long"], out["ifvg_short"] = ifvg_long, ifvg_short
    out["ifvg_low"], out["ifvg_high"] = zone_lo, zone_hi
    out["hammer_low"], out["star_high"] = hammer_low, star_high
    return out


def ifvg_hammer(day_bars: pd.DataFrame, day: pd.Timestamp, atr: float | None, cfg: Config) -> list[Signal]:
    """Hammer, then an IFVG appears (within hammer_max_bars_before candles) -> enter at that close."""
    p, s = cfg.ifvg_hammer, cfg.strategy
    sl_distance = s.sl_pips * s.pip_size
    check_stop = s.exit_mode == "fixed_pips" and p.require_stop_beyond_hammer
    signals = []
    for start, end in parse_sessions(p.sessions_utc):
        for ts, row in _window(day_bars, _at(day, start), _at(day, end)).iterrows():
            close = float(row["close"])
            zone = {"ifvg_low": float(row["ifvg_low"]), "ifvg_high": float(row["ifvg_high"])}
            if row["ifvg_long"] and not row["ifvg_short"] and not np.isnan(row["hammer_low"]):
                level = float(row["hammer_low"])
                if not check_stop or level > close - sl_distance:
                    signals.append(make_signal("ifvg_hammer", "long", close, level, ts, atr, s,
                                               hammer_low=level, **zone))
            elif row["ifvg_short"] and not row["ifvg_long"] and not np.isnan(row["star_high"]):
                level = float(row["star_high"])
                if not check_stop or level < close + sl_distance:
                    signals.append(make_signal("ifvg_hammer", "short", close, level, ts, atr, s,
                                               shooting_star_high=level, **zone))
    return sorted(signals, key=lambda sig: sig.candle_time)


@dataclass(frozen=True)
class Strategy:
    name: str
    prepare: Callable[[pd.DataFrame, pd.DataFrame, Config], pd.DataFrame]
    day_signals: Callable[[pd.DataFrame, pd.Timestamp, float | None, Config], list[Signal]]


REGISTRY = {
    "asian_breakout": Strategy("asian_breakout", _no_prepare, asian_breakout),
    "ny_orb": Strategy("ny_orb", _no_prepare, ny_orb),
    "asian_sweep": Strategy("asian_sweep", _no_prepare, asian_sweep),
    "trend_pullback": Strategy("trend_pullback", prepare_pullback, trend_pullback),
    "ifvg_hammer": Strategy("ifvg_hammer", prepare_ifvg_hammer, ifvg_hammer),
}


def get_strategy(name: str) -> Strategy:
    return REGISTRY[name]
