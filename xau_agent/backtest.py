"""Backtest the breakout RULES (not the AI agent - see README for why) on historical candles."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import StrategyConfig
from .indicators import atr, interval_delta, resample_ohlc
from .strategy import _at, day_candles, find_breakout

TIME_COLUMNS = ("time", "timestamp", "datetime", "date", "gmt time", "local time")


def load_csv(path: str | Path, timeframe: str) -> pd.DataFrame:
    """Load OHLC candles from CSV (UTC times). Works with Dukascopy exports and most others."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in TIME_COLUMNS if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"no time column found; expected one of {TIME_COLUMNS}")
    raw = df[time_col]
    if pd.api.types.is_numeric_dtype(raw):
        unit = "ms" if raw.iloc[0] > 1e11 else "s"
        index = pd.to_datetime(raw, unit=unit, utc=True)
    else:
        # Dukascopy writes "24.10.2025 07:15:00.000 GMT+0000"
        text = raw.astype(str).str.replace(r"\s*GMT[+-]\d{4}$", "", regex=True)
        dayfirst = bool(text.iloc[0][2:3] == ".")
        index = pd.to_datetime(text, utc=True, dayfirst=dayfirst, format="mixed")
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = index
    out = out.sort_index()
    out = out[~out.index.duplicated()]
    return resample_ohlc(out, interval_delta(timeframe))


def fetch_hyperliquid(coin: str, timeframe: str, days: int, testnet: bool = False) -> pd.DataFrame:
    """Recent candles from Hyperliquid's public API (no key needed). Max ~5000 candles per request."""
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    dex = coin.split(":")[0] if ":" in coin else ""
    info = Info(constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL, skip_ws=True,
                perp_dexs=["", dex] if dex else None)
    step_ms = int(interval_delta(timeframe).total_seconds() * 1000)
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows = []
    while start < end:
        chunk_end = min(end, start + 4000 * step_ms)
        batch = info.candles_snapshot(coin, timeframe, start, chunk_end)
        rows += [{"time": c["t"], "open": c["o"], "high": c["h"], "low": c["l"], "close": c["c"]} for c in batch]
        start = chunk_end
    df = pd.DataFrame(rows).drop_duplicates("time")
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    return df.astype(float).sort_index()


def run_backtest(candles: pd.DataFrame, cfg: StrategyConfig, cost_bps: float, risk_pct: float):
    """Returns (trades DataFrame, stats dict). Results are in R (1R = the amount risked per trade)."""
    daily = resample_ohlc(candles, "1D")
    if cfg.weekdays_only:
        daily = daily[daily.index.day_of_week < 5]
    # ATR known at the START of each day = ATR of the previous completed day (same as the live bot).
    atr_at_open = atr(daily, cfg.atr_period).shift(1)

    trades = []
    for day, day_atr in atr_at_open.dropna().items():
        dc = day_candles(candles, day)
        signal = find_breakout(dc, day, float(day_atr), cfg)
        if signal is None:
            continue
        risk = signal.stop_distance
        direction = 1 if signal.side == "long" else -1
        flatten_at = _at(day, cfg.flatten_utc)
        exit_px, reason, exit_time = None, None, None
        for ts, bar in dc[dc.index > signal.candle_time].iterrows():
            if ts >= flatten_at:
                exit_px, reason, exit_time = float(bar["open"]), "flatten", ts
                break
            hit_stop = bar["low"] <= signal.stop if direction == 1 else bar["high"] >= signal.stop
            hit_tp = bar["high"] >= signal.take_profit if direction == 1 else bar["low"] <= signal.take_profit
            if hit_stop:  # both in one bar -> assume the stop came first (conservative)
                exit_px, reason, exit_time = signal.stop, "stop", ts
                break
            if hit_tp:
                exit_px, reason, exit_time = signal.take_profit, "target", ts
                break
        if exit_px is None:
            exit_px, reason, exit_time = float(dc["close"].iloc[-1]), "end of data", dc.index[-1]
        cost_r = signal.entry * cost_bps / 10_000 / risk
        r = direction * (exit_px - signal.entry) / risk - cost_r
        trades.append({
            "day": day.date(), "side": signal.side, "entry_time": signal.candle_time, "entry": signal.entry,
            "stop": signal.stop, "target": signal.take_profit, "exit_time": exit_time, "exit": exit_px,
            "reason": reason, "r": r,
        })

    df = pd.DataFrame(trades)
    return df, summarize(df, risk_pct, len(atr_at_open.dropna()))


def summarize(trades: pd.DataFrame, risk_pct: float, days: int) -> dict:
    if trades.empty:
        return {"days_tested": days, "trades": 0}
    r = trades["r"]
    curve = r.cumsum()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    growth = (1 + r * risk_pct / 100).prod()
    return {
        "days_tested": days,
        "trades": len(r),
        "win_rate_pct": round((r > 0).mean() * 100, 1),
        "avg_r": round(r.mean(), 3),
        "total_r": round(r.sum(), 2),
        "profit_factor": round(wins / losses, 2) if losses > 0 else float("inf"),
        "max_drawdown_r": round((curve.cummax() - curve).max(), 2),
        "worst_losing_streak": int(max_streak(r < 0)),
        f"return_pct_at_{risk_pct}pct_risk": round((growth - 1) * 100, 1),
        "exits": trades["reason"].value_counts().to_dict(),
    }


def max_streak(mask: pd.Series) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best
