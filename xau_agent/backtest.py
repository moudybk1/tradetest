"""Backtest the entry RULES (not the AI agent - see README for why) on historical candles.

Signals come from the signal timeframe (e.g. M5). Exits are checked on the finest data you load
(ideally M1), so a 25-pip stop is tested at 1-minute precision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .indicators import atr, interval_delta, resample_ohlc
from .strategy import _at, get_strategy

TIME_COLUMNS = ("time", "timestamp", "datetime", "date", "gmt time", "local time")


@dataclass
class Costs:
    spread_pips: float = 3.0       # used when the data has no spread column
    slippage_pips: float = 0.5     # extra pips lost on every trade (entry + exit)
    commission_per_lot: float = 0.0  # round-turn commission in $ per 1.0 lot


# ---- data loading -------------------------------------------------------------------------------
def load_csv(path: str | Path, server_utc_offset_hours: float = 0.0, point: float = 0.01) -> pd.DataFrame:
    """Load OHLC candles from CSV at their native resolution.

    Understands: this tool's own export (`export-mt5`), Dukascopy exports, and MT5's
    "Export Bars" file (<DATE> <TIME> ... <SPREAD>, tab separated, broker server time ->
    pass server_utc_offset_hours, e.g. 2 or 3 for most brokers).
    A spread column in points is converted to price with `point` (0.01 for 2-decimal gold).
    """
    with open(path) as fh:
        sep = "\t" if "\t" in fh.readline() else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:  # MT5 Export Bars
        index = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), format="%Y.%m.%d %H:%M:%S",
                               errors="coerce")
        if index.isna().any():
            index = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), format="mixed")
        index = index.dt.tz_localize("UTC")
    else:
        time_col = next((c for c in TIME_COLUMNS if c in df.columns), None)
        if time_col is None:
            raise ValueError(f"no time column found; expected one of {TIME_COLUMNS} or <DATE>/<TIME>")
        raw = df[time_col]
        if pd.api.types.is_numeric_dtype(raw):
            index = pd.to_datetime(raw, unit="ms" if raw.iloc[0] > 1e11 else "s", utc=True)
        else:
            # Dukascopy writes "24.10.2025 07:15:00.000 GMT+0000"
            text = raw.astype(str).str.replace(r"\s*GMT[+-]\d{4}$", "", regex=True)
            index = pd.to_datetime(text, utc=True, dayfirst=text.iloc[0][2:3] == ".", format="mixed")
    out = df[["open", "high", "low", "close"]].astype(float)
    if "spread_price" in df.columns:
        out["spread"] = df["spread_price"].astype(float)
    elif "spread" in df.columns:
        out["spread"] = df["spread"].astype(float) * point
    out.index = pd.DatetimeIndex(index) - pd.Timedelta(hours=server_utc_offset_hours)
    out = out.sort_index()
    return out[~out.index.duplicated()]


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


def resample_with_spread(df: pd.DataFrame, rule) -> pd.DataFrame:
    out = resample_ohlc(df, rule)
    if "spread" in df.columns:
        out["spread"] = df["spread"].resample(rule, label="left", closed="left").mean().reindex(out.index)
    return out


# ---- engine -------------------------------------------------------------------------------------
def _first_exit(t, o, h, low, c, start: int, end: int, side: str, stop: float, tp: float):
    """Index and price of the first stop/target touch in bars [start, end). Stop wins ties."""
    hh, ll = h[start:end], low[start:end]
    hit_sl = ll <= stop if side == "long" else hh >= stop
    hit_tp = hh >= tp if side == "long" else ll <= tp
    i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
    i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else None
    if i_sl is not None and (i_tp is None or i_sl <= i_tp):
        return start + i_sl, stop, "stop"
    if i_tp is not None:
        return start + i_tp, tp, "target"
    if end > start:
        return end - 1, float(c[end - 1]), "flatten"
    return None


def run_backtest(candles: pd.DataFrame, cfg: Config, costs: Costs | None = None,
                 strategy: str | None = None, lots: float | None = None):
    """Returns (trades DataFrame, stats dict). Dollar results assume `lots` per trade."""
    costs = costs or Costs()
    s, r = cfg.strategy, cfg.risk
    strat = get_strategy(strategy or s.name)
    lots = r.fixed_lots if lots is None else lots
    tf = interval_delta(s.timeframe)

    base_step = candles.index.to_series().diff().median()
    if base_step > tf:
        raise ValueError(f"data is {base_step} candles but strategy.timeframe is {s.timeframe}; load finer data")
    bars = resample_with_spread(candles, tf) if base_step < tf else candles
    hourly = resample_ohlc(candles, "1h")
    daily = resample_ohlc(candles, "1D")
    if s.weekdays_only:
        daily = daily[daily.index.day_of_week < 5]
    # ATR known at the START of each day = ATR of the previous completed day (same as the live bot).
    atr_at_open = atr(daily, s.atr_period).shift(1)
    prepared = strat.prepare(bars, hourly, cfg)

    # Exit simulation on the finest bars.
    t = candles.index.as_unit("ns").asi8  # nanoseconds, to compare with Timestamp.value
    o, h, low, c = (candles[k].to_numpy() for k in ("open", "high", "low", "close"))
    spread = candles["spread"].to_numpy() if "spread" in candles.columns else None
    pip_value = s.pip_size * r.contract_oz * lots  # $ per pip for this lot size

    trades = []
    for day, day_bars in prepared.groupby(prepared.index.normalize()):
        if s.weekdays_only and day.day_of_week >= 5:
            continue
        day_atr = atr_at_open.get(day)
        day_atr = float(day_atr) if day_atr is not None and not np.isnan(day_atr) else None
        signals = strat.day_signals(day_bars, day, day_atr, cfg)
        flatten_ns = _at(day, s.flatten_utc).value
        busy_until, taken = None, 0
        for sig in signals:
            if taken >= r.max_trades_per_day:
                break
            entry_time = sig.candle_time + tf
            if busy_until is not None and entry_time < busy_until:
                continue  # already in a trade (the live bot holds one position at a time)
            start = int(np.searchsorted(t, entry_time.value, "left"))
            end = int(np.searchsorted(t, flatten_ns, "left"))
            result = _first_exit(t, o, h, low, c, start, end, sig.side, sig.stop, sig.take_profit)
            if result is None:
                continue
            idx, exit_px, reason = result
            exit_time = pd.Timestamp(t[idx], tz="UTC")
            direction = 1 if sig.side == "long" else -1
            spread_pips = (spread[start] / s.pip_size) if spread is not None and start < len(spread) \
                else costs.spread_pips
            cost_pips = spread_pips + costs.slippage_pips
            pips = direction * (exit_px - sig.entry) / s.pip_size - cost_pips
            usd = pips * pip_value - costs.commission_per_lot * lots
            trades.append({
                "strategy": sig.strategy, "day": day.date(), "side": sig.side, "entry_time": entry_time,
                "entry": round(sig.entry, 2), "stop": round(sig.stop, 2), "target": round(sig.take_profit, 2),
                "exit_time": exit_time, "exit": round(exit_px, 2), "reason": reason,
                "cost_pips": round(cost_pips, 2), "pips": round(pips, 2), "usd": round(usd, 2),
                "r": pips / (sig.stop_distance / s.pip_size),
            })
            busy_until = exit_time + pd.Timedelta(minutes=1)
            taken += 1

    df = pd.DataFrame(trades)
    days = int((pd.Series(prepared.index.normalize().unique()).dt.day_of_week < 5).sum())
    return df, summarize(df, days, s.sl_pips, s.tp_pips, lots)


def summarize(trades: pd.DataFrame, days: int, sl_pips: float, tp_pips: float, lots: float) -> dict:
    if trades.empty:
        return {"trading_days": days, "trades": 0}
    usd = trades["usd"]
    curve = usd.cumsum()
    wins, losses = usd[usd > 0].sum(), -usd[usd < 0].sum()
    avg_cost = trades["cost_pips"].mean()
    return {
        "trading_days": days,
        "trades": len(usd),
        "trades_per_week": round(len(usd) / max(days, 1) * 5, 1),
        "win_rate_pct": round((usd > 0).mean() * 100, 1),
        # Win rate needed to break even with fixed SL/TP after costs: (SL + cost) / (SL + TP)
        "breakeven_win_rate_pct": round((sl_pips + avg_cost) / (sl_pips + tp_pips) * 100, 1),
        "avg_cost_pips": round(avg_cost, 2),
        "net_pips": round(trades["pips"].sum(), 1),
        f"net_usd_at_{lots}_lot": round(usd.sum(), 2),
        "profit_factor": round(wins / losses, 2) if losses > 0 else float("inf"),
        "max_drawdown_usd": round((curve.cummax() - curve).max(), 2),
        "worst_losing_streak": int(max_streak(usd < 0)),
        "exits": trades["reason"].value_counts().to_dict(),
    }


def max_streak(mask: pd.Series) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def split_stats(trades: pd.DataFrame, split: str, cfg: Config, lots: float) -> dict:
    """Stats before and after `split` (YYYY-MM-DD): tune on the first part, judge on the second."""
    s = cfg.strategy
    cut = pd.Timestamp(split).date()
    out = {}
    for label, part in (("in_sample", trades[trades["day"] < cut]), ("out_of_sample", trades[trades["day"] >= cut])):
        days = int(np.busday_count(part["day"].min(), part["day"].max()) + 1) if not part.empty else 0
        out[label] = summarize(part.reset_index(drop=True), days, s.sl_pips, s.tp_pips, lots)
    return out


def monthly(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    month = pd.to_datetime(trades["day"]).dt.to_period("M")
    return trades.groupby(month)["usd"].sum().round(2)
