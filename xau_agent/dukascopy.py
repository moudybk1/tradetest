"""Free XAUUSD 1-minute history from Dukascopy's public datafeed, with the real bid/ask spread.

Each day is two small LZMA files (bid and ask candles). Record layout, big-endian, 24 bytes:
seconds since midnight UTC, open, close, low, high (integers, divide by the price divisor), volume.
"""

from __future__ import annotations

import logging
import lzma
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{y}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
RECORD = np.dtype([("t", ">u4"), ("open", ">u4"), ("close", ">u4"), ("low", ">u4"), ("high", ">u4"),
                   ("volume", ">f4")])


def day_url(symbol: str, day: pd.Timestamp, side: str) -> str:
    # Dukascopy months are 0-based: January = 00.
    return URL.format(symbol=symbol, y=day.year, m=day.month - 1, d=day.day, side=side)


def parse_bi5(raw: bytes, day: pd.Timestamp, divisor: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rec = np.frombuffer(lzma.decompress(raw), dtype=RECORD)
    df = pd.DataFrame({
        "open": rec["open"] / divisor, "high": rec["high"] / divisor, "low": rec["low"] / divisor,
        "close": rec["close"] / divisor, "volume": rec["volume"].astype(float),
    })
    df.index = day.normalize() + pd.to_timedelta(rec["t"].astype("int64"), unit="s")
    return df


class DukascopyRefused(RuntimeError):
    pass


def _get(session: requests.Session, url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return b""
            if resp.status_code == 429:
                raise DukascopyRefused(
                    "Dukascopy refused the automated download (HTTP 429). Use one of these instead:\n"
                    "  - MT5: python -m xau_agent export-mt5 --days 730 --out data/xauusd_m1.csv\n"
                    "  - MT5 without Python: Ctrl+U > Bars > XAUUSD, M1 > Request > Export Bars, then\n"
                    "    python -m xau_agent compare --csv <file> --server-utc-offset 2\n"
                    "  - Dukascopy's own Historical Data Export page in a browser (CSV)")
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            if attempt == tries - 1:
                raise
            log.warning("retrying %s: %s", url, exc)
            time.sleep(2**attempt)
    return b""


def fetch_dukascopy(start: pd.Timestamp, end: pd.Timestamp, symbol: str = "XAUUSD", divisor: float = 1000.0,
                    workers: int = 8, session: requests.Session | None = None) -> pd.DataFrame:
    """M1 bid candles in UTC plus `spread_price` (ask close - bid close) for [start, end)."""
    session = session or requests.Session()
    days = [d for d in pd.date_range(start.normalize(), end.normalize(), freq="1D", tz="UTC", inclusive="left")
            if d.day_of_week != 5]  # no trading on Saturdays

    def one_day(day: pd.Timestamp) -> pd.DataFrame | None:
        bid = parse_bi5(_get(session, day_url(symbol, day, "BID")), day, divisor)
        ask = parse_bi5(_get(session, day_url(symbol, day, "ASK")), day, divisor)
        bid = bid[bid["volume"] > 0]  # minutes with no ticks are flat placeholders
        if bid.empty:
            return None
        out = bid[["open", "high", "low", "close"]].copy()
        out["spread_price"] = (ask["close"].reindex(out.index) - out["close"]).clip(lower=0)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = [f for f in pool.map(one_day, days) if f is not None]
    if not frames:
        raise ValueError(f"no {symbol} data returned for {start.date()} - {end.date()}")
    df = pd.concat(frames).sort_index()
    df.index.name = "time"
    check_prices(df, symbol)
    return df


def check_prices(df: pd.DataFrame, symbol: str) -> None:
    """Catch a wrong price divisor or corrupt download before it reaches a backtest."""
    median = float(df["close"].median())
    if not 300 <= median <= 50_000:
        raise ValueError(f"{symbol} median price {median:.4f} looks wrong - check the price divisor")
    if (df["high"] < df["low"]).any() or (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9).any():
        raise ValueError(f"{symbol} candles have high below low/open/close - corrupt download?")
