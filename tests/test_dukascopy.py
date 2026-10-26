import lzma

import numpy as np
import pandas as pd
import pytest

from xau_agent.backtest import load_csv
from xau_agent.dukascopy import RECORD, day_url, fetch_dukascopy, parse_bi5

DAY = pd.Timestamp("2026-10-21", tz="UTC")


def bi5(rows) -> bytes:
    """rows: (seconds, open, close, low, high, volume) with prices in dollars -> Dukascopy file bytes."""
    rec = np.array([(t, round(o * 1000), round(c * 1000), round(lo * 1000), round(h * 1000), v)
                    for t, o, c, lo, h, v in rows], dtype=RECORD)
    return lzma.compress(rec.tobytes(), format=lzma.FORMAT_ALONE)


def test_url_months_are_zero_based():
    assert day_url("XAUUSD", DAY, "BID").endswith("/XAUUSD/2026/09/21/BID_candles_min_1.bi5")
    assert "/2026/00/05/" in day_url("XAUUSD", pd.Timestamp("2026-01-05", tz="UTC"), "ASK")


def test_parse_bi5_field_order_and_scale():
    df = parse_bi5(bi5([(60, 4012.345, 4012.9, 4011.8, 4013.1, 12.5)]), DAY, 1000)
    row = df.iloc[0]
    assert df.index[0] == pd.Timestamp("2026-10-21 00:01", tz="UTC")
    assert (row["open"], row["high"], row["low"], row["close"]) == pytest.approx((4012.345, 4013.1, 4011.8, 4012.9))


class FakeSession:
    def __init__(self, files):
        self.files = files

    def get(self, url, timeout):
        content = self.files.get(url)

        class Resp:
            status_code = 200 if content is not None else 404

            def raise_for_status(self):
                pass

        r = Resp()
        r.content = content or b""
        return r


def test_fetch_merges_spread_and_drops_empty_minutes(tmp_path):
    bid = bi5([(0, 4010.0, 4010.5, 4009.8, 4010.6, 3.0), (60, 4010.5, 4011.0, 4010.4, 4011.2, 2.0),
               (120, 4011.0, 4011.0, 4011.0, 4011.0, 0.0)])
    ask = bi5([(0, 4010.3, 4010.8, 4010.1, 4010.9, 3.0), (60, 4010.8, 4011.35, 4010.7, 4011.5, 2.0),
               (120, 4011.3, 4011.3, 4011.3, 4011.3, 0.0)])
    files = {day_url("XAUUSD", DAY, "BID"): bid, day_url("XAUUSD", DAY, "ASK"): ask}
    df = fetch_dukascopy(DAY, DAY + pd.Timedelta(days=1), session=FakeSession(files), workers=1)
    assert len(df) == 2  # the zero-volume minute is dropped
    assert df["spread_price"].tolist() == pytest.approx([0.3, 0.35])
    path = tmp_path / "m1.csv"
    df.to_csv(path)
    loaded = load_csv(path)  # round-trips into the backtester with the spread column
    assert loaded["spread"].tolist() == pytest.approx([0.3, 0.35])


def test_wrong_divisor_is_caught():
    files = {day_url("XAUUSD", DAY, "BID"): bi5([(0, 4010.0, 4010.5, 4009.8, 4010.6, 3.0)])}
    with pytest.raises(ValueError, match="price divisor"):
        fetch_dukascopy(DAY, DAY + pd.Timedelta(days=1), divisor=1.0, session=FakeSession(files), workers=1)
