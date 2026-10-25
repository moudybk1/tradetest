import pandas as pd
import pytest

from xau_agent.backtest import load_csv, run_backtest
from xau_agent.config import StrategyConfig, load_config

from .conftest import breakout_day, flat_day, history


def test_backtest_finds_target_trade():
    days = pd.bdate_range("2026-10-19", periods=3, tz="UTC")
    candles = pd.concat([history(30, "2026-10-19"), breakout_day(days[0]), flat_day(days[1]),
                         breakout_day(days[2], "short")])
    trades, stats = run_backtest(candles, StrategyConfig(), cost_bps=0, risk_pct=0.5)
    assert stats["trades"] == 2
    assert list(trades["reason"]) == ["target", "target"]
    assert trades["r"].tolist() == pytest.approx([2.0, 2.0])
    assert stats["total_r"] == 4.0


def test_backtest_costs_reduce_r():
    candles = pd.concat([history(30, "2026-10-19"), breakout_day(pd.Timestamp("2026-10-19", tz="UTC"))])
    trades, _ = run_backtest(candles, StrategyConfig(), cost_bps=10, risk_pct=0.5)
    # 10 bps of 2006 = 2.006 = 0.334R on a 6-point stop
    assert trades["r"].iloc[0] == pytest.approx(2 - 2.006 / 6)


def test_load_dukascopy_csv(tmp_path):
    path = tmp_path / "xau.csv"
    path.write_text(
        "Gmt time,Open,High,Low,Close,Volume\n"
        "21.10.2026 00:00:00.000,2000,2001,1999,2000.5,10\n"
        "21.10.2026 00:05:00.000,2000.5,2003,2000,2002,10\n"
        "21.10.2026 00:15:00.000,2002,2002,2001,2001,10\n"
    )
    df = load_csv(path, "15m")
    assert len(df) == 2
    assert df.iloc[0]["high"] == 2003 and df.iloc[0]["close"] == 2002
    assert df.index[0] == pd.Timestamp("2026-10-21 00:00", tz="UTC")


def test_example_config_loads(tmp_path):
    cfg = load_config("config.example.yaml")
    assert cfg.mode == "paper" and cfg.risk.risk_per_trade_pct == 0.5


def test_unknown_config_key_is_an_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  risk_per_trade: 1\n")
    with pytest.raises(ValueError, match="risk_per_trade"):
        load_config(path)


def test_unquoted_times_parse(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("strategy:\n  asian_end_utc: 07:00\n")
    assert load_config(path).strategy.asian_end_utc.hour == 7


def test_risk_limits_validated(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  risk_per_trade_pct: 5\n")
    with pytest.raises(ValueError, match="risk_per_trade_pct"):
        load_config(path)
