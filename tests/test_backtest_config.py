import numpy as np
import pandas as pd
import pytest

from xau_agent.backtest import Costs, _first_exit, load_csv, run_backtest, split_stats, summarize
from xau_agent.config import load_config

from .conftest import breakout_day, flat_5m, flat_day, history, ny_orb_day, to_1m

NO_COSTS = Costs(spread_pips=0, slippage_pips=0)


# ---- engine --------------------------------------------------------------------------------------
def test_asian_breakout_backtest_atr_exits(cfg):
    days = pd.bdate_range("2026-10-19", periods=3, tz="UTC")
    candles = pd.concat([history(30, "2026-10-19"), breakout_day(days[0]), flat_day(days[1]),
                         breakout_day(days[2], "short")])
    trades, stats = run_backtest(candles, cfg, NO_COSTS, lots=0.1)
    assert stats["trades"] == 2
    assert list(trades["reason"]) == ["target", "target"]
    assert trades["r"].tolist() == pytest.approx([2.0, 2.0])
    # 12-point move = 120 pips; 0.1 lot = $1 per pip
    assert trades["usd"].tolist() == pytest.approx([120.0, 120.0])


def scalp_history(sides=("long",)) -> pd.DataFrame:
    days = pd.bdate_range("2026-10-19", periods=len(sides) + 1, tz="UTC")
    frames = [flat_5m(days[0])] + [ny_orb_day(d, side=s) for d, s in zip(days[1:], sides)]
    return to_1m(pd.concat(frames))


def test_ny_orb_on_m1_data_hits_50_pip_target(scalp_cfg):
    trades, stats = run_backtest(scalp_history(("long", "short")), scalp_cfg, NO_COSTS)
    assert stats["trades"] == 2
    assert trades["pips"].tolist() == pytest.approx([50.0, 50.0])
    assert trades["usd"].tolist() == pytest.approx([50.0, 50.0])  # 0.1 lot = $1/pip
    # Entry is at the close of the 14:05 signal candle = 14:10.
    assert trades["entry_time"].iloc[0] == pd.Timestamp("2026-10-20 14:10", tz="UTC")


def test_costs_in_pips_and_breakeven(scalp_cfg):
    trades, stats = run_backtest(scalp_history(), scalp_cfg, Costs(spread_pips=3, slippage_pips=0.5,
                                                                   commission_per_lot=7))
    assert trades["pips"].iloc[0] == pytest.approx(50 - 3.5)
    assert trades["usd"].iloc[0] == pytest.approx(46.5 - 0.7)  # commission $7/lot x 0.1 lot
    assert stats["breakeven_win_rate_pct"] == pytest.approx((25 + 3.5) / 75 * 100, abs=0.1)


def test_stop_loss_costs_25_pips_plus_costs(scalp_cfg):
    df = scalp_history()
    # Make price fall through the stop (entry 2001.5 - 2.5 = 1999.0) before the target.
    crash = pd.Timestamp("2026-10-20 14:30", tz="UTC")
    df.loc[crash:crash + pd.Timedelta(minutes=4), "low"] = 1998.9
    trades, _ = run_backtest(df, scalp_cfg, Costs(spread_pips=3, slippage_pips=0))
    assert trades["reason"].iloc[0] == "stop"
    assert trades["usd"].iloc[0] == pytest.approx(-28.0)


def test_data_spread_column_overrides_fixed_spread(scalp_cfg):
    df = scalp_history()
    df["spread"] = 0.45  # 4.5 pips in price
    trades, _ = run_backtest(df, scalp_cfg, Costs(spread_pips=1, slippage_pips=0))
    assert trades["cost_pips"].iloc[0] == pytest.approx(4.5)


def test_lot_size_scales_dollars(scalp_cfg):
    trades, stats = run_backtest(scalp_history(), scalp_cfg, NO_COSTS, lots=0.5)
    assert trades["usd"].iloc[0] == pytest.approx(250.0)
    assert stats["net_usd_at_0.5_lot"] == pytest.approx(250.0)


def test_one_position_at_a_time_and_max_trades(scalp_cfg):
    scalp_cfg.strategy.name = "trend_pullback"
    scalp_cfg.risk.max_trades_per_day = 1
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-09-01", "2026-10-20", freq="1min", tz="UTC")
    idx = idx[idx.day_of_week < 5]
    close = 2000 + np.cumsum(rng.normal(0.002, 0.15, len(idx)))
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)
    trades, _ = run_backtest(df, scalp_cfg, NO_COSTS)
    assert not trades.empty
    assert trades.groupby("day").size().max() == 1
    entries, exits = trades["entry_time"].tolist(), trades["exit_time"].tolist()
    assert all(entries[i + 1] > exits[i] for i in range(len(trades) - 1))


def test_first_exit_stop_wins_ties():
    t = np.arange(3)
    o = c = np.array([100.0, 100.0, 100.0])
    h = np.array([100.5, 106.0, 100.0])
    low = np.array([99.5, 97.0, 100.0])  # bar 1 touches both stop 97.5 and target 105
    assert _first_exit(t, o, h, low, c, 0, 3, "long", 97.5, 105.0) == (1, 97.5, "stop")


def test_rejects_data_coarser_than_timeframe(scalp_cfg):
    candles = pd.concat([history(30, "2026-10-19"), breakout_day(pd.Timestamp("2026-10-19", tz="UTC"))])
    with pytest.raises(ValueError, match="finer data"):
        run_backtest(candles, scalp_cfg)  # 15m data, 5m strategy


def test_split_stats(scalp_cfg):
    trades, _ = run_backtest(scalp_history(("long", "short", "long")), scalp_cfg, NO_COSTS)
    parts = split_stats(trades, "2026-10-22", scalp_cfg, 0.1)
    assert parts["in_sample"]["trades"] == 2 and parts["out_of_sample"]["trades"] == 1


# ---- CSV loading ---------------------------------------------------------------------------------
def test_load_dukascopy_csv(tmp_path):
    path = tmp_path / "xau.csv"
    path.write_text(
        "Gmt time,Open,High,Low,Close,Volume\n"
        "21.10.2026 00:00:00.000,2000,2001,1999,2000.5,10\n"
        "21.10.2026 00:01:00.000,2000.5,2003,2000,2002,10\n"
    )
    df = load_csv(path)
    assert len(df) == 2 and df.index[1] == pd.Timestamp("2026-10-21 00:01", tz="UTC")


def test_load_mt5_export_bars_with_spread_and_offset(tmp_path):
    path = tmp_path / "XAUUSD_M1.csv"
    path.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2026.10.21\t03:00:00\t4010.10\t4011.00\t4009.50\t4010.80\t120\t0\t25\n"
        "2026.10.21\t03:01:00\t4010.80\t4012.00\t4010.20\t4011.70\t130\t0\t31\n"
    )
    df = load_csv(path, server_utc_offset_hours=3)
    assert df.index[0] == pd.Timestamp("2026-10-21 00:00", tz="UTC")
    assert df["spread"].tolist() == pytest.approx([0.25, 0.31])  # points x 0.01


def test_load_own_export_format(tmp_path):
    path = tmp_path / "m1.csv"
    path.write_text(
        "time,open,high,low,close,spread_price\n"
        "2026-10-21 00:00:00+00:00,4010.1,4011.0,4009.5,4010.8,0.22\n"
    )
    df = load_csv(path)
    assert df["spread"].iloc[0] == 0.22 and df.index[0].tzinfo is not None


# ---- config --------------------------------------------------------------------------------------
def test_example_config_loads():
    cfg = load_config("config.example.yaml")
    assert cfg.mode == "paper" and cfg.broker == "mt5"
    assert cfg.risk.sizing == "fixed_lots" and cfg.risk.fixed_lots == 0.1
    assert (cfg.strategy.sl_pips, cfg.strategy.tp_pips, cfg.strategy.pip_size) == (25, 50, 0.10)


def test_unknown_config_key_is_an_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  risk_per_trade: 1\n")
    with pytest.raises(ValueError, match="risk_per_trade"):
        load_config(path)


def test_unquoted_times_parse(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("asian_breakout:\n  asian_end_utc: 07:00\n")
    assert load_config(path).asian_breakout.asian_end_utc.hour == 7


@pytest.mark.parametrize("text, match", [
    ("risk:\n  risk_per_trade_pct: 5\n", "risk_per_trade_pct"),
    ("strategy:\n  name: martingale\n", "strategy.name"),
    ("risk:\n  sizing: yolo\n", "risk.sizing"),
])
def test_invalid_values_rejected(tmp_path, text, match):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_atr_is_an_alias_for_rr(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("strategy:\n  exit_mode: atr\n")
    assert load_config(path).strategy.exit_mode == "rr"


def test_max_stop_must_exceed_min_stop(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("strategy:\n  min_stop_pips: 20\n  max_stop_pips: 10\n")
    with pytest.raises(ValueError, match="max_stop_pips"):
        load_config(path)


def test_breakeven_with_variable_stops():
    trades = pd.DataFrame({"usd": [40.0, -22.0], "pips": [40.0, -22.0], "cost_pips": [2.0, 2.0],
                           "stop_pips": [20.0, 20.0], "target_pips": [40.0, 40.0], "risk_usd": [20.0, 20.0],
                           "r": [2.0, -1.1], "reason": ["target", "stop"]})
    st = summarize(trades, 10, 0.1)
    assert st["breakeven_win_rate_pct"] == pytest.approx((1 + 0.1) / 3 * 100, abs=0.1)
    assert st["avg_stop_pips"] == 20.0


def test_percent_fees_bybit_style(scalp_cfg):
    # 0.055% per side on a 2001.5 entry = 2 x 1.1008 = 2.2017 in price = 22.0 pips at pip 0.10
    trades, _ = run_backtest(scalp_history(), scalp_cfg, Costs(0, 0, 0, fee_pct_per_side=0.055))
    assert trades["cost_pips"].iloc[0] == pytest.approx(2 * 0.00055 * 2001.5 / 0.10, abs=0.01)
    assert trades["pips"].iloc[0] == pytest.approx(50 - 22.0, abs=0.05)
