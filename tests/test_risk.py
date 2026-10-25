import pytest

from xau_agent.brokers.mt5 import lots_for_risk
from xau_agent.config import RiskConfig
from xau_agent.risk import RiskState, leverage_block, pre_trade_blocks, risk_amount


def state(**overrides) -> RiskState:
    base = dict(equity=10_000, day_start_equity=10_000, open_positions=0, trades_today=0,
                spread_bps=2.0, news_blackout=False)
    base.update(overrides)
    return RiskState(**base)


def test_clean_state_is_allowed():
    assert pre_trade_blocks(RiskConfig(), state()) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"equity": 9_790}, "daily loss limit"),
        ({"open_positions": 1}, "max open positions"),
        ({"trades_today": 2}, "max trades per day"),
        ({"spread_pips": 6.0}, "spread too wide (6.0 pips"),
        ({"spread_bps": 12}, "spread too wide"),
        ({"news_blackout": True, "news_events": ["CPI"]}, "news blackout"),
        ({"news_blackout": None}, "calendar unavailable"),
        ({"equity": 0}, "equity is zero"),
    ],
)
def test_blocks(overrides, expected):
    blocks = pre_trade_blocks(RiskConfig(), state(**overrides))
    assert any(expected in b for b in blocks), blocks


def test_spread_pips_limit_is_optional():
    assert pre_trade_blocks(RiskConfig(max_spread_pips=None), state(spread_pips=50.0)) == []
    assert pre_trade_blocks(RiskConfig(), state(spread_pips=4.9)) == []


def test_calendar_unavailable_can_be_allowed():
    assert pre_trade_blocks(RiskConfig(block_if_calendar_unavailable=False), state(news_blackout=None)) == []


def test_daily_loss_just_below_limit_is_allowed():
    assert pre_trade_blocks(RiskConfig(), state(equity=9_801)) == []


def test_risk_amount():
    assert risk_amount(RiskConfig(risk_per_trade_pct=0.5), 10_000) == 50


def test_leverage_block():
    assert leverage_block(RiskConfig(max_leverage=5), 40_000, 10_000) is None
    assert "leverage" in leverage_block(RiskConfig(max_leverage=5), 60_000, 10_000)


def test_mt5_lots_round_down_to_step():
    # $50 risk, $1,234 loss per lot -> 0.0405 lots -> 0.04 (never round up past the planned risk)
    assert lots_for_risk(50, 1234, 0.01, 0.01, 100) == 0.04


def test_mt5_lots_below_minimum_is_zero():
    assert lots_for_risk(5, 1234, 0.01, 0.01, 100) == 0.0


def test_mt5_lots_capped_at_max():
    assert lots_for_risk(1_000_000, 10, 0.01, 0.01, 50) == 50
