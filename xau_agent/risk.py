"""Hard risk rules. The AI agent cannot override anything in this file."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RiskConfig


@dataclass
class RiskState:
    equity: float
    day_start_equity: float
    open_positions: int
    trades_today: int
    spread_bps: float
    # True = inside a news blackout, False = clear, None = calendar unavailable.
    news_blackout: bool | None
    news_events: list[str] = field(default_factory=list)
    spread_pips: float | None = None


def daily_loss_pct(state: RiskState) -> float:
    if state.day_start_equity <= 0:
        return 0.0
    return max(0.0, (state.day_start_equity - state.equity) / state.day_start_equity * 100)


def pre_trade_blocks(cfg: RiskConfig, state: RiskState) -> list[str]:
    """Every reason the bot must NOT open a trade right now. Empty list = allowed."""
    blocks = []
    if state.equity <= 0:
        blocks.append("equity is zero or unknown")
    loss = daily_loss_pct(state)
    if loss >= cfg.max_daily_loss_pct:
        blocks.append(f"daily loss limit hit ({loss:.2f}% >= {cfg.max_daily_loss_pct}%)")
    if state.open_positions >= cfg.max_open_positions:
        blocks.append(f"max open positions reached ({state.open_positions})")
    if state.trades_today >= cfg.max_trades_per_day:
        blocks.append(f"max trades per day reached ({state.trades_today})")
    if state.spread_bps > cfg.max_spread_bps:
        blocks.append(f"spread too wide ({state.spread_bps:.1f} bps > {cfg.max_spread_bps} bps)")
    if cfg.max_spread_pips is not None and state.spread_pips is not None and state.spread_pips > cfg.max_spread_pips:
        blocks.append(f"spread too wide ({state.spread_pips:.1f} pips > {cfg.max_spread_pips} pips)")
    if state.news_blackout:
        blocks.append("high-impact news blackout: " + "; ".join(state.news_events))
    elif state.news_blackout is None and cfg.block_if_calendar_unavailable:
        blocks.append("economic calendar unavailable (fail-safe)")
    return blocks


def risk_amount(cfg: RiskConfig, equity: float) -> float:
    return equity * cfg.risk_per_trade_pct / 100


def leverage_block(cfg: RiskConfig, notional: float, equity: float) -> str | None:
    if equity <= 0:
        return "equity is zero or unknown"
    lev = notional / equity
    if lev > cfg.max_leverage:
        return f"position would need {lev:.1f}x leverage (> {cfg.max_leverage}x); stop is too tight"
    return None
