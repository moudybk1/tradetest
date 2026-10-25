"""Load and validate config.yaml into typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HyperliquidConfig:
    coin: str = "PAXG"
    testnet: bool = True
    leverage: int = 5
    cross_margin: bool = True
    slippage: float = 0.003


@dataclass
class MT5Config:
    symbol: str = "XAUUSD"
    magic: int = 240925
    deviation_points: int = 30
    server_utc_offset_hours: float | None = None


@dataclass
class StrategyConfig:
    timeframe: str = "15m"
    asian_start_utc: time = time(0, 0)
    asian_end_utc: time = time(7, 0)
    entry_end_utc: time = time(16, 0)
    flatten_utc: time = time(20, 30)
    weekdays_only: bool = True
    atr_period: int = 14
    max_range_atr: float = 1.2
    min_range_atr: float = 0.15
    breakout_buffer_atr: float = 0.02
    stop_mode: str = "mid"
    min_stop_atr: float = 0.1
    reward_risk: float = 2.0


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 1
    max_trades_per_day: int = 1
    max_spread_bps: float = 8.0
    max_leverage: float = 5.0
    news_blackout_before_min: int = 15
    news_blackout_after_min: int = 15
    block_if_calendar_unavailable: bool = True


@dataclass
class AgentConfig:
    enabled: bool = True
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    min_confidence: float = 0.6
    news_search: bool = True


@dataclass
class PaperConfig:
    starting_equity: float = 10_000.0
    fee_bps: float = 4.5


@dataclass
class Config:
    mode: str = "paper"
    broker: str = "hyperliquid"
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    poll_seconds: int = 20
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


def _parse_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, int):
        # YAML reads an unquoted 07:00 as the integer 420 (minutes, sexagesimal).
        return time(value // 60, value % 60)
    hours, minutes = str(value).split(":")
    return time(int(hours), int(minutes))


def _build(cls: type, raw: dict[str, Any] | None, section: str) -> Any:
    raw = raw or {}
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}': {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in raw.items():
        default = known[name].default
        if isinstance(default, time):
            value = _parse_time(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Copy config.example.yaml to {path.name} first.")
    raw = yaml.safe_load(path.read_text()) or {}
    sections = {
        "hyperliquid": HyperliquidConfig,
        "mt5": MT5Config,
        "strategy": StrategyConfig,
        "risk": RiskConfig,
        "agent": AgentConfig,
        "paper": PaperConfig,
    }
    top = {k: v for k, v in raw.items() if k not in sections}
    cfg = _build(Config, top, "top level")
    for name, cls in sections.items():
        setattr(cfg, name, _build(cls, raw.get(name), name))
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    if cfg.broker not in ("hyperliquid", "mt5"):
        raise ValueError("broker must be 'hyperliquid' or 'mt5'")
    s, r = cfg.strategy, cfg.risk
    if s.stop_mode not in ("mid", "opposite"):
        raise ValueError("strategy.stop_mode must be 'mid' or 'opposite'")
    if not (s.asian_start_utc < s.asian_end_utc < s.entry_end_utc < s.flatten_utc):
        raise ValueError("strategy times must satisfy asian_start < asian_end < entry_end < flatten")
    if s.reward_risk <= 0:
        raise ValueError("strategy.reward_risk must be > 0")
    if not (0 < r.risk_per_trade_pct <= 2):
        raise ValueError("risk.risk_per_trade_pct must be between 0 and 2 (percent)")
    if not (0 < r.max_daily_loss_pct <= 10):
        raise ValueError("risk.max_daily_loss_pct must be between 0 and 10 (percent)")
    if r.max_daily_loss_pct < r.risk_per_trade_pct:
        raise ValueError("risk.max_daily_loss_pct must be >= risk.risk_per_trade_pct")
    if not (0 <= cfg.agent.min_confidence <= 1):
        raise ValueError("agent.min_confidence must be between 0 and 1")
