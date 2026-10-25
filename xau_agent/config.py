"""Load and validate config.yaml into typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import yaml

STRATEGIES = ("asian_breakout", "ny_orb", "asian_sweep", "trend_pullback")


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
    """Settings shared by every strategy: which one runs, timeframe and exits."""

    name: str = "ny_orb"
    timeframe: str = "5m"
    # fixed_pips = SL/TP a fixed number of pips from entry. atr = stop from the setup, target = reward_risk x stop.
    exit_mode: str = "fixed_pips"
    sl_pips: float = 25
    tp_pips: float = 50
    # XAUUSD on most MT5 brokers: 1 pip = 0.10 (10 points on a 2-decimal quote).
    pip_size: float = 0.10
    reward_risk: float = 2.0
    min_stop_atr: float = 0.1
    atr_period: int = 14
    flatten_utc: time = time(20, 30)
    weekdays_only: bool = True


@dataclass
class AsianBreakoutConfig:
    asian_start_utc: time = time(0, 0)
    asian_end_utc: time = time(7, 0)
    entry_end_utc: time = time(16, 0)
    max_range_atr: float = 1.2
    min_range_atr: float = 0.15
    breakout_buffer_atr: float = 0.02
    stop_mode: str = "mid"


@dataclass
class NYOrbConfig:
    start_ny: time = time(9, 30)
    minutes: int = 15
    entry_end_ny: time = time(12, 0)
    max_range_pips: float = 80


@dataclass
class AsianSweepConfig:
    asian_start_utc: time = time(0, 0)
    asian_end_utc: time = time(7, 0)
    end_utc: time = time(10, 0)
    max_range_atr: float = 1.2
    min_range_atr: float = 0.15


@dataclass
class TrendPullbackConfig:
    sessions_utc: str = "07:00-11:00,13:00-17:00"
    ema: int = 20
    trend_fast: int = 50
    trend_slow: int = 200
    swing_bars: int = 6
    cooldown_bars: int = 6


@dataclass
class RiskConfig:
    # percent = size from risk_per_trade_pct. fixed_lots = always trade fixed_lots.
    sizing: str = "fixed_lots"
    risk_per_trade_pct: float = 0.5
    fixed_lots: float = 0.1
    # Ounces per lot: converts lots for paper/Hyperliquid and the backtest's dollar results. MT5 uses the broker's value.
    contract_oz: float = 100
    # fixed_lots only: skip the trade if one stop-out would lose more than this % of equity.
    max_risk_per_trade_pct: float = 2.0
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 1
    max_trades_per_day: int = 2
    max_spread_bps: float = 8.0
    # Optional second spread limit in pips (strategy.pip_size). Useful with tight fixed stops.
    max_spread_pips: float | None = 5.0
    max_leverage: float = 50.0
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
    fee_bps: float = 0.0


@dataclass
class Config:
    mode: str = "paper"
    broker: str = "mt5"
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    asian_breakout: AsianBreakoutConfig = field(default_factory=AsianBreakoutConfig)
    ny_orb: NYOrbConfig = field(default_factory=NYOrbConfig)
    asian_sweep: AsianSweepConfig = field(default_factory=AsianSweepConfig)
    trend_pullback: TrendPullbackConfig = field(default_factory=TrendPullbackConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    poll_seconds: int = 10
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


SECTIONS = {f.name: f.default_factory for f in dataclasses.fields(Config) if f.default_factory is not dataclasses.MISSING}


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
    top = {k: v for k, v in raw.items() if k not in SECTIONS}
    cfg = _build(Config, top, "top level")
    for name, cls in SECTIONS.items():
        setattr(cfg, name, _build(cls, raw.get(name), name))
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    if cfg.broker not in ("hyperliquid", "mt5"):
        raise ValueError("broker must be 'hyperliquid' or 'mt5'")
    s, r = cfg.strategy, cfg.risk
    if s.name not in STRATEGIES:
        raise ValueError(f"strategy.name must be one of {', '.join(STRATEGIES)}")
    if s.exit_mode not in ("fixed_pips", "atr"):
        raise ValueError("strategy.exit_mode must be 'fixed_pips' or 'atr'")
    if s.sl_pips <= 0 or s.tp_pips <= 0 or s.pip_size <= 0 or s.reward_risk <= 0:
        raise ValueError("strategy.sl_pips, tp_pips, pip_size and reward_risk must be > 0")
    ab, sw = cfg.asian_breakout, cfg.asian_sweep
    if ab.stop_mode not in ("mid", "opposite"):
        raise ValueError("asian_breakout.stop_mode must be 'mid' or 'opposite'")
    if not (ab.asian_start_utc < ab.asian_end_utc < ab.entry_end_utc < s.flatten_utc):
        raise ValueError("asian_breakout times must satisfy asian_start < asian_end < entry_end < strategy.flatten")
    if not (sw.asian_start_utc < sw.asian_end_utc < sw.end_utc < s.flatten_utc):
        raise ValueError("asian_sweep times must satisfy asian_start < asian_end < end < strategy.flatten")
    if cfg.ny_orb.entry_end_ny <= cfg.ny_orb.start_ny:
        raise ValueError("ny_orb.entry_end_ny must be after ny_orb.start_ny")
    if r.sizing not in ("percent", "fixed_lots"):
        raise ValueError("risk.sizing must be 'percent' or 'fixed_lots'")
    if not (0 < r.risk_per_trade_pct <= 2):
        raise ValueError("risk.risk_per_trade_pct must be between 0 and 2 (percent)")
    if not (0 < r.max_risk_per_trade_pct <= 5):
        raise ValueError("risk.max_risk_per_trade_pct must be between 0 and 5 (percent)")
    if r.fixed_lots <= 0 or r.contract_oz <= 0:
        raise ValueError("risk.fixed_lots and risk.contract_oz must be > 0")
    if not (0 < r.max_daily_loss_pct <= 10):
        raise ValueError("risk.max_daily_loss_pct must be between 0 and 10 (percent)")
    if r.max_daily_loss_pct < r.risk_per_trade_pct:
        raise ValueError("risk.max_daily_loss_pct must be >= risk.risk_per_trade_pct")
    if not (0 <= cfg.agent.min_confidence <= 1):
        raise ValueError("agent.min_confidence must be between 0 and 1")
