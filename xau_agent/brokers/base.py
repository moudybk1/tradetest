"""Common broker interface so the bot works the same on Hyperliquid, MT5 and paper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class OrderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 10_000


@dataclass(frozen=True)
class Position:
    side: str  # "long" | "short"
    size: float
    entry: float


@dataclass(frozen=True)
class Fill:
    side: str
    size: float
    price: float
    stop: float
    take_profit: float


class Broker(ABC):
    name: str = "broker"

    @abstractmethod
    def candles(self, interval: str, limit: int) -> pd.DataFrame:
        """CLOSED candles only, indexed by UTC open time, columns open/high/low/close."""

    @abstractmethod
    def quote(self) -> Quote: ...

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def position(self) -> Position | None: ...

    @abstractmethod
    def size_for_risk(self, risk_amount: float, entry: float, stop: float) -> float:
        """Order size (broker units) that loses about `risk_amount` if the stop is hit. 0 = too small."""

    @abstractmethod
    def notional(self, size: float, price: float) -> float:
        """Position value in account currency."""

    def size_from_lots(self, lots: float, contract_oz: float) -> float:
        """Order size for a fixed lot amount. Default: brokers that trade in ounces/coins."""
        return round(lots * contract_oz, 6)

    def risk_usd(self, size: float, entry: float, stop: float) -> float:
        """Account-currency loss if the stop is hit."""
        return abs(entry - stop) * size

    @abstractmethod
    def open_with_bracket(self, side: str, size: float, stop: float, take_profit: float) -> Fill:
        """Market entry with stop loss and take profit attached. Must never leave a position without a stop."""

    @abstractmethod
    def close_all(self) -> None: ...

    def sync(self) -> list[str]:
        """Housekeeping each loop (paper fills, stale order cleanup). Returns messages to log."""
        return []
