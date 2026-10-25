"""Paper trading: real prices from Hyperliquid or MT5, simulated fills. No money at risk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from .base import Broker, Fill, Position, Quote

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, data: Broker, state_file: Path, starting_equity: float, fee_bps: float,
                 clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC")):
        self.data = data
        self.clock = clock
        self.state_file = state_file
        self.fee_rate = fee_bps / 10_000
        if state_file.exists():
            self.state = json.loads(state_file.read_text())
        else:
            self.state = {"cash": starting_equity, "position": None}
            self._save()

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # Prices come from the real market.
    def candles(self, interval: str, limit: int) -> pd.DataFrame:
        return self.data.candles(interval, limit)

    def quote(self) -> Quote:
        return self.data.quote()

    def equity(self) -> float:
        pos = self.state["position"]
        if not pos:
            return float(self.state["cash"])
        mark = self.quote().mid
        direction = 1 if pos["side"] == "long" else -1
        return float(self.state["cash"] + direction * (mark - pos["entry"]) * pos["size"])

    def position(self) -> Position | None:
        pos = self.state["position"]
        return Position(pos["side"], pos["size"], pos["entry"]) if pos else None

    def size_for_risk(self, risk_amount: float, entry: float, stop: float) -> float:
        distance = abs(entry - stop)
        return round(risk_amount / distance, 4) if distance > 0 else 0.0

    def notional(self, size: float, price: float) -> float:
        return size * price

    def open_with_bracket(self, side: str, size: float, stop: float, take_profit: float) -> Fill:
        q = self.quote()
        price = q.ask if side == "long" else q.bid
        self.state["cash"] -= price * size * self.fee_rate
        self.state["position"] = {
            "side": side, "size": size, "entry": price, "stop": stop, "take_profit": take_profit,
            "opened_at": self.clock().isoformat(),
        }
        self._save()
        return Fill(side, size, price, stop, take_profit)

    def _close(self, price: float, reason: str) -> str:
        pos = self.state["position"]
        direction = 1 if pos["side"] == "long" else -1
        pnl = direction * (price - pos["entry"]) * pos["size"] - price * pos["size"] * self.fee_rate
        self.state["cash"] += pnl
        self.state["position"] = None
        self._save()
        return f"paper {pos['side']} closed by {reason} at {price:.2f}, pnl {pnl:+.2f}, equity {self.state['cash']:.2f}"

    def close_all(self) -> None:
        if self.state["position"]:
            q = self.quote()
            price = q.bid if self.state["position"]["side"] == "long" else q.ask
            log.info(self._close(price, "manual/flatten"))

    def sync(self) -> list[str]:
        """Check whether the stop or target was touched since entry, using 1-minute candles."""
        pos = self.state["position"]
        if not pos:
            return []
        opened = pd.Timestamp(pos["opened_at"]).ceil("1min")
        minutes = int((self.clock() - opened).total_seconds() // 60) + 2
        bars = self.data.candles("1m", min(max(minutes, 2), 1000))
        bars = bars[bars.index >= opened]
        for _, bar in bars.iterrows():
            if pos["side"] == "long":
                # If one bar touches both, assume the stop hit first (conservative).
                if bar["low"] <= pos["stop"]:
                    return [self._close(pos["stop"], "stop loss")]
                if bar["high"] >= pos["take_profit"]:
                    return [self._close(pos["take_profit"], "take profit")]
            else:
                if bar["high"] >= pos["stop"]:
                    return [self._close(pos["stop"], "stop loss")]
                if bar["low"] <= pos["take_profit"]:
                    return [self._close(pos["take_profit"], "take profit")]
        return []
