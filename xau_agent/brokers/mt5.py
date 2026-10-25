"""MetaTrader 5 (XAUUSD at your forex/CFD broker). Windows only: needs the MT5 terminal installed."""

from __future__ import annotations

import logging
import math
import os
import time

import pandas as pd

from ..config import MT5Config
from ..indicators import interval_delta
from .base import Broker, Fill, OrderError, Position, Quote

log = logging.getLogger(__name__)

# symbol_info().filling_mode bit flags
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2


def lots_for_risk(risk_amount: float, loss_per_lot: float, step: float, vmin: float, vmax: float) -> float:
    """Round DOWN to the broker's lot step so the real risk never exceeds the plan. 0 = below min lot."""
    if loss_per_lot <= 0 or step <= 0:
        return 0.0
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    lots = round(math.floor(risk_amount / loss_per_lot / step + 1e-9) * step, decimals)
    if lots < vmin:
        return 0.0
    return min(lots, vmax)


class MT5Broker(Broker):
    name = "mt5"

    TIMEFRAMES = {"1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5", "15m": "TIMEFRAME_M15", "30m": "TIMEFRAME_M30",
                  "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4", "1d": "TIMEFRAME_D1"}

    def __init__(self, cfg: MT5Config):
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise RuntimeError("MetaTrader5 package not installed. MT5 needs Windows: pip install MetaTrader5") from None
        self.mt5 = mt5
        self.cfg = cfg
        self.symbol = cfg.symbol

        kwargs = {}
        if os.getenv("MT5_LOGIN"):
            kwargs = {"login": int(os.environ["MT5_LOGIN"]), "password": os.getenv("MT5_PASSWORD", ""),
                      "server": os.getenv("MT5_SERVER", "")}
        path = os.getenv("MT5_TERMINAL_PATH")
        ok = mt5.initialize(path, **kwargs) if path else mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            raise ValueError(f"symbol '{self.symbol}' not found in MT5 - check the exact name in Market Watch")
        self._offset_hours = cfg.server_utc_offset_hours

    # ---- helpers -----------------------------------------------------------------------------
    def _info(self):
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise OrderError(f"symbol_info({self.symbol}) failed: {self.mt5.last_error()}")
        return info

    def _server_offset_seconds(self) -> int:
        """MT5 timestamps are broker server time. Work out the UTC offset from a fresh tick."""
        if self.cfg.server_utc_offset_hours is not None:
            return int(self.cfg.server_utc_offset_hours * 3600)
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is not None:
            diff = tick.time - time.time()
            hours = round(diff / 1800) / 2  # brokers use whole or half-hour offsets
            if abs(diff - hours * 3600) < 300:  # only trust a fresh tick
                self._offset_hours = hours
        if self._offset_hours is None:
            raise OrderError("cannot detect MT5 server time offset (market closed?). Set mt5.server_utc_offset_hours")
        return int(self._offset_hours * 3600)

    # ---- market data -------------------------------------------------------------------------
    def candles(self, interval: str, limit: int) -> pd.DataFrame:
        tf = getattr(self.mt5, self.TIMEFRAMES[interval])
        rates = self.mt5.copy_rates_from_pos(self.symbol, tf, 0, limit + 1)
        if rates is None or len(rates) == 0:
            raise OrderError(f"no candles for {self.symbol}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)[["time", "open", "high", "low", "close", "tick_volume"]]
        df = df.rename(columns={"tick_volume": "volume"})
        df.index = pd.to_datetime(df.pop("time") - self._server_offset_seconds(), unit="s", utc=True)
        closed = df.index + interval_delta(interval) <= pd.Timestamp.now(tz="UTC")
        return df[closed].tail(limit)

    def quote(self) -> Quote:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise OrderError(f"no tick for {self.symbol}: {self.mt5.last_error()}")
        return Quote(float(tick.bid), float(tick.ask))

    # ---- account -----------------------------------------------------------------------------
    def equity(self) -> float:
        acc = self.mt5.account_info()
        if acc is None:
            raise OrderError(f"account_info failed: {self.mt5.last_error()}")
        return float(acc.equity)

    def _my_positions(self):
        return [p for p in (self.mt5.positions_get(symbol=self.symbol) or []) if p.magic == self.cfg.magic]

    def position(self) -> Position | None:
        positions = self._my_positions()
        if not positions:
            return None
        p = positions[0]
        side = "long" if p.type == self.mt5.POSITION_TYPE_BUY else "short"
        return Position(side, float(p.volume), float(p.price_open))

    # ---- sizing ------------------------------------------------------------------------------
    def size_for_risk(self, risk_amount: float, entry: float, stop: float) -> float:
        info = self._info()
        order_type = self.mt5.ORDER_TYPE_BUY if stop < entry else self.mt5.ORDER_TYPE_SELL
        # Exact loss for 1 lot in account currency (handles contract size and currency conversion).
        loss = self.mt5.order_calc_profit(order_type, self.symbol, 1.0, entry, stop)
        if loss is None:
            loss = -abs(entry - stop) / info.trade_tick_size * info.trade_tick_value
        return lots_for_risk(risk_amount, abs(loss), info.volume_step, info.volume_min, info.volume_max)

    def notional(self, size: float, price: float) -> float:
        return size * self._info().trade_contract_size * price

    # ---- orders ------------------------------------------------------------------------------
    def _filling(self, info) -> int:
        if info.filling_mode & SYMBOL_FILLING_FOK:
            return self.mt5.ORDER_FILLING_FOK
        if info.filling_mode & SYMBOL_FILLING_IOC:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, request: dict):
        result = self.mt5.order_send(request)
        if result is None:
            raise OrderError(f"order_send failed: {self.mt5.last_error()}")
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise OrderError(f"order rejected: retcode={result.retcode} {result.comment}")
        return result

    def open_with_bracket(self, side: str, size: float, stop: float, take_profit: float) -> Fill:
        info = self._info()
        tick = self.quote()
        is_buy = side == "long"
        price = tick.ask if is_buy else tick.bid
        min_distance = info.trade_stops_level * info.point
        if abs(price - stop) < min_distance or abs(take_profit - price) < min_distance:
            raise OrderError(f"stop/target closer than broker minimum ({min_distance})")
        # SL and TP travel with the entry order, so the position is never open without a stop.
        result = self._send({
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": size,
            "type": self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": round(stop, info.digits),
            "tp": round(take_profit, info.digits),
            "deviation": self.cfg.deviation_points,
            "magic": self.cfg.magic,
            "comment": "xau-agent",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        })
        return Fill(side, float(result.volume), float(result.price), stop, take_profit)

    def close_all(self) -> None:
        info = self._info()
        for p in self._my_positions():
            is_long = p.type == self.mt5.POSITION_TYPE_BUY
            tick = self.quote()
            self._send({
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": p.volume,
                "type": self.mt5.ORDER_TYPE_SELL if is_long else self.mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": tick.bid if is_long else tick.ask,
                "deviation": self.cfg.deviation_points,
                "magic": self.cfg.magic,
                "comment": "xau-agent close",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._filling(info),
            })
