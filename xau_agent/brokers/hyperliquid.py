"""Hyperliquid perps (PAXG or a HIP-3 gold market such as "xyz:GOLD")."""

from __future__ import annotations

import logging
import math
import os
import time

import pandas as pd

from ..config import HyperliquidConfig
from ..indicators import interval_delta
from .base import Broker, Fill, OrderError, Position, Quote

log = logging.getLogger(__name__)

MIN_ORDER_USD = 10.0  # Hyperliquid rejects orders below $10 notional


def _statuses(resp: dict) -> list[dict]:
    if not isinstance(resp, dict) or resp.get("status") != "ok":
        raise OrderError(f"Hyperliquid rejected the request: {resp}")
    statuses = resp["response"]["data"]["statuses"]
    errors = [s["error"] for s in statuses if isinstance(s, dict) and "error" in s]
    if errors:
        raise OrderError(f"Hyperliquid order error: {errors}")
    return statuses


class HyperliquidBroker(Broker):
    name = "hyperliquid"

    def __init__(self, cfg: HyperliquidConfig, trading: bool):
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        self.cfg = cfg
        self.coin = cfg.coin
        self.dex = cfg.coin.split(":")[0] if ":" in cfg.coin else ""
        perp_dexs = ["", self.dex] if self.dex else None
        self.base_url = constants.TESTNET_API_URL if cfg.testnet else constants.MAINNET_API_URL
        self.info = Info(self.base_url, skip_ws=True, perp_dexs=perp_dexs)
        try:
            asset = self.info.name_to_asset(self.coin)
        except KeyError:
            raise ValueError(f"'{self.coin}' is not listed on Hyperliquid ({self.base_url})") from None
        self.sz_decimals = self.info.asset_to_sz_decimals[asset]
        self.address = os.getenv("HL_ACCOUNT_ADDRESS") or None
        self.exchange = None

        if trading:
            import eth_account
            from hyperliquid.exchange import Exchange

            key = os.getenv("HL_API_PRIVATE_KEY")
            if not (key and self.address):
                raise ValueError("Live Hyperliquid trading needs HL_ACCOUNT_ADDRESS and HL_API_PRIVATE_KEY in .env")
            wallet = eth_account.Account.from_key(key)
            self.exchange = Exchange(wallet, self.base_url, account_address=self.address, perp_dexs=perp_dexs)
            try:
                self.exchange.update_leverage(cfg.leverage, self.coin, is_cross=cfg.cross_margin)
            except Exception as exc:  # leverage limits differ per market; sizing is risk-based anyway
                log.warning("could not set leverage to %sx: %s", cfg.leverage, exc)

    # ---- market data -------------------------------------------------------------------------
    def candles(self, interval: str, limit: int) -> pd.DataFrame:
        step = interval_delta(interval)
        now_ms = int(time.time() * 1000)
        step_ms = int(step.total_seconds() * 1000)
        raw = self.info.candles_snapshot(self.coin, interval, now_ms - (limit + 1) * step_ms, now_ms)
        rows = [
            {
                "time": int(c["t"]),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            }
            for c in raw
            if int(c["t"]) + step_ms <= now_ms  # closed candles only
        ]
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
        return df.tail(limit)

    def quote(self) -> Quote:
        book = self.info.l2_snapshot(self.coin)
        bids, asks = book["levels"]
        if not bids or not asks:
            raise OrderError(f"empty order book for {self.coin}")
        return Quote(float(bids[0]["px"]), float(asks[0]["px"]))

    # ---- account -----------------------------------------------------------------------------
    def _user_state(self) -> dict:
        if not self.address:
            raise ValueError("HL_ACCOUNT_ADDRESS is not set in .env")
        return self.info.user_state(self.address, self.dex)

    def equity(self) -> float:
        return float(self._user_state()["marginSummary"]["accountValue"])

    def position(self) -> Position | None:
        for item in self._user_state()["assetPositions"]:
            pos = item["position"]
            if pos["coin"] != self.coin:
                continue
            szi = float(pos["szi"])
            if szi == 0:
                return None
            return Position("long" if szi > 0 else "short", abs(szi), float(pos["entryPx"] or 0))
        return None

    # ---- sizing ------------------------------------------------------------------------------
    def size_for_risk(self, risk_amount: float, entry: float, stop: float) -> float:
        distance = abs(entry - stop)
        if distance <= 0:
            return 0.0
        factor = 10**self.sz_decimals
        size = round(math.floor(risk_amount / distance * factor) / factor, self.sz_decimals)
        if size * entry < MIN_ORDER_USD:
            return 0.0
        return size

    def notional(self, size: float, price: float) -> float:
        return size * price

    def size_from_lots(self, lots: float, contract_oz: float) -> float:
        factor = 10**self.sz_decimals
        return round(math.floor(lots * contract_oz * factor + 1e-9) / factor, self.sz_decimals)

    def round_px(self, px: float) -> float:
        # Hyperliquid perps: max 5 significant figures and max (6 - szDecimals) decimals.
        return round(float(f"{px:.5g}"), 6 - self.sz_decimals)

    # ---- orders ------------------------------------------------------------------------------
    def _require_exchange(self):
        if self.exchange is None:
            raise OrderError("broker was created read-only (paper mode)")
        return self.exchange

    def open_with_bracket(self, side: str, size: float, stop: float, take_profit: float) -> Fill:
        ex = self._require_exchange()
        is_buy = side == "long"
        statuses = _statuses(ex.market_open(self.coin, is_buy, size, None, self.cfg.slippage))
        filled = statuses[0].get("filled") if statuses else None
        if not filled:
            raise OrderError(f"entry order did not fill: {statuses}")
        fill_size, fill_px = float(filled["totalSz"]), float(filled["avgPx"])

        try:
            self._place_stop_and_target(is_buy, fill_size, stop, take_profit)
        except Exception:
            log.error("stop/target placement failed - closing the position so it is never unprotected")
            ex.market_close(self.coin)
            raise
        return Fill(side, fill_size, fill_px, stop, take_profit)

    def _place_stop_and_target(self, is_buy: bool, size: float, stop: float, take_profit: float) -> None:
        slip = self.cfg.slippage
        # The exit order is the opposite side; its worst acceptable price is beyond the trigger.
        worst = (lambda px: px * (1 - slip)) if is_buy else (lambda px: px * (1 + slip))
        orders = [
            {
                "coin": self.coin,
                "is_buy": not is_buy,
                "sz": size,
                "limit_px": self.round_px(worst(px)),
                "order_type": {"trigger": {"triggerPx": self.round_px(px), "isMarket": True, "tpsl": kind}},
                "reduce_only": True,
            }
            for kind, px in (("sl", stop), ("tp", take_profit))
        ]
        _statuses(self._require_exchange().bulk_orders(orders))

    def _cancel_open_orders(self) -> int:
        ex = self._require_exchange()
        orders = [o for o in self.info.open_orders(self.address, self.dex) if o["coin"] == self.coin]
        if orders:
            ex.bulk_cancel([{"coin": self.coin, "oid": o["oid"]} for o in orders])
        return len(orders)

    def close_all(self) -> None:
        ex = self._require_exchange()
        if self.position() is not None:
            _statuses(ex.market_close(self.coin))
        self._cancel_open_orders()

    def sync(self) -> list[str]:
        # After a stop or target fills, the other leg is left behind as a reduce-only order.
        # Cancel leftovers so they can never close a future position by accident.
        if self.exchange is None or self.position() is not None:
            return []
        cancelled = self._cancel_open_orders()
        return [f"cancelled {cancelled} leftover {self.coin} order(s)"] if cancelled else []
