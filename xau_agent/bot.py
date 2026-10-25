"""Main loop: trigger -> hard risk checks -> AI agent review -> order with stop and target."""

from __future__ import annotations

import logging
import time as systime
from datetime import datetime, timezone

import pandas as pd

from .agent import Decision, TradeAgent, build_context
from .brokers import Broker
from .config import Config
from .econ_calendar import EconomicCalendar
from .indicators import daily_atr
from .journal import Journal
from .notify import Notifier
from .risk import RiskState, leverage_block, pre_trade_blocks, risk_amount
from .strategy import _at, find_breakout

log = logging.getLogger(__name__)

KILL_FILE = "KILL"


def build_broker(cfg: Config, live: bool) -> Broker:
    if cfg.mode == "live" and not live:
        raise SystemExit("config.yaml says mode: live. Re-run with --live to confirm real-money trading.")
    if live and cfg.mode != "live":
        raise SystemExit("--live was passed but config.yaml says mode: paper. Change mode to live first.")

    trading = cfg.mode == "live"
    if cfg.broker == "hyperliquid":
        from .brokers.hyperliquid import HyperliquidBroker

        data: Broker = HyperliquidBroker(cfg.hyperliquid, trading=trading)
    else:
        from .brokers.mt5 import MT5Broker

        data = MT5Broker(cfg.mt5)

    if trading:
        return data
    from .brokers.paper import PaperBroker

    return PaperBroker(data, cfg.data_path / "paper_state.json", cfg.paper.starting_equity, cfg.paper.fee_bps)


def instrument_name(cfg: Config) -> str:
    if cfg.broker == "hyperliquid":
        return f"{cfg.hyperliquid.coin} perp on Hyperliquid (tracks gold, trades 24/7)"
    return f"{cfg.mt5.symbol} spot gold CFD on MT5"


class Bot:
    def __init__(self, cfg: Config, broker: Broker, agent: TradeAgent | None = None,
                 calendar: EconomicCalendar | None = None, notifier: Notifier | None = None):
        self.cfg = cfg
        self.broker = broker
        self.journal = Journal(cfg.data_path / "journal.db")
        self.calendar = calendar or EconomicCalendar(cfg.data_path / "calendar_cache.json")
        self.notifier = notifier or Notifier.from_env()
        self.agent = agent if agent is not None else (TradeAgent(cfg.agent) if cfg.agent.enabled else None)
        self.last_candle: pd.Timestamp | None = None
        self.flattened_day = None
        self.equity_day = None

    def notify(self, text: str) -> None:
        tag = "LIVE" if self.cfg.mode == "live" else "PAPER"
        self.notifier.send(f"[XAU {tag}] {text}")

    # ------------------------------------------------------------------------------------------
    def run(self, once: bool = False) -> None:
        self.notify(f"bot started: {instrument_name(self.cfg)}, risk {self.cfg.risk.risk_per_trade_pct}%/trade")
        kill_file = self.cfg.data_path / KILL_FILE
        while True:
            if kill_file.exists():
                self.broker.close_all()
                self.journal.log("kill_switch")
                self.notify("KILL switch: all positions closed, bot stopped")
                kill_file.unlink()
                return
            try:
                self.step(datetime.now(timezone.utc))
            except Exception as exc:
                log.exception("step failed")
                self.journal.log("error", error=repr(exc))
            if once:
                return
            systime.sleep(self.cfg.poll_seconds)

    def step(self, now: datetime) -> None:
        cfg = self.cfg
        for msg in self.broker.sync():
            self.journal.log("sync", message=msg)
            self.notify(msg)

        today = pd.Timestamp(now).normalize()
        if self.equity_day != today:
            # Baseline for the daily loss limit: equity at the first check of the UTC day.
            self.journal.day_start_equity(now.date(), self.broker.equity())
            self.equity_day = today

        # End-of-day flatten: intraday strategy, never hold into rollover or the weekend.
        if now >= _at(today, cfg.strategy.flatten_utc) and self.flattened_day != today:
            if self.broker.position() is not None:
                self.broker.close_all()
                self.journal.log("flatten")
                self.notify("end of session: position closed")
            self.flattened_day = today

        candles = self.broker.candles(cfg.strategy.timeframe, 200)
        if candles.empty or candles.index[-1] == self.last_candle:
            return  # wait for the next candle to close
        last = candles.index[-1]
        self.last_candle = last

        if cfg.strategy.weekdays_only and today.day_of_week >= 5:
            return

        hourly = self.broker.candles("1h", 24 * 45)
        atr = daily_atr(hourly, cfg.strategy.atr_period, today)
        signal = find_breakout(candles, today, atr, cfg.strategy)
        # Act only on a breakout that JUST closed - never chase a stale one after a restart.
        if signal is None or signal.candle_time != last:
            return
        self.handle_signal(signal, now, hourly)

    def handle_signal(self, signal, now: datetime, hourly: pd.DataFrame) -> None:
        cfg, broker = self.cfg, self.broker
        self.journal.log("signal", **signal.__dict__)
        log.info("SIGNAL %s entry %.2f stop %.2f tp %.2f", signal.side, signal.entry, signal.stop, signal.take_profit)

        # 1) Hard risk rules first - no point asking the agent about a trade we can't take.
        quote = broker.quote()
        equity = broker.equity()
        day_start = self.journal.day_start_equity(now.date(), equity)
        blackout, hits = self.calendar.blackout(
            now, cfg.risk.news_blackout_before_min, cfg.risk.news_blackout_after_min
        )
        state = RiskState(
            equity=equity,
            day_start_equity=day_start,
            open_positions=1 if broker.position() is not None else 0,
            trades_today=self.journal.count_today("entry", now.date()),
            spread_bps=quote.spread_bps,
            news_blackout=blackout,
            news_events=[e.label() for e in hits],
        )
        blocks = pre_trade_blocks(cfg.risk, state)
        if blocks:
            self._skip(signal, "risk", blocks)
            return

        # 2) Re-anchor on the live price (the candle close is already a few seconds old).
        entry_px = quote.ask if signal.side == "long" else quote.bid
        if (signal.side == "long" and entry_px <= signal.stop) or (signal.side == "short" and entry_px >= signal.stop):
            self._skip(signal, "risk", ["price already beyond the stop"])
            return
        signal = signal.with_entry(entry_px, cfg.strategy.reward_risk)

        size = broker.size_for_risk(risk_amount(cfg.risk, equity), signal.entry, signal.stop)
        if size <= 0:
            self._skip(signal, "risk", ["position size below the broker minimum for this risk %"])
            return
        lev = leverage_block(cfg.risk, broker.notional(size, signal.entry), equity)
        if lev:
            self._skip(signal, "risk", [lev])
            return

        # 3) AI agent review.
        decision = self.review(signal, now, quote, hourly)
        self.journal.log("agent", **decision.to_dict())
        if not decision.take or decision.confidence < cfg.agent.min_confidence:
            self._skip(signal, "agent", [f"{decision.reason} (confidence {decision.confidence:.2f})"])
            return

        # 4) Execute.
        fill = broker.open_with_bracket(signal.side, size, signal.stop, signal.take_profit)
        self.journal.log("entry", **fill.__dict__, risk_pct=cfg.risk.risk_per_trade_pct, agent=decision.to_dict())
        self.notify(
            f"ENTERED {fill.side.upper()} {fill.size} @ {fill.price:.2f}\n"
            f"SL {fill.stop:.2f} | TP {fill.take_profit:.2f} | risk {cfg.risk.risk_per_trade_pct}%\n"
            f"agent ({decision.source}, {decision.confidence:.2f}): {decision.reason}"
        )

    def review(self, signal, now: datetime, quote, hourly: pd.DataFrame) -> Decision:
        if self.agent is None:
            return Decision(True, 1.0, "agent disabled - rules only", source="rules-only")
        news = self.agent.news_summary(now) if self.cfg.agent.news_search else None
        context = build_context(
            signal, instrument_name(self.cfg), now, quote.mid, quote.spread_bps, hourly,
            self.calendar.upcoming(now), news,
        )
        return self.agent.review(context)

    def _skip(self, signal, stage: str, reasons: list[str]) -> None:
        self.journal.log("skip", stage=stage, reasons=reasons, side=signal.side)
        self.notify(f"signal {signal.side.upper()} @ {signal.entry:.2f} SKIPPED by {stage}: " + "; ".join(reasons))
