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
from .indicators import daily_atr, interval_delta
from .journal import Journal
from .notify import Notifier
from .risk import RiskState, leverage_block, pre_trade_blocks, risk_amount
from .strategy import Signal, _at, day_candles, get_strategy

log = logging.getLogger(__name__)

KILL_FILE = "KILL"
# Skip if price runs more than this fraction of the stop distance away from the signal close
# before we can enter (fast candle, slow agent): the setup is no longer the one that triggered.
MAX_ENTRY_DRIFT = 0.3


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
        r, st = self.cfg.risk, self.cfg.strategy
        sizing = f"{r.fixed_lots} lot/trade" if r.sizing == "fixed_lots" else f"risk {r.risk_per_trade_pct}%/trade"
        exits = f"SL {st.sl_pips}/TP {st.tp_pips} pips" if st.exit_mode == "fixed_pips" else f"{st.reward_risk}R target"
        self.notify(f"bot started: {st.name} on {st.timeframe}, {instrument_name(self.cfg)}, {sizing}, {exits}")
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

        tf = cfg.strategy.timeframe
        # A full day of bars (Asian range starts at 00:00) plus indicator warm-up.
        candles = self.broker.candles(tf, int(pd.Timedelta(days=1) / interval_delta(tf)) + 300)
        if candles.empty or candles.index[-1] == self.last_candle:
            return  # wait for the next candle to close
        last = candles.index[-1]
        self.last_candle = last

        if cfg.strategy.weekdays_only and today.day_of_week >= 5:
            return

        hourly = self.broker.candles("1h", 24 * 45)
        atr = daily_atr(hourly, cfg.strategy.atr_period, today)
        strategy = get_strategy(cfg.strategy.name)
        prepared = strategy.prepare(candles, hourly, cfg)
        signals = strategy.day_signals(day_candles(prepared, today), today, atr, cfg)
        # Act only on a signal whose candle JUST closed - never chase a stale one after a restart.
        signal = next((sig for sig in signals if sig.candle_time == last), None)
        if signal is None:
            return
        self.handle_signal(signal, now, hourly)

    def handle_signal(self, signal: Signal, now: datetime, hourly: pd.DataFrame) -> None:
        cfg, broker = self.cfg, self.broker
        self.journal.log("signal", **signal.__dict__)
        log.info("SIGNAL %s %s entry %.2f stop %.2f tp %.2f", signal.strategy, signal.side, signal.entry,
                 signal.stop, signal.take_profit)

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
            spread_pips=(quote.ask - quote.bid) / cfg.strategy.pip_size,
        )
        blocks = pre_trade_blocks(cfg.risk, state)
        if blocks:
            self._skip(signal, "risk", blocks)
            return

        # 2) Re-anchor on the live price (the candle close is already a few seconds old) and size it.
        planned, why = self._reanchor(signal, quote)
        if why:
            self._skip(signal, "risk", [why])
            return
        size, why = self._size(planned, equity)
        if why:
            self._skip(signal, "risk", [why])
            return

        # 3) AI agent review.
        decision = self.review(planned, now, quote, hourly)
        self.journal.log("agent", **decision.to_dict())
        if not decision.take or decision.confidence < cfg.agent.min_confidence:
            self._skip(signal, "agent", [f"{decision.reason} (confidence {decision.confidence:.2f})"])
            return

        # 4) The agent can take a while: re-check the price and size against a fresh quote.
        if self.agent is not None:
            planned, why = self._reanchor(signal, broker.quote())
            if not why:
                size, why = self._size(planned, equity)
            if why:
                self._skip(signal, "risk", [why + " (after agent review)"])
                return

        # 5) Execute.
        fill = broker.open_with_bracket(planned.side, size, planned.stop, planned.take_profit)
        risk_usd = broker.risk_usd(fill.size, fill.price, fill.stop)
        self.journal.log("entry", **fill.__dict__, strategy=planned.strategy, risk_usd=risk_usd,
                         agent=decision.to_dict())
        self.notify(
            f"ENTERED {fill.side.upper()} {fill.size} @ {fill.price:.2f} ({planned.strategy})\n"
            f"SL {fill.stop:.2f} | TP {fill.take_profit:.2f} | risk ${risk_usd:,.2f} "
            f"({risk_usd / equity * 100:.2f}%)\n"
            f"agent ({decision.source}, {decision.confidence:.2f}): {decision.reason}"
        )

    def _reanchor(self, signal: Signal, quote) -> tuple[Signal, str | None]:
        entry_px = quote.ask if signal.side == "long" else quote.bid
        drift = abs(entry_px - signal.entry)
        if drift > MAX_ENTRY_DRIFT * signal.stop_distance:
            return signal, f"price moved {drift:.2f} away from the signal close {signal.entry:.2f}"
        return signal.with_entry(entry_px, self.cfg.strategy), None

    def _size(self, signal: Signal, equity: float) -> tuple[float, str | None]:
        r, broker = self.cfg.risk, self.broker
        if r.sizing == "fixed_lots":
            size = broker.size_from_lots(r.fixed_lots, r.contract_oz)
            if size <= 0:
                return 0.0, f"{r.fixed_lots} lot is below the broker minimum"
            risk_pct = broker.risk_usd(size, signal.entry, signal.stop) / equity * 100 if equity > 0 else float("inf")
            if risk_pct > r.max_risk_per_trade_pct:
                return 0.0, (f"{r.fixed_lots} lot would risk {risk_pct:.2f}% of equity "
                             f"(> max_risk_per_trade_pct {r.max_risk_per_trade_pct}%)")
        else:
            size = broker.size_for_risk(risk_amount(r, equity), signal.entry, signal.stop)
            if size <= 0:
                return 0.0, "position size below the broker minimum for this risk %"
        lev = leverage_block(r, broker.notional(size, signal.entry), equity)
        return (0.0, lev) if lev else (size, None)

    def review(self, signal, now: datetime, quote, hourly: pd.DataFrame) -> Decision:
        if self.agent is None:
            return Decision(True, 1.0, "agent disabled - rules only", source="rules-only")
        news = self.agent.news_summary(now) if self.cfg.agent.news_search else None
        context = build_context(
            signal, instrument_name(self.cfg), now, quote.mid, quote.spread_bps, hourly,
            self.calendar.upcoming(now), news,
        )
        return self.agent.review(context)

    def _skip(self, signal: Signal, stage: str, reasons: list[str]) -> None:
        self.journal.log("skip", stage=stage, reasons=reasons, side=signal.side)
        self.notify(f"signal {signal.side.upper()} @ {signal.entry:.2f} SKIPPED by {stage}: " + "; ".join(reasons))
