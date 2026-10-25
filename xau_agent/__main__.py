"""Command line: python -m xau_agent <run|check|backtest|kill|journal>"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from .config import load_config


def setup_logging(cfg) -> None:
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(cfg.data_path / "bot.log")],
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_run(args) -> None:
    from .bot import Bot, build_broker

    cfg = load_config(args.config)
    setup_logging(cfg)
    Bot(cfg, build_broker(cfg, args.live)).run(once=args.once)


def cmd_check(args) -> None:
    """Read-only health check: prices, today's range, news blackout, sizing preview. Places no orders."""
    import os

    from .brokers.paper import PaperBroker
    from .econ_calendar import EconomicCalendar
    from .indicators import daily_atr
    from .risk import risk_amount
    from .strategy import asian_range, find_breakout

    cfg = load_config(args.config)
    setup_logging(cfg)
    if cfg.broker == "hyperliquid":
        from .brokers.hyperliquid import HyperliquidBroker

        data = HyperliquidBroker(cfg.hyperliquid, trading=False)
    else:
        from .brokers.mt5 import MT5Broker

        data = MT5Broker(cfg.mt5)

    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now).normalize()
    q = data.quote()
    print(f"mode={cfg.mode} broker={cfg.broker} now={now:%Y-%m-%d %H:%M} UTC")
    print(f"quote bid={q.bid:.2f} ask={q.ask:.2f} spread={q.spread_bps:.2f} bps (max {cfg.risk.max_spread_bps})")

    if cfg.mode == "live":
        equity = data.equity()
        print(f"LIVE account equity: {equity:,.2f}  open position: {data.position()}")
    else:
        paper = PaperBroker(data, cfg.data_path / "paper_state.json", cfg.paper.starting_equity, cfg.paper.fee_bps)
        equity = paper.equity()
        print(f"paper equity: {equity:,.2f}  open position: {paper.position()}")

    candles = data.candles(cfg.strategy.timeframe, 200)
    hourly = data.candles("1h", 24 * 45)
    atr = daily_atr(hourly, cfg.strategy.atr_period, today)
    print(f"last closed {cfg.strategy.timeframe} candle: {candles.index[-1]}  daily ATR: {atr}")
    rng = asian_range(candles, today, cfg.strategy)
    if rng and atr:
        width = (rng[0] - rng[1]) / atr
        print(f"today's Asian range: {rng[1]:.2f} - {rng[0]:.2f} ({width:.2f} x ATR, "
              f"allowed {cfg.strategy.min_range_atr}-{cfg.strategy.max_range_atr})")
        sig = find_breakout(candles, today, atr, cfg.strategy)
        if sig:
            size = data.size_for_risk(risk_amount(cfg.risk, equity), sig.entry, sig.stop)
            print(f"today's breakout: {sig.side} at {sig.candle_time:%H:%M} entry {sig.entry:.2f} "
                  f"stop {sig.stop:.2f} tp {sig.take_profit:.2f} -> size at "
                  f"{cfg.risk.risk_per_trade_pct}% risk = {size}")
        else:
            print("no breakout signal yet today")
    else:
        print("Asian range not available yet (before 07:00 UTC, weekend, or missing data)")

    cal = EconomicCalendar(cfg.data_path / "calendar_cache.json")
    blackout, hits = cal.blackout(now, cfg.risk.news_blackout_before_min, cfg.risk.news_blackout_after_min)
    print(f"news blackout now: {blackout} {[e.label() for e in hits]}")
    for e in cal.upcoming(now, 48):
        print(f"  upcoming: {e.label()}")
    print(f"agent: enabled={cfg.agent.enabled} model={cfg.agent.model} "
          f"ANTHROPIC_API_KEY set={bool(os.getenv('ANTHROPIC_API_KEY'))}")


def cmd_backtest(args) -> None:
    from .backtest import fetch_hyperliquid, load_csv, run_backtest

    cfg = load_config(args.config)
    s = cfg.strategy
    if args.csv:
        candles = load_csv(args.csv, s.timeframe)
    else:
        candles = fetch_hyperliquid(cfg.hyperliquid.coin, s.timeframe, args.hyperliquid_days)
    print(f"{len(candles)} candles from {candles.index[0]} to {candles.index[-1]}")
    trades, stats = run_backtest(candles, s, args.cost_bps, cfg.risk.risk_per_trade_pct)
    print(json.dumps(stats, indent=2, default=str))
    if args.trades_out and not trades.empty:
        trades.to_csv(args.trades_out, index=False)
        print(f"trades written to {args.trades_out}")


def cmd_kill(args) -> None:
    from .bot import KILL_FILE

    cfg = load_config(args.config)
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    (cfg.data_path / KILL_FILE).touch()
    print("Kill switch set. The running bot will close all positions and stop within one poll interval.")


def cmd_journal(args) -> None:
    from .journal import Journal

    cfg = load_config(args.config)
    for ts, kind, data in reversed(Journal(cfg.data_path / "journal.db").recent(args.limit)):
        print(f"{ts[:19]}  {kind:<8} {json.dumps(data, default=str)[:300]}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="xau_agent", description="Agentic XAU breakout bot")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the trading loop")
    p.add_argument("--live", action="store_true", help="confirm real-money trading (config mode must be live)")
    p.add_argument("--once", action="store_true", help="run a single loop iteration and exit")
    p.set_defaults(func=cmd_run)

    sub.add_parser("check", help="read-only status check, places no orders").set_defaults(func=cmd_check)

    p = sub.add_parser("backtest", help="backtest the breakout rules")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="OHLC candles CSV in UTC (e.g. Dukascopy XAUUSD export)")
    src.add_argument("--hyperliquid-days", type=int, help="download this many recent days from Hyperliquid")
    p.add_argument("--cost-bps", type=float, default=5.0, help="round-trip cost (spread+fees) in bps")
    p.add_argument("--trades-out", help="write every trade to this CSV")
    p.set_defaults(func=cmd_backtest)

    sub.add_parser("kill", help="close everything and stop the running bot").set_defaults(func=cmd_kill)

    p = sub.add_parser("journal", help="show recent journal events")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_journal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
