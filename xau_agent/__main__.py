"""Command line: python -m xau_agent <run|check|backtest|kill|journal>"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    from .indicators import daily_atr, interval_delta
    from .risk import risk_amount
    from .strategy import day_candles, get_strategy

    cfg = load_config(args.config)
    setup_logging(cfg)
    data = data_broker(cfg)

    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now).normalize()
    q = data.quote()
    s, r = cfg.strategy, cfg.risk
    print(f"mode={cfg.mode} broker={cfg.broker} strategy={s.name} timeframe={s.timeframe} now={now:%Y-%m-%d %H:%M} UTC")
    print(f"quote bid={q.bid:.2f} ask={q.ask:.2f} spread={(q.ask - q.bid) / s.pip_size:.1f} pips "
          f"({q.spread_bps:.2f} bps, max {r.max_spread_bps})")

    if cfg.mode == "live":
        equity = data.equity()
        print(f"LIVE account equity: {equity:,.2f}  open position: {data.position()}")
    else:
        paper = PaperBroker(data, cfg.data_path / "paper_state.json", cfg.paper.starting_equity, cfg.paper.fee_bps)
        equity = paper.equity()
        print(f"paper equity: {equity:,.2f}  open position: {paper.position()}")

    candles = data.candles(s.timeframe, int(pd.Timedelta(days=1) / interval_delta(s.timeframe)) + 300)
    hourly = data.candles("1h", 24 * 45)
    atr = daily_atr(hourly, s.atr_period, today)
    print(f"last closed {s.timeframe} candle: {candles.index[-1]}  daily ATR: {atr}")
    strategy = get_strategy(s.name)
    signals = strategy.signals(day_candles(strategy.prepare(candles, hourly, cfg), today), today, atr, cfg)
    for sig in signals:
        if r.sizing == "fixed_lots":
            size = data.size_from_lots(r.fixed_lots, r.contract_oz)
        else:
            size = data.size_for_risk(risk_amount(r, equity), sig.entry, sig.stop)
        risk_usd = data.risk_usd(size, sig.entry, sig.stop) if size else 0.0
        print(f"signal today: {sig.side} at {sig.candle_time:%H:%M} UTC entry {sig.entry:.2f} SL {sig.stop:.2f} "
              f"TP {sig.take_profit:.2f} -> size {size}, risk ${risk_usd:,.2f}")
    if not signals:
        print("no signal yet today")

    cal = EconomicCalendar(cfg.data_path / "calendar_cache.json")
    blackout, hits = cal.blackout(now, cfg.risk.news_blackout_before_min, cfg.risk.news_blackout_after_min)
    print(f"news blackout now: {blackout} {[e.label() for e in hits]}")
    for e in cal.upcoming(now, 48):
        print(f"  upcoming: {e.label()}")
    print(f"agent: enabled={cfg.agent.enabled} model={cfg.agent.model} "
          f"ANTHROPIC_API_KEY set={bool(os.getenv('ANTHROPIC_API_KEY'))}")


def data_broker(cfg):
    """Read-only market data connection (never sends orders)."""
    if cfg.broker == "hyperliquid":
        from .brokers.hyperliquid import HyperliquidBroker

        return HyperliquidBroker(cfg.hyperliquid, trading=False)
    from .brokers.mt5 import MT5Broker

    return MT5Broker(cfg.mt5)


def _load_candles(args, cfg):
    from .backtest import fetch_hyperliquid, load_csv

    if args.csv:
        candles = load_csv(args.csv, args.server_utc_offset, args.point)
    else:
        candles = fetch_hyperliquid(cfg.hyperliquid.coin, "1m" if args.hyperliquid_days <= 3 else "5m",
                                    args.hyperliquid_days)
    step = candles.index.to_series().diff().median()
    spread = "real spread from data" if "spread" in candles.columns else f"fixed spread {args.spread_pips} pips"
    print(f"{len(candles):,} candles ({step}) from {candles.index[0]} to {candles.index[-1]}, {spread}")
    return candles


def _apply_overrides(args, cfg) -> None:
    if args.timeframe:
        cfg.strategy.timeframe = args.timeframe
    if args.exit_mode:
        cfg.strategy.exit_mode = args.exit_mode
    if args.rr:
        cfg.strategy.reward_risk = args.rr


def exits_label(cfg) -> str:
    s = cfg.strategy
    if s.exit_mode == "fixed_pips":
        return f"SL {s.sl_pips} / TP {s.tp_pips} pips"
    cap = f", max {s.max_stop_pips}" if s.max_stop_pips else ""
    return f"RR 1:{s.reward_risk:g} (stop at setup + {s.stop_buffer_pips:g} pips, min {s.min_stop_pips:g}{cap} pips)"


def _costs(args):
    from .backtest import Costs

    return Costs(args.spread_pips, args.slippage_pips, args.commission)


def _print_stats(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(stats, indent=2, default=str))


def cmd_backtest(args) -> None:
    from .backtest import monthly, run_backtest, split_stats

    cfg = load_config(args.config)
    _apply_overrides(args, cfg)
    name = args.strategy or cfg.strategy.name
    lots = args.lots if args.lots is not None else cfg.risk.fixed_lots
    trades, stats = run_backtest(_load_candles(args, cfg), cfg, _costs(args), name, lots)
    _print_stats(f"{name} | {cfg.strategy.timeframe} | {exits_label(cfg)} | {lots} lot", stats)
    if args.split and not trades.empty:
        for label, part in split_stats(trades, args.split, cfg, lots).items():
            _print_stats(f"{label} ({'before' if label == 'in_sample' else 'from'} {args.split})", part)
    if not trades.empty:
        print("\nnet $ per month:")
        print(monthly(trades).to_string())
    if args.trades_out and not trades.empty:
        trades.to_csv(args.trades_out, index=False)
        print(f"trades written to {args.trades_out}")


def cmd_compare(args) -> None:
    """Run every strategy on the same data and costs, side by side."""
    from .backtest import run_backtest, split_stats
    from .config import STRATEGIES

    cfg = load_config(args.config)
    _apply_overrides(args, cfg)
    candles = _load_candles(args, cfg)
    lots = args.lots if args.lots is not None else cfg.risk.fixed_lots
    rows = []
    for name in STRATEGIES:
        trades, st = run_backtest(candles, cfg, _costs(args), name, lots)
        row = {"strategy": name, "trades": st.get("trades", 0), "per_week": st.get("trades_per_week", "-"),
               "win%": st.get("win_rate_pct", "-"), "breakeven%": st.get("breakeven_win_rate_pct", "-"),
               "avg_SL_pips": st.get("avg_stop_pips", "-"),
               "net_$": st.get(f"net_usd_at_{lots}_lot", "-"), "PF": st.get("profit_factor", "-"),
               "max_DD_$": st.get("max_drawdown_usd", "-"), "lose_streak": st.get("worst_losing_streak", "-")}
        if args.split:
            oos = split_stats(trades, args.split, cfg, lots)["out_of_sample"] if not trades.empty else {}
            row["OOS_trades"] = oos.get("trades", 0)
            row["OOS_win%"] = oos.get("win_rate_pct", "-")
            row["OOS_net_$"] = oos.get(f"net_usd_at_{lots}_lot", "-")
        rows.append(row)
    s = cfg.strategy
    print(f"\n{s.timeframe} signals | {exits_label(cfg)} | {lots} lot | "
          f"max {cfg.risk.max_trades_per_day} trades/day | flatten {s.flatten_utc:%H:%M} UTC")
    print(pd.DataFrame(rows).set_index("strategy").to_string())


def cmd_download_dukascopy(args) -> None:
    """Free M1 gold history with Dukascopy's real bid/ask spread (works on any OS, no MT5 needed)."""
    from .dukascopy import fetch_dukascopy

    end = pd.Timestamp.now(tz="UTC").normalize()
    print(f"downloading {args.days} days of {args.symbol} M1 from Dukascopy (2 small files per day)...")
    df = fetch_dukascopy(end - pd.Timedelta(days=args.days), end, args.symbol, args.price_divisor)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"saved {len(df):,} M1 bars ({df.index[0]} -> {df.index[-1]}, UTC), "
          f"median spread {df['spread_price'].median():.2f} to {args.out}")


def cmd_export_mt5(args) -> None:
    """Download M1 (or other) history with real spreads from your MT5 terminal to CSV (Windows)."""
    from .brokers.mt5 import MT5Broker

    cfg = load_config(args.config)
    broker = MT5Broker(cfg.mt5)
    end = pd.Timestamp.now(tz="UTC")
    df = broker.history(args.interval, end - pd.Timedelta(days=args.days), end)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"saved {len(df):,} {args.interval} bars ({df.index[0]} -> {df.index[-1]}, UTC) to {args.out}")


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

    def data_args(p):
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--csv", help="candles CSV: export-mt5 output, MT5 'Export Bars' file or Dukascopy (M1 best)")
        src.add_argument("--hyperliquid-days", type=int, help="download this many recent days from Hyperliquid")
        p.add_argument("--server-utc-offset", type=float, default=0.0,
                       help="hours to subtract when the CSV is in broker server time (MT5 Export Bars: often 2 or 3)")
        p.add_argument("--point", type=float, default=0.01, help="price of 1 point, to convert a spread column")
        p.add_argument("--spread-pips", type=float, default=3.0, help="spread when the data has no spread column")
        p.add_argument("--slippage-pips", type=float, default=0.5, help="extra pips lost per trade")
        p.add_argument("--commission", type=float, default=0.0, help="round-turn commission in $ per 1.0 lot")
        p.add_argument("--lots", type=float, help="lot size for $ results (default: risk.fixed_lots)")
        p.add_argument("--split", help="YYYY-MM-DD: also report results before/after this date (out-of-sample)")
        p.add_argument("--timeframe", choices=["1m", "5m", "15m"], help="override strategy.timeframe")
        p.add_argument("--exit-mode", choices=["rr", "fixed_pips"], help="override strategy.exit_mode")
        p.add_argument("--rr", type=float, help="override strategy.reward_risk (rr exit mode)")

    p = sub.add_parser("backtest", help="backtest one strategy")
    data_args(p)
    p.add_argument("--strategy", help="override strategy.name from config")
    p.add_argument("--trades-out", help="write every trade to this CSV")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("compare", help="backtest all strategies side by side")
    data_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("download-dukascopy", help="download free XAUUSD M1 history with real spread")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--price-divisor", type=float, default=1000.0, help="Dukascopy integer price scale")
    p.add_argument("--out", default="data/xauusd_m1.csv")
    p.set_defaults(func=cmd_download_dukascopy)

    p = sub.add_parser("export-mt5", help="save MT5 history (with real spread) to CSV for backtesting")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--interval", default="1m")
    p.add_argument("--out", default="data/xauusd_m1.csv")
    p.set_defaults(func=cmd_export_mt5)

    sub.add_parser("kill", help="close everything and stop the running bot").set_defaults(func=cmd_kill)

    p = sub.add_parser("journal", help="show recent journal events")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_journal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
