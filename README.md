# XAU Agent: agentic gold trading bot (MT5 / Hyperliquid)

An automated XAU/USD (gold) trading bot. A fixed rule finds the setup, hard-coded risk rules decide
whether a trade is allowed, and Claude reviews the news and market context before every entry.
If everything says yes, the bot enters with the stop loss and take profit already attached.

```
 candles ──► TRIGGER (strategy rules) ──► RISK RULES (hard code) ──► AI AGENT (Claude) ──► ORDER + SL + TP
                                             │ any "no" = skip           │ "skip" or low confidence = skip
                                             └────────── journal + Telegram alert ◄──────┘
```

Default setup (`config.example.yaml`): **MT5, 5-minute signals, SL 25 pips / TP 50 pips, 0.1 lot per
entry, max 2 trades a day.** With XAUUSD at 1 pip = $0.10 and 100 oz per lot, 0.1 lot = **$1 per pip**:
a stop-out loses about $25 plus spread, a target wins about $50 minus spread.

> ⚠️ **Trading is risky and this bot can lose money.** Nothing here is financial advice and nothing
> guarantees a profit. Backtest first, then paper trade, then a demo account, and only then risk
> small amounts you can afford to lose.

## Strategies (`strategy.name`)

| name | idea | default window (UTC unless noted) |
|---|---|---|
| `ny_orb` | first 5m close beyond the New York opening range (09:30 NY + 15 min, DST handled) | until 12:00 New York |
| `asian_sweep` | London pushes through the Asian high (or low), then a candle closes back inside → trade the reversal. Skipped if the sweep wick doesn't fit inside the 25-pip stop | 07:00-10:00 |
| `trend_pullback` | H1 EMA50 vs EMA200 sets the direction; a candle dips to the EMA20 and closes back in the trend. Skipped if the recent swing is beyond the stop | 07:00-11:00, 13:00-17:00 |
| `asian_breakout` | first close beyond the Asian (00:00-07:00) range, with ATR filters | 07:00-16:00 |

Exits: `exit_mode: fixed_pips` keeps SL/TP at exactly `sl_pips` / `tp_pips` from the real entry price.
`exit_mode: atr` puts the stop at the setup's chart level and the target at `reward_risk` × stop.
Everything open is closed at `flatten_utc` (20:30 UTC).

## Risk rules (the AI can't change them)

- `sizing: fixed_lots` → every entry is `fixed_lots` (0.1). The trade is skipped if one stop-out would
  lose more than `max_risk_per_trade_pct` (2%) of equity. For example, $25 risk needs at least $1,250
  equity; $2,500 or more keeps it at 1% or less. `sizing: percent` sizes from `risk_per_trade_pct` instead.
- Stop for the day after −2% (`max_daily_loss_pct`), max 1 open position, max 2 trades a day.
- No entries 15 minutes before or after high-impact USD news (ForexFactory calendar).
- Skip if the spread is over 5 pips, or if price ran more than 30% of the stop away from the signal
  candle's close before the order could be sent.

---

## 1. Install

You need Python 3.11 or newer. **MT5 needs Windows** with the MetaTrader 5 terminal installed and
logged in. Hyperliquid works anywhere.

```bash
git clone <this repo> && cd tradetest
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy config.example.yaml config.yaml   # Mac/Linux: cp
copy .env.example .env
```

Fill in `.env`: `ANTHROPIC_API_KEY` (https://console.anthropic.com) for the AI agent, and
`MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER`, or leave those empty to use the account already logged
in to the terminal. In `config.yaml`, check `mt5.symbol` (`XAUUSD`, `XAUUSD.m`, `GOLD`... see Market
Watch) and `strategy.pip_size` (0.10 on most brokers).

## 2. Get M1 data from your own broker

```bash
python -m xau_agent export-mt5 --days 730 --out data/xauusd_m1.csv
```

This saves 1-minute bars in UTC **with your broker's real spread per bar**, so the backtest charges
the spread you'd actually pay. If it returns fewer bars than expected, set MT5 **Tools → Options →
Charts → Max bars in chart** to Unlimited and restart the terminal.

Also supported: a file from MT5's own *Export Bars* dialog (add `--server-utc-offset 2` or `3`, your
broker's time zone), or Dukascopy CSV exports (no spread column, so `--spread-pips` is used).

## 3. Backtest

```bash
# all four strategies side by side; --split = out-of-sample check from that date
python -m xau_agent compare --csv data/xauusd_m1.csv --split 2026-01-01 --slippage-pips 0.5 --commission 0

# one strategy in detail: monthly $ and every trade to CSV
python -m xau_agent backtest --csv data/xauusd_m1.csv --strategy asian_sweep --split 2026-01-01 --trades-out trades.csv
```

Signals are built on `strategy.timeframe` candles (5m). Stops and targets are checked on every
1-minute bar, and if one bar touches both, the stop counts.

How to read the results:
- **`win%` vs `breakeven%`:** with SL 25 / TP 50 you break even at (25 + costs) / 75, about 37-38%
  with a 3-pip spread. A strategy is only interesting if it beats that clearly.
- **Look at the out-of-sample columns.** Tune settings on data before `--split`, then check the period
  after it without changing anything. If out-of-sample collapses, it was overfitting.
- **Count trades.** On purely random data, `trend_pullback` once showed +$428 over 74 trades by luck.
  Across 8 random data sets (about 2,200 trades) every strategy hit its target almost exactly 33% of the
  time, which is what pure chance gives. Want a few hundred trades before trusting a result.
- **Losing streaks are normal:** at a 40% win rate, 8-12 losses in a row will happen. At 0.1 lot that's
  about −$250 to −$350.

**Why the backtest skips the AI agent:** Claude was trained on past data and may "remember" what gold
did on a given day, so a backtest of the agent would look unrealistically good. Judge the agent
going forward in paper mode, and compare `agent.enabled: true` vs `false` over the same weeks.

## 4. Check, then paper trade

```bash
python -m xau_agent check     # prices, spread in pips, today's signals, lot size and $ risk. No orders.
python -m xau_agent run       # paper mode: real prices, simulated fills
python -m xau_agent journal   # recent signals, agent decisions, entries and skips
```

On 1m/5m, `agent.news_search: true` can add 30-60 seconds before an entry. If price moves too far
in that time, the trade is skipped. Set it to `false` for faster decisions.

## 5. Go live (in this order)

1. Log the MT5 terminal into a **demo account**, set `mode: live`, and run with the confirmation flag:
   ```bash
   python -m xau_agent run --live
   ```
   The bot refuses to start live trading without `--live`, and refuses `--live` while the config says `paper`.
2. Only after weeks of demo results that match the backtest, switch to a small real account.

**Kill switch:** `python -m xau_agent kill` closes the bot's positions and stops it. SL and TP sit on
the broker's server, so they still work if your PC or the bot goes down. The bot only touches
positions with its own `mt5.magic` number.

## Hyperliquid

Set `broker: hyperliquid` (market `PAXG`, a gold-backed token that trades 24/7, or a HIP-3 gold market
such as `xyz:GOLD`). Put `HL_ACCOUNT_ADDRESS` and an **API wallet** key (`HL_API_PRIVATE_KEY`, created at
https://app.hyperliquid.xyz/API, which can trade but cannot withdraw) in `.env`, and keep
`hyperliquid.testnet: true` until you're ready. `fixed_lots` is converted with `contract_oz`
(0.1 lot = 10 PAXG). Set `paper.fee_bps: 4.5` to simulate Hyperliquid taker fees.

## Project layout

```
xau_agent/
  strategy.py        the four triggers (pure functions, shared with the backtest)
  backtest.py        M1-precision backtest engine, CSV loaders
  risk.py            hard risk rules
  agent.py           Claude review: news search + take/skip decision
  econ_calendar.py   high-impact news blackout
  bot.py             main loop
  brokers/           mt5.py, hyperliquid.py, paper.py
tests/               pytest suite: python -m pytest
```
