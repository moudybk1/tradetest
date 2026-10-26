# XAU Agent: agentic gold trading bot (MT5 / Hyperliquid)

An automated XAU/USD (gold) trading bot. A fixed rule finds the setup, hard-coded risk rules decide
whether a trade is allowed, and Claude reviews the news and market context before every entry.
If everything says yes, the bot enters with the stop loss and take profit already attached.

```
 candles ──► TRIGGER (strategy rules) ──► RISK RULES (hard code) ──► AI AGENT (Claude) ──► ORDER + SL + TP
                                             │ any "no" = skip           │ "skip" or low confidence = skip
                                             └────────── journal + Telegram alert ◄──────┘
```

Default setup (`config.example.yaml`): **MT5, `ifvg_hammer` on 5-minute candles, both directions,
RR 1:2, 0.1 lot per entry, max 2 trades a day.** The stop goes just beyond the hammer (or shooting
star) and the target is 2× that distance. With XAUUSD at 1 pip = $0.10 and 100 oz per lot,
0.1 lot = **$1 per pip**, so the dollar risk changes with each setup. A 30-pip stop risks $30 to
win $60, and stops wider than 100 pips ($100) are skipped.

> ⚠️ **Trading is risky and this bot can lose money.** Nothing here is financial advice and nothing
> guarantees a profit. Backtest first, then paper trade, then a demo account, and only then risk
> small amounts you can afford to lose.

## Strategies (`strategy.name`)

| name | idea | default window (UTC unless noted) |
|---|---|---|
| `ny_orb` | first 5m close beyond the New York opening range (09:30 NY + 15 min, DST handled) | until 12:00 New York |
| `asian_sweep` | London pushes through the Asian high (or low), then a candle closes back inside → trade the reversal. Stop beyond the sweep wick | 07:00-10:00 |
| `trend_pullback` | H1 EMA50 vs EMA200 sets the direction; a candle dips to the EMA20 and closes back in the trend. Stop beyond the recent swing | 07:00-11:00, 13:00-17:00 |
| `asian_breakout` | first close beyond the Asian (00:00-07:00) range, with ATR filters | 07:00-16:00 |
| `ifvg_hammer` | a hammer at a swing low, then within 6 candles a candle **closes above a bearish fair value gap** (the FVG is inverted = IFVG) → long at that close. Shorts are the mirror: shooting star, then a close below a bullish FVG | 07:00-11:00, 13:00-17:00 |

`ifvg_hammer` definitions (all adjustable in `config.yaml` → `ifvg_hammer`):
- **FVG:** 3 candles where candle 1's low is above candle 3's high (bearish gap) or candle 1's high is
  below candle 3's low (bullish gap), at least `min_fvg_pips` wide, formed in the last `fvg_lookback_bars`.
- **IFVG:** the first candle that *closes* through that gap from the other side. Entry is at that close,
  without waiting for a retest.
- **Hammer:** candle range at least 5 pips, lower wick at least 2× the body, upper wick at most 25% of the
  range, and its low is the lowest of the last 10 candles. If price trades below the hammer's low
  before the IFVG, the hammer no longer counts.
- Stop (RR mode): the hammer's low minus 3 pips (`stop_buffer_pips`); shorts use the shooting star's
  high plus 3 pips.

Exits (`strategy.exit_mode`):
- **`rr` (default):** the stop sits just beyond the setup's own level: the hammer, the sweep wick, the
  swing low/high, or the opening range. The target is `reward_risk` × the stop distance (2 = RR 1:2).
  The stop is never tighter than `min_stop_pips` (10). Signals that would need a stop wider than
  `max_stop_pips` (100) are skipped.
- **`fixed_pips`:** SL/TP are always exactly `sl_pips` / `tp_pips` (25/50) from the real entry. Setups
  whose level doesn't fit inside 25 pips are skipped. For `ifvg_hammer` on M5 that is almost every
  setup, which is why `rr` is the default.

Everything open is closed at `flatten_utc` (20:30 UTC).

## Risk rules (the AI can't change them)

- `sizing: fixed_lots` → every entry is `fixed_lots` (0.1). The trade is skipped if one stop-out would
  lose more than `max_risk_per_trade_pct` (2%) of equity. At 0.1 lot, a 50-pip stop is $50 and needs at
  least $2,500 equity; the 100-pip maximum ($100) needs $5,000. `sizing: percent` sizes each trade from
  `risk_per_trade_pct` instead, so a wide stop gets a smaller lot.
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

**No MT5 or no Windows?** Download free M1 history from Dukascopy instead. It includes their real
bid/ask spread per minute, and works on any OS:

```bash
python -m xau_agent download-dukascopy --days 730 --out data/xauusd_m1.csv
```

Also supported: a file from MT5's own *Export Bars* dialog (add `--server-utc-offset 2` or `3`, your
broker's time zone), or Dukascopy CSV exports (no spread column, so `--spread-pips` is used).

## 3. Backtest

**One click:** `scripts\backtest_windows.bat` (double-click on Windows) or `bash scripts/backtest.sh`
(Mac/Linux). It installs everything, downloads 2 years of M1 gold from Dukascopy, and runs three
comparisons: M5, M1, and M5 with Bybit perp fees. Results go to `data/results_*.txt`.

```bash
# all strategies side by side (RR 1:2 from config); --split = out-of-sample check from that date
python -m xau_agent compare --csv data/xauusd_m1.csv --split 2026-01-01 --slippage-pips 0.5 --commission 0

# the same on 1-minute signals, with a different RR, or with fixed 25/50-pip exits
python -m xau_agent compare --csv data/xauusd_m1.csv --timeframe 1m --split 2026-01-01
python -m xau_agent compare --csv data/xauusd_m1.csv --rr 1.5 --split 2026-01-01
python -m xau_agent compare --csv data/xauusd_m1.csv --exit-mode fixed_pips --split 2026-01-01

# one strategy in detail: monthly $ and every trade to CSV
python -m xau_agent backtest --csv data/xauusd_m1.csv --strategy asian_sweep --split 2026-01-01 --trades-out trades.csv
```

Signals are built on `strategy.timeframe` candles (5m). Stops and targets are checked on every
1-minute bar, and if one bar touches both, the stop counts.

How to read the results:
- **`win%` vs `breakeven%`:** at RR 1:2 you break even at (1 + cost in R) / 3. A 30-pip stop with
  3.5 pips of spread and slippage needs about 37%. Tighter stops need more, because the spread is a
  bigger share of the risk. `breakeven%` is computed from the real stops and costs of every trade.
  A strategy is only interesting if it beats that clearly.
- **`avg_SL_pips` = average dollar risk at 0.1 lot.** A 40-pip average means about $40 per losing trade.
- **Look at the out-of-sample columns.** Tune settings on data before `--split`, then check the period
  after it without changing anything. If out-of-sample collapses, it was overfitting.
- **Count trades.** On purely random data, `trend_pullback` once showed +$428 over 74 trades by luck.
  On random data every strategy averages about 0R before costs, both with fixed pips and with RR 1:2,
  which is what pure chance gives. Want a few hundred trades before trusting a result.
- **Losing streaks are normal:** at a 35-40% win rate, 8-12 losses in a row will happen. At 0.1 lot
  with 40-pip stops that's about −$350 to −$500.

**Why the backtest skips the AI agent:** Claude was trained on past data and may "remember" what gold
did on a given day, so a backtest of the agent would look unrealistically good. Judge the agent
going forward in paper mode, and compare `agent.enabled: true` vs `false` over the same weeks.

### Quick check in TradingView (no Python needed)

`tradingview/ifvg_hammer.pine` is the `ifvg_hammer` strategy as a TradingView strategy, with the
same rules and both directions:

1. Open a gold chart on 1m or 5m: `BYBIT:XAUTUSDT.P`, `BYBIT:PAXGUSDT.P`, `OANDA:XAUUSD`...
2. Open **Pine Editor**, paste the file, click **Add to chart**, then open **Strategy Tester**.
   Win rate is "Percent profitable".
3. In the strategy's **Properties**, set the commission to your venue's fee (default 0.055% per side,
   Bybit's non-VIP taker fee; use 0 on a CFD chart and add slippage instead).

Limits: TradingView only backtests the candles your plan loads, which on 1m/5m is often just a few
weeks, so the sample is small. It also uses its own rule for a candle that touches both stop and
target. Use the Python backtester for 1-2 years of data.

### Costs on crypto exchanges (Bybit, Hyperliquid)

Perpetual futures charge a fee on the **position value**, not a spread. At Bybit's non-VIP taker
fee (0.055% per side) and gold around $4,000, a round trip costs about $4.40 per ounce, or
**about 44 pips**. That is more than a typical stop in these M1/M5 setups. At RR 1:2:

| average stop | cost in R | win rate needed to break even |
|---|---|---|
| 40 pips | 1.1R | ~70% |
| 75 pips | 0.6R | ~53% |
| 150 pips | 0.3R | ~43% |

Backtest with `--fee-pct 0.055` to see this. Limit (maker) orders are cheaper, and CFD brokers on MT5
usually charge only a spread of 2-4 pips. Check your exchange's fee tier.

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
  strategy.py        the five triggers (pure functions, shared with the backtest)
  backtest.py        M1-precision backtest engine, CSV loaders
  dukascopy.py       free M1 history download
  risk.py            hard risk rules
  agent.py           Claude review: news search + take/skip decision
  econ_calendar.py   high-impact news blackout
  bot.py             main loop
  brokers/           mt5.py, hyperliquid.py, paper.py
tradingview/         Pine Script version of ifvg_hammer for TradingView's Strategy Tester
scripts/             one-click backtest (Windows .bat, Mac/Linux .sh)
tests/               pytest suite: python -m pytest
```
