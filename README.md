# XAU Agent: agentic gold trading bot (Hyperliquid / MT5)

An automated XAU/USD (gold) trading bot. A fixed rule finds the setup, hard-coded risk rules decide
whether a trade is allowed, and Claude reviews the news and market context before every entry.
If everything says yes, the bot enters with the stop loss and take profit already attached.

```
 candles ──► TRIGGER (Asian range breakout) ──► RISK RULES (hard code) ──► AI AGENT (Claude) ──► ORDER + SL + TP
                                                   │ any "no" = skip             │ "skip" or low confidence = skip
                                                   └──────────── journal + Telegram alert ◄──────┘
```

- **Trigger:** the first 15-minute candle after 07:00 UTC that closes outside the Asian session
  (00:00-07:00 UTC) high/low. Stop at the middle of the range, target at 2R. One trade per day at most.
- **Risk rules (the AI can't change them):** 0.5% risk per trade, stop trading after -2% in a day,
  max 1 position, no entries 15 minutes before or after high-impact USD news, spread and leverage
  checks, everything closed at 20:30 UTC.
- **AI agent:** searches the web for today's gold news, reads the trend, ATR and economic calendar,
  then answers `take` or `skip` with a confidence score. It can only veto a trade. It never sizes,
  moves stops or opens trades on its own.

> ⚠️ **Trading is risky and this bot can lose money.** Nothing here is financial advice and nothing
> guarantees a profit. Start in paper mode, then a testnet or demo account, and only then risk small
> amounts you can afford to lose.

---

## 1. Install

You need Python 3.11 or newer.

- **Hyperliquid:** works on Windows, Mac or Linux (including a cheap Linux VPS).
- **MT5:** Windows only, with the MetaTrader 5 terminal installed and logged in to your broker.

```bash
git clone <this repo> && cd tradetest
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml     # Windows: copy config.example.yaml config.yaml
cp .env.example .env                   # Windows: copy .env.example .env
```

Fill in `.env`:

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com (needed for the AI agent) |
| `HL_ACCOUNT_ADDRESS` | your Hyperliquid main wallet address |
| `HL_API_PRIVATE_KEY` | create an **API wallet** at https://app.hyperliquid.xyz/API. It can trade but **cannot withdraw**. Never paste your main wallet's private key. |
| `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` | your broker account, or leave empty to use the account already logged in to the terminal |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | optional alerts: create a bot with @BotFather |

In `config.yaml`, set `broker: hyperliquid` or `broker: mt5`. For MT5, check the exact gold symbol
name in Market Watch (`XAUUSD`, `XAUUSD.m`, `GOLD`...).

## 2. Check that everything connects (places no orders)

```bash
python -m xau_agent check
```

This shows the live price and spread, today's Asian range, any breakout signal, the position size
it would use, and upcoming news events.

## 3. Backtest the rules

```bash
# quick look: last ~40 days from Hyperliquid (no key needed)
python -m xau_agent backtest --hyperliquid-days 40

# better: years of XAUUSD history. Export 5m or 15m candles (UTC) as CSV from
# https://www.dukascopy.com/swiss/english/marketwatch/historical/
python -m xau_agent backtest --csv xauusd_15m.csv --cost-bps 5 --trades-out trades.csv
```

Results are in **R** (1R = the amount risked on one trade). Look at `avg_r` after costs, `max_drawdown_r`
and `worst_losing_streak`. If `avg_r` isn't clearly positive across several years, tune the strategy
settings before going further. A random-walk sanity test loses about -0.1R per trade, which is
correct: random prices have no edge and costs drag them negative.

**Why the backtest skips the AI agent:** Claude was trained on data from the past, so it may "remember"
what gold did after a historical CPI release. Backtesting the agent would look unrealistically good.
The agent can only be judged going forward, in paper mode. Compare runs with `agent.enabled: true`
and `false` over the same weeks.

## 4. Paper trade (real prices, fake money)

```bash
python -m xau_agent run
```

Leave it running, for example on a VPS. It checks every 20 seconds. Paper equity is stored in
`data/paper_state.json` and every decision goes into `data/journal.db`:

```bash
python -m xau_agent journal          # recent signals, agent decisions, entries and skips
```

Run it for at least 1-3 months before going live.

## 5. Go live (in this order)

1. **Hyperliquid testnet:** in `config.yaml` set `mode: live` and keep `hyperliquid.testnet: true`. Get
   testnet funds from the Hyperliquid testnet faucet. MT5 users: use a **demo account** login.
2. Run with the extra confirmation flag:
   ```bash
   python -m xau_agent run --live
   ```
   The bot refuses to start live trading without `--live`, and refuses `--live` while the config says `paper`.
3. **Real money:** set `hyperliquid.testnet: false` (or switch MT5 to your real account). Start with a
   small account.

**Kill switch:** from another terminal run `python -m xau_agent kill`. The bot closes all its
positions and stops within one poll interval. You can also always close positions yourself in the
Hyperliquid app or MT5. The stop loss and take profit sit on the exchange or broker, so they still
work if the bot crashes.

## Notes

- **PAXG vs spot gold:** on Hyperliquid the default market is `PAXG`, a gold-backed token (1 PAXG = 1 oz)
  that trades 24/7 and can drift slightly from spot gold. The bot skips weekends. If a builder-deployed
  (HIP-3) gold market is listed, set `hyperliquid.coin` to its `dex:NAME` (for example `xyz:GOLD`).
- **Fees:** Hyperliquid's base taker fee is roughly 4.5 bps per side; check your fee tier. Use a higher
  `--cost-bps` in backtests to match.
- **MT5 times:** MT5 candles use broker server time. The bot detects the offset from the live tick, or
  you can set `mt5.server_utc_offset_hours` yourself.
- **AI cost:** at most one signal a day, and each signal makes two Claude calls: a web news search and
  the decision. The model is set in `agent.model` (default `claude-opus-5-5`). The calls use Claude's
  server-side refusal fallback (`fallbacks: "default"`). If the agent errors or can't be reached, the
  trade is **skipped**, never taken blind.
- **Calendar:** news times come from the free ForexFactory weekly feed. If it can't be loaded, trading is
  blocked by default (`risk.block_if_calendar_unavailable`).

## Project layout

```
xau_agent/
  strategy.py        trigger: Asian range breakout (pure functions, shared with the backtest)
  risk.py            hard risk rules
  agent.py           Claude review: news search + take/skip decision
  econ_calendar.py   high-impact news blackout
  bot.py             main loop
  backtest.py        rules backtest
  brokers/           hyperliquid.py, mt5.py, paper.py
tests/               pytest suite: python -m pytest
```
