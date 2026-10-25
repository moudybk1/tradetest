"""Claude as a context filter: it can veto a signal, it can never size, move stops or add trades."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import anthropic
import pandas as pd

from .config import AgentConfig
from .econ_calendar import Event
from .indicators import ema, resample_ohlc, rsi
from .strategy import Signal

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You review trade signals for an automated XAU/USD (gold) intraday trading system. A fixed rule \
(Asian-session range breakout) has just triggered. Position size, stop loss and take profit are \
already set by a separate risk manager and you cannot change them. Your only job is to decide \
whether this particular breakout is worth taking given the wider context.

Context that usually matters for gold: the direction of the US dollar and US real yields, Fed \
expectations, high-impact US data due in the next few hours, geopolitical risk, whether the \
breakout goes with or against the higher-timeframe trend, and whether price has already run far \
from the range.

Answer "skip" when the context clearly argues against the trade, or when a major event is likely \
to whipsaw price before the target can be reached. Otherwise answer "take". The system's edge \
comes from taking valid breakouts consistently, so do not skip only because the outcome is \
uncertain. `confidence` is your probability (0 to 1) that taking the trade is the better choice. \
Keep `reason` to one or two sentences and list the main risks you see in `key_risks`."""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["take", "skip"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "confidence", "reason", "key_risks"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    take: bool
    confidence: float
    reason: str
    key_risks: list[str] = field(default_factory=list)
    source: str = "agent"  # agent | rules-only | agent-error

    def to_dict(self) -> dict:
        return asdict(self)


def build_context(
    signal: Signal,
    instrument: str,
    now: datetime,
    price: float,
    spread_bps: float,
    hourly: pd.DataFrame,
    events: list[Event],
    news: str | None,
) -> dict[str, Any]:
    """Facts for the agent, computed locally so the model doesn't have to guess numbers."""
    h4 = resample_ohlc(hourly, "4h")
    daily = resample_ohlc(hourly, "1D")
    daily = daily[daily.index.day_of_week < 5]
    ema50, ema200 = ema(h4["close"], 50).iloc[-1], ema(h4["close"], 200).iloc[-1]
    risk = signal.stop_distance
    return {
        "now_utc": now.strftime("%Y-%m-%d %H:%M"),
        "instrument": instrument,
        "signal": {
            "side": signal.side,
            "breakout_candle_open_utc": signal.candle_time.strftime("%H:%M"),
            "entry": round(signal.entry, 2),
            "stop": round(signal.stop, 2),
            "take_profit": round(signal.take_profit, 2),
            "reward_risk": round(abs(signal.take_profit - signal.entry) / risk, 2) if risk else None,
            "asian_range_high": round(signal.range_high, 2),
            "asian_range_low": round(signal.range_low, 2),
            "asian_range_width_in_daily_atr": round((signal.range_high - signal.range_low) / signal.daily_atr, 2),
            "daily_atr": round(signal.daily_atr, 2),
        },
        "market": {
            "current_price": round(price, 2),
            "spread_bps": round(spread_bps, 2),
            "h4_ema50": round(float(ema50), 2),
            "h4_ema200": round(float(ema200), 2),
            "h4_trend": "up" if ema50 > ema200 else "down",
            "h1_rsi14": round(float(rsi(hourly["close"]).iloc[-1]), 1),
            "last_daily_closes": [
                {"date": ts.strftime("%a %Y-%m-%d"), "close": round(float(c), 2)}
                for ts, c in daily["close"].tail(6).items()
            ],
        },
        "high_impact_usd_events_next_24h": [e.label() for e in events],
        "news_summary": news or "not available",
    }


class TradeAgent:
    def __init__(self, cfg: AgentConfig, client: anthropic.Anthropic | None = None):
        self.cfg = cfg
        self.client = client or anthropic.Anthropic()

    def news_summary(self, now: datetime) -> str | None:
        """Web search for today's gold drivers. Separate call: search citations can't be combined
        with the structured JSON output used for the decision."""
        messages: list[dict] = [{
            "role": "user",
            "content": (
                f"It is {now:%Y-%m-%d %H:%M} UTC. Search for the latest news moving gold (XAU/USD) today: "
                "US dollar, Treasury yields, Fed speakers or expectations, US data releases, geopolitics, "
                "central bank gold buying. Reply with at most 8 short bullet points, newest first, each with "
                "its date. Say clearly if you found nothing from the last 24 hours."
            ),
        }]
        try:
            for _ in range(4):
                response = self.client.beta.messages.create(
                    model=self.cfg.model,
                    max_tokens=16000,
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                    output_config={"effort": self.cfg.effort},
                    tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
                    messages=messages,
                )
                if response.stop_reason != "pause_turn":
                    break
                # Server-side search loop paused; send the turn back so it resumes.
                messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "refusal":
                return None
            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            return text or None
        except anthropic.APIConnectionError as exc:
            log.warning("news search: network error: %s", exc)
        except anthropic.APIStatusError as exc:
            log.warning("news search: API error %s: %s", exc.status_code, exc.message)
        return None

    def review(self, context: dict[str, Any]) -> Decision:
        try:
            response = self.client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=16000,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={
                    "effort": self.cfg.effort,
                    "format": {"type": "json_schema", "schema": DECISION_SCHEMA},
                },
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(context, indent=2)}],
            )
        except anthropic.AuthenticationError:
            return Decision(False, 0.0, "ANTHROPIC_API_KEY is missing or invalid", source="agent-error")
        except anthropic.APIConnectionError as exc:
            return Decision(False, 0.0, f"network error: {exc}", source="agent-error")
        except anthropic.APIStatusError as exc:
            return Decision(False, 0.0, f"API error {exc.status_code}: {exc.message}", source="agent-error")
        return self._parse(response)

    @staticmethod
    def _parse(response: Any) -> Decision:
        # Fail safe: anything unexpected means "don't trade".
        if response.stop_reason in ("refusal", "max_tokens"):
            return Decision(False, 0.0, f"agent stopped early ({response.stop_reason})", source="agent-error")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
            confidence = min(1.0, max(0.0, float(data["confidence"])))
            return Decision(data["decision"] == "take", confidence, str(data["reason"]), list(data["key_risks"]))
        except (ValueError, KeyError, TypeError) as exc:
            return Decision(False, 0.0, f"could not parse agent reply: {exc}", source="agent-error")
