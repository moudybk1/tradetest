import json
from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd

from xau_agent.agent import TradeAgent, build_context
from xau_agent.config import AgentConfig, StrategyConfig
from xau_agent.strategy import find_breakout

from .conftest import breakout_day, history


def text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def decision_json(decision="take", confidence=0.8):
    return json.dumps({"decision": decision, "confidence": confidence, "reason": "trend aligned", "key_risks": []})


def test_take_decision_and_request_shape():
    client = FakeClient(text_response(decision_json()))
    d = TradeAgent(AgentConfig(), client).review({"signal": {}})
    assert d.take and d.confidence == 0.8 and d.source == "agent"
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"


def test_skip_decision():
    d = TradeAgent(AgentConfig(), FakeClient(text_response(decision_json("skip", 0.9)))).review({})
    assert not d.take


def test_confidence_is_clamped():
    d = TradeAgent(AgentConfig(), FakeClient(text_response(decision_json(confidence=7)))).review({})
    assert d.confidence == 1.0


def test_refusal_and_bad_json_fail_safe():
    for resp in (text_response("", "refusal"), text_response("not json"), text_response(decision_json(), "max_tokens")):
        d = TradeAgent(AgentConfig(), FakeClient(resp)).review({})
        assert not d.take and d.source == "agent-error"


def test_api_errors_fail_safe():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    errors = [
        anthropic.APIConnectionError(request=request),
        anthropic.InternalServerError("boom", response=httpx2.Response(500, request=request), body=None),
    ]
    for err in errors:
        d = TradeAgent(AgentConfig(), FakeClient(err)).review({})
        assert not d.take and d.source == "agent-error"


def test_news_search_resumes_pause_turn():
    paused = SimpleNamespace(stop_reason="pause_turn", content=[SimpleNamespace(type="server_tool_use")])
    client = FakeClient(paused, text_response("- Dollar weaker after CPI"))
    news = TradeAgent(AgentConfig(), client).news_summary(datetime(2026, 10, 21, 7, 15, tzinfo=timezone.utc))
    assert news == "- Dollar weaker after CPI"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "assistant"


def test_build_context_has_numbers():
    day = pd.Timestamp("2026-10-21", tz="UTC")
    candles = pd.concat([history(60), breakout_day(day)])
    signal = find_breakout(candles, day, 20.0, StrategyConfig())
    hourly = candles.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    ctx = build_context(signal, "PAXG", datetime(2026, 10, 21, 7, 30, tzinfo=timezone.utc), 2006.2, 1.5,
                        hourly, [], None)
    assert ctx["signal"]["reward_risk"] == 2.0
    assert ctx["market"]["h4_trend"] in ("up", "down")
    assert ctx["news_summary"] == "not available"
    json.dumps(ctx)  # must be serialisable for the API call
