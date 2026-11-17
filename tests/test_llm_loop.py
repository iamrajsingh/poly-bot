from types import SimpleNamespace as NS

import pytest

from polyengine.config import Settings
from polyengine.llm import LLM, BudgetExceeded
from polyengine.storage import Store


def resp(stop, *blocks):
    usage = NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0,
               server_tool_use=NS(web_search_requests=1))
    return NS(stop_reason=stop, content=list(blocks), usage=usage, model="claude-opus-5-5", stop_details=None)


def tool(name, inp):
    return NS(type="tool_use", name=name, input=inp, id="tu_1")


def text(t):
    return NS(type="text", text=t)


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, **kw):
        self.calls.append([dict(m) for m in kw["messages"]])
        return self.script.pop(0)


def make(tmp_path, script, **kw):
    s = Settings(_env_file=None, db_path=str(tmp_path / "t.db"), **kw)
    llm = LLM.__new__(LLM)
    llm.s, llm.store, llm.disabled_reason = s, Store(s.db_path), None
    fake = FakeMessages(script)
    llm.client = NS(beta=NS(messages=fake))
    return llm, fake


async def run(llm):
    return await llm.run_until_tool(purpose="t", system="s", user="u", tools=[], final_tool="submit",
                                    effort="low")


async def test_pause_then_answer(tmp_path):
    llm, fake = make(tmp_path, [resp("pause_turn", text("searching")),
                                resp("tool_use", tool("submit", {"probability": 0.6}))])
    assert await run(llm) == {"probability": 0.6}
    # resume after pause_turn re-sends history ending in the assistant turn - no extra user message
    assert fake.calls[1][-1]["role"] == "assistant"
    assert llm.store.llm_usage_today()[0] == 2


async def test_nudge_once_then_give_up(tmp_path):
    llm, fake = make(tmp_path, [resp("end_turn", text("I think 60%")),
                                resp("tool_use", tool("submit", {"probability": 0.6}))])
    assert await run(llm) == {"probability": 0.6}
    assert fake.calls[1][-1]["role"] == "user"

    llm, _ = make(tmp_path, [resp("end_turn", text("a")), resp("end_turn", text("b"))])
    assert await run(llm) is None


async def test_refusal_returns_none(tmp_path):
    r = resp("refusal")
    r.stop_details = NS(category="cyber")
    llm, _ = make(tmp_path, [r])
    assert await run(llm) is None


async def test_budget_guard(tmp_path):
    llm, _ = make(tmp_path, [], max_llm_calls_per_day=0)
    with pytest.raises(BudgetExceeded):
        await run(llm)


async def test_caching_request_shape_and_cache_aware_cost(tmp_path):
    r = resp("tool_use", tool("submit", {"probability": 0.6}))
    r.usage.input_tokens, r.usage.cache_read_input_tokens, r.usage.cache_creation_input_tokens = 1000, 40000, 8000
    r.usage.output_tokens = 1500
    llm, fake = make(tmp_path, [r])
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return r

    fake.create = create
    await llm.run_until_tool(purpose="forecast:1", system="static", user="u", tools=[], final_tool="submit",
                             effort="medium", cache=True)
    assert seen["cache_control"] == {"type": "ephemeral"}
    assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    row = llm.store._x("SELECT usd, cache_read, cache_write FROM llm_calls").fetchone()
    # 1000*4 + 8000*1.25*4 + 40000*0.20 + 1500*20 per MTok, + 1 search * $0.01
    assert row["usd"] == pytest.approx((4000 + 40000 + 8000 + 30000) / 1e6 + 0.01)
    assert (row["cache_read"], row["cache_write"]) == (40000, 8000)


async def test_refresh_forecast_uses_fewer_searches_and_builds_on_prior(tmp_path):
    from datetime import timedelta
    from polyengine.models import Forecast, Market, utcnow
    from polyengine.research.forecaster import Forecaster
    s = Settings(_env_file=None, db_path=str(tmp_path / "t.db"))
    calls = []

    class FakeLLM:
        async def run_until_tool(self, **kw):
            calls.append(kw)
            return {"probability": 0.4, "confidence": 0.7, "base_rate": "", "rationale": "", "key_factors": [],
                    "sources": [], "resolution_risks": []}

    fc = Forecaster(s, FakeLLM())
    m = Market("1", "c", "Will X?", "rules", "x", "ev", "E", utcnow() + timedelta(days=9), "Y", "N", 0.4, 1e5, 1e5)
    await fc.forecast(m, [])
    prior = Forecast(0.35, 0.8, "because Y", [], [], [], "claude-opus-5-5")
    await fc.forecast(m, [], prior=prior)
    first, refresh = calls
    assert first["tools"][0]["max_uses"] == s.web_search_max_uses and first["cache"] is True
    assert refresh["tools"][0]["max_uses"] == s.web_search_refresh_max_uses < s.web_search_max_uses
    assert "your_previous_forecast" in refresh["user"] and "since your previous forecast" in refresh["user"]
    assert "your_previous_forecast" not in first["user"]
