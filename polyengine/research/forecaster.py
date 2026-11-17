"""Blind LLM forecaster.

The model is deliberately NOT shown the Polymarket price or other venues' prices: an
estimate that never saw the market is an *independent* source, which is what the
ensemble needs. Anchored estimates just echo the market back and create fake agreement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import Settings
from ..llm import LLM
from ..models import Forecast, Market, clamp
from .news import Headline

log = logging.getLogger(__name__)

SYSTEM = """You are a calibrated forecaster estimating the probability that a prediction-market \
question resolves YES. Your estimates are scored with a log score, so overconfidence is costly \
and well-calibrated uncertainty is rewarded.

How to work:
- Read the resolution criteria literally. What exactly must happen, by when, according to which \
source? Edge cases in the wording often decide the outcome.
- Start from a base rate for this kind of event, then update on specific, current evidence.
- Use web search to find the latest developments, official sources, schedules, polls, data \
releases or statements. Prefer primary and reputable sources; note publication dates.
- Weigh evidence on both sides. Consider the time remaining until resolution and how much can \
still change.
- If evidence is thin or conflicting, keep the estimate closer to the base rate and report low \
confidence.
- Form an independent view from primary evidence. Do not look up or rely on prediction-market or \
betting-exchange prices for this question (Polymarket, Kalshi, Manifold, PredictIt and similar). \
Those prices are already used elsewhere, so echoing them adds no information.

When finished, call submit_forecast exactly once. `confidence` (0-1) is how well-informed the \
estimate is (quality and recency of evidence), not how extreme the probability is."""

FORECAST_TOOL = {
    "name": "submit_forecast",
    "description": "Submit the final probability that the question resolves YES.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "probability": {"type": "number", "description": "P(YES), between 0.01 and 0.99"},
            "confidence": {"type": "number", "description": "0-1: how well-informed this estimate is"},
            "base_rate": {"type": "string", "description": "Reference class and base rate used"},
            "rationale": {"type": "string", "description": "Concise reasoning, max ~150 words"},
            "key_factors": {"type": "array", "items": {"type": "string"}},
            "sources": {"type": "array", "items": {"type": "string"}, "description": "URLs relied on"},
            "resolution_risks": {"type": "array", "items": {"type": "string"},
                                 "description": "Ambiguities in wording / resolution source that could surprise"},
        },
        "required": ["probability", "confidence", "base_rate", "rationale", "key_factors",
                     "sources", "resolution_risks"],
        "additionalProperties": False,
    },
}


def build_prompt(market: Market, news: list[Headline], prior: Forecast | None = None,
                 feedback: str | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    end = market.end_date.strftime("%Y-%m-%d %H:%M UTC") if market.end_date else "unknown"
    lines = [
        f"Current time: {now}",
        f"Question: {market.question}",
    ]
    if not market.is_yes_no:
        a, b = market.outcomes
        lines.append(f'Outcomes: "{a}" vs "{b}". In this task, YES means "{a}" and NO means "{b}".')
    if market.event_title and market.event_title != market.question:
        lines.append(f"Event: {market.event_title}")
    lines += [
        f"Scheduled resolution/close: {end}",
        "",
        "<resolution_criteria>",
        market.description.strip()[:2500],
        "</resolution_criteria>",
    ]
    if news:
        lines += ["", "<recent_headlines> (automated news search - verify before relying on them)"]
        lines += [f"- [{h.published}] {h.source}: {h.title}" for h in news]
        lines.append("</recent_headlines>")
    if feedback:
        # The engine's learning loop: how this forecaster's past probabilities turned out.
        lines += ["", "<your_track_record>", feedback, "</your_track_record>"]
    if prior:
        # Refresh: build on the earlier work instead of redoing it - far fewer searches and tokens.
        lines += ["", f"<your_previous_forecast made=\"{prior.ts:%Y-%m-%d %H:%M} UTC\">",
                  f"P(YES)={prior.prob:.2f}, confidence={prior.confidence:.2f}",
                  prior.rationale.strip()[:1200],
                  "</your_previous_forecast>", "",
                  "Search only for developments since your previous forecast, then update it (keep it if "
                  "nothing material changed) and call submit_forecast."]
    else:
        lines += ["", "Research this question, then call submit_forecast."]
    return "\n".join(lines)


class Forecaster:
    def __init__(self, settings: Settings, llm: LLM):
        self.s = settings
        self.llm = llm

    async def forecast(self, market: Market, news: list[Headline], prior: Forecast | None = None,
                       feedback: str | None = None) -> Forecast | None:
        max_uses = self.s.web_search_refresh_max_uses if prior else self.s.web_search_max_uses
        tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses,
             "blocked_domains": self.s.forecast_blocked_domains},
            FORECAST_TOOL,
        ]
        out = await self.llm.run_until_tool(
            purpose=f"forecast:{market.id}",
            system=SYSTEM,
            user=build_prompt(market, news, prior, feedback),
            tools=tools,
            final_tool="submit_forecast",
            effort=self.s.forecast_effort,
            max_turns=8,
            cache=self.s.prompt_caching,
        )
        if not out:
            return None
        try:
            prob = clamp(float(out["probability"]), 0.01, 0.99)
            conf = clamp(float(out["confidence"]), 0.0, 1.0)
        except (KeyError, TypeError, ValueError):
            log.warning("forecast for %s had invalid fields: %s", market.id, out)
            return None
        return Forecast(
            prob=prob, confidence=conf,
            rationale=f"[base rate] {out.get('base_rate', '')}\n{out.get('rationale', '')}",
            key_factors=list(out.get("key_factors", []))[:10],
            sources=list(out.get("sources", []))[:15],
            resolution_risks=list(out.get("resolution_risks", []))[:8],
            model=self.s.anthropic_model,
        )
