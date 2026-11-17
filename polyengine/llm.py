"""Thin wrapper around Claude: budget guard, refusal handling, pause_turn resume, final-answer tool."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)

# $ per million tokens (input, output, cache read) and $ per web search - for the daily spend guard.
# Cache writes bill at 1.25x input (5-minute TTL).
_PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
_WEB_SEARCH_USD = 0.01

# Server-side fallback: if a safety classifier declines, the API retries on Anthropic's
# recommended model for that refusal category inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept `fallbacks: "default"`. Claude Haiku 5.5 has no server-side fallback (the
# parameter must not be sent); its refusals are handled client-side like any other.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, settings: Settings, store: Store):
        self.s = settings
        self.store = store
        # 5-minute cap per request (forecasts take ~1 min); the SDK default of 10 min would let a dead
        # connection stall a research slot far too long.
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None, max_retries=3,
                                               timeout=300.0)
        self.disabled_reason: str | None = None

    def budget_ok(self) -> bool:
        if self.disabled_reason:
            return False
        calls, usd = self.store.llm_usage_today()
        return calls < self.s.max_llm_calls_per_day and usd < self.s.max_llm_usd_per_day

    async def run_until_tool(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        final_tool: str,
        effort: str,
        max_turns: int = 6,
        cache: bool = False,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Run a turn loop until Claude calls `final_tool`; return that tool's input (or None).

        Forced tool_choice isn't available on this model family, so we use `auto`, instruct the
        model to finish with the tool, and nudge once if it ends its turn without calling it.
        """
        if self.disabled_reason:
            raise BudgetExceeded(f"LLM disabled: {self.disabled_reason}")
        if not self.budget_ok():
            raise BudgetExceeded("daily LLM budget used up (PE_MAX_LLM_CALLS_PER_DAY / PE_MAX_LLM_USD_PER_DAY)")

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        nudged = False
        model = model or self.s.anthropic_model
        extra: dict[str, Any] = {}
        if model in _FALLBACK_MODELS:
            extra.update(betas=[_FALLBACK_BETA], fallbacks="default")
        system_param: Any = system
        if cache:
            # Explicit breakpoint on the static prefix (tools + system, shared by every task in a
            # cycle) plus automatic caching for the growing tail. With caching on, web search also
            # writes cache entries after its results, so later steps of the search loop re-read
            # them at the cache-read rate instead of full input price.
            system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            extra["cache_control"] = {"type": "ephemeral"}
        for _ in range(max_turns):
            try:
                resp = await self.client.beta.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=system_param,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "auto"},
                    output_config={"effort": effort},
                    **extra,
                )
            except anthropic.AuthenticationError as e:
                self.disabled_reason = f"authentication failed ({e.message})"
                log.error("LLM research disabled: %s", self.disabled_reason)
                return None
            except TypeError as e:
                # Raised by the SDK before sending when no credential source resolves.
                if "authentication" not in str(e).lower():
                    raise
                self.disabled_reason = "no Anthropic credentials (set ANTHROPIC_API_KEY)"
                log.error("LLM research disabled: %s", self.disabled_reason)
                return None
            except anthropic.RateLimitError as e:
                log.warning("%s: rate limited (%s)", purpose, e.message)
                return None
            except anthropic.BadRequestError as e:
                log.error("%s: bad request: %s", purpose, e.message)
                return None
            except anthropic.APIStatusError as e:
                log.error("%s: API error %s: %s", purpose, e.status_code, e.message)
                return None
            except anthropic.APIConnectionError as e:
                log.warning("%s: connection error: %s", purpose, e)
                return None

            self._record(purpose, resp)

            if resp.stop_reason == "refusal":
                cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
                log.warning("%s: model declined (category=%s)", purpose, cat)
                return None

            # Append content unchanged - thinking blocks must be passed back as-is.
            messages.append({"role": "assistant", "content": resp.content})

            for block in resp.content:
                if block.type == "tool_use" and block.name == final_tool:
                    return dict(block.input)

            if resp.stop_reason == "pause_turn":
                continue  # server-side tool loop paused; re-send to resume, no extra user message
            if resp.stop_reason == "tool_use":
                # Only server tools + final_tool are offered; answer any stray client call with an error.
                results = [{"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                            "content": f"Unknown tool. Call {final_tool} with your answer."}
                           for b in resp.content if b.type == "tool_use"]
                messages.append({"role": "user", "content": results})
                continue
            if resp.stop_reason == "max_tokens" or nudged:
                log.warning("%s: ended without %s (stop=%s)", purpose, final_tool, resp.stop_reason)
                return None
            nudged = True
            messages.append({"role": "user", "content": f"Submit your final answer now by calling {final_tool}."})
        log.warning("%s: no answer after %d turns", purpose, max_turns)
        return None

    def _record(self, purpose: str, resp: Any) -> None:
        u = resp.usage
        pin, pout, pread = _PRICES.get(resp.model, _PRICES["claude-opus-5-5"])
        searches = 0
        stu = getattr(u, "server_tool_use", None)
        if stu is not None:
            searches = getattr(stu, "web_search_requests", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usd = ((u.input_tokens + cache_write * 1.25) * pin + cache_read * pread
               + u.output_tokens * pout) / 1e6 + searches * _WEB_SEARCH_USD
        self.store.record_llm_call(purpose, resp.model, u.input_tokens, u.output_tokens, searches, usd,
                                   cache_read, cache_write)
        log.info("LLM %s: in=%d cache_read=%d cache_write=%d out=%d searches=%d ~$%.3f", purpose,
                 u.input_tokens, cache_read, cache_write, u.output_tokens, searches, usd)
