"""Provider clients: one async ``complete()`` per provider, returning text plus token usage.

Keys come from the environment (never from arguments or files in the repository):
ANTHROPIC_API_KEY (or an ``ant auth login`` profile), OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from conviction.evaluation.api import convert

# OpenAI-compatible Chat Completions endpoints. Verify against each provider's docs before a paid run.
# ("openai-chat" is kept for non-reasoning OpenAI models; OpenAI models default to the Responses API.)
OPENAI_COMPATIBLE = {
    "openai-chat": (None, "OPENAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}
PROVIDERS = ["anthropic", "openai", *OPENAI_COMPATIBLE, "mock"]


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    stop: str
    meta: dict = field(default_factory=dict)


class AnthropicProvider:
    """Claude via the Messages API. Sampling parameters are not sent: current models reject them.
    Thinking is left at the model default unless ``thinking``/``effort`` are given; only text blocks
    are scored (thinking blocks are ignored)."""

    def __init__(self, model: str, max_output_tokens: int, effort: str | None = None,
                 extra: dict | None = None, max_retries: int = 6, http_client=None):
        import anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries, **({"http_client": http_client} if http_client else {}))
        self.model, self.max_output_tokens, self.effort, self.extra = model, max_output_tokens, effort, extra or {}

    def request_params(self, system: str, turns: list[dict]) -> dict:
        p = {"model": self.model, "max_tokens": self.max_output_tokens, "system": system,
             "messages": convert.to_anthropic(turns), "tools": convert.ANTHROPIC_TOOLS,
             "tool_choice": {"type": "none"}}
        if self.effort:
            p["output_config"] = {"effort": self.effort}
        p.update(self.extra)
        return p

    async def complete(self, params: dict, episode: dict | None = None) -> Completion:
        r = await self.client.messages.create(**params)
        text = "".join(b.text for b in r.content if b.type == "text")
        meta = {"stop_reason": r.stop_reason, "id": r.id}
        if r.stop_reason == "refusal" and getattr(r, "stop_details", None):
            meta["refusal_category"] = getattr(r.stop_details, "category", None)
        return Completion(text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason or "", meta)


class OpenAICompatibleProvider:
    """OpenAI, xAI and Google through the OpenAI-compatible Chat Completions API."""

    def __init__(self, provider: str, model: str, max_output_tokens: int, effort: str | None = None,
                 temperature: float | None = None, base_url: str | None = None, extra: dict | None = None,
                 max_retries: int = 6, http_client=None):
        import openai
        default_url, key_env = OPENAI_COMPATIBLE[provider]
        key = os.environ.get(key_env)
        if not key:
            raise SystemExit(f"{key_env} is not set; export it in your shell (it is never read from files)")
        self.client = openai.AsyncOpenAI(api_key=key, base_url=base_url or default_url, max_retries=max_retries,
                                         **({"http_client": http_client} if http_client else {}))
        self.provider, self.model = provider, model
        self.max_output_tokens, self.effort, self.temperature, self.extra = max_output_tokens, effort, temperature, extra or {}

    def request_params(self, system: str, turns: list[dict]) -> dict:
        p = {"model": self.model, "messages": convert.to_openai(system, turns), "tools": convert.OPENAI_TOOLS,
             "tool_choice": "none", "max_completion_tokens": self.max_output_tokens}
        if self.effort:
            p["reasoning_effort"] = self.effort
        if self.temperature is not None:
            p["temperature"] = self.temperature
        p.update(self.extra)
        return p

    async def complete(self, params: dict, episode: dict | None = None) -> Completion:
        r = await self.client.chat.completions.create(**params)
        ch = r.choices[0]
        u = r.usage
        meta = {"finish_reason": ch.finish_reason, "id": r.id}
        details = getattr(u, "completion_tokens_details", None)
        if details is not None and getattr(details, "reasoning_tokens", None) is not None:
            meta["reasoning_tokens"] = details.reasoning_tokens
        return Completion(ch.message.content or "", u.prompt_tokens, u.completion_tokens, ch.finish_reason or "", meta)


class OpenAIResponsesProvider:
    """OpenAI models through the Responses API (required by OpenAI for function tools with reasoning).
    ``store`` is false so OpenAI does not retain the responses (matters for the private split)."""

    def __init__(self, model: str, max_output_tokens: int, effort: str | None = None,
                 temperature: float | None = None, base_url: str | None = None, extra: dict | None = None,
                 max_retries: int = 6, http_client=None):
        import openai
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set; export it in your shell (it is never read from files)")
        self.client = openai.AsyncOpenAI(api_key=key, base_url=base_url, max_retries=max_retries,
                                         **({"http_client": http_client} if http_client else {}))
        self.model, self.max_output_tokens, self.effort, self.temperature, self.extra = \
            model, max_output_tokens, effort, temperature, extra or {}

    def request_params(self, system: str, turns: list[dict]) -> dict:
        p = {"model": self.model, "instructions": system, "input": convert.to_openai_responses(turns),
             "tools": convert.RESPONSES_TOOLS, "tool_choice": "none", "max_output_tokens": self.max_output_tokens,
             "store": False}
        if self.effort:
            p["reasoning"] = {"effort": self.effort}
        if self.temperature is not None:
            p["temperature"] = self.temperature
        p.update(self.extra)
        return p

    async def complete(self, params: dict, episode: dict | None = None) -> Completion:
        r = await self.client.responses.create(**params)
        u = r.usage
        meta = {"status": r.status, "id": r.id}
        details = getattr(u, "output_tokens_details", None)
        if details is not None:
            meta["reasoning_tokens"] = details.reasoning_tokens
        inc = getattr(r, "incomplete_details", None)
        if inc is not None:
            meta["incomplete_reason"] = getattr(inc, "reason", None)
        return Completion(r.output_text or "", u.input_tokens, u.output_tokens, r.status or "", meta)


class MockProvider:
    """Offline stand-in for tests and dry runs: answers with the episode's oracle reference response,
    so a correct pipeline scores perfectly. Uses Anthropic-format conversion to exercise it."""

    def __init__(self, model: str = "mock", **_):
        self.model = model

    def request_params(self, system: str, turns: list[dict]) -> dict:
        return {"model": self.model, "system": system, "messages": convert.to_anthropic(turns)}

    async def complete(self, params: dict, episode: dict | None = None) -> Completion:
        text = (episode or {}).get("reference_response") or json.dumps(
            {"reason": "mock", "answer": None, "decision": "insufficient"})
        n_in = len(json.dumps(params)) // 4
        return Completion(text, n_in, len(text) // 4, "end_turn", {"mock": True})


def make_provider(name: str, model: str, max_output_tokens: int, effort: str | None, temperature: float | None,
                  base_url: str | None, extra: dict | None):
    if name == "anthropic":
        if temperature is not None:
            raise SystemExit("--temperature is not supported for Anthropic models (current models reject it)")
        return AnthropicProvider(model, max_output_tokens, effort, extra)
    if name == "openai":
        return OpenAIResponsesProvider(model, max_output_tokens, effort, temperature, base_url, extra)
    if name in OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider(name, model, max_output_tokens, effort, temperature, base_url, extra)
    if name == "mock":
        return MockProvider(model)
    raise SystemExit(f"unknown provider {name!r}; choose from {PROVIDERS}")
