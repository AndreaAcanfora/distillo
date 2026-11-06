"""Claude through the official Anthropic SDK, with structured JSON output."""

from __future__ import annotations

import json
from typing import Any, Literal, cast

import anthropic

from .base import LLMError, QuotaError

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Models that accept the server-side refusal fallback in its "default" form.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        *,
        effort: str = "medium",
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = cast(Effort, effort)
        self._client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}:{self.effort}"

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        try:
            if self.model in _FALLBACK_MODELS:
                response: Any = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={
                        "effort": self.effort,
                        "format": {"type": "json_schema", "schema": schema},
                    },
                    betas=[_FALLBACK_BETA],
                    fallbacks="default",
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={
                        "effort": self.effort,
                        "format": {"type": "json_schema", "schema": schema},
                    },
                )
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic rejected the credentials. Set ANTHROPIC_API_KEY "
                "(https://platform.claude.com/settings/keys) or use --provider ollama."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"model not found: {self.model}") from exc
        except anthropic.APIConnectionError as exc:
            raise QuotaError(f"cannot reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            message = f"Anthropic API error {exc.status_code}: {exc.message}"
            if exc.status_code == 429 or exc.status_code >= 500:
                raise QuotaError(message) from exc
            raise LLMError(message) from exc

        if response.stop_reason == "refusal":
            raise LLMError("the model declined this request")
        if response.stop_reason == "max_tokens":
            raise LLMError("the response was cut off; try a lower --chunk-chars")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("the model returned no text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"the model returned invalid JSON: {exc}") from exc
