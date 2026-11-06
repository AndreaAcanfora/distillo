"""Groq's OpenAI-compatible API (free tier): chat models and hosted Whisper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._http import post_json
from .base import LLMError

API = "https://api.groq.com/openai/v1"


def groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


def _require_key(api_key: str | None) -> str:
    key = api_key or groq_key()
    if not key:
        raise LLMError(
            "Groq needs an API key: set GROQ_API_KEY (free at https://console.groq.com/keys)."
        )
    return key


class GroqLLM:
    # Groq's free tier allows 8,000 tokens per minute per model, counting the room left
    # for the answer: keep each request well under that, and send one at a time.
    max_chunk_chars = 7_000
    max_parallel = 1
    max_output_tokens = 4_000

    def __init__(self, model: str, *, api_key: str | None = None, retries: int = 3) -> None:
        self.model = model
        self.retries = retries
        self.api_key = _require_key(api_key)

    @property
    def name(self) -> str:
        return f"groq:{self.model}"

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": min(max_tokens, self.max_output_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": _without_enums(schema), "strict": True},
            },
        }
        if self.model.startswith("openai/gpt-oss"):
            # At the default (medium) effort gpt-oss can spend its whole token budget
            # reasoning and return empty JSON; low is fast and enough for extraction.
            payload["reasoning_effort"] = "low"
        body = post_json(
            f"{API}/chat/completions",
            provider="Groq",
            retries=self.retries,
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
        )
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Groq returned an unexpected response") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("the Groq response was cut off; try a lower --chunk-chars")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"Groq returned invalid JSON: {exc}") from exc


def _without_enums(schema: Any) -> Any:
    """Groq rejects a whole answer when one value misses an ``enum``; the pipeline
    normalizes such values itself, so send the schema without them."""
    if isinstance(schema, dict):
        return {k: _without_enums(v) for k, v in schema.items() if k != "enum"}
    if isinstance(schema, list):
        return [_without_enums(v) for v in schema]
    return schema


# Groq's free tier accepts uploads up to this size for speech-to-text.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def groq_transcribe(
    path: Path,
    *,
    model: str = "whisper-large-v3-turbo",
    language: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return Groq's ``verbose_json`` transcription (``language`` and timed ``segments``)."""
    key = _require_key(api_key)
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise LLMError(f"{path.name} is {size / 2**20:.0f} MB, over Groq's 25 MB upload limit")
    data: dict[str, Any] = {"model": model, "response_format": "verbose_json", "temperature": "0"}
    if language:
        data["language"] = language
    with path.open("rb") as handle:
        return dict(
            post_json(
                f"{API}/audio/transcriptions",
                provider="Groq",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (path.name, handle)},
                data=data,
                timeout=900.0,
            )
        )
