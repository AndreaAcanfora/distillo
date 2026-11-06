"""Google Gemini through the Generative Language REST API (free tier in Google AI Studio)."""

from __future__ import annotations

import json
import os
from typing import Any

from ._http import post_json
from .base import LLMError

API = "https://generativelanguage.googleapis.com/v1beta/models"
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY")


def gemini_key() -> str | None:
    return next((os.environ[v] for v in KEY_VARS if os.environ.get(v)), None)


class GeminiLLM:
    def __init__(self, model: str, *, api_key: str | None = None, retries: int = 3) -> None:
        self.model = model
        self.retries = retries
        self.api_key = api_key or gemini_key()
        if not self.api_key:
            raise LLMError(
                "Gemini needs an API key: set GEMINI_API_KEY "
                "(free at https://aistudio.google.com/apikey)."
            )

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        body = post_json(
            f"{API}/{self.model}:generateContent",
            provider="Gemini",
            retries=self.retries,
            # When overloaded, Gemini can hold a request for minutes before failing.
            timeout=90.0,
            headers={"x-goog-api-key": str(self.api_key)},
            payload={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                },
            },
        )
        candidates = body.get("candidates") or []
        if not candidates:
            reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer ({reason})")
        candidate = candidates[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise LLMError("the Gemini response was cut off; try a lower --chunk-chars")
        text = "".join(
            part.get("text", "") for part in (candidate.get("content") or {}).get("parts", [])
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Gemini returned invalid JSON: {exc}") from exc
