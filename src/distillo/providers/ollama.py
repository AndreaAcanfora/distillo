"""Local models through Ollama's native chat API: nothing leaves your machine."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .base import LLMError


class OllamaLLM:
    def __init__(self, model: str, *, host: str | None = None, timeout: float = 600.0) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if "://" not in self.host:
            self.host = f"http://{self.host}"
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16_384},
        }
        try:
            response = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.host} ({exc}). Is `ollama serve` running?"
            ) from exc
        if response.status_code == 404:
            raise LLMError(f"Ollama model not found; run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
        try:
            return json.loads(response.json()["message"]["content"])
        except (KeyError, ValueError) as exc:
            raise LLMError(f"Ollama returned an unexpected response: {exc}") from exc
