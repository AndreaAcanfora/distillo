"""Try providers in order, moving on when one is rate limited or out of quota."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from .base import LLM, LLMError, QuotaError


class FallbackLLM:
    def __init__(self, providers: Sequence[LLM]) -> None:
        if not providers:
            raise LLMError("no provider available")
        self.providers = list(providers)
        self._exhausted: set[int] = set()
        self._lock = threading.Lock()
        self.served_by: set[str] = set()

    @property
    def name(self) -> str:
        return " > ".join(p.name for p in self.providers)

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        errors: list[str] = []
        for index, provider in enumerate(self.providers):
            if index in self._exhausted:
                continue
            try:
                result = provider.complete_json(
                    system=system, prompt=prompt, schema=schema, max_tokens=max_tokens
                )
            except QuotaError as exc:
                # Stop asking this provider for the rest of the run.
                with self._lock:
                    self._exhausted.add(index)
                errors.append(str(exc))
                continue
            with self._lock:
                self.served_by.add(provider.name)
            return result
        raise LLMError("every provider failed:\n- " + "\n- ".join(errors or ["all exhausted"]))
