from __future__ import annotations

from typing import Any, Protocol


class LLMError(RuntimeError):
    pass


class QuotaError(LLMError):
    """The provider is rate limited, out of quota or down: another provider may still work."""


class LLM(Protocol):
    """A model backend.

    Backends may also define ``max_chunk_chars`` and ``max_parallel`` (ints) when their
    limits, such as a small free-tier token budget, call for smaller or fewer requests.
    """

    @property
    def name(self) -> str:
        """Stable identifier such as ``gemini:gemini-3.6-flash``, used for caching."""
        ...

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        """Return the parsed JSON response, which must validate against ``schema``."""
        ...
