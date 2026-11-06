from __future__ import annotations

import os

from .base import LLM, LLMError

DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-opus-5-5",
    "ollama": "qwen3:8b",
}
PROVIDERS = ("auto", *DEFAULT_MODELS)

NO_KEY_HELP = """\
No model provider is configured. distillo works for free with either of these:
  - Gemini: get a key at https://aistudio.google.com/apikey, then export GEMINI_API_KEY=...
  - Groq:   get a key at https://console.groq.com/keys,     then export GROQ_API_KEY=...
Set both and distillo moves to Groq automatically when Gemini runs out of free quota.
You can also use Claude (ANTHROPIC_API_KEY) or a local model with --provider ollama."""


def make_llm(provider: str, model: str | None = None, *, effort: str = "medium") -> LLM:
    if provider == "auto":
        if model:
            raise LLMError("--model needs an explicit --provider (gemini, groq, anthropic, ollama)")
        return _auto(effort)
    if provider == "gemini":
        from .gemini import GeminiLLM

        return GeminiLLM(model or DEFAULT_MODELS["gemini"])
    if provider == "groq":
        from .groq import GroqLLM

        return GroqLLM(model or DEFAULT_MODELS["groq"])
    if provider == "anthropic":
        from .anthropic import AnthropicLLM

        return AnthropicLLM(model or DEFAULT_MODELS["anthropic"], effort=effort)
    if provider == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM(model or DEFAULT_MODELS["ollama"])
    raise LLMError(f"unknown provider {provider!r}; choose one of: {', '.join(PROVIDERS)}")


def _auto(effort: str) -> LLM:
    """Every provider with a key in the environment, free tiers first."""
    from .gemini import gemini_key
    from .groq import groq_key

    return free_chain(
        gemini_api_key=gemini_key(),
        groq_api_key=groq_key(),
        with_anthropic=bool(os.environ.get("ANTHROPIC_API_KEY")),
        effort=effort,
    )


def free_chain(
    *,
    gemini_api_key: str | None,
    groq_api_key: str | None,
    with_anthropic: bool = False,
    effort: str = "medium",
) -> LLM:
    """Gemini, then Groq (then Claude), for the keys given: used by ``auto`` and by apps
    that hold keys per request instead of in the environment."""
    from .anthropic import AnthropicLLM
    from .fallback import FallbackLLM
    from .gemini import GeminiLLM
    from .groq import GroqLLM

    available = [
        name
        for name, present in (
            ("gemini", gemini_api_key),
            ("groq", groq_api_key),
            ("anthropic", with_anthropic),
        )
        if present
    ]
    if not available:
        raise LLMError(NO_KEY_HELP)
    chain: list[LLM] = []
    for position, name in enumerate(available):
        # Free tiers are often briefly overloaded: with a fallback behind it, a provider
        # gets one quick retry instead of minutes of backoff.
        retries = 3 if position == len(available) - 1 else 1
        if name == "gemini":
            chain.append(
                GeminiLLM(DEFAULT_MODELS["gemini"], api_key=gemini_api_key, retries=retries)
            )
        elif name == "groq":
            chain.append(GroqLLM(DEFAULT_MODELS["groq"], api_key=groq_api_key, retries=retries))
        else:
            chain.append(AnthropicLLM(DEFAULT_MODELS["anthropic"], effort=effort))
    return chain[0] if len(chain) == 1 else FallbackLLM(chain)
