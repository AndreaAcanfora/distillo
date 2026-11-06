from __future__ import annotations

import os

from .base import LLM, LLMError

DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
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
    """Every provider with a key, free tiers first."""
    from .fallback import FallbackLLM
    from .gemini import gemini_key
    from .groq import groq_key

    chain: list[LLM] = []
    if gemini_key():
        chain.append(make_llm("gemini"))
    if groq_key():
        chain.append(make_llm("groq"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        chain.append(make_llm("anthropic", effort=effort))
    if not chain:
        raise LLMError(NO_KEY_HELP)
    return chain[0] if len(chain) == 1 else FallbackLLM(chain)
