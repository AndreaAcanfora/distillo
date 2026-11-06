from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from distillo.providers import LLMError, make_llm
from distillo.providers.anthropic import AnthropicLLM
from distillo.providers.ollama import OllamaLLM

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _client(text: str = '{"ok": true}', stop_reason: str = "end_turn") -> Any:
    response = SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )
    return SimpleNamespace(
        messages=FakeMessages(response), beta=SimpleNamespace(messages=FakeMessages(response))
    )


def test_anthropic_default_model_uses_fallbacks() -> None:
    client = _client()
    llm = AnthropicLLM("claude-opus-5-5", client=client)
    assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"ok": True}
    sent = client.beta.messages.kwargs
    assert sent["fallbacks"] == "default"
    assert sent["output_config"]["format"]["schema"] == SCHEMA
    assert sent["output_config"]["effort"] == "medium"


def test_anthropic_other_models_use_plain_endpoint() -> None:
    client = _client()
    AnthropicLLM("claude-haiku-5-5", effort="low", client=client).complete_json(
        system="s", prompt="p", schema=SCHEMA
    )
    assert client.messages.kwargs["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in client.messages.kwargs


@pytest.mark.parametrize(
    ("stop_reason", "text", "message"),
    [
        ("refusal", "{}", "declined"),
        ("max_tokens", "{", "cut off"),
        ("end_turn", "{", "invalid JSON"),
    ],
)
def test_anthropic_errors(stop_reason: str, text: str, message: str) -> None:
    llm = AnthropicLLM("claude-haiku-5-5", client=_client(text, stop_reason))
    with pytest.raises(LLMError, match=message):
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)


def test_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        seen.update(url=url, payload=json)
        return httpx.Response(200, json={"message": {"content": '{"points": []}'}})

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaLLM("qwen3:8b", host="localhost:11434")
    assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"points": []}
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["payload"]["format"] == SCHEMA
    assert seen["payload"]["stream"] is False


def test_ollama_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(404, text="not found"))
    with pytest.raises(LLMError, match="ollama pull"):
        OllamaLLM("nope").complete_json(system="s", prompt="p", schema=SCHEMA)


def test_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaLLM("qwen3:8b").complete_json(system="s", prompt="p", schema=SCHEMA)


def test_registry() -> None:
    assert make_llm("anthropic").name == "anthropic:claude-opus-5-5:medium"
    assert make_llm("ollama", "llama3.2").name == "ollama:llama3.2"
    with pytest.raises(LLMError, match="unknown provider"):
        make_llm("nope")


def test_schemas_are_strict() -> None:
    from distillo.prompts import EXTRACT_SCHEMA, RANK_SCHEMA

    for schema in (EXTRACT_SCHEMA, RANK_SCHEMA):
        text = json.dumps(schema)
        assert '"additionalProperties": false' in text
        assert "minimum" not in text
