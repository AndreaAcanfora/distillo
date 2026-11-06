from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from distillo.providers import LLMError, QuotaError, _http, make_llm
from distillo.providers.fallback import FallbackLLM
from distillo.providers.gemini import GeminiLLM
from distillo.providers.groq import GroqLLM
from distillo.transcribe import TranscriptionError, load_transcript

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def no_keys_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(_http.time, "sleep", lambda _seconds: None)


class Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def gemini_ok(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(payload)}]}}
            ]
        },
    )


def groq_ok(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]},
    )


def test_gemini_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(gemini_ok({"points": []}))
    monkeypatch.setattr(httpx, "post", recorder)
    llm = GeminiLLM("gemini-3.8-flash", api_key="k")
    assert llm.complete_json(system="sys", prompt="p", schema=SCHEMA) == {"points": []}
    sent = recorder.requests[0]
    assert sent["url"].endswith("/models/gemini-3.8-flash:generateContent")
    assert sent["headers"] == {"x-goog-api-key": "k"}
    config = sent["json"]["generationConfig"]
    assert config["responseJsonSchema"] == SCHEMA
    assert config["responseMimeType"] == "application/json"
    assert sent["json"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_reads_any_of_its_key_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "from-onchain-copilot")
    assert GeminiLLM("m").api_key == "from-onchain-copilot"


def test_gemini_without_key() -> None:
    with pytest.raises(LLMError, match="aistudio"):
        GeminiLLM("m")


def test_gemini_blocked_and_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    cut = httpx.Response(200, json={"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]})
    monkeypatch.setattr(httpx, "post", Recorder(blocked, cut))
    llm = GeminiLLM("m", api_key="k")
    with pytest.raises(LLMError, match="SAFETY"):
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)
    with pytest.raises(LLMError, match="cut off"):
        llm.complete_json(system="s", prompt="p", schema=SCHEMA)


def test_rate_limit_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(httpx.Response(429, headers={"retry-after": "1"}), gemini_ok({"a": 1}))
    monkeypatch.setattr(httpx, "post", recorder)
    assert GeminiLLM("m", api_key="k").complete_json(system="s", prompt="p", schema=SCHEMA) == {
        "a": 1
    }
    assert len(recorder.requests) == 2


def test_persistent_rate_limit_is_a_quota_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", Recorder(*[httpx.Response(429)] * 4))
    with pytest.raises(QuotaError):
        GeminiLLM("m", api_key="k").complete_json(system="s", prompt="p", schema=SCHEMA)


def test_bad_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(httpx.Response(401))
    monkeypatch.setattr(httpx, "post", recorder)
    with pytest.raises(LLMError, match="rejected the API key") as info:
        GroqLLM("m", api_key="k").complete_json(system="s", prompt="p", schema=SCHEMA)
    assert not isinstance(info.value, QuotaError)
    assert len(recorder.requests) == 1


def test_groq_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(groq_ok({"ranking": []}))
    monkeypatch.setattr(httpx, "post", recorder)
    llm = GroqLLM("openai/gpt-oss-120b", api_key="k")
    assert llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"ranking": []}
    sent = recorder.requests[0]
    assert sent["headers"] == {"Authorization": "Bearer k"}
    assert sent["json"]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert sent["json"]["messages"][0] == {"role": "system", "content": "s"}


class Scripted:
    def __init__(self, name: str, *outcomes: Any) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete_json(self, **_kwargs: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_fallback_moves_on_and_stays_moved() -> None:
    gemini = Scripted("gemini:x", QuotaError("429"))
    groq = Scripted("groq:y", {"n": 1}, {"n": 2})
    chain = FallbackLLM([gemini, groq])
    assert chain.complete_json(system="s", prompt="p", schema=SCHEMA) == {"n": 1}
    assert chain.complete_json(system="s", prompt="p", schema=SCHEMA) == {"n": 2}
    assert gemini.calls == 1  # not asked again once out of quota
    assert chain.served_by == {"groq:y"}
    assert chain.name == "gemini:x > groq:y"


def test_fallback_does_not_hide_real_errors() -> None:
    chain = FallbackLLM([Scripted("a", LLMError("bad schema")), Scripted("b", {"ok": 1})])
    with pytest.raises(LLMError, match="bad schema"):
        chain.complete_json(system="s", prompt="p", schema=SCHEMA)


def test_fallback_all_exhausted() -> None:
    chain = FallbackLLM([Scripted("a", QuotaError("a down")), Scripted("b", QuotaError("b down"))])
    with pytest.raises(LLMError, match="every provider failed"):
        chain.complete_json(system="s", prompt="p", schema=SCHEMA)


def test_auto_without_keys_explains_free_options() -> None:
    with pytest.raises(LLMError, match=r"aistudio.*\n.*console\.groq"):
        make_llm("auto")


def test_auto_orders_free_providers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("GEMINI_API_KEY", "m")
    assert make_llm("auto").name.startswith(
        "gemini:gemini-3.8-flash > groq:openai/gpt-oss-120b > anthropic:"
    )


def test_auto_with_one_key_is_that_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert make_llm("auto").name == "groq:openai/gpt-oss-120b"
    with pytest.raises(LLMError, match="explicit --provider"):
        make_llm("auto", "some-model")


def test_groq_transcription(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "g")
    audio = tmp_path / "lesson.m4a"
    audio.write_bytes(b"fake audio")
    body = {
        "language": "italian",
        "segments": [
            {"start": 0.0, "end": 4.2, "text": " Oggi parliamo di Bayes."},
            {"start": 4.2, "end": 5.0, "text": "  "},
        ],
    }
    recorder = Recorder(httpx.Response(200, json=body))
    monkeypatch.setattr(httpx, "post", recorder)
    transcript = load_transcript(audio, language="it")
    assert [s.text for s in transcript.segments] == ["Oggi parliamo di Bayes."]
    assert transcript.language == "italian"
    sent = recorder.requests[0]
    assert sent["url"].endswith("/audio/transcriptions")
    assert sent["data"]["response_format"] == "verbose_json"
    assert sent["data"]["language"] == "it"


def test_groq_transcription_size_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from distillo.providers import groq

    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(groq, "MAX_UPLOAD_BYTES", 4)
    audio = tmp_path / "long.mp3"
    audio.write_bytes(b"0123456789")
    with pytest.raises(TranscriptionError, match="25 MB"):
        load_transcript(audio, transcriber="groq")
