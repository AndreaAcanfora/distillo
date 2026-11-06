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
    llm = GeminiLLM("gemini-3.6-flash", api_key="k")
    assert llm.complete_json(system="sys", prompt="p", schema=SCHEMA) == {"points": []}
    sent = recorder.requests[0]
    assert sent["url"].endswith("/models/gemini-3.6-flash:generateContent")
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
        "gemini:gemini-3.6-flash > groq:openai/gpt-oss-120b > anthropic:"
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


def test_large_recordings_are_split_and_timestamps_shifted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from distillo import audio

    monkeypatch.setenv("GROQ_API_KEY", "g")
    big = tmp_path / "long.mp3"
    big.write_bytes(b"x")
    first, second = tmp_path / "p1.ogg", tmp_path / "p2.ogg"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    monkeypatch.setattr(
        audio,
        "prepare_for_upload",
        lambda *a, **k: [audio.Part(first, 0.0), audio.Part(second, 3600.0)],
    )
    segment = {"start": 1.0, "end": 2.0, "text": "ciao"}
    recorder = Recorder(
        httpx.Response(200, json={"language": "italian", "segments": [segment]}),
        httpx.Response(200, json={"language": "italian", "segments": [segment]}),
    )
    monkeypatch.setattr(httpx, "post", recorder)
    transcript = load_transcript(big, transcriber="groq")
    assert [s.start for s in transcript.segments] == [1.0, 3601.0]
    assert len(recorder.requests) == 2


def test_unreadable_large_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from distillo.providers import groq

    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(groq, "MAX_UPLOAD_BYTES", 4)
    audio_file = tmp_path / "long.mp3"
    audio_file.write_bytes(b"0123456789")
    with pytest.raises(TranscriptionError, match="cannot read the audio"):
        load_transcript(audio_file, transcriber="groq")


def test_auto_retries_little_before_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "m")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    chain = make_llm("auto")
    assert isinstance(chain, FallbackLLM)
    gemini, groq = chain.providers
    assert isinstance(gemini, GeminiLLM)
    assert isinstance(groq, GroqLLM)
    assert (gemini.retries, groq.retries) == (1, 3)

    recorder = Recorder(httpx.Response(503), httpx.Response(503), groq_ok({"ok": 1}))
    monkeypatch.setattr(httpx, "post", recorder)
    assert chain.complete_json(system="s", prompt="p", schema=SCHEMA) == {"ok": 1}
    assert [r["url"].split("/")[2] for r in recorder.requests] == [
        "generativelanguage.googleapis.com",
        "generativelanguage.googleapis.com",
        "api.groq.com",
    ]


def test_request_too_large_moves_on_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(httpx.Response(413, text="tokens per minute"))
    monkeypatch.setattr(httpx, "post", recorder)
    with pytest.raises(QuotaError, match="too large"):
        GroqLLM("m", api_key="k").complete_json(system="s", prompt="p", schema=SCHEMA)
    assert len(recorder.requests) == 1


def test_groq_caps_output_for_the_free_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(groq_ok({}))
    monkeypatch.setattr(httpx, "post", recorder)
    GroqLLM("m", api_key="k").complete_json(system="s", prompt="p", schema=SCHEMA)
    assert recorder.requests[0]["json"]["max_completion_tokens"] == GroqLLM.max_output_tokens


def test_chain_uses_the_tightest_limits() -> None:
    chain = FallbackLLM([Scripted("a"), GroqLLM("m", api_key="k")])
    assert chain.max_chunk_chars == GroqLLM.max_chunk_chars
    assert chain.max_parallel == 1
    assert FallbackLLM([Scripted("a")]).max_chunk_chars is None


def test_groq_schema_has_no_enums(monkeypatch: pytest.MonkeyPatch) -> None:
    from distillo.prompts import EXTRACT_SCHEMA

    recorder = Recorder(groq_ok({"points": []}))
    monkeypatch.setattr(httpx, "post", recorder)
    GroqLLM("openai/gpt-oss-120b", api_key="k").complete_json(
        system="s", prompt="p", schema=EXTRACT_SCHEMA
    )
    sent = recorder.requests[0]["json"]
    assert "enum" not in json.dumps(sent["response_format"])
    assert sent["reasoning_effort"] == "low"
    assert "enum" in json.dumps(EXTRACT_SCHEMA)  # the shared schema is untouched
