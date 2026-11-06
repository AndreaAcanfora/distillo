from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from distillo.models import Transcript
from distillo.timefmt import format_clock, parse_clock
from distillo.transcribe import (
    TranscriptionError,
    load_transcript,
    parse_plain_text,
    parse_subtitles,
    to_srt,
)


def test_srt_fixture(lecture: Transcript) -> None:
    assert len(lecture.segments) == 9
    assert lecture.segments[4].start == 70.0
    assert lecture.duration == 160.0
    assert lecture.source == "lezione-statistica.srt"
    assert lecture.has_timestamps


def test_vtt_strips_tags_and_repeated_auto_captions() -> None:
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.500 align:start
<c>Hello</c> <00:00:02.000><c>world</c>

00:00:03.500 --> 00:00:04.000
Hello world

intro
00:00:04.000 --> 00:01:05.250
Second line
"""
    transcript = parse_subtitles(vtt)
    assert [s.text for s in transcript.segments] == ["Hello world", "Second line"]
    assert transcript.segments[1].end == 65.25


def test_plain_text_has_no_timestamps() -> None:
    transcript = parse_plain_text("First paragraph.\n\nSecond\nparagraph.\n")
    assert len(transcript.segments) == 2
    assert not transcript.has_timestamps


def test_srt_round_trip(lecture: Transcript) -> None:
    again = parse_subtitles(to_srt(lecture))
    assert again.segments == lecture.segments


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="not found"):
        load_transcript(tmp_path / "nope.mp3")


def test_media_without_local_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "lesson.mp3"
    audio.write_bytes(b"\0")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "faster_whisper":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TranscriptionError, match=r"distillo\[local\]"):
        load_transcript(audio)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("01:10", 70.0), ("1:02:03", 3723.0), ("00:00:01,500", 1.5), ("00:01.25", 1.25)],
)
def test_parse_clock(text: str, seconds: float) -> None:
    assert parse_clock(text) == seconds


def test_format_clock() -> None:
    assert format_clock(70.9) == "01:10"
    assert format_clock(3723) == "1:02:03"
    with pytest.raises(ValueError, match="timestamp"):
        parse_clock("soon")


def test_rolling_youtube_captions_are_not_repeated() -> None:
    blank = " "  # YouTube puts a whitespace-only line in its rolling cues
    vtt = "\n".join(
        [
            "WEBVTT",
            "Kind: captions",
            "Language: it",
            "",
            "00:00:00.160 --> 00:00:02.990 align:start position:0%",
            blank,
            "oggi<00:00:00.480><c> parliamo</c><00:00:00.800><c> di</c><00:00:01.000><c> Bayes</c>",
            "",
            "00:00:02.990 --> 00:00:03.000 align:start position:0%",
            "oggi parliamo di Bayes",
            blank,
            "",
            "00:00:03.000 --> 00:00:05.270 align:start position:0%",
            "oggi parliamo di Bayes",
            "e<00:00:03.360><c> del</c><00:00:03.600><c> teorema</c>",
            "",
            "00:00:05.270 --> 00:00:05.280 align:start position:0%",
            "e del teorema",
            blank,
        ]
    )
    transcript = parse_subtitles(vtt)
    assert [s.text for s in transcript.segments] == ["oggi parliamo di Bayes", "e del teorema"]
    assert transcript.segments[1].start == 3.0
