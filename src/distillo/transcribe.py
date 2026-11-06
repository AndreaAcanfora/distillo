"""Turn an input file into a :class:`Transcript`.

Subtitle and text files are parsed directly. Audio and video files are transcribed with
Whisper, either on Groq's free API (``GROQ_API_KEY``) or offline with faster-whisper
(``pip install "distillo[local]"``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Segment, Transcript
from .timefmt import parse_clock

SUBTITLE_SUFFIXES = {".srt", ".vtt"}
TEXT_SUFFIXES = {".txt", ".md"}
TRANSCRIPT_SUFFIXES = {".json"}

_CUE_TIMING = re.compile(r"^\s*([\d:.,]+)\s*-->\s*([\d:.,]+)")
_TAG = re.compile(r"<[^>]+>")


class TranscriptionError(RuntimeError):
    pass


def load_transcript(
    path: Path,
    *,
    transcriber: str = "auto",
    whisper_model: str = "small",
    language: str | None = None,
) -> Transcript:
    """Load ``path``. ``transcriber`` is ``groq``, ``local`` or ``auto`` (Groq if keyed)."""
    suffix = path.suffix.lower()
    if not path.is_file():
        raise TranscriptionError(f"file not found: {path}")
    if suffix in SUBTITLE_SUFFIXES:
        transcript = parse_subtitles(path.read_text(encoding="utf-8-sig"))
    elif suffix in TEXT_SUFFIXES:
        transcript = parse_plain_text(path.read_text(encoding="utf-8-sig"))
    elif suffix in TRANSCRIPT_SUFFIXES:
        transcript = Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
    else:
        transcript = transcribe_media(
            path, transcriber=transcriber, model_size=whisper_model, language=language
        )
    if not transcript.segments:
        raise TranscriptionError(f"no speech or text found in {path}")
    transcript.source = transcript.source or path.name
    if language and not transcript.language:
        transcript.language = language
    return transcript


def parse_subtitles(text: str) -> Transcript:
    """Parse SRT or WebVTT. Consecutive duplicate lines (common in auto captions) are dropped."""
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [line for line in block.strip().split("\n") if line.strip()]
        for i, line in enumerate(lines):
            timing = _CUE_TIMING.match(line)
            if not timing:
                continue
            body = " ".join(_TAG.sub("", rest).strip() for rest in lines[i + 1 :]).strip()
            if body and not (segments and segments[-1].text == body):
                segments.append(
                    Segment(
                        start=parse_clock(timing.group(1)),
                        end=parse_clock(timing.group(2)),
                        text=body,
                    )
                )
            break
    return Transcript(segments=segments)


def parse_plain_text(text: str) -> Transcript:
    """A transcript without timing: one segment per paragraph, all at time zero."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return Transcript(segments=[Segment(start=0.0, end=0.0, text=p) for p in paragraphs])


def transcribe_media(
    path: Path, *, transcriber: str = "auto", model_size: str = "small", language: str | None
) -> Transcript:
    from .providers.groq import groq_key

    if transcriber == "auto":
        transcriber = "groq" if groq_key() else "local"
    if transcriber == "groq":
        return transcribe_with_groq(path, language=language)
    if transcriber == "local":
        return transcribe_locally(path, model_size=model_size, language=language)
    raise TranscriptionError(f"unknown transcriber {transcriber!r}; use auto, groq or local")


def transcribe_with_groq(path: Path, *, language: str | None) -> Transcript:
    from .providers import LLMError
    from .providers.groq import groq_transcribe

    try:
        body = groq_transcribe(path, language=language)
    except LLMError as exc:
        raise TranscriptionError(str(exc)) from exc
    segments = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip())
        for s in body.get("segments") or []
        if str(s.get("text", "")).strip()
    ]
    return Transcript(segments=segments, language=body.get("language"))


def transcribe_locally(path: Path, *, model_size: str, language: str | None) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise TranscriptionError(
            "Transcribing audio or video needs either a free Groq key (export GROQ_API_KEY=...,\n"
            "https://console.groq.com/keys) or the offline extra: pip install 'distillo[local]'.\n"
            "You can also pass a subtitle (.srt/.vtt) or text file."
        ) from exc

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    raw_segments, info = model.transcribe(str(path), language=language, vad_filter=True)
    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in raw_segments
        if s.text.strip()
    ]
    return Transcript(segments=segments, language=info.language)


def to_srt(transcript: Transcript) -> str:
    def stamp(seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, rest = divmod(millis, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        secs, millis = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    blocks = [
        f"{i}\n{stamp(s.start)} --> {stamp(s.end)}\n{s.text}\n"
        for i, s in enumerate(transcript.segments, start=1)
    ]
    return "\n".join(blocks)
