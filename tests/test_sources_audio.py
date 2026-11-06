from __future__ import annotations

import itertools
import math
import struct
import sys
import types
import wave
from pathlib import Path
from typing import Any, ClassVar

import pytest

from distillo import cli
from distillo.audio import AudioError, compress, prepare_for_upload
from distillo.sources import SourceError, fetch, is_url, link_template, pick_captions

from .conftest import FakeLLM


def _wav(path: Path, seconds: float, rate: int = 8000) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(seconds * rate))
        )
        out.writeframes(frames)
    return path


def test_small_files_are_uploaded_as_they_are(tmp_path: Path) -> None:
    source = _wav(tmp_path / "short.wav", 0.5)
    assert prepare_for_upload(source, tmp_path / "w", max_bytes=10**6)[0].path == source


def test_compress_splits_with_offsets(tmp_path: Path) -> None:
    source = _wav(tmp_path / "talk.wav", 3.2)
    parts = compress(source, tmp_path / "w", part_seconds=1.0)
    assert len(parts) in (3, 4)
    assert parts[0].offset == 0.0
    assert all(
        b.offset - a.offset == pytest.approx(1.0, abs=0.15) for a, b in itertools.pairwise(parts)
    )
    assert all(p.path.suffix == ".ogg" and p.path.stat().st_size > 0 for p in parts)


def test_compress_shrinks_large_files(tmp_path: Path) -> None:
    source = _wav(tmp_path / "talk.wav", 4.0, rate=48_000)
    parts = prepare_for_upload(source, tmp_path / "w", max_bytes=source.stat().st_size - 1)
    assert sum(p.path.stat().st_size for p in parts) < source.stat().st_size / 5


def test_compress_rejects_non_audio(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(AudioError, match="cannot read"):
        compress(junk, tmp_path / "w")


def test_is_url() -> None:
    assert is_url("https://youtu.be/abc")
    assert not is_url("lesson.mp3")


@pytest.mark.parametrize(
    ("info", "language", "expected"),
    [
        ({"subtitles": {"it": []}, "automatic_captions": {"it-orig": []}}, "it", ("it", False)),
        (
            {"automatic_captions": {"it-orig": [], "en": []}, "language": "it"},
            None,
            ("it-orig", True),
        ),
        ({"automatic_captions": {"en": [], "it": []}}, "it-IT", ("it", True)),
        ({"automatic_captions": {"it": []}}, None, None),  # unknown spoken language
        ({"subtitles": {"en": []}}, "it", None),  # never a translation
    ],
)
def test_pick_captions(info: dict[str, Any], language: str | None, expected: Any) -> None:
    assert pick_captions(info, language) == expected


def test_link_template() -> None:
    assert (
        link_template({"id": "abc", "extractor_key": "Youtube"})
        == "https://youtu.be/abc?t={seconds}"
    )
    vimeo = {"id": "1", "extractor_key": "Vimeo", "webpage_url": "https://vimeo.com/1"}
    assert link_template(vimeo) == "https://vimeo.com/1#t={seconds}s"
    assert link_template({"id": "x", "extractor_key": "Generic"}) is None


class FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL that writes files instead of downloading."""

    info: ClassVar[dict[str, Any]] = {}
    error: ClassVar[str | None] = None
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def __enter__(self) -> FakeYDL:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, Any]:
        if self.error:
            raise sys.modules["yt_dlp"].utils.DownloadError(self.error)
        return self.info

    def process_ie_result(self, info: dict[str, Any], download: bool) -> dict[str, Any]:
        FakeYDL.calls.append(self.options)
        folder = Path(self.options["outtmpl"]).parent
        if self.options.get("skip_download"):
            key = self.options["subtitleslangs"][0]
            (folder / f"{info['id']}.{key}.vtt").write_text(
                "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nciao a tutti\n", encoding="utf-8"
            )
            return info
        path = folder / f"{info['id']}.m4a"
        path.write_bytes(b"audio")
        return {"requested_downloads": [{"filepath": str(path)}]}


@pytest.fixture
def fake_ytdlp(monkeypatch: pytest.MonkeyPatch) -> type[FakeYDL]:
    class DownloadError(Exception):
        pass

    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYDL  # type: ignore[attr-defined]
    module.utils = types.SimpleNamespace(DownloadError=DownloadError)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    FakeYDL.info = {
        "id": "vid1",
        "title": "Lezione 3",
        "extractor_key": "Youtube",
        "language": "it",
    }
    FakeYDL.error = None
    FakeYDL.calls = []
    return FakeYDL


def test_fetch_prefers_captions(fake_ytdlp: type[FakeYDL], tmp_path: Path) -> None:
    fake_ytdlp.info["automatic_captions"] = {"it-orig": []}
    fetched = fetch("https://youtu.be/vid1", tmp_path)
    assert fetched.captions
    assert fetched.path.name == "vid1.it-orig.vtt"
    assert fetched.title == "Lezione 3"
    assert fetched.link == "https://youtu.be/vid1?t={seconds}"
    assert fake_ytdlp.calls[0]["writeautomaticsub"] is True


def test_fetch_downloads_audio_without_captions(fake_ytdlp: type[FakeYDL], tmp_path: Path) -> None:
    fetched = fetch(
        "https://youtu.be/vid1", tmp_path, captions=False, cookies_from_browser="firefox"
    )
    assert not fetched.captions
    assert fetched.path.name == "vid1.m4a"
    assert fake_ytdlp.calls[0]["format"].startswith("bestaudio")
    assert fake_ytdlp.calls[0]["cookiesfrombrowser"] == ("firefox",)


def test_fetch_explains_bot_checks(fake_ytdlp: type[FakeYDL], tmp_path: Path) -> None:
    fake_ytdlp.error = "ERROR: [youtube] vid1: Sign in to confirm you're not a bot."
    with pytest.raises(SourceError, match="cookies-from-browser"):
        fetch("https://youtu.be/vid1", tmp_path)


def test_cli_with_a_link(
    fake_ytdlp: type[FakeYDL], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    fake_ytdlp.info["subtitles"] = {"it": []}
    fake_ytdlp.info["language"] = "it"
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setenv("DISTILLO_CACHE_DIR", str(tmp_path / "cache"))
    result = CliRunner().invoke(cli.app, ["https://youtu.be/vid1", "-n", "1"])
    assert result.exit_code == 0, result.output
    assert "# Lezione 3\n" in result.output
    assert "Using the video's captions." in result.output
