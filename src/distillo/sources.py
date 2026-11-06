"""Fetch recordings from links (YouTube and the other sites yt-dlp supports).

Captions are preferred when the video has them in the spoken language: no audio to
download and nothing to transcribe. Otherwise only the audio track is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fetched:
    path: Path
    title: str
    link: str | None  # URL template with {seconds}, for clickable timestamps
    captions: bool


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def fetch(
    url: str,
    workdir: Path,
    *,
    language: str | None = None,
    captions: bool = True,
    cookies_from_browser: str | None = None,
) -> Fetched:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - yt-dlp is a core dependency
        raise SourceError("links need yt-dlp: pip install yt-dlp") from exc

    base: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
    }
    if cookies_from_browser:
        base["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise SourceError(_explain(str(exc))) from exc
    if not isinstance(info, dict):
        raise SourceError(f"nothing to download at {url}")

    title = str(info.get("title") or info.get("id") or url)
    link = link_template(info)
    track = pick_captions(info, language) if captions else None
    try:
        if track is not None:
            key, automatic = track
            options = {
                **base,
                "skip_download": True,
                "writesubtitles": not automatic,
                "writeautomaticsub": automatic,
                "subtitleslangs": [key],
                "subtitlesformat": "vtt",
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.process_ie_result(info, download=True)
            found = sorted(workdir.glob(f"{info['id']}*.vtt"))
            if found:
                return Fetched(found[0], title, link, captions=True)
        options = {**base, "format": "bestaudio[ext=m4a]/bestaudio/best"}
        with yt_dlp.YoutubeDL(options) as ydl:
            result = ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise SourceError(_explain(str(exc))) from exc
    downloads = (result or {}).get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads else None
    if path is None or not path.is_file():
        raise SourceError(f"could not download the audio of {url}")
    return Fetched(path, title, link, captions=False)


def pick_captions(info: dict[str, Any], language: str | None) -> tuple[str, bool] | None:
    """Choose a caption track in the spoken language: ``(key, is_automatic)``.

    Uploaded captions win over automatic ones. Automatic tracks are only used in the
    original language (YouTube marks it ``xx-orig`` or ``xx``), never machine-translated.
    """
    language = (language or info.get("language") or "").split("-")[0].lower()
    if not language:
        return None
    manual = info.get("subtitles") or {}
    for key in manual:
        if key.split("-")[0].lower() == language:
            return key, False
    automatic = info.get("automatic_captions") or {}
    for key in (f"{language}-orig", language):
        if key in automatic:
            return key, True
    return None


def link_template(info: dict[str, Any]) -> str | None:
    video_id = info.get("id")
    extractor = str(info.get("extractor_key") or info.get("extractor") or "").lower()
    if extractor.startswith("youtube") and video_id:
        return f"https://youtu.be/{video_id}?t={{seconds}}"
    if extractor.startswith("vimeo") and info.get("webpage_url"):
        return f"{info['webpage_url']}#t={{seconds}}s"
    return None


def _explain(message: str) -> str:
    message = message.removeprefix("ERROR: ").strip()
    if "not a bot" in message or "Sign in to confirm" in message:
        message += (
            "\nYouTube blocks some networks (servers, VPNs). Try --cookies-from-browser firefox "
            "(or chrome), or download the video yourself and pass the file."
        )
    return message
