"""Command-line interface."""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .cache import Cache
from .pipeline import Distiller, Options
from .providers import DEFAULT_MODELS, LLMError, make_llm
from .render import to_json, to_markdown
from .sources import SourceError, fetch, is_url
from .transcribe import TranscriptionError, load_transcript, to_srt

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
err = Console(stderr=True)

LANGUAGE_NAMES = {
    "en": "English",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ro": "Romanian",
    "ja": "Japanese",
    "zh": "Chinese",
}


class Density(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Format(str, Enum):
    md = "md"
    json = "json"


class Order(str, Enum):
    time = "time"
    rank = "rank"


class Provider(str, Enum):
    auto = "auto"
    gemini = "gemini"
    groq = "groq"
    anthropic = "anthropic"
    ollama = "ollama"


class Transcriber(str, Enum):
    auto = "auto"
    groq = "groq"
    local = "local"


class Effort(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


def _version(value: bool) -> None:
    if value:
        typer.echo(f"distillo {__version__}")
        raise typer.Exit()


@app.command()
def main(
    source: Annotated[
        str,
        typer.Argument(
            help="Audio or video file, .srt/.vtt/.txt transcript, or a link "
            "(YouTube and the other sites yt-dlp supports).",
            show_default=False,
        ),
    ],
    points: Annotated[
        int | None,
        typer.Option(
            "--points", "-n", min=1, help="Exact number of key points. Overrides --density."
        ),
    ] = None,
    density: Annotated[
        Density,
        typer.Option("--density", "-d", help="low ≈ 6, medium ≈ 12, high ≈ 24 points per hour."),
    ] = Density.medium,
    lang: Annotated[
        str | None,
        typer.Option(
            "--lang", "-l", help="Language of the notes (it, en, Spanish…). Default: as spoken."
        ),
    ] = None,
    context: Annotated[
        str | None,
        typer.Option(
            "--context", "-c", help='What the recording is, e.g. "university statistics lecture".'
        ),
    ] = None,
    fmt: Annotated[Format, typer.Option("--format", "-f", help="Output format.")] = Format.md,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write to a file instead of stdout.")
    ] = None,
    order: Annotated[
        Order, typer.Option(help="List points in the order they were said, or by importance.")
    ] = Order.time,
    quotes: Annotated[bool, typer.Option(help="Include the quote each point comes from.")] = True,
    link: Annotated[
        str | None,
        typer.Option(
            help='Turn timestamps into links: a URL with {seconds}, e.g. "https://youtu.be/ID?t={seconds}".'
        ),
    ] = None,
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            "-p",
            envvar="DISTILLO_PROVIDER",
            help="Model backend. auto = Gemini, then Groq, then Claude, by available keys.",
        ),
    ] = Provider.auto,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            envvar="DISTILLO_MODEL",
            help="Model name for an explicit --provider. Defaults: "
            + ", ".join(f"{m} ({p})" for p, m in DEFAULT_MODELS.items())
            + ".",
        ),
    ] = None,
    effort: Annotated[
        Effort, typer.Option(help="Reasoning effort (anthropic only).")
    ] = Effort.medium,
    transcriber: Annotated[
        Transcriber,
        typer.Option(
            help="Speech to text: groq (free API), local (offline), auto = groq if keyed."
        ),
    ] = Transcriber.auto,
    whisper_model: Annotated[
        str,
        typer.Option(
            help="Local faster-whisper model: tiny, base, small, medium, large-v3, turbo."
        ),
    ] = "small",
    source_lang: Annotated[
        str | None, typer.Option(help="Spoken language code for transcription. Default: detect.")
    ] = None,
    save_transcript: Annotated[
        Path | None, typer.Option(help="Also save the transcript as .srt (reusable as input).")
    ] = None,
    chunk_chars: Annotated[
        int, typer.Option(min=2_000, help="Transcript characters per extraction request.")
    ] = 16_000,
    captions: Annotated[
        bool, typer.Option(help="For links, use the video's captions when available.")
    ] = True,
    cookies_from_browser: Annotated[
        str | None,
        typer.Option(help="For links, use this browser's cookies (firefox, chrome…) if blocked."),
    ] = None,
    cache: Annotated[bool, typer.Option(help="Reuse earlier model responses.")] = True,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = None,
) -> None:
    """Distill a lecture, course or podcast into the key points worth learning."""
    try:
        # Check the model configuration first: no point transcribing an hour of audio
        # only to find out there is no API key.
        llm = make_llm(provider.value, model, effort=effort.value)
        with tempfile.TemporaryDirectory(prefix="distillo-") as workdir:
            path, title = Path(source), None
            if is_url(source):
                with err.status("Downloading…"):
                    fetched = fetch(
                        source,
                        Path(workdir),
                        language=source_lang,
                        captions=captions,
                        cookies_from_browser=cookies_from_browser,
                    )
                path, title = fetched.path, fetched.title
                link = link or fetched.link
                if fetched.captions:
                    err.print("[dim]Using the video's captions.[/dim]")
            with err.status("Reading the recording…"):
                transcript = load_transcript(
                    path,
                    transcriber=transcriber.value,
                    whisper_model=whisper_model,
                    language=source_lang,
                )
            if title:
                transcript.source = title
        if save_transcript:
            save_transcript.write_text(to_srt(transcript), encoding="utf-8")
            err.print(f"[dim]Transcript saved to {save_transcript}[/dim]")

        language = LANGUAGE_NAMES.get(lang.lower(), lang) if lang else None
        options = Options(
            points=points,
            density=density.value,
            language=language,
            context=context,
            order=order.value,
            chunk_chars=chunk_chars,
        )
        with err.status("Distilling…") as status:
            distiller = Distiller(llm, Cache(None, enabled=cache), progress=status.update)
            result, report = distiller.run(transcript, options)
    except (TranscriptionError, LLMError, SourceError) as exc:
        err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    rendered = (
        to_json(result)
        if fmt is Format.json
        else to_markdown(result, language=lang, quotes=quotes, link=link)
    )
    if output:
        output.write_text(rendered, encoding="utf-8")
        err.print(f"[green]✓[/green] {len(result.points)} key points written to {output}")
    else:
        typer.echo(rendered, nl=False)
    if report.rejected:
        err.print(
            f"[dim]{len(report.rejected)} of {report.candidates} extracted points were dropped "
            "because their quote was not found in the transcript.[/dim]"
        )
