"""Web demo of distillo (Gradio), ready for Hugging Face Spaces.

Shared keys come from the Space secrets GEMINI_API_KEY and GROQ_API_KEY. Visitors can paste
their own free keys instead, which lifts the demo limits. Keys are only used for the
request in progress: never stored, logged or put in the environment.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import gradio as gr

from distillo.cache import Cache
from distillo.pipeline import Distiller, Options
from distillo.providers import LLMError, free_chain
from distillo.render import to_markdown
from distillo.sources import SourceError, fetch, is_url
from distillo.transcribe import TranscriptionError, load_transcript, to_srt

SHARED_GEMINI = os.environ.get("GEMINI_API_KEY") or None
SHARED_GROQ = os.environ.get("GROQ_API_KEY") or None

# Limits that only apply when the shared (free-tier) keys are used.
RUNS_PER_HOUR = int(os.environ.get("DISTILLO_DEMO_RUNS_PER_HOUR", "4"))
MAX_MINUTES = int(os.environ.get("DISTILLO_DEMO_MAX_MINUTES", "90"))

LANGUAGES = {
    "Come nella registrazione": (None, None),
    "Italiano": ("Italian", "it"),
    "English": ("English", "en"),
    "Español": ("Spanish", "es"),
    "Français": ("French", "fr"),
    "Deutsch": ("German", "de"),
}

HERE = Path(__file__).parent
CACHE = Cache(Path(tempfile.gettempdir()) / "distillo-demo-cache")
OUTPUTS = Path(tempfile.mkdtemp(prefix="distillo-demo-out-"))

_runs: dict[str, deque[float]] = defaultdict(deque)
_runs_lock = threading.Lock()


def _client_id(request: gr.Request | None) -> str:
    if request is None:
        return "local"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate(client: str) -> None:
    now = time.time()
    with _runs_lock:
        runs = _runs[client]
        while runs and now - runs[0] > 3600:
            runs.popleft()
        if len(runs) >= RUNS_PER_HOUR:
            wait = int((3600 - (now - runs[0])) / 60) + 1
            raise gr.Error(
                f"Hai usato le {RUNS_PER_HOUR} prove dell'ultima ora. Riprova tra {wait} minuti, "
                "oppure inserisci le tue chiavi gratuite in “Usa le tue chiavi”."
            )
        runs.append(now)


def _duration_minutes(path: Path) -> float | None:
    try:
        import av

        with av.open(str(path)) as container:
            return container.duration / 1_000_000 / 60 if container.duration else None
    except Exception:
        return None


def distill(
    upload: str | None,
    url: str | None,
    points: float,
    language: str,
    context: str | None,
    gemini_key: str | None,
    groq_key: str | None,
    request: gr.Request,
    progress: Any = gr.Progress(),  # noqa: B008 - how Gradio injects progress
) -> tuple[str, list[str]]:
    url = (url or "").strip()
    if not upload and not url:
        raise gr.Error("Carica un file oppure incolla un link.")
    if url and not is_url(url):
        raise gr.Error("Il link deve iniziare con http:// o https://")

    gemini_key, groq_key = (gemini_key or "").strip(), (groq_key or "").strip()
    own_keys = bool(gemini_key or groq_key)
    gemini = gemini_key or SHARED_GEMINI
    groq = groq_key or SHARED_GROQ
    if not (gemini or groq):
        raise gr.Error(
            "La demo non ha chiavi configurate: inserisci le tue in “Usa le tue chiavi”."
        )
    if not own_keys:
        _check_rate(_client_id(request))

    output_language, label_language = LANGUAGES.get(language, (None, None))
    try:
        with tempfile.TemporaryDirectory(prefix="distillo-demo-") as workdir:
            path: Path
            title = None
            link = None
            if url:
                progress(0.05, desc="Scarico la registrazione…")
                fetched = fetch(url, Path(workdir))
                path, title, link = fetched.path, fetched.title, fetched.link
            else:
                path = Path(str(upload))

            minutes = _duration_minutes(path)
            if minutes and minutes > MAX_MINUTES and not own_keys:
                raise gr.Error(
                    f"La registrazione dura {minutes:.0f} minuti: con le chiavi condivise il limite "
                    f"è {MAX_MINUTES}. Usa le tue chiavi gratuite per registrazioni più lunghe."
                )
            if path.suffix.lower() not in {".srt", ".vtt", ".txt"} and not groq:
                raise gr.Error("Per trascrivere audio e video serve una chiave Groq.")

            progress(0.2, desc="Trascrivo (Whisper)…")
            transcript = load_transcript(
                path, transcriber="groq" if groq else "auto", groq_api_key=groq
            )
            if title:
                transcript.source = title
            elif upload:
                transcript.source = Path(str(upload)).name

            progress(0.5, desc="Estraggo e ordino le nozioni…")
            llm = free_chain(gemini_api_key=gemini, groq_api_key=groq)
            result, report = Distiller(llm, CACHE).run(
                transcript,
                Options(points=int(points), language=output_language, context=context or None),
            )
    except (SourceError, TranscriptionError, LLMError) as exc:
        raise gr.Error(str(exc)) from exc

    # The page is in Italian, so headings and labels default to Italian.
    markdown = to_markdown(
        result, language=label_language or transcript.language or "it", link=link
    )
    if report.rejected:
        markdown += (
            f"\n_{len(report.rejected)} nozioni proposte dall'AI sono state scartate perché la "
            "citazione non si trova nella registrazione._\n"
        )
    stem = f"{int(time.time())}-{Path(result.source).stem[:40] or 'lezione'}"
    notes = OUTPUTS / f"{stem}.md"
    notes.write_text(markdown, encoding="utf-8")
    files = [str(notes)]
    if transcript.has_timestamps:
        subtitles = OUTPUTS / f"{stem}.srt"
        subtitles.write_text(to_srt(transcript), encoding="utf-8")
        files.append(str(subtitles))
    progress(1.0, desc="Fatto")
    return markdown, files


INTRO = """
# distillo
**Da una lezione registrata alle poche nozioni che contano, ognuna con il minuto in cui è stata detta.**

Carica l'audio o il video di una lezione (oppure i sottotitoli), scegli quante nozioni vuoi e ricevi una scheda di
ripasso. Ogni nozione è verificata: l'AI deve citare la frase esatta e distillo la cerca nella registrazione; se non
c'è, la nozione viene scartata. [Codice open source su GitHub](https://github.com/AndreaAcanfora/distillo).
"""

PRIVACY = """
**Privacy.** Con le chiavi condivise di questa demo l'audio e il testo passano dai piani gratuiti di Groq e Google,
che possono usarli per migliorare i loro servizi. Non caricare registrazioni riservate: per quelle usa distillo
[sul tuo computer](https://github.com/AndreaAcanfora/distillo#install), anche completamente offline.
I file vengono cancellati alla fine dell'elaborazione.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="distillo: le nozioni chiave di una lezione") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="Lezione: audio, video o sottotitoli (.srt, .vtt, .txt)",
                    file_types=["audio", "video", ".srt", ".vtt", ".txt"],
                    type="filepath",
                )
                url = gr.Textbox(
                    label="…oppure un link",
                    placeholder="https://archive.org/…  ·  link diretto a un file audio/video",
                    info="YouTube spesso blocca i server: se non funziona, scarica il video e caricalo.",
                )
                points = gr.Slider(3, 30, value=8, step=1, label="Quante nozioni")
                language = gr.Dropdown(
                    list(LANGUAGES), value="Come nella registrazione", label="Lingua degli appunti"
                )
                context = gr.Textbox(
                    label="Di cosa si tratta (facoltativo)",
                    placeholder="es. lezione di statistica, secondo anno di economia",
                )
                with gr.Accordion("Usa le tue chiavi (gratuite, nessun limite)", open=False):
                    gr.Markdown(
                        "Chiavi gratuite: [Gemini](https://aistudio.google.com/apikey) · "
                        "[Groq](https://console.groq.com/keys). Vengono usate solo per questa "
                        "richiesta e non vengono salvate."
                    )
                    gemini_key = gr.Textbox(label="GEMINI_API_KEY", type="password")
                    groq_key = gr.Textbox(label="GROQ_API_KEY", type="password")
                run = gr.Button("Estrai le nozioni", variant="primary")
                gr.Examples(
                    examples=[[str(HERE / "examples" / "lezione-statistica.srt"), "", 4]],
                    inputs=[upload, url, points],
                    label="Prova con una lezione di esempio",
                )
            with gr.Column(scale=1):
                notes = gr.Markdown(label="Nozioni chiave", value="_Le nozioni appariranno qui._")
                downloads = gr.File(
                    label="Scarica appunti (.md) e trascrizione (.srt)", file_count="multiple"
                )
        gr.Markdown(PRIVACY)
        run.click(
            distill,
            inputs=[upload, url, points, language, context, gemini_key, groq_key],
            outputs=[notes, downloads],
            concurrency_limit=2,
        )
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860"))
    )
