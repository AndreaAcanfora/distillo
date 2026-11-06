# Contributing

Thanks for helping! A few notes to get started.

## Setup

```bash
git clone https://github.com/AndreaAcanfora/distillo && cd distillo
uv sync                    # creates .venv with the dev tools
uv run pytest              # no API key or network needed: tests use a fake model
uv run ruff format . && uv run ruff check . && uv run mypy
```

## Where things live

| Module | What it does |
| --- | --- |
| `transcribe.py` | files → `Transcript` (subtitles, text, Whisper via Groq or faster-whisper) |
| `chunking.py` | splits the transcript into overlapping parts |
| `prompts.py` | prompts and JSON schemas (bump `PROMPT_VERSION` when you change them) |
| `grounding.py` | finds each quote in the transcript, drops what isn't there |
| `pipeline.py` | extract → ground → rank → select |
| `providers/` | Gemini, Groq, Claude, Ollama, and the fallback chain |
| `render.py`, `cli.py` | output formats and the command line |

## Adding a provider

Implement the `LLM` protocol in `providers/base.py` (a `name` and a `complete_json` method that returns JSON
matching the given schema), raise `QuotaError` for rate limits and outages so the fallback chain can move on,
and register it in `providers/registry.py`. Add tests that mock the HTTP layer.

## Pull requests

Keep them focused, add tests, and describe what you checked by hand (which recording, which provider).
