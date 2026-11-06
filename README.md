<div align="center">

# distillo

**An hour of lecture in, the handful of things worth learning out. Each one linked to the second it was said.**

distillo turns lectures, online courses, podcasts and meeting recordings into a short, ranked list of
key points: definitions, formulas, facts, procedures and warnings. You choose how many: 5 for a quick
review, 30 for a full revision sheet. It runs **free** on the Gemini and Groq free tiers, or **fully offline**.

[![CI](https://github.com/AndreaAcanfora/distillo/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreaAcanfora/distillo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Donate with PayPal](https://img.shields.io/badge/donate-PayPal-00457C?logo=paypal)](https://www.paypal.com/paypalme/AndreaAca)

</div>

---

```console
$ distillo lezione-statistica.m4a --points 4 --lang it
```

```markdown
# lezione-statistica.m4a

**4 nozioni chiave** (su 17) · 48:12 · `gemini:gemini-3.6-flash`

1. **Probabilità condizionata** `03:41` · _definizione_
   P(A|B) = P(A ∩ B) / P(B), definita solo quando P(B) > 0.

   > “la probabilità condizionata di A dato B è la probabilità di A e B diviso la probabilità di B”

2. **Teorema di Bayes** `12:34` · _formula_
   P(A|B) = P(B|A) · P(A) / P(B). Il docente ha detto che sarà all'esame.

   > “ora il teorema di Bayes, che vi chiederò sicuramente all'esame”
...
```

_Example output, shortened._

## Why another summarizer?

Most AI summarizers give you a paragraph you can't check. distillo is built for studying:

- **You choose the density.** `--points 5` or `--density low|medium|high` (about 6, 12 or 24 points per hour).
  Every candidate point is ranked once, so the 5 points are always the top 5 of the 20. Changing the number reuses
  the cached ranking: it is instant and free.
- **Every point is grounded.** The model must quote the transcript word for word. distillo looks the quote up with
  fuzzy matching. If it isn't there, the point is dropped as a probable hallucination. If it is there, the quote
  tells us *when* the point was made, so the timestamp comes from the recording, not from the model's guess.
- **Duplicates are merged.** Teachers repeat the important things. Repetitions become one point, linked to the
  first time it was said.
- **Free, or private.** Gemini and Groq free tiers, with automatic fallback from one to the other when you hit
  a quota. Or `--provider ollama --transcriber local` to keep a recorded lesson on your machine.
- **Notes in your language.** A lecture in English, notes in Italian: `--lang it`.

## Install

```bash
pipx install git+https://github.com/AndreaAcanfora/distillo              # or: uv tool install git+https://...
pipx install "distillo[local] @ git+https://github.com/AndreaAcanfora/distillo"   # + offline transcription
```

A PyPI release (`pipx install distillo`) is planned for v1.0. From source:

```bash
git clone https://github.com/AndreaAcanfora/distillo && cd distillo
uv sync && uv run distillo --help
```

## Set up a free model (two minutes)

distillo needs a language model to read the transcript. The free tiers are more than enough for lectures:

| Provider | Free key | Used for |
| --- | --- | --- |
| **Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | key points (first choice) |
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | key points (fallback) **and** speech-to-text with Whisper large-v3-turbo |

```bash
export GEMINI_API_KEY=...   # also read: GOOGLE_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY
export GROQ_API_KEY=...
```

With both keys, `--provider auto` (the default) uses Gemini (`gemini-3.6-flash`) and moves to Groq
(`openai/gpt-oss-120b`) when Gemini is overloaded, rate limited or out of quota. Groq's free tier allows 8,000 tokens
per minute, so distillo sends it smaller pieces, one at a time: a 40-minute lecture takes about two minutes. With `ANTHROPIC_API_KEY` set, Claude is added at the end of the chain. You can also pick one explicitly:

```bash
distillo talk.mp3 --provider anthropic                       # claude-opus-5-5
distillo talk.mp3 --provider anthropic -m claude-haiku-5-5   # cheaper
distillo talk.mp3 --provider ollama -m qwen3:8b              # local, nothing leaves your machine
```

> Free tiers may use your prompts to improve the provider's products. For recordings that must stay private,
> use `--provider ollama --transcriber local`.

## Usage

```bash
distillo lesson.mp4                                  # ~12 points per hour, in the spoken language
distillo lesson.mp4 -n 5                             # exactly 5 points
distillo lesson.mp4 --density high --lang en         # dense notes, in English
distillo lesson.mp4 -c "2nd year organic chemistry"  # context helps the model
distillo lesson.mp4 -o notes.md --save-transcript lesson.srt
distillo lesson.srt -n 20 --order rank               # reuse the transcript, most important first
distillo captions.vtt --link "https://youtu.be/VIDEO_ID?t={seconds}"   # clickable timestamps
distillo lesson.mp4 -f json                          # for scripts and other apps
```

Inputs: any audio or video file, `.srt` / `.vtt` subtitles, plain `.txt` transcripts (no timestamps).

### Speech to text

| `--transcriber` | How | Notes |
| --- | --- | --- |
| `auto` (default) | Groq if `GROQ_API_KEY` is set, otherwise local | |
| `groq` | Whisper large-v3-turbo on Groq | fast, free; files up to 25 MB (about 1h45 of mono MP3 at 32 kbps) |
| `local` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your CPU/GPU | needs `distillo[local]`; `--whisper-model tiny…large-v3` |

Long recording over 25 MB? Compress it first: `ffmpeg -i lesson.m4a -ac 1 -b:a 32k lesson.mp3`.

## How it works

```text
audio/video ──Whisper──▶ timed transcript ──split──▶ parts of ~15 min
                                                         │ extract: every candidate point + verbatim quote
                                                         ▼
                                    grounding: find each quote in the transcript (fuzzy match)
                                       ├── not found → dropped (probable invention)
                                       └── found → timestamp of the matching segment
                                                         │ one ranking call over all points:
                                                         ▼ merge duplicates, order by importance
                                              top N ──▶ Markdown or JSON
```

Model responses are cached in `~/.cache/distillo` (override with `DISTILLO_CACHE_DIR`), so trying a different number
of points or another output format costs nothing. `--no-cache` forces fresh answers.

## Roadmap

- [x] v0.1: files and subtitles, Whisper (Groq or local), grounded points, density, merge, Markdown/JSON, free providers with fallback
- [ ] v0.2: YouTube links (captions when available), chapters for long recordings, automatic compression for Groq
- [ ] v0.3: Anki deck export and self-test questions
- [ ] v0.4: `--merge` several lectures into one course revision sheet
- [ ] v1.0: PyPI release, web demo on Hugging Face Spaces

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md): the test suite needs no API key.

## Support

distillo is free and open source. If it saves you hours of re-watching lectures, you can
[buy me a coffee on PayPal](https://www.paypal.com/paypalme/AndreaAca). Thank you!

## License

[MIT](LICENSE) © Andrea Acanfora
