# Changelog

## 0.2.0 (unreleased)

- Links as input: YouTube and the other sites yt-dlp supports. Captions in the spoken language are used when
  available (uploaded before automatic, never machine-translated), otherwise only the audio is downloaded.
  YouTube and Vimeo timestamps become links; the video title becomes the heading.
- `--no-captions` and `--cookies-from-browser` for links.
- Recordings over Groq's 25 MB limit are compressed to mono Opus (about 11 MB per hour) and split into one-hour
  parts with their timestamps joined back.
- Rolling captions (YouTube's automatic ones) are no longer read twice.
- PyAV and yt-dlp are now regular dependencies; the `local` extra only adds faster-whisper.

## 0.1.0 (unreleased)

First version.

- Audio and video transcription with Whisper: Groq's free API or offline with faster-whisper (`distillo[local]`).
- `.srt`, `.vtt` and plain text transcripts as input.
- Key point extraction per part of the recording, with a verbatim quote for each point.
- Grounding: quotes are looked up in the transcript; points whose quote is missing are dropped, and timestamps come
  from where the quote was found.
- One ranking pass that merges duplicates, so `--points N` is always a prefix of a longer list.
- `--points` and `--density low|medium|high`; `--order time|rank`; `--lang` for the language of the notes.
- Providers: Gemini and Groq (free tiers) with automatic fallback, Claude, and Ollama for fully local use.
- Free-tier aware: smaller requests and one at a time for Groq (8k tokens/minute), quick fallback when a
  provider is overloaded, tolerant parsing of small schema slips.
- Markdown and JSON output, clickable timestamps with `--link`, on-disk response cache.
