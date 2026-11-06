# Changelog

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
- Markdown and JSON output, clickable timestamps with `--link`, on-disk response cache.
