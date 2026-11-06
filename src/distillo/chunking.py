"""Split a transcript into windows that fit comfortably in one extraction request."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Segment, Transcript
from .timefmt import format_clock


@dataclass(frozen=True)
class Chunk:
    index: int
    segments: list[Segment]
    text: str

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end


def render_segments(segments: list[Segment], *, timestamps: bool) -> str:
    if not timestamps:
        return "\n\n".join(s.text for s in segments)
    return "\n".join(f"[{format_clock(s.start)}] {s.text}" for s in segments)


def chunk_transcript(
    transcript: Transcript, *, max_chars: int = 16_000, overlap_chars: int = 600
) -> list[Chunk]:
    """Greedy split on segment boundaries.

    Each chunk repeats the last ``overlap_chars`` of the previous one so a point
    that straddles the boundary is still seen whole; duplicates are merged later.
    """
    timestamps = transcript.has_timestamps
    chunks: list[Chunk] = []
    current: list[Segment] = []
    size = 0

    def flush() -> None:
        chunks.append(
            Chunk(len(chunks), list(current), render_segments(current, timestamps=timestamps))
        )

    for segment in transcript.segments:
        length = len(segment.text) + 10
        if current and size + length > max_chars:
            flush()
            carried: list[Segment] = []
            carried_size = 0
            for previous in reversed(current):
                if carried_size + len(previous.text) > overlap_chars:
                    break
                carried.insert(0, previous)
                carried_size += len(previous.text) + 10
            current, size = carried, carried_size
        current.append(segment)
        size += length
    if current:
        flush()
    return chunks
