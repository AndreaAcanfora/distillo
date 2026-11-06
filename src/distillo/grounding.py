"""Check every point against the transcript and find where it was said.

The model is asked for a verbatim quote, not a timestamp. We look the quote up in the
transcript with fuzzy matching: if it is there, the point is grounded and its timestamp
is the start of the segment where the quote begins; if not, the point is dropped as a
probable invention.
"""

from __future__ import annotations

import bisect
import re
import unicodedata

from rapidfuzz import fuzz

from .models import Candidate, Segment

MIN_SCORE = 80.0
_NON_WORD = re.compile(r"[^\w]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", " ", text)
    return _NON_WORD.sub(" ", text).strip()


class SegmentIndex:
    """Normalized text of a run of segments, with a map from offsets back to segments."""

    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        parts: list[str] = []
        self.offsets: list[int] = []
        position = 0
        for segment in segments:
            self.offsets.append(position)
            normalized = normalize(segment.text)
            parts.append(normalized)
            position += len(normalized) + 1
        self.text = " ".join(parts)

    def locate(self, quote: str) -> tuple[float, Segment | None]:
        needle = normalize(quote)
        if not needle or not self.text:
            return 0.0, None
        if len(needle) > len(self.text):
            needle = needle[: len(self.text)]
        alignment = fuzz.partial_ratio_alignment(needle, self.text)
        if alignment is None:
            return 0.0, None
        index = bisect.bisect_right(self.offsets, alignment.dest_start) - 1
        return alignment.score, self.segments[max(index, 0)]


def ground(
    candidates: list[Candidate],
    index: SegmentIndex,
    *,
    has_timestamps: bool,
    min_score: float = MIN_SCORE,
) -> tuple[list[Candidate], list[Candidate]]:
    """Return ``(grounded, rejected)``; grounded candidates get ``start`` and ``match_score``."""
    grounded: list[Candidate] = []
    rejected: list[Candidate] = []
    for candidate in candidates:
        score, segment = index.locate(candidate.quote)
        candidate.match_score = round(score, 1)
        if segment is None or score < min_score:
            rejected.append(candidate)
            continue
        candidate.start = segment.start if has_timestamps else None
        grounded.append(candidate)
    return grounded, rejected
