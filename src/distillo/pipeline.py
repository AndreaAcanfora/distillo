"""The distillation pipeline: extract per chunk, ground, merge and rank once, then cut.

Ranking is computed once over every grounded candidate, and the requested number of
points is a prefix of that ranking. The 5 points you get with ``--points 5`` are
therefore always the first 5 of the 20 you get with ``--points 20``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from .cache import Cache
from .chunking import Chunk, chunk_transcript
from .grounding import SegmentIndex, ground
from .models import Candidate, Distillation, KeyPoint, Transcript
from .prompts import (
    EXTRACT_PROMPT,
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM,
    PROMPT_VERSION,
    RANK_PROMPT,
    RANK_SCHEMA,
    RANK_SYSTEM,
    context_line,
)
from .providers import LLM, LLMError
from .timefmt import format_clock

Density = Literal["low", "medium", "high"]
Order = Literal["time", "rank"]

# Points per hour of recording for each density level.
DENSITY_PER_HOUR: dict[str, int] = {"low": 6, "medium": 12, "high": 24}
MIN_POINTS = 3
# Untimed text: points per this many characters.
CHARS_PER_HOUR = 50_000


@dataclass
class Options:
    points: int | None = None
    density: Density = "medium"
    language: str | None = None
    context: str | None = None
    order: Order = "time"
    chunk_chars: int = 16_000
    workers: int = 4


@dataclass
class Report:
    chunks: int = 0
    candidates: int = 0
    rejected: list[Candidate] = field(default_factory=list)


Progress = Callable[[str], None]


def target_count(transcript: Transcript, options: Options) -> int:
    if options.points is not None:
        return max(1, options.points)
    per_hour = DENSITY_PER_HOUR[options.density]
    if transcript.has_timestamps:
        hours = transcript.duration / 3600
    else:
        hours = sum(len(s.text) for s in transcript.segments) / CHARS_PER_HOUR
    return max(MIN_POINTS, math.ceil(per_hour * hours))


class Distiller:
    def __init__(self, llm: LLM, cache: Cache | None = None, progress: Progress | None = None):
        self.llm = llm
        self.cache = cache or Cache(None, enabled=False)
        self.progress = progress or (lambda _message: None)

    def run(self, transcript: Transcript, options: Options) -> tuple[Distillation, Report]:
        report = Report()
        language = options.language or "the same language as the transcript"
        chunks = chunk_transcript(transcript, max_chars=options.chunk_chars)
        report.chunks = len(chunks)

        self.progress(f"Extracting key points from {len(chunks)} part(s)")
        with ThreadPoolExecutor(max_workers=max(1, options.workers)) as pool:
            per_chunk = list(
                pool.map(
                    lambda chunk: self._extract(chunk, len(chunks), transcript, language, options),
                    chunks,
                )
            )

        candidates: list[Candidate] = []
        for chunk, extracted in zip(chunks, per_chunk, strict=False):
            grounded, rejected = ground(
                extracted, SegmentIndex(chunk.segments), has_timestamps=transcript.has_timestamps
            )
            candidates.extend(grounded)
            report.rejected.extend(rejected)
        report.candidates = len(candidates) + len(report.rejected)

        self.progress(f"Ranking {len(candidates)} grounded point(s)")
        groups = self._rank(candidates, options) if candidates else []

        wanted = target_count(transcript, options)
        by_id = {c.id: c for c in candidates}
        selected: list[KeyPoint] = []
        for rank, (lead, duplicates) in enumerate(groups[:wanted], start=1):
            best = by_id[lead]
            starts = [s for i in (lead, *duplicates) if (s := by_id[i].start) is not None]
            selected.append(
                KeyPoint(
                    rank=rank,
                    title=best.title,
                    statement=best.statement,
                    kind=best.kind,
                    # The first time the idea came up is the most useful place to jump to.
                    start=min(starts) if starts else None,
                    quote=best.quote,
                    merged_from=list(duplicates),
                )
            )
        if options.order == "time":
            selected.sort(key=lambda p: (p.start if p.start is not None else math.inf, p.rank))

        distillation = Distillation(
            source=transcript.source,
            language=transcript.language,
            duration=transcript.duration,
            model=_served_by(self.llm),
            requested=wanted,
            total_available=len(groups),
            points=selected,
        )
        return distillation, report

    def _extract(
        self, chunk: Chunk, total: int, transcript: Transcript, language: str, options: Options
    ) -> list[Candidate]:
        span = (
            f" ({format_clock(chunk.start)} to {format_clock(chunk.end)})"
            if transcript.has_timestamps
            else ""
        )
        prompt = EXTRACT_PROMPT.format(
            context_line=context_line(options.context),
            index=chunk.index + 1,
            total=total,
            span=span,
            text=chunk.text,
            language=language,
        )
        data = self._call(EXTRACT_SYSTEM, prompt, EXTRACT_SCHEMA, kind="extract")
        candidates: list[Candidate] = []
        for n, raw in enumerate(data.get("points", []), start=1):
            try:
                candidates.append(Candidate(id=f"c{chunk.index + 1}.{n}", **raw))
            except (TypeError, ValidationError):
                continue
        return candidates

    def _rank(self, candidates: list[Candidate], options: Options) -> list[tuple[str, list[str]]]:
        lines = "\n".join(f"{c.id} | {c.importance} | {c.title}: {c.statement}" for c in candidates)
        prompt = RANK_PROMPT.format(context_line=context_line(options.context), candidates=lines)
        data = self._call(RANK_SYSTEM, prompt, RANK_SCHEMA, kind="rank")
        return repair_ranking(data.get("ranking", []), candidates)

    def _call(self, system: str, prompt: str, schema: dict[str, Any], *, kind: str) -> Any:
        key = Cache.key(PROMPT_VERSION, kind, self.llm.name, system, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.llm.complete_json(system=system, prompt=prompt, schema=schema)
        if not isinstance(result, dict):
            raise LLMError(f"expected a JSON object from the {kind} step")
        self.cache.set(key, result)
        return result


def _served_by(llm: LLM) -> str:
    """The models that actually answered (a fallback chain may have used several)."""
    served: set[str] = getattr(llm, "served_by", set())
    return ", ".join(sorted(served)) if served else llm.name


def repair_ranking(
    ranking: list[dict[str, Any]], candidates: list[Candidate]
) -> list[tuple[str, list[str]]]:
    """Make the model's ranking safe to use.

    Unknown ids are ignored, an id used twice keeps its first place, and candidates the
    model forgot are appended after the ranked ones, by extraction importance.
    """
    known = {c.id for c in candidates}
    seen: set[str] = set()
    groups: list[tuple[str, list[str]]] = []
    for item in ranking:
        lead = item.get("id")
        if not isinstance(lead, str) or lead not in known or lead in seen:
            continue
        seen.add(lead)
        duplicates = []
        for dup in item.get("duplicates") or []:
            if isinstance(dup, str) and dup in known and dup not in seen:
                seen.add(dup)
                duplicates.append(dup)
        groups.append((lead, duplicates))
    missing = sorted(
        (c for c in candidates if c.id not in seen),
        key=lambda c: (-c.importance, c.start if c.start is not None else math.inf),
    )
    groups.extend((c.id, []) for c in missing)
    return groups
