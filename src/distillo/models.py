"""Data model shared by every stage of the pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A stretch of speech with its position in the recording, in seconds."""

    start: float
    end: float
    text: str


class Transcript(BaseModel):
    segments: list[Segment]
    language: str | None = None
    source: str = ""

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def has_timestamps(self) -> bool:
        return self.duration > 0


class Kind(str, Enum):
    DEFINITION = "definition"
    CONCEPT = "concept"
    FORMULA = "formula"
    FACT = "fact"
    PROCEDURE = "procedure"
    EXAMPLE = "example"
    WARNING = "warning"


class Candidate(BaseModel):
    """A key point proposed by the model for one chunk, before ranking."""

    id: str
    title: str
    statement: str
    kind: Kind
    importance: int = Field(ge=1, le=5)
    quote: str
    # Filled in by grounding: where the quote was found in the transcript.
    start: float | None = None
    match_score: float = 0.0


class KeyPoint(BaseModel):
    rank: int
    title: str
    statement: str
    kind: Kind
    start: float | None
    quote: str
    merged_from: list[str] = Field(default_factory=list)


class Distillation(BaseModel):
    source: str
    language: str | None
    duration: float
    model: str
    requested: int
    total_available: int
    points: list[KeyPoint]
