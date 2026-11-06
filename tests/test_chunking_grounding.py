from __future__ import annotations

import itertools

from distillo.chunking import chunk_transcript
from distillo.grounding import SegmentIndex, ground, normalize
from distillo.models import Candidate, Kind, Transcript


def _candidate(quote: str, cid: str = "c1.1") -> Candidate:
    return Candidate(id=cid, title="t", statement="s", kind=Kind.FACT, importance=3, quote=quote)


def test_single_chunk_has_timestamps(lecture: Transcript) -> None:
    chunks = chunk_transcript(lecture)
    assert len(chunks) == 1
    assert chunks[0].text.splitlines()[4].startswith("[01:10] Ora il teorema di Bayes")


def test_small_chunks_cover_everything_with_overlap(lecture: Transcript) -> None:
    chunks = chunk_transcript(lecture, max_chars=400, overlap_chars=200)
    assert len(chunks) > 2
    covered = [s for chunk in chunks for s in chunk.segments]
    assert all(s in covered for s in lecture.segments)
    for previous, current in itertools.pairwise(chunks):
        assert current.segments[0].start <= previous.segments[-1].start


def test_normalize_ignores_case_punctuation_and_markers() -> None:
    assert normalize("[01:10] Il Teorema, di BAYES!") == "il teorema di bayes"


def test_grounding_finds_the_segment(lecture: Transcript) -> None:
    index = SegmentIndex(lecture.segments)
    quote = "Attenzione all'errore più comune: confondere la probabilità di A dato B"
    grounded, rejected = ground([_candidate(quote)], index, has_timestamps=True)
    assert not rejected
    assert grounded[0].start == 84.5
    assert grounded[0].match_score >= 95


def test_grounding_tolerates_small_differences(lecture: Transcript) -> None:
    index = SegmentIndex(lecture.segments)
    quote = "due eventi si dicono indipendenti quando la probabilita della loro intersezione"
    grounded, _ = ground([_candidate(quote)], index, has_timestamps=True)
    assert grounded
    assert grounded[0].start == 125.0


def test_grounding_rejects_invented_quotes(lecture: Transcript) -> None:
    index = SegmentIndex(lecture.segments)
    quote = "la varianza di una costante è sempre uguale a quarantadue unità"
    grounded, rejected = ground([_candidate(quote)], index, has_timestamps=True)
    assert not grounded
    assert rejected[0].match_score < 80


def test_grounding_without_timestamps() -> None:
    transcript = Transcript.model_validate(
        {"segments": [{"start": 0, "end": 0, "text": "Water boils at one hundred degrees."}]}
    )
    grounded, _ = ground(
        [_candidate("water boils at one hundred degrees")],
        SegmentIndex(transcript.segments),
        has_timestamps=False,
    )
    assert grounded[0].start is None
