from __future__ import annotations

from pathlib import Path

import pytest

from distillo.cache import Cache
from distillo.models import Candidate, Kind, Segment, Transcript
from distillo.pipeline import Distiller, Options, repair_ranking, target_count

from .conftest import FakeLLM


def test_invented_points_are_dropped(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, report = Distiller(fake_llm).run(lecture, Options(points=10, order="rank"))
    assert all(p.title != "Invented" for p in result.points)
    assert len(report.rejected) == 1
    assert report.chunks == 1


def test_duplicates_merge_and_keep_first_mention(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, _ = Distiller(fake_llm).run(lecture, Options(points=10, order="rank"))
    top = result.points[0]
    assert top.title == "Teorema di Bayes"
    assert top.rank == 1
    assert top.merged_from  # the repetition at 02:18 was merged
    assert top.start == 70.0  # first time it was said
    assert sum(p.title == "Teorema di Bayes" for p in result.points) == 1


def test_fewer_points_are_a_prefix_of_more_points(fake_llm: FakeLLM, lecture: Transcript) -> None:
    distiller = Distiller(fake_llm)
    few, _ = distiller.run(lecture, Options(points=2, order="rank"))
    many, _ = distiller.run(lecture, Options(points=5, order="rank"))
    assert [p.title for p in few.points] == [p.title for p in many.points[:2]]
    assert many.total_available == 5


def test_time_order(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, _ = Distiller(fake_llm).run(lecture, Options(points=5))
    starts = [p.start for p in result.points]
    assert starts == sorted(starts)  # type: ignore[type-var]


def test_forgotten_candidates_are_appended(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, _ = Distiller(fake_llm).run(lecture, Options(points=50, order="rank"))
    # The fake ranker drops its last candidate; repair puts it back at the end.
    assert result.total_available == 5
    assert result.points[-1].title == "Esempio con le carte"


def test_cache_makes_reruns_free(tmp_path: Path, fake_llm: FakeLLM, lecture: Transcript) -> None:
    cache = Cache(tmp_path)
    Distiller(fake_llm, cache).run(lecture, Options(points=3))
    calls = len(fake_llm.calls)
    again, _ = Distiller(fake_llm, cache).run(lecture, Options(points=1))
    assert len(fake_llm.calls) == calls
    assert len(again.points) == 1


def test_disabled_cache_calls_again(tmp_path: Path, fake_llm: FakeLLM, lecture: Transcript) -> None:
    cache = Cache(tmp_path, enabled=False)
    Distiller(fake_llm, cache).run(lecture, Options(points=3))
    Distiller(fake_llm, cache).run(lecture, Options(points=3))
    assert fake_llm.calls.count("rank") == 2


def test_many_chunks(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, report = Distiller(fake_llm).run(lecture, Options(points=3, chunk_chars=300))
    assert report.chunks > 1
    assert len(result.points) == 3


@pytest.mark.parametrize(
    ("minutes", "density", "expected"),
    [(30, "low", 3), (30, "medium", 6), (30, "high", 12), (90, "medium", 18), (5, "high", 3)],
)
def test_density(minutes: int, density: str, expected: int) -> None:
    transcript = Transcript(segments=[Segment(start=0, end=minutes * 60, text="x")])
    assert target_count(transcript, Options(density=density)) == expected  # type: ignore[arg-type]


def test_explicit_points_win() -> None:
    transcript = Transcript(segments=[Segment(start=0, end=7200, text="x")])
    assert target_count(transcript, Options(points=4, density="high")) == 4


def test_repair_ranking_handles_bad_ids() -> None:
    candidates = [
        Candidate(id=f"c{i}", title="t", statement="s", kind=Kind.FACT, importance=i, quote="q")
        for i in (1, 2, 3)
    ]
    ranking = [
        {"id": "c1", "duplicates": ["c1", "nope"]},
        {"id": "c1", "duplicates": []},
        {"id": "ghost", "duplicates": []},
    ]
    assert repair_ranking(ranking, candidates) == [("c1", []), ("c3", []), ("c2", [])]


def test_provider_limits_shrink_chunks_and_parallelism(lecture: Transcript) -> None:
    class Limited(FakeLLM):
        max_chunk_chars = 300
        max_parallel = 1

    _, report = Distiller(Limited()).run(lecture, Options(points=3))
    assert report.chunks > 1


def test_long_statements_are_shortened_for_ranking() -> None:
    from distillo.pipeline import _shorten

    assert _shorten("word " * 100).endswith("…")
    assert len(_shorten("word " * 100)) == 220
    assert _shorten("short  text") == "short text"


def test_schema_slips_are_normalized() -> None:
    from distillo.pipeline import _normalize

    assert _normalize({"kind": "Theorem", "importance": 9}) == {"kind": "concept", "importance": 5}
    assert _normalize({"kind": " FORMULA ", "importance": "2"}) == {
        "kind": "formula",
        "importance": 2,
    }
    assert _normalize({"kind": "fact", "importance": None})["importance"] == 3
