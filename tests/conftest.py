from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import pytest

from distillo.models import Transcript
from distillo.transcribe import load_transcript

FIXTURES = Path(__file__).parent / "fixtures"


class FakeLLM:
    """Deterministic stand-in for a model.

    Extraction: one point per transcript line that contains a keyword, quoting it
    verbatim, plus one invented point whose quote is not in the transcript.
    Ranking: Bayes first, the repeated Bayes line merged into it, the rest by
    importance, and one candidate deliberately left out to exercise the repair step.
    """

    name = "fake:test"

    KEYWORDS: ClassVar[dict[str, tuple[str, str, int]]] = {
        "Bayes": ("Teorema di Bayes", "formula", 5),
        "condizionata di A": ("Probabilità condizionata", "definition", 5),
        "indipendenti": ("Eventi indipendenti", "definition", 4),
        "errore più comune": ("Non scambiare A|B con B|A", "warning", 4),
        "mazzo di carte": ("Esempio con le carte", "example", 2),
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16_000
    ) -> Any:
        if "ranking" in schema["properties"]:
            self.calls.append("rank")
            return self._rank(prompt)
        self.calls.append("extract")
        return self._extract(prompt)

    def _extract(self, prompt: str) -> dict[str, Any]:
        text = prompt.split("<transcript>")[1].split("</transcript>")[0]
        points = []
        for line in text.strip().splitlines():
            sentence = re.sub(r"^\[[\d:]+\]\s*", "", line)
            for keyword, (title, kind, importance) in self.KEYWORDS.items():
                if keyword in sentence:
                    words = sentence.split()
                    points.append(
                        {
                            "title": title,
                            "statement": sentence,
                            "kind": kind,
                            "importance": importance,
                            "quote": " ".join(words[:14]),
                        }
                    )
        points.append(
            {
                "title": "Invented",
                "statement": "Something never said.",
                "kind": "fact",
                "importance": 5,
                "quote": "la varianza di una costante è sempre uguale a quarantadue unità",
            }
        )
        return {"points": points}

    def _rank(self, prompt: str) -> dict[str, Any]:
        rows = re.findall(r"^(c[\d.]+) \| (\d) \| ([^:]+):", prompt, flags=re.M)
        bayes = [cid for cid, _, title in rows if title == "Teorema di Bayes"]
        others = sorted(
            ((cid, int(imp)) for cid, imp, title in rows if title != "Teorema di Bayes"),
            key=lambda row: -row[1],
        )
        ranking = []
        if bayes:
            ranking.append({"id": bayes[0], "duplicates": bayes[1:]})
        ranking += [{"id": cid, "duplicates": []} for cid, _ in others[:-1]]
        ranking.append({"id": "c99.9", "duplicates": []})  # unknown id: must be ignored
        return {"ranking": ranking}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def lecture() -> Transcript:
    return load_transcript(FIXTURES / "lezione-statistica.srt")
