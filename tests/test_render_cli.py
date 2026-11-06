from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from distillo import cli
from distillo.models import Transcript
from distillo.pipeline import Distiller, Options
from distillo.render import to_json, to_markdown

from .conftest import FIXTURES, FakeLLM

runner = CliRunner()


def test_markdown(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, _ = Distiller(fake_llm).run(lecture, Options(points=3))
    text = to_markdown(result, language="it")
    assert text.startswith("# lezione-statistica.srt\n")
    assert "**3 nozioni chiave** (su 5) · 02:40" in text
    assert "**Teorema di Bayes** `01:10` · _formula_" in text
    assert "> “" in text


def test_markdown_links_and_no_quotes(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, _ = Distiller(fake_llm).run(lecture, Options(points=3))
    text = to_markdown(result, quotes=False, link="https://youtu.be/abc?t={seconds}")
    assert "[01:10](https://youtu.be/abc?t=70)" in text
    assert "key points" in text
    assert "> " not in text


def test_json(fake_llm: FakeLLM, lecture: Transcript) -> None:
    result, _ = Distiller(fake_llm).run(lecture, Options(points=2))
    data = json.loads(to_json(result))
    assert data["requested"] == 2
    assert {"rank", "title", "statement", "kind", "start", "quote"} <= set(data["points"][0])


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(cli, "make_llm", lambda *args, **kwargs: llm)
    monkeypatch.setenv("DISTILLO_CACHE_DIR", str(tmp_path / "cache"))
    return llm


def test_cli_to_file(patched: FakeLLM, tmp_path: Path) -> None:
    out = tmp_path / "notes.md"
    srt = tmp_path / "copy.srt"
    result = runner.invoke(
        cli.app,
        [
            str(FIXTURES / "lezione-statistica.srt"),
            "-n",
            "2",
            "-o",
            str(out),
            "--save-transcript",
            str(srt),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").count("**") >= 4
    assert srt.read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> ")


def test_cli_json_stdout(patched: FakeLLM) -> None:
    result = runner.invoke(
        cli.app, [str(FIXTURES / "lezione-statistica.srt"), "--density", "low", "-f", "json"]
    )
    assert result.exit_code == 0, result.output
    assert '"requested": 3' in result.output


def test_cli_missing_file(patched: FakeLLM) -> None:
    result = runner.invoke(cli.app, ["missing.srt"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("distillo ")
