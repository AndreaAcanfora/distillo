"""Output formats."""

from __future__ import annotations

from .models import Distillation
from .timefmt import format_clock

_LABELS = {
    "en": {
        "points": "key points",
        "of": "of",
        "definition": "definition",
        "concept": "concept",
        "formula": "formula",
        "fact": "fact",
        "procedure": "procedure",
        "example": "example",
        "warning": "watch out",
    },
    "it": {
        "points": "nozioni chiave",
        "of": "su",
        "definition": "definizione",
        "concept": "concetto",
        "formula": "formula",
        "fact": "fatto",
        "procedure": "procedura",
        "example": "esempio",
        "warning": "attenzione",
    },
}


_ALIASES = {"english": "en", "italian": "it", "italiano": "it"}


def _labels(language: str | None) -> dict[str, str]:
    code = (language or "en").strip().lower()
    code = _ALIASES.get(code, code.split("-")[0])
    return _LABELS.get(code, _LABELS["en"])


def to_markdown(
    result: Distillation,
    *,
    language: str | None = None,
    quotes: bool = True,
    link: str | None = None,
) -> str:
    """Render as Markdown. ``link``, if given, is a URL template with ``{seconds}``."""
    labels = _labels(language or result.language)
    header = [f"# {result.source}", ""]
    meta = [f"**{len(result.points)} {labels['points']}**"]
    if result.total_available > len(result.points):
        meta[0] += f" ({labels['of']} {result.total_available})"
    if result.duration:
        meta.append(format_clock(result.duration))
    meta.append(f"`{result.model}`")
    header += [" · ".join(meta), ""]

    body: list[str] = []
    for n, point in enumerate(result.points, start=1):
        stamp = ""
        if point.start is not None:
            clock = format_clock(point.start)
            stamp = (
                f" [{clock}]({link.format(seconds=int(point.start))})" if link else f" `{clock}`"
            )
        body.append(f"{n}. **{point.title}**{stamp} · _{labels[point.kind.value]}_  ")
        body.append(f"   {point.statement}")
        if quotes:
            body += ["", f"   > “{point.quote.strip()}”"]
        body.append("")
    return "\n".join(header + body).rstrip() + "\n"


def to_json(result: Distillation) -> str:
    return result.model_dump_json(indent=2) + "\n"
