"""Prompts and JSON schemas.

Bump PROMPT_VERSION whenever either changes: it is part of the cache key.
"""

from __future__ import annotations

from typing import Any

from .models import Kind

PROMPT_VERSION = "2"

EXTRACT_SYSTEM = """\
You turn recordings of lessons, lectures, courses and talks into study notes.
You read a transcript excerpt and pull out every distinct piece of knowledge that a \
student should retain from it.

What counts as a key point:
- definitions, core concepts, formulas, laws and theorems, important facts and figures,
  procedures and methods, illuminating examples, common mistakes and warnings;
- anything the speaker flags as important, repeats, or says will be on the exam.

What does not: greetings, logistics, jokes, digressions, filler, and questions from the \
audience unless the answer teaches something.

Rules:
- A point must teach something. Do not make a point that only says a topic is \
important, will be covered, or is the basis of the course: put that emphasis in the \
importance score of the point that explains the topic.
- One idea per point. Write the statement so it is understandable on its own, without \
the transcript: precise, concrete, at most three sentences. Keep numbers, names and \
formulas exact.
- Use only what is said in the transcript. Do not add knowledge of your own. You may fix \
obvious speech-to-text errors (a misheard technical term, for example).
- "quote" must be copied verbatim from the transcript excerpt: 6 to 30 consecutive words \
where the point is made, in the original language, without the [mm:ss] markers.
- importance: 5 = a central idea of the lesson or explicitly flagged as essential; \
4 = important; 3 = useful supporting detail; 2 = minor; 1 = trivia.
- If the excerpt teaches nothing, return an empty list."""

EXTRACT_PROMPT = """\
{context_line}Transcript excerpt {index} of {total}{span}:

<transcript>
{text}
</transcript>

Extract the key points. Write "title" and "statement" in {language}."""

RANK_SYSTEM = """\
You are an expert teacher preparing a revision sheet. You receive candidate key points \
extracted from consecutive parts of one recording. Some say the same thing in different \
words because the parts overlap or the speaker repeated themselves.

Do two things:
1. Merge duplicates: points that state the same idea form one group, and so do a \
vague point and a more precise point about the same thing. Choose as "id" the \
best formulated member and list the others in "duplicates".
2. Order the groups from most to least worth learning, for a student who may only read \
the first few. Rank first the ideas the rest of the lesson depends on, those the speaker \
stressed or tied to the exam, and general principles before the examples that illustrate \
them. Make the top of the list cover the main topics of the recording instead of \
clustering on one.

Every candidate id must appear exactly once, either as an "id" or inside "duplicates"."""

RANK_PROMPT = """\
{context_line}Candidates (id | importance given at extraction | title: statement):

{candidates}

Return the merged groups in ranked order."""


def _kind_values() -> list[str]:
    return [kind.value for kind in Kind]


EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "statement": {"type": "string"},
                    "kind": {"type": "string", "enum": _kind_values()},
                    "importance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "quote": {"type": "string"},
                },
                "required": ["title", "statement", "kind", "importance", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["points"],
    "additionalProperties": False,
}

RANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "duplicates": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "duplicates"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


def context_line(context: str | None) -> str:
    return f"About this recording: {context.strip()}\n\n" if context and context.strip() else ""
